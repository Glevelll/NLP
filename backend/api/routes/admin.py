"""Администрирование справочника и базы знаний: CRUD, импорт CSV/JSON, переиндексация."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_container, get_db, require_admin, require_editor, submit_task
from backend.core.models import GlossaryIn, GlossaryOut, KnowledgeIn, KnowledgeOut
from backend.data.examples_store import (
    count_rows,
    import_glossary,
    import_knowledge,
    kb_to_record,
    parse_csv_bytes,
)
from backend.db.models import Glossary, KnowledgeBase, User
from backend.services.container import AppContainer
from backend.worker.tasks import reindex_knowledge_task

router = APIRouter(prefix="/admin", tags=["admin"])


# ------------------------- справочник -------------------------
@router.get("/glossary", response_model=list[GlossaryOut])
async def list_glossary(kind: str | None = None, category: str | None = None, q: str | None = None,
                        limit: int = Query(100, le=1000), offset: int = 0, _: User = Depends(require_editor),
                        db: AsyncSession = Depends(get_db)) -> list[Glossary]:
    query = select(Glossary).order_by(Glossary.id)
    if kind:
        query = query.where(Glossary.kind == kind)
    if category:
        query = query.where(Glossary.category == category)
    if q:
        query = query.where(Glossary.term.ilike(f"%{q}%"))
    return list((await db.execute(query.limit(limit).offset(offset))).scalars())


@router.post("/glossary", response_model=GlossaryOut, status_code=status.HTTP_201_CREATED)
async def create_glossary(data: GlossaryIn, _: User = Depends(require_editor), db: AsyncSession = Depends(get_db),
                          c: AppContainer = Depends(get_container)) -> Glossary:
    row = Glossary(**data.model_dump())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    c.knowledge.glossary[row.kind].append({"id": row.id, **data.model_dump()})
    return row


@router.put("/glossary/{item_id}", response_model=GlossaryOut)
async def update_glossary(item_id: int, data: GlossaryIn, _: User = Depends(require_editor),
                          db: AsyncSession = Depends(get_db)) -> Glossary:
    row = await db.get(Glossary, item_id)
    if row is None:
        raise HTTPException(404, "Запись не найдена")
    for k, v in data.model_dump().items():
        setattr(row, k, v)
    await db.commit()
    await db.refresh(row)
    return row


@router.delete("/glossary/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_glossary(item_id: int, _: User = Depends(require_admin), db: AsyncSession = Depends(get_db)) -> None:
    row = await db.get(Glossary, item_id)
    if row is None:
        raise HTTPException(404, "Запись не найдена")
    await db.delete(row)
    await db.commit()


# ------------------------- база знаний -------------------------
@router.get("/knowledge", response_model=list[KnowledgeOut])
async def list_knowledge(category: str | None = None, platform: str | None = None, q: str | None = None,
                         limit: int = Query(50, le=1000), offset: int = 0, _: User = Depends(require_editor),
                         db: AsyncSession = Depends(get_db)) -> list[KnowledgeBase]:
    query = select(KnowledgeBase).order_by(KnowledgeBase.id.desc())
    if category:
        query = query.where(KnowledgeBase.category == category)
    if platform:
        query = query.where(KnowledgeBase.platform == platform)
    if q:
        query = query.where(KnowledgeBase.source_text.ilike(f"%{q}%"))
    return list((await db.execute(query.limit(limit).offset(offset))).scalars())


@router.post("/knowledge", response_model=KnowledgeOut, status_code=status.HTTP_201_CREATED)
async def create_knowledge(data: KnowledgeIn, _: User = Depends(require_editor), db: AsyncSession = Depends(get_db),
                           c: AppContainer = Depends(get_container)) -> KnowledgeBase:
    row = KnowledgeBase(**data.model_dump(), source="manual")
    db.add(row)
    await db.commit()
    await db.refresh(row)
    c.knowledge.add_example(kb_to_record(row))
    return row


@router.put("/knowledge/{item_id}", response_model=KnowledgeOut)
async def update_knowledge(item_id: int, data: KnowledgeIn, _: User = Depends(require_editor),
                           db: AsyncSession = Depends(get_db), c: AppContainer = Depends(get_container)
                           ) -> KnowledgeBase:
    row = await db.get(KnowledgeBase, item_id)
    if row is None:
        raise HTTPException(404, "Запись не найдена")
    for k, v in data.model_dump().items():
        setattr(row, k, v)
    await db.commit()
    await db.refresh(row)
    c.knowledge.remove_example(f"kb:{row.id}")
    c.knowledge.add_example(kb_to_record(row))
    return row


@router.delete("/knowledge/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge(item_id: int, _: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
                           c: AppContainer = Depends(get_container)) -> None:
    row = await db.get(KnowledgeBase, item_id)
    if row is None:
        raise HTTPException(404, "Запись не найдена")
    await db.delete(row)
    await db.commit()
    c.knowledge.remove_example(f"kb:{item_id}")


# ------------------------- импорт и индексы -------------------------
@router.post("/import/{kind}")
async def import_file(kind: str, file: UploadFile = File(...), _: User = Depends(require_editor),
                      db: AsyncSession = Depends(get_db), c: AppContainer = Depends(get_container)) -> dict:
    if kind not in ("knowledge", "glossary"):
        raise HTTPException(422, "kind должен быть knowledge или glossary")
    content = await file.read()
    if (file.filename or "").lower().endswith(".json"):
        records = json.loads(content.decode("utf-8"))
    else:
        records = parse_csv_bytes(content)
    added = await (import_knowledge(db, records, source="upload") if kind == "knowledge"
                   else import_glossary(db, records))
    await c.reload_knowledge()
    return {"kind": kind, "received": len(records), "added": added}


@router.post("/reindex")
async def reindex(background: bool = False, _: User = Depends(require_editor),
                  c: AppContainer = Depends(get_container)) -> dict:
    if background:
        return await submit_task(reindex_knowledge_task, c)
    return {"status": "ok", "stats": await c.reload_knowledge()}


@router.get("/stats")
async def stats(_: User = Depends(require_editor), db: AsyncSession = Depends(get_db),
                c: AppContainer = Depends(get_container)) -> dict:
    by_cat = (await db.execute(select(KnowledgeBase.category, func.count()).group_by(KnowledgeBase.category))).all()
    by_kind = (await db.execute(select(Glossary.kind, func.count()).group_by(Glossary.kind))).all()
    return {"rows": await count_rows(db), "knowledge_by_category": dict(by_cat), "glossary_by_kind": dict(by_kind),
            "index": c.knowledge.stats()}

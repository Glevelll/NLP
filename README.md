# 🛒 Генератор продающих описаний товаров

Интеллектуальная система для интернет-магазинов (тема №14 курса NLP-2026). Пользователь вводит характеристики
товара: название, категорию, ключевые особенности, цену, целевую аудиторию, бренд. Система генерирует продающее
описание для карточки с учётом стиля бренда и площадки (**Ozon**, **Wildberries**, собственный сайт).

Ключевые компоненты:

- **Мультиагентная система без фреймворков** (LangChain/LangGraph не используются): `DescriptionGenerator`,
  `Validator`, `Critic` (LLM-as-a-Judge), `Editor` и оркестратор `Supervisor`.
- **RAG с гибридным поиском**: BM25 + векторный поиск в Qdrant, объединение через Reciprocal Rank Fusion.
  Поиск идёт по 1200 эталонным описаниям; справочник содержит 737 записей.
- **LLM-бэкенды**: Ollama (`qwen2.5:7b/3b/1.5b`), HuggingFace Transformers (`Qwen2.5-0.5B-Instruct`), внешние API
  (OpenAI, Anthropic, OpenRouter, Groq) и офлайн-заглушка на правилах `mock`.
- **Оценка качества**: BLEU, chrF++, ROUGE-L, METEOR, BERTScore, покрытие ключевых особенностей, тональность,
  классификатор стиля. Выборки 20/50/100, режимы `baseline / +dict / +rag / +agents / +ensemble`, bootstrap-интервалы
  и тесты значимости.
- **Сквозное логирование**: лог рассуждений агентов (`reasoning_logs`) и трассировка каждого вызова LLM
  (`llm_calls`). Оба доступны через API и UI.
- **Остальная инфраструктура**: FastAPI, JWT с ролями, Celery + Redis, кэш, Human-in-the-Loop, Streamlit, Docker
  Compose (10 сервисов), Nginx, pytest (покрытие 95%), GitHub Actions.

Подробный отчёт об архитектуре, выборе моделей и результатах лежит в [docs/REPORT.md](docs/REPORT.md).

---

## Архитектура

```
                 ┌────────────── Streamlit UI (7 страниц) ──────────────┐
                 │                                                       │
   Nginx :8080 ──┤ /ui → frontend:8501            /api → app:8000        │
                 └───────────────────────────────────┬──────────────────┘
                                                     │ FastAPI (JWT, роли)
                         ┌───────────────────────────┴─────────────────────────┐
                         │                  GenerationService                  │
                         │  кэш → подготовка контекста → Supervisor → история  │
                         └──────┬───────────────┬──────────────────┬───────────┘
              справочник        │      RAG      │                  │  HITL-очередь
        (фразы, выгоды, запреты)│ BM25 + Qdrant │                  │  (низкая оценка)
                                ▼  + RRF        ▼                  ▼
   Supervisor:  DescriptionGenerator → Validator → Critic → (Editor → Validator → Critic)*
                                 │            │          │
                                 └──── LLMManager: Ollama → OpenAI/Anthropic/OpenRouter/Groq → HF → mock
                                              │ каждый вызов → llm_calls + logs/llm_calls.log
   PostgreSQL: users, knowledge_base, glossary, cache, uncertain_examples, feedback,
               generation_requests, reasoning_logs, llm_calls, evaluation_results, model_comparisons
   Celery (Redis): пакетная генерация, оценка/сравнение моделей, переиндексация, очистка кэша (beat)
```

**Цикл агентов.** Generator пишет черновик. Validator проверяет формальные правила: длину под площадку,
упоминание каждой ключевой особенности, запрещённые формулировки, соответствие цены, ссылки и эмодзи, тон бренда,
повторы, капс. Critic ставит оценки 0–10 по пяти критериям: привлекательность, убедительность, релевантность для
ЦА, фактическая точность, стиль. Если описание не прошло проверку или оценка ниже 7, Editor исправляет текст, и
цикл повторяется, но не больше `MAX_ITERATIONS=3` раз. Если результат остался неуверенным (оценка < 6, ошибки
валидации или запасная оценка Critic), он уходит в очередь Human-in-the-Loop.

## Режимы генерации

| Режим | Что используется |
|---|---|
| `baseline` | только LLM по характеристикам |
| `+dict` | + справочник: выгоды для особенностей, продающие фразы, шаблоны структуры, запрещённые формулировки |
| `+rag` | + гибридный поиск похожих товаров (примеры описаний в промпте) |
| `+agents` | + цикл Validator → Critic → Editor |
| `+ensemble` | несколько моделей генерируют кандидатов, Critic и Validator выбирают лучший |

---

## Быстрый старт (Docker, одна команда)

```bash
cp .env.example .env
```

```bash
docker compose up -d --build
```

После запуска доступны:

- UI: http://localhost:8080/ui/
- Swagger: http://localhost:8080/docs
- Health: http://localhost:8080/api/v1/health

При первом старте `ollama-init` скачивает модели из `OLLAMA_PULL_MODELS` (по умолчанию `qwen2.5:7b qwen2.5:3b`,
около 6.6 ГБ), а приложение само импортирует датасет из `data/import/` в PostgreSQL и индексирует его в Qdrant.
**Первый зарегистрированный пользователь получает роль `admin`.** Повысить пользователя можно через
`PATCH /api/v1/auth/users/{id}/role` или скриптом `scripts/create_admin.py`.

Облегчённый образ без torch собирается заметно быстрее (эмбеддинги hashing, BERTScore через эмбеддинги):
`INSTALL_ML=false docker compose up -d --build`. На слабой машине можно взять маленькую модель:
`OLLAMA_MODEL=qwen2.5:1.5b OLLAMA_PULL_MODELS=qwen2.5:1.5b`.

## Локальный запуск без Docker

Нужен Python 3.11–3.12.

```bash
poetry install --with dev
```

ML-зависимости (sentence-transformers, transformers, bert-score) ставятся отдельно:

```bash
poetry install --with dev,ml
```

Сгенерировать датасет (он уже лежит в репозитории):

```bash
python scripts/generate_data.py
```

Запустить API (SQLite, in-memory Qdrant, Celery в eager-режиме):

```bash
python run.py
```

Запустить UI:

```bash
streamlit run frontend/app.py
```

Без Ollama и API-ключей система работает на офлайн-заглушке `mock`, так что все сценарии можно пройти без GPU.
Для реальной генерации установите [Ollama](https://ollama.com) и выполните `ollama pull qwen2.5:7b`.

## Тесты

```bash
pytest --cov=backend
```

Результат: **88 тестов, покрытие 95%**. Внешние сервисы изолированы: Ollama и HTTP API подменены через
`httpx.MockTransport`, HF pipeline — через monkeypatch, Qdrant работает in-memory, БД — SQLite. Есть юнит-тесты
каждого агента, retrieval (BM25, вектор, RRF), метрик, логирования (`reasoning_logs`, `llm_calls`), кэша, HITL,
Celery-задач и скриптов, а также интеграционные тесты всех групп эндпоинтов через `TestClient`.

---

## Данные

`scripts/generate_data.py` собирает воспроизводимый синтетический русскоязычный датасет из каталога предметной
области (`scripts/data_catalog.py`: 12 категорий, по 8 типов товаров, 6 брендов и 12 особенностей с выгодами):

| Файл | Объём | Содержимое |
|---|---|---|
| `data/import/knowledge_base.json/.csv` | **1200** | пары «характеристики → описание» с разметкой категории, площадки и стиля |
| `data/import/glossary.json/.csv` | **737** | 528 продающих фраз, 144 выгоды, 50 запрещённых формулировок, 15 шаблонов |
| `data/import/test.json` | **200** | тестовые пары; товары не пересекаются с базой знаний |
| `data/import/batch_example.csv` | 10 | пример файла для пакетной генерации |

Импорт в БД без дубликатов выполняет `scripts/import_data.py` (он же вызывается автоматически при старте, если
таблицы пусты). Через UI или API можно дозагрузить CSV/JSON (`POST /api/v1/admin/import/{knowledge|glossary}`).

## Оценка качества

```bash
python scripts/run_eval.py --sample-sizes 20 50 100 --modes baseline +dict +rag +agents +ensemble --models ollama:qwen2.5:7b ollama:qwen2.5:3b --output results/
```

Скрипт делает стратифицированные по категориям выборки, прогоняет каждую модель в каждом режиме и считает метрики
с bootstrap 95%-интервалами. Значимость отличий от `baseline` проверяется парным bootstrap и парным t-тестом
(`significance.json`). Результаты сохраняются в JSON, CSV и Markdown, графики Plotly (bar и radar) — в HTML.
То же доступно через API (`POST /evaluate`, `GET /evaluate/compare`, `POST /evaluate/batch`,
`GET /evaluate/history`), Celery-задачи и страницу «Оценка качества» в UI.

### Результаты

> **Важно.** Таблица ниже получена на офлайн-заглушке `mock` (rule-based, без весов) с эмбеддингами `hashing-384`.
> Она подтверждает, что конвейер оценки работает от начала до конца и что компоненты (справочник, RAG) влияют на
> метрики. **Качество реальной LLM она не отражает.** Для отчёта по модели выполните:
>
> ```bash
> python scripts/run_eval.py --sample-sizes 20 50 100 --models ollama:qwen2.5:7b ollama:qwen2.5:3b --output results/qwen
> ```

Полные файлы лежат в [results/mock/](results/mock/): `eval_results.json`, `eval_summary.csv`, `significance.json`,
графики `eval_bar.html` и `eval_radar.html`.

| режим | n | BLEU | chrF++ | ROUGE-L | METEOR | BERTScore* | keyword cov. | style acc. | composite |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 0.243 | 0.452 | 0.413 | 0.423 | 0.711 | 0.965 | 0.450 | 0.535 |
| +dict | 20 | 0.410 | 0.624 | 0.640 | 0.649 | 0.823 | 1.000 | 0.450 | 0.691 |
| +rag | 20 | 0.413 | 0.618 | 0.626 | 0.630 | 0.814 | 1.000 | 0.850 | 0.684 |
| baseline | 50 | 0.229 | 0.446 | 0.395 | 0.405 | 0.707 | 0.936 | 0.360 | 0.520 |
| +dict | 50 | 0.404 | 0.626 | 0.632 | 0.640 | 0.824 | 1.000 | 0.320 | 0.688 |
| +rag | 50 | 0.428 | 0.638 | 0.642 | 0.645 | 0.827 | 1.000 | 0.720 | 0.697 |
| baseline | 100 | 0.233 | 0.446 | 0.400 | 0.406 | 0.704 | 0.929 | 0.370 | 0.520 |
| +dict | 100 | 0.407 | 0.627 | 0.631 | 0.642 | 0.825 | 1.000 | 0.340 | 0.689 |
| +rag | 100 | 0.435 | 0.642 | 0.647 | 0.652 | 0.829 | 1.000 | 0.640 | 0.701 |

\* BERTScore посчитан через косинусную близость hashing-эмбеддингов (torch не использовался); с установленной
ML-группой считается настоящий `bert-score`.

Наблюдения:

- **Справочник** даёт основной прирост n-граммных метрик (chrF++ +0.18 при n=100, p < 0.001 по парному bootstrap
  и t-тесту): выгоды для каждой особенности совпадают с формулировками эталона. Покрытие ключевых особенностей
  растёт с 0.93 до 1.00.
- **RAG** почти не меняет n-граммные метрики, зато сильно повышает точность стиля (0.34 → 0.64 при n=100): модель
  перенимает вступление и концовку примеров того же стиля и площадки.
- `+agents` и `+ensemble` на заглушке совпадают с `+rag` (полная таблица в `results/mock`): заглушка проходит
  Validator и получает оценку Critic ≥ 7 с первой итерации, поэтому Editor не вызывается, а варианты заглушки для
  ансамбля дают одинаковый текст. Эти режимы раскрываются на реальных моделях.
- С ростом выборки от 20 до 100 средние стабилизируются, 95%-интервал chrF++ для `+rag` при n=100 — [0.628, 0.658].

---

## API (префикс `/api/v1`)

| Группа | Эндпоинты |
|---|---|
| Аутентификация | `POST /auth/register`, `POST /auth/login`, `GET /auth/me`, `GET /auth/users`, `PATCH /auth/users/{id}/role` |
| Генерация | `POST /generate`, `POST /generate/multi`, `POST /generate/batch`, `POST /generate/batch/upload`, `POST /generate/ensemble` |
| Администрирование | CRUD `/admin/glossary`, `/admin/knowledge`, `POST /admin/import/{kind}`, `POST /admin/reindex`, `GET /admin/stats` |
| Оценка | `POST /evaluate`, `GET /evaluate/compare`, `POST /evaluate/batch`, `GET /evaluate/history`, `GET /evaluate/comparisons` |
| HITL | `GET /hitl/uncertain`, `POST /hitl/uncertain/{id}/approve`, `POST /hitl/uncertain/{id}/reject`, `POST /hitl/feedback`, `GET /hitl/stats` |
| Логи | `GET /requests`, `GET /requests/{id}`, **`GET /requests/{id}/reasoning_log`**, `GET /requests/{id}/llm_calls` |
| Мониторинг | `GET /health`, `GET /healthz`, `GET /monitoring/stats`, `POST /monitoring/cache/clear`, `GET /llm/providers`, `GET /tasks/{id}` |

Роли: `user` генерирует, смотрит свои запросы и оставляет отзывы. `editor` дополнительно работает со справочником,
базой знаний и очередью HITL. `admin` ещё удаляет записи, управляет ролями и очищает кэш.

Пример запроса:

```bash
curl -X POST localhost:8080/api/v1/generate -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"product": {"name": "Термокружка Thermo Go 450 мл", "category": "Дом и кухня", "features": ["двойные стенки", "держит тепло 8 часов"], "price": 1290, "audience": "путешественники", "style": "эмоциональный", "platform": "wildberries"}, "mode": "+agents"}'
```

## Интерфейс (Streamlit)

1. **Главная**: вход и регистрация, форма товара, выбор режима, площадки и модели. Показывает результат с
   оценками Critic, ошибками валидации, найденными примерами RAG, кандидатами ансамбля и формой отзыва.
2. **Пакетная обработка**: загрузка CSV, фоновая задача Celery с прогрессом, выгрузка результата.
3. **Справочник и база знаний**: просмотр, поиск, добавление и удаление записей, импорт CSV/JSON, переиндексация.
4. **Проверка редактором (HITL)**: подтверждение с правкой (пример попадает в базу знаний и сразу в индекс) или
   отклонение.
5. **Оценка качества**: запуск по моделям, режимам и размерам, таблица сравнения, bar, radar и зависимость от
   размера выборки.
6. **Логи рассуждений**: таймлайн шагов агентов, вход и выход каждого шага, все вызовы LLM (промпт, ответ,
   задержка, токены, хэши).
7. **Мониторинг**: статусы сервисов и LLM-провайдеров, статистика кэша и вызовов.

## Структура проекта

```
backend/
  agents/       base, generator, validator, critic, editor, supervisor, prompts
  api/          main.py, deps.py, routes/ (auth, generate, admin, evaluate, hitl, requests, monitoring)
  core/         config (pydantic-settings), constants, models (Pydantic), metrics, exceptions
  data/         llm_manager, api_llm, hf_model, mock_llm, embeddings, knowledge_base, examples_store, cache
  db/           models (SQLAlchemy), session (async), database
  evaluation/   evaluator, metrics_calculator, significance, report
  hitl/         active_learning, feedback, updater
  retrieval/    bm25_index, vector_index (Qdrant), hybrid_retriever (RRF), reranker
  services/     generation (режимы, ансамбль, кэш), container (DI)
  utils/        llm_client (Ollama), llm_logger, reasoning_logger, tracing, logging, security, text_processing
  worker/       celery_app (beat), tasks
frontend/       app.py, pages/, utils/ (api_client, constants, session)
scripts/        generate_data, data_catalog, import_data, create_admin, run_eval
tests/          unit/, integration/
infrastructure/ Dockerfile, docker-compose.yml, nginx.conf
data/import/    датасет;   results/ — результаты оценки;   docs/REPORT.md — отчёт
```

## Ограничения

- Датасет синтетический: описания собраны из шаблонов, поэтому n-граммные метрики завышены для моделей, которые
  перенимают структуру примеров (RAG). Особенности выбираются в пределах категории, так что у части товаров
  сочетания «тип товара + особенность» неестественны.
- Заглушка `mock` детерминирована, поэтому ансамбль из её вариантов не даёт прироста. Ансамбль имеет смысл на
  разных реальных моделях.
- Без torch BERTScore заменяется косинусной близостью эмбеддингов (это указано в `bertscore_backend` результата).

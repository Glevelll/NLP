"""Отчёты по оценке: JSON, CSV, Markdown-таблица и графики Plotly (bar + radar)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

SUMMARY_METRICS = ["bleu", "chrf_pp", "rouge_l", "meteor", "bertscore", "keyword_coverage", "sentiment",
                   "style_accuracy", "composite"]


def summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        if "metrics" not in r:
            continue
        row = {"model": r["model_name"], "mode": r["mode"], "sample_size": r["sample_size"],
               "time_s": r["evaluation_time"]}
        row.update({k: r["metrics"].get(k) for k in SUMMARY_METRICS})
        rows.append(row)
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_нет результатов_"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = [f"{v:.3f}" if isinstance(v, float) and h not in ("time_s",) else str(v if v is not None else "—")
                 for h, v in row.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(results: list[dict[str, Any]], out_dir: Path | str, prefix: str = "eval",
                 extra: dict[str, Any] | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / f"{prefix}_results.json", "csv": out / f"{prefix}_summary.csv",
             "md": out / f"{prefix}_summary.md"}
    slim = [{k: v for k, v in r.items() if k != "per_sample"} for r in results]
    paths["json"].write_text(json.dumps({"results": slim, **(extra or {})}, ensure_ascii=False, indent=2,
                                        default=str), encoding="utf-8")
    rows = summary_rows(results)
    if rows:
        with paths["csv"].open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    paths["md"].write_text(markdown_table(rows) + "\n", encoding="utf-8")
    return paths


def bar_figure(rows: list[dict[str, Any]], metric: str = "composite"):
    import plotly.graph_objects as go

    fig = go.Figure()
    for size in sorted({r["sample_size"] for r in rows}):
        sub = [r for r in rows if r["sample_size"] == size]
        fig.add_bar(name=f"n={size}", x=[f"{r['model']} {r['mode']}" for r in sub], y=[r.get(metric) for r in sub])
    fig.update_layout(title=f"{metric} по моделям и режимам", barmode="group", yaxis_title=metric,
                      template="plotly_white")
    return fig


def radar_figure(rows: list[dict[str, Any]], metrics: list[str] | None = None):
    import plotly.graph_objects as go

    metrics = metrics or ["bleu", "chrf_pp", "rouge_l", "meteor", "bertscore", "keyword_coverage"]
    fig = go.Figure()
    for r in rows:
        values = [r.get(m) or 0 for m in metrics]
        fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=metrics + metrics[:1], fill="toself",
                                      name=f"{r['model']} {r['mode']} n={r['sample_size']}"))
    fig.update_layout(title="Профиль метрик", polar={"radialaxis": {"range": [0, 1]}}, template="plotly_white")
    return fig


def save_charts(results: list[dict[str, Any]], out_dir: Path | str, prefix: str = "eval") -> list[Path]:
    rows = summary_rows(results)
    if not rows:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    largest = max(r["sample_size"] for r in rows)
    figs = {"bar": bar_figure(rows), "radar": radar_figure([r for r in rows if r["sample_size"] == largest])}
    saved = []
    for name, fig in figs.items():
        html = out / f"{prefix}_{name}.html"
        fig.write_html(html, include_plotlyjs="cdn")
        saved.append(html)
        try:  # PNG требует kaleido — необязательная зависимость
            png = out / f"{prefix}_{name}.png"
            fig.write_image(png, width=1100, height=650)
            saved.append(png)
        except Exception:
            pass
    return saved

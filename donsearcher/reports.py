#!/usr/bin/env python3
"""CSV / JSONL report writers. Pure stdlib, no heavy dependencies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- events_meta.jsonl: сайдкар для раздельного запуска стадий -----------------
# Сервер A (только YOLO, --skip-vlm --images-schema 1) пишет кропы + этот файл;
# сервер B (только VLM) по нему связывает OCR-результаты с событиями стрима.
# Первая строка — запись о прогоне (type=run), дальше — по строке на событие.

# Поля события, известные ДО VLM (детекторная сторона event_row) — схема "full".
FULL_META_FIELDS = [
    "event_id", "video_name", "start_time", "end_time", "duration",
    "first_frame", "last_frame", "detections_count",
    "best_detection_confidence", "best_detection_time", "best_detection_frame",
    "best_detection_score", "base_box_json", "padded_box_json", "crop_path",
]


def build_run_record(meta_mode: str, video_name: str, streamer: str, platform: str,
                     stream_date: str, **full_fields: Any) -> dict[str, Any]:
    """Первая строка events_meta.jsonl — провенанс прогона.

    minimal — только то, чего нет в видео (дата/платформа/стример);
    full — плюс произвольные поля прогона (run_name, fps, conf и т.п.).
    """
    rec: dict[str, Any] = {
        "type": "run",
        "meta": meta_mode,
        "video": video_name,
        "streamer": streamer,
        "platform": platform,
        "stream_date": stream_date,
    }
    if meta_mode == "full":
        rec.update(full_fields)
    return rec


def build_events_meta_rows(event_rows: list[dict[str, Any]], meta_mode: str) -> list[dict[str, Any]]:
    """Строки событий для events_meta.jsonl.

    minimal — сокращённые ключи для экономии: id (event_id), crop (имя файла
    кропа), t (время появления доната в стриме, start_time).
    full — все детекторные поля event_row (без VLM-результатов).
    """
    if meta_mode == "minimal":
        return [{"id": r["event_id"], "crop": r["crop_path"], "t": r["start_time"]}
                for r in event_rows]
    return [{k: r.get(k, "") for k in FULL_META_FIELDS} for r in event_rows]


def write_events_meta(path: Path, run_record: dict[str, Any],
                      event_rows: list[dict[str, Any]], meta_mode: str) -> None:
    write_jsonl(path, [run_record] + build_events_meta_rows(event_rows, meta_mode))


def meta_from_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Обратное к build_events_meta_rows: строка события из events_meta.jsonl ->
    словарь детекторных полей (FULL_META_FIELDS) для build_event_rows.

    full-запись содержит поля под своими именами; minimal — сокращённые
    id/crop/t, остальные поля остаются пустыми.
    """
    if "event_id" in rec:  # full
        return {k: rec.get(k, "") for k in FULL_META_FIELDS}
    meta = {k: "" for k in FULL_META_FIELDS}
    meta["event_id"] = rec.get("id", "")
    meta["crop_path"] = rec.get("crop", "")
    meta["start_time"] = rec.get("t", "")
    return meta

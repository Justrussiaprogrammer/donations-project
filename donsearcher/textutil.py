#!/usr/bin/env python3
"""Small string / time / JSON formatting helpers.

No heavy dependencies — safe to import from anywhere without pulling in
cv2/numpy/ultralytics.
"""

from __future__ import annotations

import json
import re
from typing import Any


def safe_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ._-]+", "_", text).strip("_")
    return text[:max_len] or "run"


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_img_size(value: str) -> int | list[int]:
    """Разбор --img-size: '640' -> 640, '576,1024' / '576x1024' -> [576, 1024].

    Пара — это (высота, ширина) для ultralytics imgsz; нужна прямоугольным
    статичным OpenVINO-экспортам (например 576x1024 под кадр 16:9), которым
    квадратный леттербокс по одному числу не подходит.
    """
    parts = [p for p in re.split(r"[x,×]", str(value).strip()) if p]
    try:
        dims = [int(p) for p in parts]
    except ValueError:
        dims = []
    if len(dims) == 1:
        return dims[0]
    if len(dims) == 2:
        return dims
    raise ValueError(
        f"--img-size: ожидается '640' или '576,1024' (высота,ширина), получено {value!r}"
    )


def unpack_images_schema(schema: int) -> tuple[bool, bool, bool]:
    """Разбор битовой маски --images-schema -> (crops, annotated, original).

    Бит 0 (1) — best_crops, бит 1 (2) — annotated_frames, бит 2 (4) — original_frames.
    7 — сохранять всё (прежнее поведение по умолчанию), 0 — ничего (бывший
    --no-save-images), 1 — только кропы (YOLO-стадия, кропы уезжают на отдельный
    VLM-сервер), 5 — кропы + оригинальные кадры и т.п.
    """
    if not 0 <= schema <= 7:
        raise ValueError(f"--images-schema должен быть в диапазоне 0..7, получено {schema}")
    return bool(schema & 1), bool(schema & 2), bool(schema & 4)

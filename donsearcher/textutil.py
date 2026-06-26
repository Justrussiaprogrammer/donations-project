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

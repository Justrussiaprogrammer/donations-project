#!/usr/bin/env python3
"""VLM prompt, the OpenAI-compatible chat call, and response parsing.

This module is self-contained on purpose: it imports cv2/numpy/requests but NOT
ultralytics, so eval/iteration scripts (eval_vlm.py, prompt_iter.py) can use
``call_vlm_for_image`` / ``load_prompt`` without paying the torch import cost.

Prompt text lives in prompts/*.txt (single source of truth); ``load_prompt``
resolves a version name ("v7") or an arbitrary .txt path.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests


def image_bgr_to_data_url(image_bgr: np.ndarray, ext: str = ".png") -> str:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image crop")

    ext = ext.lower()
    mime = "image/png"
    encode_ext = ".png"
    if ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
        encode_ext = ".jpg"

    ok, buf = cv2.imencode(encode_ext, image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image crop")

    data = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Parse a JSON object. Falls back to extracting the first {...} block."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON object not found in model response")
    return json.loads(text[start : end + 1])


# -----------------------------
# VLM prompt and call
# -----------------------------

# Промпты лежат в prompts/*.txt (единственный источник; в коде текста промпта нет).
# Версия по умолчанию — лучшая по бенчмарку на test/gt (см. CLAUDE.md).
DEFAULT_PROMPT_VERSION = "v7"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(spec: str = DEFAULT_PROMPT_VERSION) -> tuple[str, str]:
    """Загрузить промпт по спецификации: имя версии ('v7' -> prompts/v7.txt)
    или путь к произвольному .txt-файлу. Возвращает (текст, метка версии).
    """
    p = Path(spec).expanduser()
    looks_like_path = p.suffix == ".txt" or "/" in spec or "\\" in spec
    path = p if looks_like_path else PROMPTS_DIR / f"{spec}.txt"
    if not path.is_file():
        available = sorted(f.stem for f in PROMPTS_DIR.glob("*.txt")) if PROMPTS_DIR.is_dir() else []
        raise FileNotFoundError(
            f"Промпт не найден: {path}\n"
            f"Доступные версии в {PROMPTS_DIR}: {', '.join(available) or '(нет)'}; "
            f"либо укажите путь к .txt-файлу."
        )
    return path.read_text(encoding="utf-8"), path.stem


def call_vlm_for_image(
    crop_bgr: np.ndarray,
    server_url: str = "http://127.0.0.1:8081/v1/chat/completions",
    model_name: str = "Qwen3-VL-8b-Q4-K-M",
    timeout_sec: int = 300,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    prompt: Optional[str] = None,
    retries: int = 2,
) -> tuple[str, Optional[dict[str, Any]], str]:
    """
    Returns: raw_text, parsed_json_or_none, error_message.
    No semantic repair is performed here.
    Network/server failures are retried up to `retries` times with backoff;
    JSON parse failures are not retried (the model answer is deterministic at temp 0).
    prompt=None -> the default prompt version is loaded from prompts/.
    """
    if prompt is None:
        prompt, _ = load_prompt()

    donation_schema = {
        "type": "object",
        "properties": {
            "donor": {
                "type": ["string", "null"],
                "description": "Имя донатера из верхнего заголовка"
            },
            "amount": {
                "type": ["number", "null"],
                "description": "Сумма доната (число, без валюты и знаков препинания)"
            },
            "currency": {
                "type": ["string", "null"],
                "description": "Валюта: RUB, USD, EUR, $, € и т.п. или null"
            },
            "message": {
                "type": ["string", "null"],
                "description": "Текст сообщения ниже заголовка (включая ссылки)"
            },
            "fee_covered": {
                "type": "boolean",
                "description": "true, если слева от имени есть сердце (донатер покрыл комиссию)"
            },
            "needs_review": {
                "type": "boolean",
                "description": "true, если что-то нечитаемо или неоднозначно"
            }
        },
        "required": ["donor", "amount", "currency", "message", "fee_covered", "needs_review"],
        "additionalProperties": False
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_bgr_to_data_url(crop_bgr)},
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "donation_extraction",
                "schema": donation_schema,
                "strict": True
            }
        }
    }

    last_error = ""
    raw_text = ""
    for attempt in range(retries + 1):
        try:
            response = requests.post(server_url, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"]
            break
        except Exception as exc:
            last_error = f"request_failed: {type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(1.0)
    else:
        return "", None, last_error

    try:
        parsed = extract_json_from_text(raw_text)
        return raw_text, parsed, ""
    except Exception as exc:
        return raw_text, None, f"json_parse_failed: {type(exc).__name__}: {exc}"

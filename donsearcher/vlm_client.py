#!/usr/bin/env python3
"""VLM prompt, the OpenAI-compatible chat call, and response parsing.

This module is self-contained on purpose: it imports cv2/numpy/requests but NOT
ultralytics, so eval/iteration scripts (test_qwen_ocr.py, prompt_iter.py) can use
``call_vlm_for_image`` / ``VLM_PROMPT`` without paying the torch import cost.
"""

from __future__ import annotations

import base64
import json
import time
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

VLM_PROMPT_VERSION = "v7"

VLM_PROMPT = """
Ты извлекаешь донат из ОДНОГО crop изображения стрима.

Верни ТОЛЬКО валидный JSON.
Без markdown.
Без пояснений.
Без текста до или после JSON.

## ОБЫЧНО ИЗОБРАЖЕНИЕ СОДЕРЖИТ:
1. ВЕРХНИЙ крупный заголовок доната.
2. Иногда заголовок визуально разбит на две строки.
3. НИЖНИЙ текст сообщения, обычно меньшим шрифтом.
4. Иногда СЛЕВА от имени донатера маленькое СЕРДЦЕ (♥/❤) — пометка, что донатер покрыл комиссию перевода.

## ОЧЕНЬ ВАЖНЫЕ ПРАВИЛА LAYOUT:
- donor находится в ВЕРХНЕМ заголовке доната, до тире/дефиса перед суммой.
- amount находится в ВЕРХНЕМ заголовке доната, после donor и тире/дефиса.
- message находится НИЖЕ заголовка и суммы, обычно меньшим шрифтом.
- Никогда не используй нижнее сообщение как donor, если сверху есть заголовок вида "donor - amount".
- Не включай donor и amount в message.

## ГРАНИЦА ДОНАТА — что переписывать, а что НЕТ:
- Плашка доната — это отдельная панель (подложка/рамка) поверх видео. Извлекай текст ТОЛЬКО с этой панели: заголовок донатера и его сообщение.
- Crop часто захватывает кусок самого видео/стрима ПОД и ВОКРУГ плашки. Текст, который принадлежит фоновому видео, — НЕ часть доната. Полностью игнорируй его, не переписывай ни в одно поле.
- НЕ включай в message текст фона, например: субтитры и реплики из видео, название ролика, имя канала, элементы плеера (таймкоды вида "9:06 / 12:50", "HD", "FULL HD", стрелки ▶), водяные знаки, любой интерфейс.
- Признаки фона: текст лежит за краем панели доната, перекрывается ею, набран другим шрифтом/стилем, обрывается на полуслове у края crop, не связан по смыслу с сообщением доната.
- Бери ровно тот текст, что находится ВНУТРИ панели доната. Если строка за её пределами — не трогай.
- Если не уверен, относится строка к донату или к фону — НЕ включай её и поставь needs_review = true.

## ПРАВИЛО СЕРДЦА (покрытие комиссии):
- Если СЛЕВА от имени донатера есть символ сердца (♥, ❤ и похожие) — поставь fee_covered = true.
- Если сердца нет — fee_covered = false.
- Сердце НЕ часть имени: никогда не добавляй символ сердца в donor, верни только чистое имя.

## ПРАВИЛА СУММЫ:
- Если написано "200!", amount = 200, currency = null.
- Если написано "200 RUB!", amount = 200, currency = "RUB".
- Если сумма разбита на строки, объедини её.
  Пример:
  строка 1: "Великая Рогатая Крыса - 1"
  строка 2: "500 RUB!"
  означает:
  donor = "Великая Рогатая Крыса"
  amount = 1500
  currency = "RUB"
- Если видишь "1 500", "1.500" или "1,500" как сумму, трактуй это как 1500.
- Не теряй старший разряд суммы.

## ПРАВИЛА ЧТЕНИЯ ТЕКСТА:
- message — это ВЕСЬ текст сообщения доната под заголовком, переписанный дословно (но только текст самой плашки, см. «ГРАНИЦА ДОНАТА»).
- Если под заголовком есть любой текст, ссылка (http, https, t.me, youtu.be и т.п.) или их сочетание — обязательно перепиши его ПОЛНОСТЬЮ в message.
- Ссылка — это тоже сообщение. Никогда не выбрасывай message и не ставь null только потому, что это ссылка или «не похоже на сообщение».
- message = null ТОЛЬКО если под заголовком реально нет никакого текста доната.
- Если сообщение доната занимает несколько строк, СКЛЕЙ их в одну сплошную строку через пробел. НИКОГДА не вставляй символ переноса "\\n" и переводы строк — message всегда одна строка без переносов.
- Извлекай только то, что реально видно.
- Не выдумывай продолжение.
- Не повторяй фразы несколько раз.
- Сохраняй мат, сленг, ошибки, имена и разговорную лексику как есть.
- Не цензурируй.
- Не перефразируй.
- Не исправляй грамматику по смыслу.
- Не заменяй русские буквы похожими латинскими и наоборот.

## КРИТЕРИИ needs_review = true:
- Текст сильно размыт или обрезан краем изображения.
- Неясно, где заканчивается имя и начинается сумма.
- Есть несколько интерпретаций layout.
- Сумма содержит нестандартные символы.
- Непонятно, относится ли часть текста к донату или к фоновому видео.

Верни JSON строго по этой схеме:
{
  "donor": string или null,
  "amount": number или null,
  "currency": string или null,
  "message": string или null,
  "fee_covered": boolean,
  "needs_review": boolean
}
"""


def call_vlm_for_image(
    crop_bgr: np.ndarray,
    server_url: str = "http://127.0.0.1:8081/v1/chat/completions",
    model_name: str = "Qwen3-VL-8b-Q4-K-M",
    timeout_sec: int = 300,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    prompt: str = VLM_PROMPT,
    retries: int = 2,
) -> tuple[str, Optional[dict[str, Any]], str]:
    """
    Returns: raw_text, parsed_json_or_none, error_message.
    No semantic repair is performed here.
    Network/server failures are retried up to `retries` times with backoff;
    JSON parse failures are not retried (the model answer is deterministic at temp 0).
    """

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

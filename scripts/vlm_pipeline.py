#!/usr/bin/env python3
"""
Donation extraction pipeline v5 - memory-optimised fork (vlm_pipeline_old).

Changes vs vlm_pipeline.py:
  1. CandidateCrop stores file paths instead of numpy arrays.
     Images are written to a tmp dir immediately during YOLO stage,
     evicted candidates' files are deleted right away.
     This keeps RAM proportional to keep_top_candidates * events_alive_at_once,
     not to the whole video.
  2. find_matching_event only searches active events (end_sec within gap).
     Old events are moved to a closed list once per frame, keeping the
     search O(active) instead of O(all).

Everything else is identical to the original.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import queue
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
from ultralytics import YOLO

import os

# Использовать только при запуске YOLO в сборке openvino
os.environ["OPENVINO_DEVICE"] = "GPU"

import time


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CandidateCrop:
    event_id: int
    video_name: str
    frame_idx: int
    timestamp_sec: float
    confidence: float
    class_id: int
    base_box: tuple[int, int, int, int]
    padded_box: tuple[int, int, int, int]
    crop_path: Path
    annotated_frame_path: Optional[Path]  # None when --no-save-images
    original_frame_path: Optional[Path]   # None when --no-save-images
    score: float


@dataclass
class DonationEvent:
    event_id: int
    video_name: str
    start_sec: float
    end_sec: float
    first_frame: int
    last_frame: int
    last_box: tuple[int, int, int, int]
    detections_count: int = 0
    best_confidence: float = 0.0
    best_timestamp_sec: float = 0.0
    best_frame_idx: int = 0
    candidates: list[CandidateCrop] = field(default_factory=list)


# -----------------------------
# Basic helpers
# -----------------------------

def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(v, high))


def safe_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ._-]+", "_", text).strip("_")
    return text[:max_len] or "run"


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        clamp(x1 - pad_x, 0, width - 1),
        clamp(y1 - pad_y, 0, height - 1),
        clamp(x2 + pad_x, 0, width - 1),
        clamp(y2 + pad_y, 0, height - 1),
    )


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_distance_norm(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    width: int,
    height: int,
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    diag = max(1.0, (width ** 2 + height ** 2) ** 0.5)
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / diag


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


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# -----------------------------
# VLM prompt and call
# -----------------------------

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

## ОЧЕНЬ ВАЖНЫЕ ПРАВИЛА LAYOUT:
- donor находится в ВЕРХНЕМ заголовке доната, до тире/дефиса перед суммой.
- amount находится в ВЕРХНЕМ заголовке доната, после donor и тире/дефиса.
- message находится НИЖЕ заголовка и суммы, обычно меньшим шрифтом.
- Никогда не используй нижнее сообщение как donor, если сверху есть заголовок вида "donor - amount".
- Не включай donor и amount в message.

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

Верни JSON строго по этой схеме:
{
  "donor": string или null,
  "amount": number или null,
  "currency": string или null,
  "message": string или null,
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
    prompt: str = VLM_PROMPT
) -> tuple[str, Optional[dict[str, Any]], str]:
    """
    Returns: raw_text, parsed_json_or_none, error_message.
    No semantic repair is performed here.
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
                "description": "Текст сообщения ниже заголовка"
            },
            "needs_review": {
                "type": "boolean",
                "description": "true, если что-то нечитаемо или неоднозначно"
            }
        },
        "required": ["donor", "amount", "currency", "message", "needs_review"],
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

    try:
        response = requests.post(server_url, json=payload, timeout=timeout_sec)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        return "", None, f"request_failed: {type(exc).__name__}: {exc}"

    try:
        parsed = extract_json_from_text(raw_text)
        return raw_text, parsed, ""
    except Exception as exc:
        return raw_text, None, f"json_parse_failed: {type(exc).__name__}: {exc}"


# -----------------------------
# Event merging
# -----------------------------

def find_matching_event(
    events: list[DonationEvent],
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    iou_thr: float,
    center_thr: float,
    current_frame_idx: int,
) -> Optional[DonationEvent]:
    """Search only the provided event list (caller is responsible for passing active events).

    Same-frame events (last_frame == current_frame_idx) can only match by IoU.
    Center-distance matching is intentionally disabled for same-frame events because
    it would incorrectly merge two separate donations visible simultaneously.
    """
    best: Optional[DonationEvent] = None
    best_score = -999.0

    for ev in events:
        iou = box_iou(ev.last_box, box)
        dist = center_distance_norm(ev.last_box, box, frame_w, frame_h)

        # print(f"iou: {round(iou, 2)}, dist: {round(dist, 2)}, razn: {round(iou - dist, 2)}, best: {ev.last_box}, {box}, {frame_w}, {frame_h}")

        if ev.last_frame == current_frame_idx:
            # Same frame: require real spatial overlap — no center-distance fallback.
            if iou < iou_thr:
                continue
        else:
            # Different frame: IoU or center distance (temporal tracking of a moved box).
            if iou < iou_thr and dist > center_thr:
                continue

        score = iou - dist
        if score > best_score:
            best_score = score
            best = ev

    return best


def _delete_candidate_files(candidate: CandidateCrop) -> None:
    for p in (candidate.crop_path, candidate.annotated_frame_path, candidate.original_frame_path):
        if p is None:
            continue
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def add_candidate(event: DonationEvent, candidate: CandidateCrop, max_candidates: int) -> None:
    event.candidates.append(candidate)
    event.candidates.sort(key=lambda c: c.score, reverse=True)
    if len(event.candidates) > max_candidates:
        _delete_candidate_files(event.candidates[max_candidates])
        event.candidates = event.candidates[:max_candidates]


# -----------------------------
# CSV helpers
# -----------------------------

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


def build_totals_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}

    for row in event_rows:
        if not row.get("parsed_ok"):
            continue
        if row.get("needs_review") is True or row.get("best_detection_confidence") < 0.5:
            continue

        amount = row.get("amount")
        if amount in (None, ""):
            continue

        try:
            amount_float = float(amount)
        except Exception:
            continue

        currency = row.get("currency") or "NO_CURRENCY"
        currency = str(currency)

        if currency not in totals:
            totals[currency] = {
                "currency": currency,
                "events_count": 0,
                "amount_count": 0,
                "amount_sum": 0.0,
            }

        totals[currency]["events_count"] += 1
        totals[currency]["amount_count"] += 1
        totals[currency]["amount_sum"] += amount_float

    out = []
    for currency, data in sorted(totals.items()):
        amount_sum = data["amount_sum"]
        data["amount_sum"] = int(amount_sum) if amount_sum.is_integer() else round(amount_sum, 2)
        out.append(data)
    return out


# -----------------------------
# Per-event VLM processing (runs in worker thread)
# -----------------------------

def _process_event_vlm(
    ev: DonationEvent,
    crops_dir: Path,
    frames_dir: Path,
    original_frames_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process one closed event through VLM. Returns (event_row, jsonl_row)."""
    best_det = max(ev.candidates, key=lambda c: c.score, default=None)

    if best_det is None:
        event_row: dict[str, Any] = {
            "event_id": ev.event_id,
            "video_name": ev.video_name,
            "start_time": seconds_to_timestamp(ev.start_sec),
            "end_time": seconds_to_timestamp(ev.end_sec),
            "duration_sec": round(ev.end_sec - ev.start_sec, 3),
            "first_frame": ev.first_frame,
            "last_frame": ev.last_frame,
            "detections_count": ev.detections_count,
            "best_detection_confidence": round(ev.best_confidence, 4),
            "best_detection_time": "",
            "best_detection_frame": "",
            "best_detection_score": "",
            "base_box_json": "",
            "padded_box_json": "",
            "crop_path": "",
            "parsed_ok": False,
            "donor": "",
            "amount": "",
            "currency": "",
            "message": "",
            "needs_review": True,
            "raw_model_response": "",
            "model_error": "no_candidate_crop",
        }
        jsonl_row: dict[str, Any] = {
            "file_name": "",
            "parsed_ok": False,
            "error": "no_candidate_crop",
            "donor": "",
            "amount": "",
            "currency": "",
            "message": "",
            "needs_review": True,
        }
        return event_row, jsonl_row

    detection_time = seconds_to_timestamp(best_det.timestamp_sec)
    detection_time_safe = detection_time.replace(":", "-")

    if not args.no_save_images:
        best_crop_path = crops_dir / f"event_{ev.event_id:04d}_best_detector_crop.png"
        best_frame_path = frames_dir / f"event_{ev.event_id:04d}_best_detector_frame.jpg"
        best_original_frame_path = original_frames_dir / f"event_{ev.event_id:04d}_{detection_time_safe}.png"
        shutil.copy2(str(best_det.crop_path), str(best_crop_path))
        if best_det.annotated_frame_path:
            shutil.copy2(str(best_det.annotated_frame_path), str(best_frame_path))
        if best_det.original_frame_path:
            shutil.copy2(str(best_det.original_frame_path), str(best_original_frame_path))
        saved_crop_path = str(best_crop_path.name)
    else:
        saved_crop_path = ""

    crop_bgr = cv2.imread(str(best_det.crop_path))

    raw_text = ""
    parsed: Optional[dict[str, Any]] = None
    model_error = ""

    if args.skip_vlm or crop_bgr is None:
        model_error = "vlm_skipped" if args.skip_vlm else "crop_load_failed"
    else:
        raw_text, parsed, model_error = call_vlm_for_image(
            crop_bgr=crop_bgr,
            server_url=args.vlm_server_url,
            model_name=args.vlm_model,
            timeout_sec=args.vlm_timeout,
            max_tokens=args.vlm_max_tokens,
            temperature=args.vlm_temperature,
        )

    parsed_ok = parsed is not None
    parsed = parsed or {}

    event_row = {
        "event_id": ev.event_id,
        "video_name": ev.video_name,
        "start_time": seconds_to_timestamp(ev.start_sec),
        "end_time": seconds_to_timestamp(ev.end_sec),
        "duration_sec": round(ev.end_sec - ev.start_sec, 3),
        "first_frame": ev.first_frame,
        "last_frame": ev.last_frame,
        "detections_count": ev.detections_count,
        "best_detection_confidence": round(ev.best_confidence, 4),
        "best_detection_time": detection_time,
        "best_detection_frame": best_det.frame_idx,
        "best_detection_score": round(best_det.score, 4),
        "base_box_json": json_dumps_compact(best_det.base_box),
        "padded_box_json": json_dumps_compact(best_det.padded_box),
        "crop_path": saved_crop_path,
        "parsed_ok": parsed_ok,
        "donor": parsed.get("donor", ""),
        "amount": parsed.get("amount", ""),
        "currency": parsed.get("currency", ""),
        "message": parsed.get("message", ""),
        "needs_review": parsed.get("needs_review", True),
        "raw_model_response": raw_text,
        "model_error": model_error,
    }
    jsonl_row = {
        "file_name": saved_crop_path,
        "parsed_ok": parsed_ok,
        "error": model_error,
        "donor": parsed.get("donor", ""),
        "amount": parsed.get("amount", ""),
        "currency": parsed.get("currency", ""),
        "message": parsed.get("message", ""),
        "needs_review": parsed.get("needs_review", True),
    }
    return event_row, jsonl_row


def _vlm_worker(
    vlm_queue: "queue.Queue[Optional[DonationEvent]]",
    results: list[tuple[dict[str, Any], dict[str, Any]]],
    results_lock: threading.Lock,
    crops_dir: Path,
    frames_dir: Path,
    original_frames_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Thread worker: drain vlm_queue and call VLM for each event."""
    while True:
        ev = vlm_queue.get()
        if ev is None:  # sentinel — no more events
            vlm_queue.task_done()
            break
        try:
            event_row, jsonl_row = _process_event_vlm(
                ev, crops_dir, frames_dir, original_frames_dir, args
            )
            with results_lock:
                results.append((event_row, jsonl_row))
            print(f"[VLM] event {ev.event_id}: {seconds_to_timestamp(ev.start_sec)} "
                  f"donor={event_row.get('donor') or '?'} amount={event_row.get('amount') or '?'}")
        except Exception as exc:
            print(f"[VLM] event {ev.event_id} failed: {exc}")
        finally:
            vlm_queue.task_done()


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).expanduser().resolve()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_dir / model_path

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = project_dir / video_path

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    run_name = args.run_name or f"{safe_filename(video_path.stem)}_vlm_v5_run"
    output_dir = (project_dir / args.output_dir / run_name).resolve()
    events_dir = output_dir / "events"
    crops_dir = events_dir / "best_crops"
    frames_dir = events_dir / "annotated_frames"
    original_frames_dir = events_dir / "original_frames"
    tmp_dir = output_dir / "_tmp_candidates"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_images:
        events_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        original_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project:      {project_dir}")
    print(f"YOLO model:   {model_path}")
    print(f"Video:        {video_path}")
    print(f"Output:       {output_dir}")
    print(f"VLM server:   {args.vlm_server_url}")
    print(f"VLM model:    {args.vlm_model}")

    vlm_queue: queue.Queue = queue.Queue()
    vlm_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if not args.sequential:
        vlm_results_lock = threading.Lock()
        vlm_thread = threading.Thread(
            target=_vlm_worker,
            args=(vlm_queue, vlm_results, vlm_results_lock, crops_dir, frames_dir, original_frames_dir, args),
            daemon=True,
            name="vlm-worker",
        )
        vlm_thread.start()
        print("VLM worker thread started - will process events as they close.")

    model = YOLO(str(model_path), task="detect")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"FPS: {fps:.3f}, frames: {total_frames}, frame_step: {args.frame_step}]\n")

    all_events: list[DonationEvent] = []
    active_events: list[DonationEvent] = []

    frame_idx = 0
    processed_frames = 0
    raw_detections_count = 0
    detection_counter = 0  # unique ID for tmp file naming

    print("Starting YOLO detection stage...")
    stage_start_time = time.time()
    batch_start_time = stage_start_time

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        processed_frames += 1
        timestamp_sec = frame_idx / fps
        h, w = frame.shape[:2]

        still_active = []
        for ev in active_events:
            if timestamp_sec - ev.end_sec <= args.event_gap_sec:
                still_active.append(ev)
            else:
                for evicted in ev.candidates[1:]:
                    _delete_candidate_files(evicted)
                ev.candidates = ev.candidates[:1]
                if not args.sequential:
                    vlm_queue.put(ev)
        active_events = still_active

        result = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.img_size,
            device=args.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            annotated = result.plot()

            for box in boxes:
                cls_id = int(box.cls[0].cpu().item())
                conf = float(box.conf[0].cpu().item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

                base_box = (
                    clamp(x1, 0, w - 1),
                    clamp(y1, 0, h - 1),
                    clamp(x2, 0, w - 1),
                    clamp(y2, 0, h - 1),
                )
                padded_box = expand_box(base_box, w, h, args.padding_x, args.padding_y)
                px1, py1, px2, py2 = padded_box
                crop = frame[py1:py2, px1:px2].copy()

                bx1, by1, bx2, by2 = base_box

                # Area: larger donation = more pixels for VLM to read (0–0.3)
                area_norm = box_area(base_box) / max(1, w * h)
                area_score = min(area_norm * 10.0, 0.3)

                # Sharpness: blurry crops give bad VLM results (0–0.4).
                # Laplacian variance: ~20 for motion-blurred, ~500+ for sharp text.
                # Threshold of 500 may need tuning for your stream resolution.
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 0.4)

                # Clipping penalty: padding was cut off by the frame edge (-0.15)
                clipped = (
                    (px1 == 0 and bx1 > args.padding_x) or
                    (py1 == 0 and by1 > args.padding_y) or
                    (px2 == w - 1 and bx2 < w - 1 - args.padding_x) or
                    (py2 == h - 1 and by2 < h - 1 - args.padding_y)
                )
                clip_penalty = -0.15 if clipped else 0.0

                candidate_score = conf + area_score + sharpness_score + clip_penalty

                det_id = detection_counter
                detection_counter += 1
                tmp_crop_path = tmp_dir / f"det{det_id:07d}_crop.png"
                cv2.imwrite(str(tmp_crop_path), crop)
                if not args.no_save_images:
                    tmp_ann_path = tmp_dir / f"det{det_id:07d}_ann.jpg"
                    tmp_orig_path = tmp_dir / f"det{det_id:07d}_orig.png"
                    cv2.imwrite(str(tmp_ann_path), annotated)
                    cv2.imwrite(str(tmp_orig_path), frame)
                else:
                    tmp_ann_path = None
                    tmp_orig_path = None

                matched = find_matching_event(
                    events=active_events,
                    box=base_box,
                    frame_w=w,
                    frame_h=h,
                    iou_thr=args.event_iou_thr,
                    center_thr=args.event_center_thr,
                    current_frame_idx=frame_idx,
                )

                if matched is None:
                    matched = DonationEvent(
                        event_id=len(all_events) + 1,
                        video_name=video_path.name,
                        start_sec=timestamp_sec,
                        end_sec=timestamp_sec,
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        last_box=base_box,
                    )
                    all_events.append(matched)
                    active_events.append(matched)

                matched.end_sec = timestamp_sec
                matched.last_frame = frame_idx
                matched.last_box = base_box
                matched.detections_count += 1

                if conf >= matched.best_confidence:
                    matched.best_confidence = conf
                    matched.best_timestamp_sec = timestamp_sec
                    matched.best_frame_idx = frame_idx

                candidate = CandidateCrop(
                    event_id=matched.event_id,
                    video_name=video_path.name,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    confidence=conf,
                    class_id=cls_id,
                    base_box=base_box,
                    padded_box=padded_box,
                    crop_path=tmp_crop_path,
                    annotated_frame_path=tmp_ann_path,
                    original_frame_path=tmp_orig_path,
                    score=candidate_score,
                )
                add_candidate(matched, candidate, args.keep_top_candidates)
                raw_detections_count += 1

        if args.max_processed_frames and processed_frames >= args.max_processed_frames:
            break

        if processed_frames % 100 == 0:
            batch_elapsed = time.time() - batch_start_time
            print(
                f"Processed sampled frames: {processed_frames}, "
                f"source frame: {frame_idx}, detections: {raw_detections_count}, events: {len(all_events)}, "
                f"batch time: {round(batch_elapsed, 2)}с"
            )
            batch_start_time = time.time()

        frame_idx += 1

    cap.release()
    yolo_elapsed = round(time.time() - stage_start_time, 2)
    print(f"\nYOLO stage done in {yolo_elapsed}s.")
    print(f"Sampled frames: {processed_frames}, detections: {raw_detections_count}, events: {len(all_events)}")

    # Clean up non-best candidates for events still active at end of video.
    for ev in active_events:
        for evicted in ev.candidates[1:]:
            _delete_candidate_files(evicted)
        ev.candidates = ev.candidates[:1]

    vlm_elapsed: Optional[float] = None

    if args.sequential:
        print(f"\nStarting VLM stage (sequential)...")
        vlm_start = time.time()
        for ev in all_events:
            try:
                event_row, jsonl_row = _process_event_vlm(
                    ev, crops_dir, frames_dir, original_frames_dir, args
                )
                vlm_results.append((event_row, jsonl_row))
                print(f"[VLM] event {ev.event_id}/{len(all_events)}: "
                      f"donor={event_row.get('donor') or '?'} amount={event_row.get('amount') or '?'}")
            except Exception as exc:
                print(f"[VLM] event {ev.event_id} failed: {exc}")
        vlm_elapsed = round(time.time() - vlm_start, 2)
        print(f"VLM stage done in {vlm_elapsed}s.")
    else:
        # Flush remaining active events to worker thread, then wait for it to finish.
        for ev in active_events:
            vlm_queue.put(ev)
        vlm_queue.put(None)
        print("Waiting for VLM worker to finish...")
        vlm_thread.join()
        vlm_results.sort(key=lambda r: r[0]["event_id"])

    event_rows = [r[0] for r in vlm_results]
    jsonl_rows = [r[1] for r in vlm_results]

    shutil.rmtree(tmp_dir, ignore_errors=True)

    events_csv = output_dir / "events_summary.csv"
    totals_csv = output_dir / "totals_by_currency.csv"
    jsonl_path = output_dir / "donations.jsonl"
    metadata_json = output_dir / "run_metadata.json"

    totals_rows = build_totals_rows(event_rows)

    write_csv(events_csv, event_rows)
    write_csv(totals_csv, totals_rows)
    write_jsonl(jsonl_path, jsonl_rows)

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "project_dir": str(project_dir),
                "yolo_model": str(model_path),
                "video": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "frame_step": args.frame_step,
                "sampled_frames_processed": processed_frames,
                "raw_detections": raw_detections_count,
                "events": len(all_events),
                "yolo_conf": args.conf,
                "yolo_img_size": args.img_size,
                "padding_x": args.padding_x,
                "padding_y": args.padding_y,
                "event_gap_sec": args.event_gap_sec,
                "event_iou_thr": args.event_iou_thr,
                "event_center_thr": args.event_center_thr,
                "vlm_server_url": args.vlm_server_url,
                "vlm_model": args.vlm_model,
                "vlm_timeout": args.vlm_timeout,
                "vlm_max_tokens": args.vlm_max_tokens,
                "vlm_temperature": args.vlm_temperature,
                "skip_vlm": args.skip_vlm,
                "sequential": args.sequential,
                "no_save_images": args.no_save_images,
                "yolo_elapsed_sec": yolo_elapsed,
                "vlm_elapsed_sec": vlm_elapsed,
                "outputs": {
                    "events_summary_csv": str(events_csv),
                    "totals_by_currency_csv": str(totals_csv),
                    "donations_jsonl": str(jsonl_path),
                    "events_dir": "" if args.no_save_images else str(events_dir)
                }
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone.")
    print(f"  YOLO stage:        {yolo_elapsed}s")
    if vlm_elapsed is not None:
        print(f"  VLM stage:         {vlm_elapsed}s")
    else:
        print(f"  VLM stage:         ran in parallel with YOLO (no isolated time)")
    print(f"  Sampled frames:    {processed_frames}")
    print(f"  Raw detections:    {raw_detections_count}")
    print(f"  Donation events:   {len(all_events)}")
    print(f"Events summary:      {events_csv}")
    print(f"Totals by currency:  {totals_csv}")
    print(f"Raw JSONL:           {jsonl_path}")
    if not args.no_save_images:
        print(f"Best crops/frames:   {events_dir}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO + local Qwen3-VL donation extraction pipeline v5 (memory-optimised)"
    )

    p.add_argument("--project-dir", default=".")
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--video", default="video_tests/test.mp4")
    p.add_argument("--output-dir", default="vlm_runs")
    p.add_argument("--run-name", default="")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--frame-step", type=int, default=10)
    p.add_argument("--padding-x", type=int, default=20)
    p.add_argument("--padding-y", type=int, default=12)

    p.add_argument("--event-gap-sec", type=float, default=3)
    p.add_argument("--event-iou-thr", type=float, default=0.25)
    p.add_argument("--event-center-thr", type=float, default=0.05)
    p.add_argument("--keep-top-candidates", type=int, default=3)
    p.add_argument("--max-processed-frames", type=int, default=0)

    p.add_argument("--vlm-server-url", default="http://127.0.0.1:8081/v1/chat/completions")
    p.add_argument("--vlm-model", default="Qwen3-VL")
    p.add_argument("--vlm-timeout", type=int, default=300)
    p.add_argument("--vlm-max-tokens", type=int, default=1024)
    p.add_argument("--vlm-temperature", type=float, default=0.0)
    p.add_argument("--skip-vlm", action="store_true",
                   help="Run YOLO/event grouping only, no VLM calls")
    p.add_argument("--sequential", action="store_true",
                   help="Run YOLO fully, then VLM fully (no overlap). "
                        "Gives clean per-stage timing for benchmarking")
    p.add_argument("--no-save-images", action="store_true",
                   help="Do not save any image files (crops, frames). "
                        "Output is CSV/JSONL only")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    if args.keep_top_candidates <= 0:
        raise ValueError("--keep-top-candidates must be >= 1")
    if args.vlm_max_tokens <= 0:
        raise ValueError("--vlm-max-tokens must be >= 1")
    if args.vlm_timeout <= 0:
        raise ValueError("--vlm-timeout must be >= 1")

    run_pipeline(args)


if __name__ == "__main__":
    main()

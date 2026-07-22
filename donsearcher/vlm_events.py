#!/usr/bin/env python3
"""Per-event VLM processing — runs in the worker thread (parallel mode) or inline
(sequential mode). Picks the best crop of a closed event, calls the VLM, and builds
the event_row / jsonl_row records. Shared by both the Python and C++ engines
(fast_pipeline.py reuses ``_vlm_worker`` / ``_process_event_vlm``).
"""

from __future__ import annotations

import argparse
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

import cv2

from .events import DonationEvent
from .reports import FULL_META_FIELDS
from .textutil import json_dumps_compact, seconds_to_timestamp, unpack_images_schema
from .vlm_client import call_vlm_for_image


def build_event_rows(
    meta: dict[str, Any],
    raw_text: str,
    parsed: Optional[dict[str, Any]],
    model_error: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Собрать (event_row, jsonl_row) из детекторных метаданных и ответа VLM.

    meta — словарь с полями FULL_META_FIELDS (отсутствующие -> ""). Единая точка
    формата строк: её используют и обычный пайплайн (_process_event_vlm), и
    отдельная VLM-стадия по готовым кропам (scripts/vlm_stage.py) — вывод обеих
    схем запуска совпадает.
    """
    parsed_ok = parsed is not None
    parsed = parsed or {}
    event_row: dict[str, Any] = {k: meta.get(k, "") for k in FULL_META_FIELDS}
    event_row.update({
        "parsed_ok": parsed_ok,
        "donor": parsed.get("donor", ""),
        "amount": parsed.get("amount", ""),
        "currency": parsed.get("currency", ""),
        "message": parsed.get("message", ""),
        "fee_covered": parsed.get("fee_covered", False),
        "needs_review": parsed.get("needs_review", True),
        "raw_model_response": raw_text,
        "model_error": model_error,
        "is_duplicate": False,
        "duplicate_of_event_id": "",
    })
    jsonl_row: dict[str, Any] = {
        "file_name": meta.get("crop_path", ""),
        "parsed_ok": parsed_ok,
        "error": model_error,
        "donor": parsed.get("donor", ""),
        "amount": parsed.get("amount", ""),
        "currency": parsed.get("currency", ""),
        "message": parsed.get("message", ""),
        "fee_covered": parsed.get("fee_covered", False),
        "needs_review": parsed.get("needs_review", True),
        "is_duplicate": False,
        "duplicate_of_event_id": "",
    }
    return event_row, jsonl_row


def _event_meta(ev: DonationEvent, best_det: Any = None, crop_name: str = "") -> dict[str, Any]:
    """Детекторные поля события (FULL_META_FIELDS) для build_event_rows."""
    meta: dict[str, Any] = {
        "event_id": ev.event_id,
        "video_name": ev.video_name,
        "start_time": seconds_to_timestamp(ev.start_sec),
        "end_time": seconds_to_timestamp(ev.end_sec),
        "duration": seconds_to_timestamp(ev.end_sec - ev.start_sec),
        "first_frame": ev.first_frame,
        "last_frame": ev.last_frame,
        "detections_count": ev.detections_count,
        "best_detection_confidence": round(ev.best_confidence, 4),
        "best_detection_time": "",
        "best_detection_frame": "",
        "best_detection_score": "",
        "base_box_json": "",
        "padded_box_json": "",
        "crop_path": crop_name,
    }
    if best_det is not None:
        meta["best_detection_time"] = seconds_to_timestamp(best_det.timestamp_sec)
        meta["best_detection_frame"] = best_det.frame_idx
        meta["best_detection_score"] = round(best_det.score, 4)
        meta["base_box_json"] = json_dumps_compact(best_det.base_box)
        meta["padded_box_json"] = json_dumps_compact(best_det.padded_box)
    return meta


def _make_error_rows(ev: DonationEvent, error: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rows for an event that produced no VLM result, so it still appears in CSV/JSONL."""
    return build_event_rows(_event_meta(ev), "", None, error)


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
        return _make_error_rows(ev, "no_candidate_crop")

    detection_time = seconds_to_timestamp(best_det.timestamp_sec)
    detection_time_safe = detection_time.replace(":", "-")

    save_crops, save_annotated, save_original = unpack_images_schema(args.images_schema)
    saved_crop_path = ""
    if save_crops:
        best_crop_path = crops_dir / f"event_{ev.event_id:04d}_best_detector_crop.png"
        shutil.copy2(str(best_det.crop_path), str(best_crop_path))
        saved_crop_path = str(best_crop_path.name)
    if save_annotated and best_det.annotated_frame_path:
        best_frame_path = frames_dir / f"event_{ev.event_id:04d}_best_detector_frame.jpg"
        shutil.copy2(str(best_det.annotated_frame_path), str(best_frame_path))
    if save_original and best_det.original_frame_path:
        best_original_frame_path = original_frames_dir / f"event_{ev.event_id:04d}_{detection_time_safe}.png"
        shutil.copy2(str(best_det.original_frame_path), str(best_original_frame_path))

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
            prompt=getattr(args, "vlm_prompt_text", None),
            retries=args.vlm_retries,
        )

    meta = _event_meta(ev, best_det, saved_crop_path)
    return build_event_rows(meta, raw_text, parsed, model_error)


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
            with results_lock:
                results.append(_make_error_rows(ev, f"worker_exception: {type(exc).__name__}: {exc}"))
        finally:
            vlm_queue.task_done()

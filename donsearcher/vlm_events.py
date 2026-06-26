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
from .textutil import json_dumps_compact, seconds_to_timestamp
from .vlm_client import call_vlm_for_image


def _make_error_rows(ev: DonationEvent, error: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rows for an event that produced no VLM result, so it still appears in CSV/JSONL."""
    event_row: dict[str, Any] = {
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
        "crop_path": "",
        "parsed_ok": False,
        "donor": "",
        "amount": "",
        "currency": "",
        "message": "",
        "fee_covered": False,
        "needs_review": True,
        "raw_model_response": "",
        "model_error": error,
        "is_duplicate": False,
        "duplicate_of_event_id": "",
    }
    jsonl_row: dict[str, Any] = {
        "file_name": "",
        "parsed_ok": False,
        "error": error,
        "donor": "",
        "amount": "",
        "currency": "",
        "message": "",
        "fee_covered": False,
        "needs_review": True,
        "is_duplicate": False,
        "duplicate_of_event_id": "",
    }
    return event_row, jsonl_row


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
            retries=args.vlm_retries,
        )

    parsed_ok = parsed is not None
    parsed = parsed or {}

    event_row = {
        "event_id": ev.event_id,
        "video_name": ev.video_name,
        "start_time": seconds_to_timestamp(ev.start_sec),
        "end_time": seconds_to_timestamp(ev.end_sec),
        "duration": seconds_to_timestamp(ev.end_sec - ev.start_sec),
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
        "fee_covered": parsed.get("fee_covered", False),
        "needs_review": parsed.get("needs_review", True),
        "raw_model_response": raw_text,
        "model_error": model_error,
        "is_duplicate": False,
        "duplicate_of_event_id": "",
    }
    jsonl_row = {
        "file_name": saved_crop_path,
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

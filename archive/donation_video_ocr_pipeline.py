#!/usr/bin/env python3
"""
Prototype pipeline for stream donation extraction:
video -> YOLO detector -> padded crops -> PaddleOCR -> event merging -> CSV files for manual review.

Expected project structure by default:
~/donation_project/
  models/donation_detector_yolo26n_v1.pt
  video_tests/test.mp4
  ocr_runs/

Example:
python donation_video_ocr_pipeline.py \
  --project-dir ~/donation_project \
  --model models/donation_detector_yolo26n_v1.pt \
  --video video_tests/2_donate.mp4 \
  --device cpu \
  --frame-step 10 \
  --ocr-lang ru
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class DetectionRecord:
    detection_id: int
    event_id: int
    video_name: str
    frame_idx: int
    timestamp_sec: float
    confidence: float
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    ocr_text: str
    ocr_avg_conf: float
    crop_path: str = ""


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
    best_ocr_text: str = ""
    best_ocr_avg_conf: float = 0.0
    best_crop_path: str = ""
    best_frame_path: str = ""
    ocr_variants: list[str] = field(default_factory=list)


# -----------------------------
# Geometry and text helpers
# -----------------------------

def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(v, high))


def expand_box(box: tuple[int, int, int, int], width: int, height: int, pad_x: int, pad_y: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        clamp(x1 - pad_x, 0, width - 1),
        clamp(y1 - pad_y, 0, height - 1),
        clamp(x2 + pad_x, 0, width - 1),
        clamp(y2 + pad_y, 0, height - 1),
    )


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance_norm(a: tuple[int, int, int, int], b: tuple[int, int, int, int], width: int, height: int) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    diag = max(1.0, (width ** 2 + height ** 2) ** 0.5)
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / diag


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-zа-яё$€₽.,:;!?+\- ]+", "", text, flags=re.IGNORECASE)
    return text.strip()


def text_similarity(a: str, b: str) -> float:
    a_norm, b_norm = normalize_text(a), normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ._-]+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


# -----------------------------
# OCR helpers with PaddleOCR v2/v3 compatibility
# -----------------------------

def create_ocr(lang: str, use_ocr: bool):
    if not use_ocr:
        return None
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR is not installed or failed to import. Install it with: pip install paddlepaddle paddleocr"
        ) from exc

    # PaddleOCR 3.x often uses use_textline_orientation; older 2.x uses use_angle_cls.
    init_attempts = [
        {"use_textline_orientation": False, "lang": lang},
        {"use_angle_cls": True, "lang": lang, "show_log": False},
        {"use_angle_cls": True, "lang": lang},
        {"lang": lang},
    ]
    last_exc = None
    for kwargs in init_attempts:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to initialize PaddleOCR for lang={lang!r}: {last_exc}")


def collect_text_scores(obj: Any) -> list[tuple[str, float]]:
    """Recursively extract (text, confidence) from PaddleOCR v2/v3-ish outputs."""
    pairs: list[tuple[str, float]] = []

    if obj is None:
        return pairs

    # PaddleOCR v3 result objects may expose json-like methods/properties.
    for attr in ("json", "to_json"):
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                value = value() if callable(value) else value
                pairs.extend(collect_text_scores(value))
                if pairs:
                    return pairs
            except Exception:
                pass

    if isinstance(obj, dict):
        # Common v3 keys.
        if "rec_texts" in obj:
            texts = obj.get("rec_texts") or []
            scores = obj.get("rec_scores") or []
            for i, text in enumerate(texts):
                if text:
                    score = float(scores[i]) if i < len(scores) else 0.0
                    pairs.append((str(text), score))

        # Other possible keys.
        for text_key in ("text", "rec_text", "transcription"):
            if text_key in obj and isinstance(obj[text_key], str):
                score = float(obj.get("score", obj.get("confidence", 0.0)) or 0.0)
                pairs.append((obj[text_key], score))

        for value in obj.values():
            if isinstance(value, (list, tuple, dict)):
                pairs.extend(collect_text_scores(value))
        return pairs

    if isinstance(obj, tuple):
        # PaddleOCR v2 line usually: [box, (text, conf)] or (text, conf)
        if len(obj) >= 2 and isinstance(obj[0], str):
            try:
                pairs.append((obj[0], float(obj[1])))
                return pairs
            except Exception:
                pass
        if len(obj) >= 2 and isinstance(obj[1], tuple) and len(obj[1]) >= 2 and isinstance(obj[1][0], str):
            try:
                pairs.append((obj[1][0], float(obj[1][1])))
                return pairs
            except Exception:
                pass
        for value in obj:
            pairs.extend(collect_text_scores(value))
        return pairs

    if isinstance(obj, list):
        for value in obj:
            pairs.extend(collect_text_scores(value))
        return pairs

    return pairs


def run_ocr_on_crop(ocr: Any, crop_bgr: np.ndarray, tmp_path: Path) -> tuple[str, float]:
    if ocr is None or crop_bgr.size == 0:
        return "", 0.0

    cv2.imwrite(str(tmp_path), crop_bgr)

    result = None
    # PaddleOCR v3 style.
    if hasattr(ocr, "predict"):
        try:
            result = ocr.predict(str(tmp_path))
        except NotImplementedError:
            result = None
        except Exception:
            result = None

    # PaddleOCR v2 style.
    if result is None and hasattr(ocr, "ocr"):
        try:
            result = ocr.ocr(str(tmp_path), cls=True)
        except TypeError:
            result = ocr.ocr(str(tmp_path))

    pairs = collect_text_scores(result)
    texts = [t.strip() for t, _ in pairs if t and t.strip()]
    scores = [s for t, s in pairs if t and t.strip()]

    raw_text = " | ".join(texts)
    avg_conf = float(sum(scores) / len(scores)) if scores else 0.0
    return raw_text, avg_conf


# -----------------------------
# Optional rough parser
# -----------------------------

def parse_donation_text(raw_text: str) -> dict[str, str]:
    """Very rough parser. It is intentionally conservative; manual review still matters."""
    text = raw_text.replace("\n", " | ").strip()
    amount = ""
    currency = ""

    patterns = [
        r"(?P<amount>\d+[\d\s]*(?:[.,]\d{1,2})?)\s*(?P<currency>₽|руб\.?|rub|р|usd|eur|€|\$|bits?|бит(?:ов|ы)?|мемкоин\w*)",
        r"(?P<currency>₽|€|\$)\s*(?P<amount>\d+[\d\s]*(?:[.,]\d{1,2})?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            amount = re.sub(r"\s+", "", m.group("amount"))
            currency = m.group("currency")
            break

    # Donor/message extraction is intentionally left mostly blank because streamer layouts differ.
    return {
        "donor_guess": "",
        "amount_guess": amount,
        "currency_guess": currency,
        "message_guess": "",
    }


# -----------------------------
# Event merge logic
# -----------------------------

def find_matching_event(
    events: list[DonationEvent],
    timestamp_sec: float,
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    ocr_text: str,
    max_gap_sec: float,
    iou_thr: float,
    center_thr: float,
    text_split_similarity_thr: float,
) -> Optional[DonationEvent]:
    candidates: list[tuple[float, DonationEvent]] = []

    for ev in events:
        if timestamp_sec - ev.end_sec > max_gap_sec:
            continue

        iou = box_iou(ev.last_box, box)
        center_dist = center_distance_norm(ev.last_box, box, frame_w, frame_h)
        spatial_ok = iou >= iou_thr or center_dist <= center_thr
        if not spatial_ok:
            continue

        # If OCR is strong and clearly different, do not merge into old event.
        # This protects against two different donations appearing in the same overlay area.
        if ev.best_ocr_text and ocr_text and len(normalize_text(ev.best_ocr_text)) > 10 and len(normalize_text(ocr_text)) > 10:
            sim = text_similarity(ev.best_ocr_text, ocr_text)
            if sim < text_split_similarity_thr and (timestamp_sec - ev.start_sec) > 2.0:
                continue

        score = iou - center_dist
        candidates.append((score, ev))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def update_event_best_files(
    event: DonationEvent,
    crop: np.ndarray,
    annotated_frame: np.ndarray,
    events_dir: Path,
):
    crop_path = events_dir / f"event_{event.event_id:04d}_best_crop.png"
    frame_path = events_dir / f"event_{event.event_id:04d}_best_frame.jpg"
    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(frame_path), annotated_frame)
    event.best_crop_path = str(crop_path)
    event.best_frame_path = str(frame_path)


def should_replace_best(event: DonationEvent, confidence: float, ocr_text: str, ocr_avg_conf: float) -> bool:
    old_score = event.best_confidence + 0.010 * len(event.best_ocr_text) + 0.20 * event.best_ocr_avg_conf
    new_score = confidence + 0.010 * len(ocr_text) + 0.20 * ocr_avg_conf
    return new_score > old_score


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).expanduser().resolve()
    model_path = (project_dir / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    video_path = (project_dir / args.video).resolve() if not Path(args.video).is_absolute() else Path(args.video).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    run_name = args.run_name or f"{safe_filename(video_path.stem)}_ocr_run"
    output_dir = (project_dir / args.output_dir / run_name).resolve()
    events_dir = output_dir / "events"
    raw_crops_dir = output_dir / "raw_crops"
    tmp_dir = output_dir / "tmp"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    events_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw_crops:
        raw_crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project: {project_dir}")
    print(f"Model:   {model_path}")
    print(f"Video:   {video_path}")
    print(f"Output:  {output_dir}")

    model = YOLO(str(model_path))
    ocr = create_ocr(args.ocr_lang, use_ocr=not args.no_ocr)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"FPS: {fps:.3f}, frames: {total_frames}, frame_step: {args.frame_step}")

    events: list[DonationEvent] = []
    raw_rows: list[DetectionRecord] = []
    detection_id = 0
    processed_frames = 0
    frame_idx = 0
    tmp_ocr_path = tmp_dir / "current_crop.png"

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

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.img_size,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            annotated = result.plot()

            for box in boxes:
                cls_id = int(box.cls[0].cpu().item())
                conf = float(box.conf[0].cpu().item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                base_box = (clamp(x1, 0, w - 1), clamp(y1, 0, h - 1), clamp(x2, 0, w - 1), clamp(y2, 0, h - 1))
                padded_box = expand_box(base_box, w, h, args.padding_x, args.padding_y)
                px1, py1, px2, py2 = padded_box
                crop = frame[py1:py2, px1:px2]

                ocr_text, ocr_avg_conf = run_ocr_on_crop(ocr, crop, tmp_ocr_path)

                matched = find_matching_event(
                    events=events,
                    timestamp_sec=timestamp_sec,
                    box=base_box,
                    frame_w=w,
                    frame_h=h,
                    ocr_text=ocr_text,
                    max_gap_sec=args.event_gap_sec,
                    iou_thr=args.event_iou_thr,
                    center_thr=args.event_center_thr,
                    text_split_similarity_thr=args.text_split_similarity_thr,
                )

                if matched is None:
                    event_id = len(events) + 1
                    matched = DonationEvent(
                        event_id=event_id,
                        video_name=video_path.name,
                        start_sec=timestamp_sec,
                        end_sec=timestamp_sec,
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        last_box=base_box,
                    )
                    events.append(matched)

                matched.end_sec = timestamp_sec
                matched.last_frame = frame_idx
                matched.last_box = base_box
                matched.detections_count += 1
                if ocr_text and ocr_text not in matched.ocr_variants:
                    matched.ocr_variants.append(ocr_text)

                if should_replace_best(matched, conf, ocr_text, ocr_avg_conf):
                    matched.best_confidence = conf
                    matched.best_timestamp_sec = timestamp_sec
                    matched.best_frame_idx = frame_idx
                    matched.best_ocr_text = ocr_text
                    matched.best_ocr_avg_conf = ocr_avg_conf
                    update_event_best_files(matched, crop, annotated, events_dir)

                raw_crop_path = ""
                if args.save_raw_crops:
                    raw_crop_file = raw_crops_dir / f"det_{detection_id:06d}_event_{matched.event_id:04d}_frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(raw_crop_file), crop)
                    raw_crop_path = str(raw_crop_file)

                raw_rows.append(
                    DetectionRecord(
                        detection_id=detection_id,
                        event_id=matched.event_id,
                        video_name=video_path.name,
                        frame_idx=frame_idx,
                        timestamp_sec=round(timestamp_sec, 3),
                        confidence=round(conf, 4),
                        class_id=cls_id,
                        x1=base_box[0],
                        y1=base_box[1],
                        x2=base_box[2],
                        y2=base_box[3],
                        ocr_text=ocr_text,
                        ocr_avg_conf=round(ocr_avg_conf, 4),
                        crop_path=raw_crop_path,
                    )
                )
                detection_id += 1

        if args.max_processed_frames and processed_frames >= args.max_processed_frames:
            break

        if processed_frames % 100 == 0:
            print(f"Processed frames sampled: {processed_frames}, source frame: {frame_idx}, detections: {len(raw_rows)}, events: {len(events)}")

        frame_idx += 1

    cap.release()

    # Save raw detections CSV.
    raw_csv = output_dir / "detections_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(DetectionRecord.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row.__dict__)

    # Save event summary CSV.
    event_csv = output_dir / "events_summary.csv"
    event_rows: list[dict[str, Any]] = []
    for ev in events:
        parsed = parse_donation_text(ev.best_ocr_text)
        event_rows.append({
            "event_id": ev.event_id,
            "video_name": ev.video_name,
            "start_sec": round(ev.start_sec, 3),
            "end_sec": round(ev.end_sec, 3),
            "start_time": seconds_to_timestamp(ev.start_sec),
            "end_time": seconds_to_timestamp(ev.end_sec),
            "duration_sec": round(ev.end_sec - ev.start_sec, 3),
            "first_frame": ev.first_frame,
            "last_frame": ev.last_frame,
            "detections_count": ev.detections_count,
            "best_confidence": round(ev.best_confidence, 4),
            "best_timestamp_sec": round(ev.best_timestamp_sec, 3),
            "best_time": seconds_to_timestamp(ev.best_timestamp_sec),
            "best_frame_idx": ev.best_frame_idx,
            "ocr_avg_conf": round(ev.best_ocr_avg_conf, 4),
            "ocr_text": ev.best_ocr_text,
            "donor_guess": parsed["donor_guess"],
            "amount_guess": parsed["amount_guess"],
            "currency_guess": parsed["currency_guess"],
            "message_guess": parsed["message_guess"],
            "best_crop_path": ev.best_crop_path,
            "best_frame_path": ev.best_frame_path,
            "ocr_variants_count": len(ev.ocr_variants),
        })

    with event_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "event_id", "video_name", "start_sec", "end_sec", "start_time", "end_time", "duration_sec",
            "first_frame", "last_frame", "detections_count", "best_confidence", "best_timestamp_sec",
            "best_time", "best_frame_idx", "ocr_avg_conf", "ocr_text", "donor_guess", "amount_guess",
            "currency_guess", "message_guess", "best_crop_path", "best_frame_path", "ocr_variants_count"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_rows)

    # Save OCR variants as JSON for manual debugging.
    variants_json = output_dir / "events_ocr_variants.json"
    with variants_json.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "event_id": ev.event_id,
                    "start_time": seconds_to_timestamp(ev.start_sec),
                    "end_time": seconds_to_timestamp(ev.end_sec),
                    "best_ocr_text": ev.best_ocr_text,
                    "ocr_variants": ev.ocr_variants,
                }
                for ev in events
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Remove temporary files.
    if tmp_dir.exists() and not args.keep_tmp:
        shutil.rmtree(tmp_dir)

    print("\nDone.")
    print(f"Sampled frames processed: {processed_frames}")
    print(f"Raw detections: {len(raw_rows)}")
    print(f"Merged donation events: {len(events)}")
    print(f"Events summary: {event_csv}")
    print(f"Raw detections:  {raw_csv}")
    print(f"Best crops/frames: {events_dir}")
    print(f"OCR variants: {variants_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + PaddleOCR donation extraction prototype")
    parser.add_argument("--project-dir", default=".", help="Project root directory")
    parser.add_argument("--model", default="models/donation_detector_yolo26n_v1.pt", help="Model path, absolute or relative to project dir")
    parser.add_argument("--video", default="video_tests/test.mp4", help="Video path, absolute or relative to project dir")
    parser.add_argument("--output-dir", default="ocr_runs", help="Output directory relative to project dir")
    parser.add_argument("--run-name", default="", help="Optional run folder name")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing run folder before writing")

    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    parser.add_argument("--img-size", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--frame-step", type=int, default=10, help="Run YOLO every Nth frame")
    parser.add_argument("--padding-x", type=int, default=28, help="Horizontal padding for OCR crop")
    parser.add_argument("--padding-y", type=int, default=18, help="Vertical padding for OCR crop")

    parser.add_argument("--ocr-lang", default="ru", help="PaddleOCR language, e.g. ru, en")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR and test detection/grouping only")

    parser.add_argument("--event-gap-sec", type=float, default=8.0, help="Max time gap between detections of one event")
    parser.add_argument("--event-iou-thr", type=float, default=0.15, help="IoU threshold for merging detections")
    parser.add_argument("--event-center-thr", type=float, default=0.25, help="Normalized center distance threshold for merging")
    parser.add_argument("--text-split-similarity-thr", type=float, default=0.25, help="If OCR texts are very different, split event")

    parser.add_argument("--save-raw-crops", action="store_true", help="Save crop for every raw detection; can create many files")
    parser.add_argument("--max-processed-frames", type=int, default=0, help="Debug limit for sampled frames; 0 means no limit")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary OCR folder")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    run_pipeline(args)


if __name__ == "__main__":
    main()

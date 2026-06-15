#!/usr/bin/env python3
"""
Donation extraction prototype v2:
video -> YOLO donation detector -> event grouping -> multi-variant OCR -> amount/donor/message parsing.

Compared with v1, this version:
- groups repeated YOLO detections BEFORE OCR;
- runs OCR only on top crop candidates per event;
- upscales and preprocesses crops before OCR;
- stores OCR blocks with coordinates;
- tries to parse donor, amount, currency and message;
- handles simple two-line amounts like "1" + "500 RUB" -> 1500 RUB.

Recommended install for local Ubuntu with PaddleOCR 2.x:
  python3.10 -m venv donate_env310
  source donate_env310/bin/activate
  pip install -U pip
  pip install "numpy==1.26.4" "opencv-python==4.6.0.66" "opencv-contrib-python==4.6.0.66"
  pip install ultralytics pandas paddlepaddle==2.6.2 paddleocr==2.7.3

Example:
  python scripts/donation_video_ocr_pipeline_v2.py \
    --project-dir . \
    --model models/best.pt \
    --video video_tests/test_fragment.mp4 \
    --device cpu \
    --frame-step 10 \
    --conf 0.25 \
    --img-size 640 \
    --ocr-lang ru \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


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
    crop: np.ndarray
    annotated_frame: np.ndarray
    score: float


@dataclass
class OcrBlock:
    event_id: int
    candidate_id: int
    variant: str
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    line_id: int = -1


@dataclass
class OcrVariant:
    event_id: int
    candidate_id: int
    variant: str
    text: str
    avg_confidence: float
    chars_count: int
    has_digit: bool
    has_currency: bool
    cyrillic_count: int
    score: float
    image_path: str = ""


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
    return text[:max_len]


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def expand_box(box: tuple[int, int, int, int], width: int, height: int, pad_x: int, pad_y: int) -> tuple[int, int, int, int]:
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


def center_distance_norm(a: tuple[int, int, int, int], b: tuple[int, int, int, int], width: int, height: int) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    diag = max(1.0, (width ** 2 + height ** 2) ** 0.5)
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / diag


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-zа-я$€₽.,:;!?+\- ]+", "", text, flags=re.IGNORECASE)
    return text.strip()


def text_similarity(a: str, b: str) -> float:
    a_norm, b_norm = normalize_text(a), normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def has_currency(text: str) -> bool:
    return bool(re.search(r"\b(rub|руб|р|usd|eur|bits?|бит(?:ов|ы)?|мемкоин\w*)\b|[₽€$]", text, re.IGNORECASE))


def cyrillic_count(text: str) -> int:
    return len(re.findall(r"[а-яёА-ЯЁ]", text))


# -----------------------------
# OCR creation and compatibility
# -----------------------------

def create_ocr(lang: str, use_ocr: bool):
    if not use_ocr:
        return None
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError("PaddleOCR import failed. Check paddlepaddle/paddleocr installation.") from exc

    attempts = [
        {"use_angle_cls": True, "lang": lang, "show_log": False},  # PaddleOCR 2.x
        {"use_angle_cls": True, "lang": lang},
        {"use_textline_orientation": False, "lang": lang},  # PaddleOCR 3.x
        {"lang": lang},
    ]
    last_exc = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not initialize PaddleOCR lang={lang}: {last_exc}")


def is_point_list(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)) or len(obj) < 4:
        return False
    try:
        for p in obj[:4]:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                return False
            float(p[0]); float(p[1])
        return True
    except Exception:
        return False


def points_to_xyxy(points: Any, scale: float) -> tuple[float, float, float, float]:
    xs = [float(p[0]) / scale for p in points]
    ys = [float(p[1]) / scale for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def extract_ocr_blocks(obj: Any, event_id: int, candidate_id: int, variant: str, scale: float) -> list[OcrBlock]:
    """Extract text blocks from PaddleOCR 2.x/3.x-like outputs.

    Coordinates are normalized back to the original crop size by dividing by `scale`.
    """
    blocks: list[OcrBlock] = []

    def walk(x: Any):
        if x is None:
            return

        # PaddleOCR 3.x result objects may have json/to_json.
        for attr in ("json", "to_json"):
            if hasattr(x, attr):
                try:
                    v = getattr(x, attr)
                    v = v() if callable(v) else v
                    walk(v)
                    return
                except Exception:
                    pass

        if isinstance(x, dict):
            texts = x.get("rec_texts") or x.get("texts")
            scores = x.get("rec_scores") or x.get("scores") or []
            boxes = x.get("rec_polys") or x.get("dt_polys") or x.get("rec_boxes") or x.get("boxes") or []
            if isinstance(texts, list) and texts:
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    score = float(scores[i]) if isinstance(scores, list) and i < len(scores) else 0.0
                    if isinstance(boxes, list) and i < len(boxes) and is_point_list(boxes[i]):
                        x1, y1, x2, y2 = points_to_xyxy(boxes[i], scale)
                    else:
                        x1 = y1 = x2 = y2 = -1.0
                    blocks.append(OcrBlock(event_id, candidate_id, variant, str(text), score, x1, y1, x2, y2))
            for v in x.values():
                if isinstance(v, (list, tuple, dict)):
                    walk(v)
            return

        if isinstance(x, (list, tuple)):
            # PaddleOCR 2.x line: [points, (text, confidence)]
            if len(x) >= 2 and is_point_list(x[0]) and isinstance(x[1], (list, tuple)) and len(x[1]) >= 2:
                text = x[1][0]
                if isinstance(text, str) and text.strip():
                    try:
                        score = float(x[1][1])
                    except Exception:
                        score = 0.0
                    x1, y1, x2, y2 = points_to_xyxy(x[0], scale)
                    blocks.append(OcrBlock(event_id, candidate_id, variant, text.strip(), score, x1, y1, x2, y2))
                    return
            # Sometimes pair is just (text, confidence) without box.
            if len(x) >= 2 and isinstance(x[0], str):
                try:
                    score = float(x[1])
                except Exception:
                    score = 0.0
                blocks.append(OcrBlock(event_id, candidate_id, variant, x[0].strip(), score, -1, -1, -1, -1))
                return
            for v in x:
                walk(v)

    walk(obj)
    # Deduplicate identical text/box pairs.
    unique: dict[tuple[str, int, int, int, int], OcrBlock] = {}
    for b in blocks:
        key = (b.text, int(b.x1), int(b.y1), int(b.x2), int(b.y2))
        if key not in unique or b.confidence > unique[key].confidence:
            unique[key] = b
    return list(unique.values())


def run_paddle_ocr(ocr: Any, image_bgr: np.ndarray, tmp_path: Path, event_id: int, candidate_id: int, variant: str, scale: float) -> list[OcrBlock]:
    if ocr is None or image_bgr.size == 0:
        return []
    cv2.imwrite(str(tmp_path), image_bgr)
    result = None

    # PaddleOCR 2.x usually works best with ocr(..., cls=True).
    if hasattr(ocr, "ocr"):
        try:
            result = ocr.ocr(str(tmp_path), cls=True)
        except TypeError:
            result = ocr.ocr(str(tmp_path))
        except Exception:
            result = None

    # PaddleOCR 3.x fallback.
    if result is None and hasattr(ocr, "predict"):
        result = ocr.predict(str(tmp_path))

    return extract_ocr_blocks(result, event_id, candidate_id, variant, scale)


# -----------------------------
# Crop preprocessing
# -----------------------------

def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def increase_contrast(img: np.ndarray, alpha: float = 1.35, beta: int = 4) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def preprocess_variants(crop_bgr: np.ndarray, scales: list[float]) -> list[tuple[str, np.ndarray, float]]:
    variants: list[tuple[str, np.ndarray, float]] = [("original", crop_bgr, 1.0)]
    for scale in scales:
        if scale <= 1.0:
            continue
        up = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((f"up{scale:g}", up, scale))
        variants.append((f"up{scale:g}_contrast", increase_contrast(up), scale))
        variants.append((f"up{scale:g}_sharp", sharpen(up), scale))

        # Grayscale contrast sometimes helps thin text, but can hurt colored text; keep as optional variant.
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
        variants.append((f"up{scale:g}_gray_clahe", gray_bgr, scale))
    return variants


def blocks_to_lines(blocks: list[OcrBlock], crop_h: int) -> list[dict[str, Any]]:
    good = [b for b in blocks if b.text.strip()]
    if not good:
        return []

    # If no coordinates, return in OCR order as one line.
    if all(b.y1 < 0 for b in good):
        return [{"line_id": 0, "text": " ".join(b.text for b in good), "blocks": good, "x1": -1, "y1": -1, "x2": -1, "y2": -1}]

    good.sort(key=lambda b: ((b.y1 + b.y2) / 2 if b.y1 >= 0 else 1e9, b.x1 if b.x1 >= 0 else 1e9))
    threshold = max(12.0, crop_h * 0.08)
    lines: list[list[OcrBlock]] = []
    centers: list[float] = []

    for b in good:
        cy = (b.y1 + b.y2) / 2 if b.y1 >= 0 else 1e9
        placed = False
        for idx, c in enumerate(centers):
            if abs(cy - c) <= threshold:
                lines[idx].append(b)
                centers[idx] = float(np.mean([(x.y1 + x.y2) / 2 for x in lines[idx] if x.y1 >= 0]))
                placed = True
                break
        if not placed:
            lines.append([b])
            centers.append(cy)

    out: list[dict[str, Any]] = []
    for line_id, line_blocks in enumerate(lines):
        line_blocks.sort(key=lambda b: b.x1 if b.x1 >= 0 else 1e9)
        for b in line_blocks:
            b.line_id = line_id
        text = " ".join(b.text.strip() for b in line_blocks if b.text.strip())
        xs1 = [b.x1 for b in line_blocks if b.x1 >= 0]
        ys1 = [b.y1 for b in line_blocks if b.y1 >= 0]
        xs2 = [b.x2 for b in line_blocks if b.x2 >= 0]
        ys2 = [b.y2 for b in line_blocks if b.y2 >= 0]
        out.append({
            "line_id": line_id,
            "text": text,
            "blocks": line_blocks,
            "x1": min(xs1) if xs1 else -1,
            "y1": min(ys1) if ys1 else -1,
            "x2": max(xs2) if xs2 else -1,
            "y2": max(ys2) if ys2 else -1,
        })
    return out


def variant_score(blocks: list[OcrBlock], crop_h: int) -> tuple[str, float, dict[str, Any]]:
    lines = blocks_to_lines(blocks, crop_h)
    text = " | ".join(line["text"] for line in lines if line["text"].strip())
    confs = [b.confidence for b in blocks if b.text.strip() and b.confidence > 0]
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    chars = len(re.sub(r"\s+", "", text))
    score = (
        avg_conf * 1.2
        + min(chars, 180) / 120.0
        + (0.25 if re.search(r"\d", text) else 0.0)
        + (0.25 if has_currency(text) else 0.0)
        + min(cyrillic_count(text), 120) / 200.0
    )
    return text, score, {"avg_conf": avg_conf, "chars": chars, "has_digit": bool(re.search(r"\d", text)), "has_currency": has_currency(text), "cyrillic_count": cyrillic_count(text), "lines": lines}


# -----------------------------
# Text parsing
# -----------------------------

LATIN_TO_CYR = str.maketrans({
    "A": "А", "a": "а", "B": "В", "E": "Е", "e": "е", "K": "К", "k": "к",
    "M": "М", "H": "Н", "O": "О", "o": "о", "P": "Р", "p": "р", "C": "С", "c": "с",
    "T": "Т", "X": "Х", "x": "х", "Y": "У", "y": "у",
})


def normalize_ocr_text_for_output(text: str) -> str:
    text = text.replace(" | ", "\n")
    text = text.replace("MеHя", "меня").replace("Mеня", "меня").replace("Tы", "Ты")
    text = text.replace("Бечерний", "Вечерний")
    text = text.replace(" оха", " Тоха")
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_number(s: str) -> str:
    s = re.sub(r"[^0-9]", "", s)
    return s


def parse_from_lines(lines: list[dict[str, Any]]) -> dict[str, str]:
    line_texts = [line["text"].strip() for line in lines if line["text"].strip()]
    full_text = "\n".join(line_texts)
    donor = ""
    amount = ""
    currency = ""
    message = ""
    amount_line_idx: Optional[int] = None

    currency_re = r"(?:₽|руб\.?|rub|р\.?|usd|eur|€|\$|bits?|бит(?:ов|ы)?)"

    # 1) Strong pattern: donor - amount on one line, currency optional.
    for i, line in enumerate(line_texts[:3]):
        m = re.search(r"^(.{2,80}?)\s*[-–—:]+\s*(\d[\d\s.,]*)\s*(!?\s*(" + currency_re + r"))?", line, re.IGNORECASE)
        if m:
            donor = m.group(1).strip(" -–—:|")
            amount = clean_number(m.group(2))
            currency = (m.group(4) or "").strip()
            amount_line_idx = i
            break

    # 2) Currency line, including two-line amount: previous line ends with "- 1", current line starts with "500 RUB".
    if not amount:
        for i, line in enumerate(line_texts):
            m = re.search(r"(\d[\d\s.,]*)\s*(" + currency_re + r")", line, re.IGNORECASE)
            if not m:
                continue
            num = clean_number(m.group(1))
            cur = m.group(2).strip()
            if len(num) == 3 and i > 0:
                prev = line_texts[i - 1]
                pm = re.search(r"[-–—:]\s*(\d{1,3})\s*$", prev)
                if pm:
                    prefix = clean_number(pm.group(1))
                    if prefix:
                        num = prefix + num
                        donor = re.sub(r"[-–—:]\s*\d{1,3}\s*$", "", prev).strip()
            amount = num
            currency = cur
            amount_line_idx = i
            break

    # 3) If donor still absent, use text before dash on first line.
    if not donor and line_texts:
        m = re.search(r"^(.{2,80}?)\s*[-–—:]+", line_texts[0])
        if m:
            donor = m.group(1).strip()
        else:
            donor = line_texts[0].strip()

    # 4) Message: lines not used as donor/amount. For one-line donor+amount, message starts after line 0.
    used = set()
    if amount_line_idx is not None:
        used.add(amount_line_idx)
        if amount_line_idx > 0 and donor and donor in line_texts[amount_line_idx - 1]:
            used.add(amount_line_idx - 1)
    elif line_texts:
        used.add(0)

    # If donor and amount are on first line, all following lines are message.
    if amount_line_idx == 0:
        msg_lines = line_texts[1:]
    else:
        msg_lines = [t for idx, t in enumerate(line_texts) if idx not in used]
    message = "\n".join(msg_lines).strip()

    # Remove currency punctuation cleanup.
    if currency:
        currency = currency.upper().replace("РУБ", "RUB").replace("Р.", "RUB").replace("Р", "RUB")
    return {
        "donor_guess": donor,
        "amount_guess": amount,
        "currency_guess": currency,
        "message_guess": message,
        "full_text_lines": full_text,
    }


# -----------------------------
# Event merging
# -----------------------------

def find_matching_event(
    events: list[DonationEvent],
    timestamp_sec: float,
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    max_gap_sec: float,
    iou_thr: float,
    center_thr: float,
) -> Optional[DonationEvent]:
    best: Optional[DonationEvent] = None
    best_score = -999.0
    for ev in events:
        if timestamp_sec - ev.end_sec > max_gap_sec:
            continue
        iou = box_iou(ev.last_box, box)
        dist = center_distance_norm(ev.last_box, box, frame_w, frame_h)
        if iou >= iou_thr or dist <= center_thr:
            score = iou - dist
            if score > best_score:
                best_score = score
                best = ev
    return best


def add_candidate(event: DonationEvent, candidate: CandidateCrop, max_candidates: int) -> None:
    event.candidates.append(candidate)
    event.candidates.sort(key=lambda c: c.score, reverse=True)
    if len(event.candidates) > max_candidates:
        # Keep only best in memory. This is enough for OCR and avoids RAM growth.
        event.candidates = event.candidates[:max_candidates]


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
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    run_name = args.run_name or f"{safe_filename(video_path.stem)}_ocr_v2_run"
    output_dir = (project_dir / args.output_dir / run_name).resolve()
    events_dir = output_dir / "events"
    variants_dir = output_dir / "ocr_variants_images"
    tmp_dir = output_dir / "tmp"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    events_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project: {project_dir}")
    print(f"Model:   {model_path}")
    print(f"Video:   {video_path}")
    print(f"Output:  {output_dir}")

    model = YOLO(str(model_path))
    ocr = create_ocr(args.ocr_lang, use_ocr=not args.no_ocr)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"FPS: {fps:.3f}, frames: {total_frames}, frame_step: {args.frame_step}")

    events: list[DonationEvent] = []
    frame_idx = 0
    processed_frames = 0
    raw_detections_count = 0

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
                base_box = (clamp(x1, 0, w - 1), clamp(y1, 0, h - 1), clamp(x2, 0, w - 1), clamp(y2, 0, h - 1))
                padded_box = expand_box(base_box, w, h, args.padding_x, args.padding_y)
                px1, py1, px2, py2 = padded_box
                crop = frame[py1:py2, px1:px2].copy()
                area_norm = box_area(base_box) / max(1, w * h)
                candidate_score = conf + min(area_norm * 10.0, 0.5)

                matched = find_matching_event(
                    events,
                    timestamp_sec,
                    base_box,
                    w,
                    h,
                    args.event_gap_sec,
                    args.event_iou_thr,
                    args.event_center_thr,
                )
                if matched is None:
                    matched = DonationEvent(
                        event_id=len(events) + 1,
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
                matched.best_confidence = max(matched.best_confidence, conf)
                if conf >= matched.best_confidence:
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
                    crop=crop,
                    annotated_frame=annotated.copy(),
                    score=candidate_score,
                )
                add_candidate(matched, candidate, args.keep_top_candidates)
                raw_detections_count += 1

        if args.max_processed_frames and processed_frames >= args.max_processed_frames:
            break
        if processed_frames % 100 == 0:
            print(f"Processed sampled frames: {processed_frames}, source frame: {frame_idx}, detections: {raw_detections_count}, events: {len(events)}")
        frame_idx += 1

    cap.release()

    print(f"Detection stage done. Events: {len(events)}, raw detections: {raw_detections_count}")
    print("Starting OCR stage...")

    all_variant_rows: list[dict[str, Any]] = []
    all_block_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    variant_images_saved = 0

    scales = [float(x) for x in args.ocr_scales.split(",") if x.strip()]

    for ev in events:
        event_blocks: list[OcrBlock] = []
        event_variants: list[OcrVariant] = []

        # Save best candidate crop/frame by detection score.
        if ev.candidates:
            best_det = max(ev.candidates, key=lambda c: c.score)
            best_crop_path = events_dir / f"event_{ev.event_id:04d}_best_detector_crop.png"
            best_frame_path = events_dir / f"event_{ev.event_id:04d}_best_detector_frame.jpg"
            cv2.imwrite(str(best_crop_path), best_det.crop)
            cv2.imwrite(str(best_frame_path), best_det.annotated_frame)
        else:
            best_crop_path = Path("")
            best_frame_path = Path("")

        for candidate_id, cand in enumerate(ev.candidates[:args.ocr_top_candidates]):
            for variant_name, img, scale in preprocess_variants(cand.crop, scales):
                if args.save_ocr_variant_images:
                    var_path = variants_dir / f"event_{ev.event_id:04d}_cand_{candidate_id:02d}_{variant_name}.png"
                    cv2.imwrite(str(var_path), img)
                    variant_images_saved += 1
                    image_path_str = str(var_path)
                else:
                    var_path = tmp_dir / f"event_{ev.event_id:04d}_cand_{candidate_id:02d}_{variant_name}.png"
                    image_path_str = ""

                blocks = run_paddle_ocr(ocr, img, var_path, ev.event_id, candidate_id, variant_name, scale)
                crop_h = cand.crop.shape[0]
                text, score, info = variant_score(blocks, crop_h)
                event_blocks.extend(blocks)
                ov = OcrVariant(
                    event_id=ev.event_id,
                    candidate_id=candidate_id,
                    variant=variant_name,
                    text=text,
                    avg_confidence=round(info["avg_conf"], 4),
                    chars_count=info["chars"],
                    has_digit=info["has_digit"],
                    has_currency=info["has_currency"],
                    cyrillic_count=info["cyrillic_count"],
                    score=round(score, 4),
                    image_path=image_path_str,
                )
                event_variants.append(ov)
                all_variant_rows.append(asdict(ov))

        # Pick best OCR variant by score. Use its blocks for line-based parse.
        best_variant = max(event_variants, key=lambda v: v.score, default=None)
        best_blocks = [b for b in event_blocks if best_variant and b.candidate_id == best_variant.candidate_id and b.variant == best_variant.variant]
        crop_h = ev.candidates[best_variant.candidate_id].crop.shape[0] if best_variant and ev.candidates else 1
        lines = blocks_to_lines(best_blocks, crop_h)
        parsed = parse_from_lines(lines)
        best_text = normalize_ocr_text_for_output(best_variant.text if best_variant else "")
        parsed["message_guess"] = normalize_ocr_text_for_output(parsed.get("message_guess", ""))

        # Save OCR blocks rows with line ids.
        for b in event_blocks:
            all_block_rows.append({
                "event_id": b.event_id,
                "candidate_id": b.candidate_id,
                "variant": b.variant,
                "line_id": b.line_id,
                "text": b.text,
                "confidence": round(b.confidence, 4),
                "x1": round(b.x1, 2),
                "y1": round(b.y1, 2),
                "x2": round(b.x2, 2),
                "y2": round(b.y2, 2),
            })

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
            "best_detection_confidence": round(ev.best_confidence, 4),
            "best_ocr_variant": best_variant.variant if best_variant else "",
            "best_ocr_candidate_id": best_variant.candidate_id if best_variant else "",
            "best_ocr_score": best_variant.score if best_variant else "",
            "ocr_text_raw": best_variant.text if best_variant else "",
            "ocr_text_clean": best_text,
            "donor_guess": parsed.get("donor_guess", ""),
            "amount_guess": parsed.get("amount_guess", ""),
            "currency_guess": parsed.get("currency_guess", ""),
            "message_guess": parsed.get("message_guess", ""),
            "full_text_lines": parsed.get("full_text_lines", ""),
            "best_detector_crop_path": str(best_crop_path),
            "best_detector_frame_path": str(best_frame_path),
            "ocr_variants_count": len(event_variants),
            "ocr_blocks_count": len(event_blocks),
        })

    # Write outputs.
    event_csv = output_dir / "events_summary.csv"
    variant_csv = output_dir / "ocr_variants.csv"
    block_csv = output_dir / "ocr_blocks.csv"
    metadata_json = output_dir / "run_metadata.json"

    def write_csv(path: Path, rows: list[dict[str, Any]]):
        with path.open("w", newline="", encoding="utf-8") as f:
            if not rows:
                f.write("")
                return
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(event_csv, event_rows)
    write_csv(variant_csv, all_variant_rows)
    write_csv(block_csv, all_block_rows)

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump({
            "project_dir": str(project_dir),
            "model": str(model_path),
            "video": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "frame_step": args.frame_step,
            "sampled_frames_processed": processed_frames,
            "raw_detections": raw_detections_count,
            "events": len(events),
            "ocr_scales": scales,
            "ocr_top_candidates": args.ocr_top_candidates,
            "variant_images_saved": variant_images_saved,
        }, f, ensure_ascii=False, indent=2)

    if tmp_dir.exists() and not args.keep_tmp:
        shutil.rmtree(tmp_dir)

    print("\nDone.")
    print(f"Sampled frames processed: {processed_frames}")
    print(f"Raw detections: {raw_detections_count}")
    print(f"Merged donation events: {len(events)}")
    print(f"Events summary: {event_csv}")
    print(f"OCR variants:   {variant_csv}")
    print(f"OCR blocks:     {block_csv}")
    print(f"Best crops/frames: {events_dir}")
    if args.save_ocr_variant_images:
        print(f"OCR variant images: {variants_dir}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO + multi-variant PaddleOCR donation extraction prototype v2")
    p.add_argument("--project-dir", default="~/donation_project")
    p.add_argument("--model", default="models/donation_detector_yolo26n_v1.pt")
    p.add_argument("--video", default="video_tests/test.mp4")
    p.add_argument("--output-dir", default="ocr_runs")
    p.add_argument("--run-name", default="")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--frame-step", type=int, default=10)
    p.add_argument("--padding-x", type=int, default=30)
    p.add_argument("--padding-y", type=int, default=20)

    p.add_argument("--ocr-lang", default="ru")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--ocr-scales", default="2.0,3.0", help="Comma-separated upscale factors for OCR variants")
    p.add_argument("--ocr-top-candidates", type=int, default=5, help="Run OCR on top N crop candidates per event")
    p.add_argument("--keep-top-candidates", type=int, default=12, help="Keep top N candidates in memory while scanning video")
    p.add_argument("--save-ocr-variant-images", action="store_true", help="Save preprocessed images used for OCR; can create many files")

    p.add_argument("--event-gap-sec", type=float, default=8.0)
    p.add_argument("--event-iou-thr", type=float, default=0.15)
    p.add_argument("--event-center-thr", type=float, default=0.25)
    p.add_argument("--max-processed-frames", type=int, default=0)
    p.add_argument("--keep-tmp", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    if args.ocr_top_candidates <= 0:
        raise ValueError("--ocr-top-candidates must be >= 1")
    if args.keep_top_candidates < args.ocr_top_candidates:
        args.keep_top_candidates = args.ocr_top_candidates
    run_pipeline(args)


if __name__ == "__main__":
    main()

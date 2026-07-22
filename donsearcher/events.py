#!/usr/bin/env python3
"""Donation event data structures and detection grouping.

``CandidateCrop`` / ``DonationEvent`` model the memory-optimised pipeline (crops
live on disk as file paths, not numpy arrays). ``find_matching_event`` decides
whether a new detection extends an existing event or starts a new one — including
the plaque-swap height guard. Geometry-only and free of cv2/numpy/ultralytics, so
the decision logic stays in lockstep with the C++ engine's 1:1 port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .geometry import box_height, box_iou, center_distance_norm, reference_height


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
    annotated_frame_path: Optional[Path]  # None когда бит annotated выключен в --images-schema
    original_frame_path: Optional[Path]   # None когда бит original выключен в --images-schema
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
    # Last few box heights — reference geometry for the plaque-swap split guard.
    # A sharp height change in the same overlay slot means the plaque was replaced
    # (a new donation), so the detection must NOT merge into this event.
    recent_heights: list[int] = field(default_factory=list)


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
    split_height_frac: float = 0.0,
) -> Optional[DonationEvent]:
    """Search only the provided event list (caller is responsible for passing active events).

    Same-frame events (last_frame == current_frame_idx) can only match by IoU.
    Center-distance matching is intentionally disabled for same-frame events because
    it would incorrectly merge two separate donations visible simultaneously.

    Plaque-swap guard (split_height_frac > 0): two donations shown back-to-back in
    the same overlay slot overlap heavily (high IoU), so geometry alone merges them
    and the second donation is lost. But the plaque box height jumps when the plaque
    is replaced (different message length), while it stays ~stable within one
    donation. So a candidate whose height differs from the event's reference height
    by more than split_height_frac is treated as a DIFFERENT plaque and does not
    match — forcing a new event. The decision uses only integer box coords, which
    both the Python and C++ engines produce identically, so their output stays in
    sync. It will not separate two donations of the same plaque height (accepted).
    """
    best: Optional[DonationEvent] = None
    best_score = -999.0
    box_h = box_height(box)

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

        if split_height_frac > 0.0:
            ref_h = reference_height(ev.recent_heights)
            if ref_h > 0 and abs(box_h - ref_h) > split_height_frac * ref_h:
                continue  # plaque height changed → different donation, don't merge

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

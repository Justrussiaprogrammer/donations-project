#!/usr/bin/env python3
"""Pure box / coordinate geometry helpers.

No heavy dependencies (no cv2/numpy/ultralytics) — just integer box math shared by
event grouping and the YOLO stage. The plaque-swap split guard
(``reference_height``) lives here too because it is a geometry decision and the C++
engine mirrors it 1:1.
"""

from __future__ import annotations

RECENT_HEIGHTS_MAXLEN = 5


def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(v, high))


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


def box_height(box: tuple[int, int, int, int]) -> int:
    return box[3] - box[1]


def reference_height(heights: list[int]) -> int:
    """Median height of an event's recent detections — reference geometry for the
    plaque-swap guard. Robust to per-frame jitter and the appear animation.
    For an even count returns the upper of the two middles, a simple rule the C++
    engine mirrors 1:1 (so both engines decide identically)."""
    if not heights:
        return 0
    s = sorted(heights)
    return s[len(s) // 2]


def push_recent_height(heights: list[int], h: int, maxlen: int = RECENT_HEIGHTS_MAXLEN) -> None:
    heights.append(h)
    if len(heights) > maxlen:
        del heights[0]

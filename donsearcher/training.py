#!/usr/bin/env python3
"""Optional training-data collection for improving the YOLO detector.

Harvests full frames + YOLO labels + annotated previews into a self-contained
detector dataset, written straight into the run dir. Independent of the VLM stage —
training mode skips VLM entirely (no events_summary/totals/donations are written;
this module's ``manifest.csv`` replaces them). Needs cv2 for image writes but not
ultralytics; Python engine only.

Strategies (any subset of TRAIN_STRATEGIES):
  best       — the highest-score confident frame of each event (1/event): the same
               frame the VLM would receive, i.e. the cleanest view of the donation.
  worst      — the lowest-score confident frame(s) of each event (worst_per_event).
  random     — ONE uniformly-random confident frame per event (size-1 reservoir, not
               a burst). Keeps the set from over-fitting one animation phase.
  uncertain  — frames with a weak fire: a box in [uncertain_min, conf). Decision-
               boundary cases. Capped at `budget`, spread across the stream.
  negatives  — frames with NO detection at all. Capped at `budget`, spread across the
               stream.

Both budget-capped strategies use a time-stratified reservoir: the stream is split
into `budget` equal buckets and at most one (uniformly-random) sample is kept per
bucket, so the kept frames are spread over the whole stream instead of clustering on
nearby frames of the same moment.

Output layout (images lossless so no JPEG artefacts; previews are JPEG — not used for
training, only for human review):
  images/    — PNG full frames
  labels/    — YOLO txt (`class cx cy w h`, ALL confident boxes; empty for negatives)
  previews/  — JPG annotated frames (when boxes are present / for uncertain)
  manifest.csv — one row per saved image: strategy, model confidence, box, event_id,
                 frame index, timestamp (hh:mm:ss), etc.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

import cv2

from .textutil import seconds_to_timestamp

# Order is also the strategy listing shown in --train-select help.
TRAIN_STRATEGIES = ("best", "worst", "random", "uncertain", "negatives")

MANIFEST_FIELDS = (
    "image", "strategy", "confidence", "select_score", "event_id",
    "frame_idx", "time", "num_boxes", "box_xyxy", "box_cxcywh",
    "frame_w", "frame_h", "label", "preview", "video",
)

Box = tuple[int, int, int, int]


def yolo_label_lines(
    boxes: list[tuple[Box, float, int]],
    frame_w: int,
    frame_h: int,
) -> list[str]:
    """YOLO-format label lines (`class cx cy w h`, normalized) for every box in a frame.

    A training label must list ALL objects in the image — a partially-labelled
    frame would teach the detector that a real plaque is background.
    """
    lines: list[str] = []
    if frame_w <= 0 or frame_h <= 0:
        return lines
    for (x1, y1, x2, y2), _conf, cls_id in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        cx = (x1 + x2) / 2.0 / frame_w
        cy = (y1 + y2) / 2.0 / frame_h
        lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw / frame_w:.6f} {bh / frame_h:.6f}")
    return lines


def _cxcywh(box: Box, frame_w: int, frame_h: int) -> str:
    """Normalized `cx cy w h` for a single box (for the manifest), or "" if degenerate."""
    if frame_w <= 0 or frame_h <= 0:
        return ""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return ""
    cx = (x1 + x2) / 2.0 / frame_w
    cy = (y1 + y2) / 2.0 / frame_h
    return f"{cx:.6f} {cy:.6f} {bw / frame_w:.6f} {bh / frame_h:.6f}"


class _StratifiedReservoir:
    """Keep <= budget samples, spread across the stream.

    The stream (frames 0..total_frames) is split into `budget` equal buckets; each
    bucket holds a size-1 reservoir (a uniformly-random frame within the bucket). The
    result is at most `budget` samples, one per occupied bucket — as evenly spread
    over the stream as the candidates allow, with no clustering of near-duplicate
    frames. Falls back to a plain size-`budget` reservoir when total_frames is unknown.
    """

    def __init__(self, budget: int, total_frames: int, rng: random.Random) -> None:
        self.budget = max(1, budget)
        self.total = total_frames if total_frames and total_frames > 0 else 0
        self._rng = rng
        self._buckets: dict[int, dict[str, Any]] = {}
        self._seen = 0
        self._items: list[dict[str, Any]] = []  # plain-reservoir fallback

    def offer(self, frame_idx: int, write_fn: Callable[[], dict], delete_fn: Callable[[dict], None]) -> None:
        if self.total > 0:
            b = min(frame_idx * self.budget // self.total, self.budget - 1)
            st = self._buckets.setdefault(b, {"seen": 0, "rec": None})
            st["seen"] += 1
            if st["rec"] is None:
                st["rec"] = write_fn()
            elif self._rng.randint(0, st["seen"] - 1) == 0:
                delete_fn(st["rec"])
                st["rec"] = write_fn()
        else:
            self._seen += 1
            if len(self._items) < self.budget:
                self._items.append(write_fn())
            else:
                j = self._rng.randint(0, self._seen - 1)
                if j < self.budget:
                    delete_fn(self._items[j])
                    self._items[j] = write_fn()

    def records(self) -> list[dict[str, Any]]:
        if self.total > 0:
            return [st["rec"] for st in self._buckets.values() if st["rec"] is not None]
        return list(self._items)


class TrainingCollector:
    """Harvest full frames + YOLO labels + previews + a manifest for detector retraining.

    Box classification (confident vs uncertain) is done by the caller; this class just
    routes already-classified boxes. ``observe_event`` is called once per event present
    in a sampled frame (driving best/worst/random); ``observe_frame`` once per sampled
    frame (driving uncertain/negatives).
    """

    def __init__(
        self,
        train_dir: Path,
        strategies: set[str],
        budget: int,
        worst_per_event: int,
        conf_thr: float,
        uncertain_min: float,
        video_name: str,
        fps: float,
        total_frames: int,
        image_ext: str = ".png",
        preview_ext: str = ".jpg",
    ) -> None:
        self.dir = train_dir
        self.images_dir = train_dir / "images"
        self.labels_dir = train_dir / "labels"
        self.previews_dir = train_dir / "previews"
        for d in (self.images_dir, self.labels_dir, self.previews_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.strategies = set(strategies)
        self.budget = max(1, budget)
        self.worst_per_event = max(1, worst_per_event)
        self.conf_thr = conf_thr
        self.uncertain_min = uncertain_min
        self.video_name = video_name
        self.fps = float(fps) if fps else 0.0
        self.total_frames = total_frames
        self.ext = image_ext
        self.prev_ext = preview_ext
        self._seq = 0
        self._rng = random.Random(0)  # deterministic sampling across runs
        self._events_seen: set[int] = set()
        self._best: dict[int, dict[str, Any]] = {}          # eid -> record (max score)
        self._worst: dict[int, list[dict[str, Any]]] = {}   # eid -> records (min scores)
        self._random: dict[int, dict[str, Any]] = {}        # eid -> {"seen", "rec"}
        self._uncertain = _StratifiedReservoir(budget, total_frames, self._rng)
        self._neg = _StratifiedReservoir(budget, total_frames, self._rng)
        self.saved_counts: dict[str, int] = {k: 0 for k in TRAIN_STRATEGIES}

    # -- low-level write/delete -------------------------------------------------

    def _write(self, frame, label_lines, annotated, frame_idx, meta, force_preview=False) -> dict[str, Any]:
        self._seq += 1
        strategy = meta["strategy"]
        stem = f"{strategy}_{self._seq:06d}_f{frame_idx:07d}"
        img_name = f"{stem}{self.ext}"
        img_path = self.images_dir / img_name
        lbl_path = self.labels_dir / f"{stem}.txt"
        cv2.imwrite(str(img_path), frame)
        lbl_path.write_text(
            ("\n".join(label_lines) + "\n") if label_lines else "", encoding="utf-8"
        )
        prev_path = None
        prev_name = ""
        if annotated is not None and (label_lines or force_preview):
            prev_path = self.previews_dir / f"{stem}{self.prev_ext}"
            cv2.imwrite(str(prev_path), annotated)
            prev_name = prev_path.name
        row = {
            "image": img_name,
            "strategy": strategy,
            "confidence": meta.get("confidence", ""),
            "select_score": meta.get("select_score", ""),
            "event_id": meta.get("event_id", ""),
            "frame_idx": frame_idx,
            "time": seconds_to_timestamp(frame_idx / self.fps) if self.fps else "",
            "num_boxes": len(label_lines),
            "box_xyxy": meta.get("box_xyxy", ""),
            "box_cxcywh": meta.get("box_cxcywh", ""),
            "frame_w": meta.get("frame_w", ""),
            "frame_h": meta.get("frame_h", ""),
            "label": f"{stem}.txt",
            "preview": prev_name,
            "video": self.video_name,
        }
        return {"img": img_path, "lbl": lbl_path, "prev": prev_path,
                "frame_idx": frame_idx, "row": row}

    @staticmethod
    def _delete(record: dict[str, Any]) -> None:
        for key in ("img", "lbl", "prev"):
            p = record.get(key)
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    def _event_record(self, strategy, eid, det, frame, label_lines, annotated, frame_idx, w, h) -> dict[str, Any]:
        score, conf, box = det
        meta = {
            "strategy": strategy,
            "event_id": eid,
            "confidence": round(conf, 4),
            "select_score": round(score, 4),
            "box_xyxy": json.dumps(list(box)),
            "box_cxcywh": _cxcywh(box, w, h),
            "frame_w": w,
            "frame_h": h,
        }
        rec = self._write(frame, label_lines, annotated, frame_idx, meta)
        rec["score"] = score
        return rec

    # -- per-event strategies (best / worst / random) ---------------------------

    def observe_event(self, eid, frame_recs, frame, confident_boxes, frame_idx, w, h, annotated) -> None:
        """Called once per event present in a sampled frame.

        frame_recs — list of (candidate_score, confidence, base_box) for THIS event in
        this frame (usually one). best keeps the highest-score frame of the event,
        worst the lowest, random a uniformly-random one.
        """
        self._events_seen.add(eid)
        if not frame_recs:
            return
        label_lines = yolo_label_lines(confident_boxes, w, h)
        max_rec = max(frame_recs, key=lambda r: r[0])
        min_rec = min(frame_recs, key=lambda r: r[0])

        if "best" in self.strategies:
            cur = self._best.get(eid)
            if cur is None or max_rec[0] > cur["score"]:
                if cur is not None:
                    self._delete(cur)
                self._best[eid] = self._event_record(
                    "best", eid, max_rec, frame, label_lines, annotated, frame_idx, w, h)

        if "worst" in self.strategies:
            records = self._worst.setdefault(eid, [])
            if not any(r["frame_idx"] == frame_idx for r in records):
                if len(records) < self.worst_per_event:
                    records.append(self._event_record(
                        "worst", eid, min_rec, frame, label_lines, annotated, frame_idx, w, h))
                else:
                    hi = max(range(len(records)), key=lambda i: records[i]["score"])
                    if min_rec[0] < records[hi]["score"]:
                        self._delete(records[hi])
                        records[hi] = self._event_record(
                            "worst", eid, min_rec, frame, label_lines, annotated, frame_idx, w, h)

        if "random" in self.strategies:
            st = self._random.setdefault(eid, {"seen": 0, "rec": None})
            st["seen"] += 1
            # size-1 reservoir: keep the new frame with probability 1/seen.
            if st["rec"] is None:
                st["rec"] = self._event_record(
                    "random", eid, max_rec, frame, label_lines, annotated, frame_idx, w, h)
            elif self._rng.randint(0, st["seen"] - 1) == 0:
                self._delete(st["rec"])
                st["rec"] = self._event_record(
                    "random", eid, max_rec, frame, label_lines, annotated, frame_idx, w, h)

    # -- frame-level strategies (uncertain / negatives) -------------------------

    def observe_frame(self, frame, confident_boxes, uncertain_boxes, frame_idx, w, h, annotated) -> None:
        """Called once per sampled frame, after observe_event for any events present.

        confident_boxes — conf >= conf_thr; uncertain_boxes — conf in [uncertain_min,
        conf_thr). The label always lists only confident boxes (weak fires are left for
        a human to confirm); for uncertain the annotated preview is forced so the weak
        box is visible. Both strategies are budget-capped and spread across the stream.
        """
        if "uncertain" in self.strategies and uncertain_boxes:
            label_lines = yolo_label_lines(confident_boxes, w, h)
            box, conf, _cls = max(uncertain_boxes, key=lambda b: b[1])
            meta = {
                "strategy": "uncertain",
                "event_id": "",
                "confidence": round(conf, 4),
                "select_score": "",
                "box_xyxy": json.dumps(list(box)),
                "box_cxcywh": _cxcywh(box, w, h),
                "frame_w": w,
                "frame_h": h,
            }
            self._uncertain.offer(
                frame_idx,
                lambda: self._write(frame, label_lines, annotated, frame_idx, meta, force_preview=True),
                self._delete,
            )

        if "negatives" in self.strategies and not confident_boxes and not uncertain_boxes:
            meta = {
                "strategy": "negatives", "event_id": "", "confidence": "",
                "select_score": "", "box_xyxy": "", "box_cxcywh": "",
                "frame_w": w, "frame_h": h,
            }
            self._neg.offer(
                frame_idx,
                lambda: self._write(frame, [], None, frame_idx, meta),
                self._delete,
            )

    # -- finalize ---------------------------------------------------------------

    def _all_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = list(self._best.values())
        for recs in self._worst.values():
            records.extend(recs)
        records.extend(st["rec"] for st in self._random.values() if st["rec"] is not None)
        records.extend(self._uncertain.records())
        records.extend(self._neg.records())
        return records

    def finalize(self) -> dict[str, int]:
        self.saved_counts["best"] = len(self._best)
        self.saved_counts["worst"] = sum(len(v) for v in self._worst.values())
        self.saved_counts["random"] = sum(1 for st in self._random.values() if st["rec"] is not None)
        self.saved_counts["uncertain"] = len(self._uncertain.records())
        self.saved_counts["negatives"] = len(self._neg.records())

        rows = [r["row"] for r in self._all_records()]
        # Group by strategy (all of a kind together), chronological within a group.
        strat_order = {s: i for i, s in enumerate(TRAIN_STRATEGIES)}
        rows.sort(key=lambda x: (strat_order.get(x["strategy"], 99), x["frame_idx"]))
        with (self.dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
            writer.writeheader()
            writer.writerows(rows)

        info = {
            "strategies": sorted(self.strategies),
            "budget": self.budget,
            "budget_applies_to": ["uncertain", "negatives"],
            "worst_per_event": self.worst_per_event,
            "uncertain_conf_band": [self.uncertain_min, self.conf_thr],
            "image_format": self.ext.lstrip("."),
            "preview_format": self.prev_ext.lstrip("."),
            "sampling": "uncertain/negatives: time-stratified reservoir (budget buckets, spread across stream)",
            "saved_counts": {k: self.saved_counts[k] for k in sorted(self.strategies)},
            "events_total": len(self._events_seen),
            "layout": "images/ (png) + labels/ (YOLO txt) + previews/ (jpg) + manifest.csv",
        }
        (self.dir / "_training_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.saved_counts

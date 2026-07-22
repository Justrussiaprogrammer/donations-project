#!/usr/bin/env python3
"""donsearcher — donation extraction pipeline (core logic, importable as a package).

`import donsearcher` exposes the whole public surface: geometry/text helpers, the
VLM client, event grouping, per-event VLM processing, dedup/totals, report writers,
training-data collection, and the orchestrator (run_pipeline) + CLI (main).

The heavy ultralytics/torch import is deferred into run_pipeline(), so importing
the package for its helpers (e.g. call_vlm_for_image) does not pay that cost.
Service scripts and tools live in ../scripts/ and import from here.
"""

from __future__ import annotations

from .geometry import (
    RECENT_HEIGHTS_MAXLEN,
    box_area,
    box_height,
    box_iou,
    center_distance_norm,
    clamp,
    expand_box,
    push_recent_height,
    reference_height,
)
from .textutil import (
    json_dumps_compact,
    parse_img_size,
    safe_filename,
    seconds_to_timestamp,
    unpack_images_schema,
)
from .vlm_client import (
    DEFAULT_PROMPT_VERSION,
    PROMPTS_DIR,
    call_vlm_for_image,
    extract_json_from_text,
    image_bgr_to_data_url,
    load_prompt,
)
from .events import (
    CandidateCrop,
    DonationEvent,
    _delete_candidate_files,
    add_candidate,
    find_matching_event,
)
from .vlm_events import (
    _make_error_rows,
    _process_event_vlm,
    _vlm_worker,
    build_event_rows,
)
from .dedup import (
    GENERIC_DONORS,
    GENERIC_MESSAGES,
    build_totals_rows,
    dedup_events,
    is_generic_donor,
    is_generic_message,
    is_identifying,
    normalize_amount,
    normalize_currency,
    normalize_donor,
    normalize_message,
)
from .reports import (
    FULL_META_FIELDS,
    build_events_meta_rows,
    build_run_record,
    meta_from_record,
    write_csv,
    write_events_meta,
    write_jsonl,
)
from .training import TRAIN_STRATEGIES, TrainingCollector, yolo_label_lines
from .pipeline import build_parser, main, run_pipeline

__all__ = [
    # geometry
    "RECENT_HEIGHTS_MAXLEN", "box_area", "box_height", "box_iou",
    "center_distance_norm", "clamp", "expand_box", "push_recent_height",
    "reference_height",
    # textutil
    "json_dumps_compact", "parse_img_size", "safe_filename", "seconds_to_timestamp",
    "unpack_images_schema",
    # vlm_client
    "DEFAULT_PROMPT_VERSION", "PROMPTS_DIR", "call_vlm_for_image",
    "extract_json_from_text", "image_bgr_to_data_url", "load_prompt",
    # events
    "CandidateCrop", "DonationEvent", "_delete_candidate_files",
    "add_candidate", "find_matching_event",
    # vlm_events
    "_make_error_rows", "_process_event_vlm", "_vlm_worker", "build_event_rows",
    # dedup
    "GENERIC_DONORS", "GENERIC_MESSAGES", "build_totals_rows", "dedup_events",
    "is_generic_donor", "is_generic_message", "is_identifying",
    "normalize_amount", "normalize_currency", "normalize_donor", "normalize_message",
    # reports
    "FULL_META_FIELDS", "build_events_meta_rows", "build_run_record",
    "meta_from_record", "write_csv", "write_events_meta", "write_jsonl",
    # training
    "TRAIN_STRATEGIES", "TrainingCollector", "yolo_label_lines",
    # pipeline
    "build_parser", "main", "run_pipeline",
]

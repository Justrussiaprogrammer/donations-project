#!/usr/bin/env python3
"""Donation dedup (re-shown / replayed donations), field normalization, and totals.

A streamer can replay an already-shown donation, which would otherwise be counted
twice. Dedup marks later exact-match repeats as duplicates (non-destructive flags)
and ``build_totals_rows`` excludes them. Pure stdlib (re/typing) — no heavy deps.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Generic donor names many people share — not identifying on their own (a streamer
# replaying an "Аноним" donation vs two different anonymous donors look the same).
GENERIC_DONORS = {"аноним", "анонім", "anonymous", "anon"}

# Platform-hidden / placeholder message contents — not identifying on their own.
GENERIC_MESSAGES = {"ссылка удалена", "(ссылка удалена)", "сообщение удалено", "link removed"}

_WS_RE = re.compile(r"\s+")
# Bracketed "ссылка удалена" placeholder in various bracket/spacing variants.
_LINK_REMOVED_RE = re.compile(r"[(\[{]\s*ссылка\s+удалена\s*[)\]}]", re.IGNORECASE)


def _collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def normalize_donor(donor: Any) -> str:
    """Casefold + collapse whitespace. Empty for None (so OCR/case noise on the same
    name still matches exactly, per the user's exact-match dedup policy)."""
    if donor is None:
        return ""
    return _collapse_ws(str(donor)).casefold()


def normalize_message(message: Any) -> str:
    """Casefold + collapse whitespace; canonicalize bracketed '(ссылка удалена)'
    placeholder so bracket/case noise doesn't defeat exact matching."""
    if message is None:
        return ""
    s = _LINK_REMOVED_RE.sub("(ссылка удалена)", _collapse_ws(str(message)))
    return s.casefold()


def normalize_currency(currency: Any) -> str:
    if currency is None:
        return ""
    s = _collapse_ws(str(currency)).upper()
    return "" if s in ("", "NO_CURRENCY") else s


def normalize_amount(amount: Any) -> Optional[float]:
    if amount in (None, ""):
        return None
    try:
        return float(amount)
    except (TypeError, ValueError):
        return None


def is_generic_donor(donor_norm: str) -> bool:
    return donor_norm == "" or donor_norm in GENERIC_DONORS


def is_generic_message(message_norm: str) -> bool:
    return message_norm == "" or message_norm in GENERIC_MESSAGES


def is_identifying(donor_norm: str, message_norm: str) -> bool:
    """True if the donation carries at least one identifying field. A donation with
    both a generic/empty donor AND a generic/empty message cannot be called a
    duplicate (two different 'Аноним' 200₽ with no message may be different people)."""
    return not is_generic_donor(donor_norm) or not is_generic_message(message_norm)


def dedup_events(
    event_rows: list[dict[str, Any]],
    jsonl_rows: list[dict[str, Any]],
) -> int:
    """Mark re-shown / replayed donations as duplicates IN PLACE (rows are flagged,
    not removed). A later event duplicates an earlier one when their normalized
    (donor, amount, currency, message) match EXACTLY and the donation is identifying.
    event_rows/jsonl_rows are parallel and sorted by event_id (= time order), so the
    earliest occurrence stays canonical. Duplicates are excluded from totals.
    Returns the number of events flagged."""
    seen: dict[tuple, Any] = {}
    dups = 0
    for ev_row, js_row in zip(event_rows, jsonl_rows):
        if not ev_row.get("parsed_ok"):
            continue
        amount = normalize_amount(ev_row.get("amount"))
        if amount is None:
            continue  # no amount -> not a totals donation, nothing to dedup
        donor_norm = normalize_donor(ev_row.get("donor"))
        msg_norm = normalize_message(ev_row.get("message"))
        if not is_identifying(donor_norm, msg_norm):
            continue  # no identifying info -> never call it a duplicate
        key = (donor_norm, amount, normalize_currency(ev_row.get("currency")), msg_norm)
        canon = seen.get(key)
        if canon is not None:
            ev_row["is_duplicate"] = True
            ev_row["duplicate_of_event_id"] = canon
            js_row["is_duplicate"] = True
            js_row["duplicate_of_event_id"] = canon
            dups += 1
        else:
            seen[key] = ev_row.get("event_id")
    return dups


def build_totals_rows(event_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Returns (per-currency totals, count of events excluded from totals).

    Excluded: parse failures, needs_review, low detection confidence, missing amount.
    Duplicates (is_duplicate) are also excluded but reported separately (dedup_events),
    so they are not counted here. The skipped count is reported so the totals are not
    silently undercounted.
    """
    totals: dict[str, dict[str, Any]] = {}
    skipped = 0

    for row in event_rows:
        if row.get("is_duplicate") is True:
            continue  # re-shown donation — counted once via the canonical event
        if not row.get("parsed_ok"):
            skipped += 1
            continue
        # Нечисловой conf ("" при minimal-метаданных VLM-стадии) — не фильтруем:
        # порог по confidence уже применил детектор на YOLO-стадии.
        conf = row.get("best_detection_confidence")
        low_conf = isinstance(conf, (int, float)) and conf < 0.5
        if row.get("needs_review") is True or low_conf:
            skipped += 1
            continue

        amount = row.get("amount")
        if amount in (None, ""):
            skipped += 1
            continue

        try:
            amount_float = float(amount)
        except Exception:
            skipped += 1
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
    return out, skipped

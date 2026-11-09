"""Cross-task musical consistency checks.

These checks are diagnostics, not hard schema errors.  Borrowed harmony,
modulation, rubato, and annotation conventions can all produce legitimate
exceptions, so warnings must be reviewed rather than silently "corrected".
"""

from __future__ import annotations

import re
from typing import Any


PITCH_CLASS = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}
QUALITY_INTERVALS = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10},
}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _parse_key(value: Any) -> tuple[int, str] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([A-Ga-g](?:#|b)?)\s+(major|minor)\s*", value)
    if not match:
        return None
    tonic = PITCH_CLASS.get(match.group(1).upper())
    return (tonic, match.group(2)) if tonic is not None else None


def _parse_chord(value: Any) -> set[int] | None:
    if value == "N":
        return set()
    if not isinstance(value, str):
        return None
    match = re.match(r"^([A-Ga-g](?:#|b)?):([^/(]+)", value)
    if not match:
        return None
    root = PITCH_CLASS.get(match.group(1).upper())
    quality = match.group(2)
    intervals = QUALITY_INTERVALS.get(quality)
    if root is None or intervals is None:
        return None
    return {(root + interval) % 12 for interval in intervals}


def _audit_timed_coverage(
    items: list[dict[str, Any]],
    *,
    duration_seconds: float | None,
    label: str,
) -> list[dict[str, Any]]:
    """Check ordering, overlap, internal gaps, and end coverage."""

    warnings: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, item in enumerate(items):
        start = _number(item.get("start_seconds"))
        end = _number(item.get("end_seconds"))
        if start is None or end is None:
            continue
        if end <= start:
            warnings.append(
                {
                    "code": f"{label}_NON_POSITIVE_DURATION",
                    "severity": "error",
                    "message": f"{label.lower()} item {index} must end after it starts",
                    "observed": {"start_seconds": start, "end_seconds": end},
                }
            )
        if start < previous_end - 1e-3:
            warnings.append(
                {
                    "code": f"{label}_OVERLAP_OR_ORDER",
                    "severity": "error",
                    "message": f"{label.lower()} item {index} starts before the previous item ends",
                    "observed": {"start_seconds": start, "previous_end_seconds": previous_end},
                }
            )
        elif index > 0 and start > previous_end + 0.5:
            warnings.append(
                {
                    "code": f"{label}_INTERNAL_GAP",
                    "severity": "warning",
                    "message": f"{label.lower()} proposals contain an internal temporal gap",
                    "observed": {"start_seconds": start, "previous_end_seconds": previous_end},
                }
            )
        previous_end = max(previous_end, end)
    if items and duration_seconds and previous_end < duration_seconds - 0.5:
        warnings.append(
            {
                "code": f"{label}_COVERAGE_GAP",
                "severity": "warning",
                "message": f"{label.lower()} proposals do not cover the end of the audio",
                "observed": {"last_end_seconds": previous_end, "duration_seconds": duration_seconds},
            }
        )
    return warnings


def audit_task_results(
    global_result: dict[str, Any] | None,
    structure_result: dict[str, Any] | None,
    harmony_result: dict[str, Any] | None,
    *,
    duration_seconds: float | None,
) -> list[dict[str, Any]]:
    """Return reviewable warnings for cross-task and temporal contradictions."""

    global_result = global_result or {}
    structure_result = structure_result or {}
    harmony_result = harmony_result or {}
    warnings: list[dict[str, Any]] = []

    meter = global_result.get("time_signature")
    if meter is not None and (not isinstance(meter, str) or not re.fullmatch(r"\d{1,2}/\d{1,2}", meter.strip())):
        warnings.append(
            {
                "code": "NON_CANONICAL_METER",
                "severity": "warning",
                "message": "time_signature must contain one canonical value such as 4/4; alternatives belong in an Or choice",
                "observed": meter,
            }
        )

    tempo = _number(global_result.get("tempo_bpm"))
    if tempo is not None and not 20 <= tempo <= 300:
        warnings.append(
            {
                "code": "IMPLAUSIBLE_TEMPO",
                "severity": "warning",
                "message": "tempo_bpm is outside the broad validation range [20, 300]",
                "observed": tempo,
            }
        )

    global_key = global_result.get("global_key")
    parsed_key = _parse_key(global_key)
    if global_key is not None and parsed_key is None:
        warnings.append(
            {
                "code": "NON_CANONICAL_KEY",
                "severity": "warning",
                "message": "global_key must contain one canonical value such as E major; alternatives belong in an Or choice",
                "observed": global_key,
            }
        )

    sections = [item for item in structure_result.get("sections", []) if isinstance(item, dict)]
    warnings.extend(
        _audit_timed_coverage(sections, duration_seconds=duration_seconds, label="SECTION")
    )
    seen_section_ids: set[str] = set()
    for index, section in enumerate(sections):
        repeated_from = section.get("repeated_from")
        if repeated_from is not None and str(repeated_from) not in seen_section_ids:
            warnings.append(
                {
                    "code": "INVALID_SECTION_REPEAT_REFERENCE",
                    "severity": "error",
                    "message": f"section {index} repeated_from must reference an earlier section",
                    "observed": repeated_from,
                }
            )
        if section.get("id") is not None:
            seen_section_ids.add(str(section["id"]))

    key_changes = [item for item in global_result.get("key_changes", []) if isinstance(item, dict)]
    warnings.extend(
        _audit_timed_coverage(key_changes, duration_seconds=duration_seconds, label="KEY_SEGMENT")
    )

    chords = [item for item in harmony_result.get("chords", []) if isinstance(item, dict)]
    warnings.extend(
        _audit_timed_coverage(chords, duration_seconds=duration_seconds, label="HARMONY")
    )
    if parsed_key and chords:
        tonic, mode = parsed_key
        scale_intervals = {0, 2, 4, 5, 7, 9, 11} if mode == "major" else {0, 2, 3, 5, 7, 8, 10}
        scale = {(tonic + interval) % 12 for interval in scale_intervals}
        weighted_in_scale = 0.0
        weighted_total = 0.0
        unsupported: list[str] = []
        for chord in chords:
            pitches = _parse_chord(chord.get("label"))
            if pitches is None:
                unsupported.append(str(chord.get("label")))
                continue
            if not pitches:
                continue
            start = _number(chord.get("start_seconds")) or 0.0
            end = _number(chord.get("end_seconds")) or start
            weight = max(0.001, end - start)
            weighted_in_scale += weight * len(pitches & scale) / len(pitches)
            weighted_total += weight
        if weighted_total:
            consistency = weighted_in_scale / weighted_total
            if consistency < 0.6:
                warnings.append(
                    {
                        "code": "LOW_KEY_CHORD_CONSISTENCY",
                        "severity": "warning",
                        "message": "global key and timestamped chord proposals have low pitch-class consistency; review both tasks",
                        "observed": {
                            "global_key": global_result.get("global_key"),
                            "weighted_in_scale_ratio": round(consistency, 4),
                        },
                    }
                )
        if unsupported:
            warnings.append(
                {
                    "code": "UNSUPPORTED_CHORD_LABELS",
                    "severity": "info",
                    "message": "some chord labels were not included in the basic consistency audit",
                    "observed": sorted(set(unsupported)),
                }
            )

    return warnings

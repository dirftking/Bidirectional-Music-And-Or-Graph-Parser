"""Deterministic assembly of task-level observations into an AOG candidate set."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .validation import REPRESENTATION, SCHEMA_VERSION


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _confidence(value: Any, default: float = 0.5) -> float:
    return min(1.0, max(0.0, _number(value, default)))


def _span(start: Any, end: Any) -> dict[str, float]:
    clean_start = max(0.0, _number(start))
    clean_end = max(clean_start, _number(end, clean_start))
    return {"start_seconds": round(clean_start, 6), "end_seconds": round(clean_end, 6)}


def _has_positive_span(item: dict[str, Any]) -> bool:
    """Return whether an observation can become a temporal graph node."""

    start = _number(item.get("start_seconds"))
    end = _number(item.get("end_seconds"), start)
    return start >= 0.0 and end > start


def _node(
    node_id: str,
    kind: str,
    level: str,
    node_type: str,
    span: dict[str, float],
    attributes: dict[str, Any],
    confidence: float,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": kind,
        "level": level,
        "type": node_type,
        "span": span,
        "attributes": attributes,
        "confidence": _confidence(confidence),
        "evidence": evidence or [],
    }


def _normalise_choices(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    deduplicated: dict[str, float] = {}
    for label, score in items:
        if label:
            deduplicated[str(label)] = max(deduplicated.get(str(label), 0.0), _confidence(score))
    total = sum(deduplicated.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(deduplicated))
        return [(label, uniform) for label in deduplicated]
    return [(label, score / total) for label, score in deduplicated.items()]


def _best_section_for_chord(chord: dict[str, Any], sections: list[dict[str, Any]]) -> int | None:
    start = _number(chord.get("start_seconds"))
    end = _number(chord.get("end_seconds"), start)
    for index, section in enumerate(sections):
        section_start = _number(section.get("start_seconds"))
        section_end = _number(section.get("end_seconds"), section_start)
        if start >= section_start - 1e-3 and end <= section_end + 1e-3:
            return index
    # A chord crossing a section boundary belongs to the song-level timeline.
    # Splitting it here would mutate the source annotation and hide uncertainty.
    return None


def assemble_candidate_graph(
    global_result: dict[str, Any] | None,
    structure_result: dict[str, Any] | None,
    harmony_result: dict[str, Any] | None,
    *,
    audio_sha256: str,
    audio_filename: str,
    duration_seconds: float | None = None,
    model: str = "MOSS-Music",
) -> dict[str, Any]:
    """Build a graph while preserving ambiguous section labels as Or nodes."""

    global_result = global_result or {}
    structure_result = structure_result or {}
    harmony_result = harmony_result or {}
    raw_sections = [item for item in structure_result.get("sections", []) if isinstance(item, dict)]
    raw_chords = [item for item in harmony_result.get("chords", []) if isinstance(item, dict)]
    sections = [item for item in raw_sections if _has_positive_span(item)]
    chords = [item for item in raw_chords if _has_positive_span(item)]
    omitted_sections = [item for item in raw_sections if not _has_positive_span(item)]
    omitted_chords = [item for item in raw_chords if not _has_positive_span(item)]

    observed_end = max(
        [_number(duration_seconds), _number(global_result.get("duration_seconds"))]
        + [_number(item.get("end_seconds")) for item in sections]
        + [_number(item.get("end_seconds")) for item in chords]
    )
    root_span = _span(0.0, observed_end)
    nodes: list[dict[str, Any]] = []
    productions: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    uncertainties: list[dict[str, Any]] = []

    for source_name, omitted in (("structure", omitted_sections), ("harmony", omitted_chords)):
        for item in omitted:
            uncertainties.append(
                {
                    "scope_id": None,
                    "field": f"{source_name}_observation_omitted",
                    "alternatives": [],
                    "explanation": (
                        "Observation omitted from the graph because its time span is not positive: "
                        f"start={item.get('start_seconds')!r}, end={item.get('end_seconds')!r}, "
                        f"source_id={item.get('id')!r}, label={item.get('label')!r}."
                    ),
                }
            )

    root_attributes = {
        "tempo_bpm": global_result.get("tempo_bpm"),
        "time_signature": global_result.get("time_signature"),
        "global_key": global_result.get("global_key"),
        "key_changes": global_result.get("key_changes", []),
    }
    nodes.append(
        _node(
            "song_000",
            "and",
            "song",
            "song",
            root_span,
            root_attributes,
            global_result.get("confidence", 0.5),
        )
    )

    section_ids: list[str] = []
    section_children: dict[int, list[str]] = {}
    input_id_to_node: dict[str, str] = {}
    for index, section in enumerate(sections, start=1):
        section_id = f"section_{index:03d}"
        section_ids.append(section_id)
        if section.get("id"):
            input_id_to_node[str(section["id"])] = section_id
        span = _span(section.get("start_seconds"), section.get("end_seconds"))
        confidence = _confidence(section.get("confidence"))
        nodes.append(
            _node(
                section_id,
                "and",
                "section",
                "section",
                span,
                {"source_id": section.get("id"), "selected_label": section.get("label")},
                confidence,
            )
        )

        label_or_id = f"section_label_choice_{index:03d}"
        nodes.append(_node(label_or_id, "or", "section", "section_label", span, {}, confidence))
        section_children[index - 1] = [label_or_id]
        choices = [(str(section.get("label") or "unknown"), confidence)]
        for alternative in section.get("alternative_labels", []) or []:
            if isinstance(alternative, dict):
                choices.append((str(alternative.get("label") or ""), _confidence(alternative.get("confidence"))))
            elif isinstance(alternative, str):
                choices.append((alternative, max(0.05, 1.0 - confidence)))
        normalised = _normalise_choices(choices)
        for choice_index, (label, probability) in enumerate(normalised, start=1):
            label_id = f"section_label_{index:03d}_{choice_index:02d}"
            nodes.append(
                _node(
                    label_id,
                    "terminal",
                    "section",
                    "section_label",
                    span,
                    {"label": label},
                    probability,
                    [
                        {
                            "start_seconds": span["start_seconds"],
                            "end_seconds": span["end_seconds"],
                            "description": str(section.get("evidence") or "MOSS-Music section proposal"),
                            "source": "moss_music",
                        }
                    ],
                )
            )
            productions.append(
                {
                    "id": f"prod_section_label_{index:03d}_{choice_index:02d}",
                    "parent": label_or_id,
                    "children": [label_id],
                    "rule": "or_choice",
                    "probability": probability,
                    "selected": choice_index == 1,
                }
            )
        if len(normalised) > 1:
            uncertainties.append(
                {
                    "scope_id": label_or_id,
                    "field": "section_label",
                    "alternatives": [
                        {"value": label, "confidence": probability, "source": "moss_music"}
                        for label, probability in normalised
                    ],
                    "explanation": "Multiple plausible functional labels for the same time span.",
                }
            )

    chord_ids: list[str] = []
    for index, chord in enumerate(chords, start=1):
        chord_id = f"chord_{index:04d}"
        chord_ids.append(chord_id)
        span = _span(chord.get("start_seconds"), chord.get("end_seconds"))
        confidence = _confidence(chord.get("confidence"))
        nodes.append(
            _node(
                chord_id,
                "terminal",
                "harmony",
                "chord",
                span,
                {"label": chord.get("label") or "N"},
                confidence,
                [
                    {
                        "start_seconds": span["start_seconds"],
                        "end_seconds": span["end_seconds"],
                        "description": str(chord.get("evidence") or "MOSS-Music chord proposal"),
                        "source": "moss_music",
                    }
                ],
            )
        )
        owner = _best_section_for_chord(chord, sections)
        if owner is not None:
            section_children[owner].append(chord_id)

    root_children = list(section_ids)
    assigned_chords = {child for children in section_children.values() for child in children if child.startswith("chord_")}
    root_children.extend(chord_id for chord_id in chord_ids if chord_id not in assigned_chords)
    if root_children:
        productions.append(
            {
                "id": "prod_song_decomposition",
                "parent": "song_000",
                "children": root_children,
                "rule": "and_decomposition",
                "probability": 1.0,
                "selected": True,
            }
        )
    for index, section_id in enumerate(section_ids):
        productions.append(
            {
                "id": f"prod_section_decomposition_{index + 1:03d}",
                "parent": section_id,
                "children": section_children[index],
                "rule": "and_decomposition",
                "probability": 1.0,
                "selected": True,
            }
        )

    for index, (source, target) in enumerate(zip(section_ids, section_ids[1:]), start=1):
        relations.append(
            {
                "id": f"rel_section_succession_{index:03d}",
                "source": source,
                "target": target,
                "type": "succession",
                "attributes": {},
                "confidence": 1.0,
                "evidence": [],
            }
        )
    for index, (source, target) in enumerate(zip(chord_ids, chord_ids[1:]), start=1):
        relations.append(
            {
                "id": f"rel_chord_succession_{index:04d}",
                "source": source,
                "target": target,
                "type": "succession",
                "attributes": {},
                "confidence": min(nodes[-len(chord_ids) + index - 1]["confidence"], nodes[-len(chord_ids) + index]["confidence"])
                if chord_ids
                else 1.0,
                "evidence": [],
            }
        )
    for index, section in enumerate(sections):
        repeated_from = section.get("repeated_from")
        source = input_id_to_node.get(str(repeated_from)) if repeated_from is not None else None
        target = section_ids[index] if index < len(section_ids) else None
        if source and target and source != target:
            relations.append(
                {
                    "id": f"rel_section_variation_{index + 1:03d}",
                    "source": source,
                    "target": target,
                    "type": "variation",
                    "attributes": {"subtype": "repetition_or_variation"},
                    "confidence": _confidence(section.get("confidence")),
                    "evidence": [],
                }
            )

    for source_name, payload in (
        ("global", global_result),
        ("structure", structure_result),
        ("harmony", harmony_result),
    ):
        for item in payload.get("uncertainties", []) or []:
            uncertainties.append(
                {
                    "scope_id": None,
                    "field": source_name,
                    "alternatives": [],
                    "explanation": str(item),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "representation": REPRESENTATION,
        "root": "song_000",
        "nodes": nodes,
        "productions": productions,
        "relations": relations,
        "uncertainties": uncertainties,
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "audio_sha256": audio_sha256,
            "audio_filename": audio_filename,
            "model": model,
            "assembler": "audio2musicaog/0.2",
        },
    }

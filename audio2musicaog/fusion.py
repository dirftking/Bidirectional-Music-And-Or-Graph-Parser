"""Fuse MOSS semantic observations with beat-synchronous MIR candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .validation import validate_graph


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _confidence(value: Any, default: float = 0.5) -> float:
    return min(1.0, max(0.0, _number(value, default)))


def _artifact_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _merge_candidates(items: list[dict[str, Any]], selected_label: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        entry = merged.setdefault(
            label, {"label": label, "score": 0.0, "sources": [], "source_scores": {}}
        )
        confidence = _confidence(item.get("confidence"))
        entry["score"] = max(entry["score"], confidence)
        source = str(item.get("source") or "unknown")
        if source not in entry["sources"]:
            entry["sources"].append(source)
        entry["source_scores"][source] = max(entry["source_scores"].get(source, 0.0), confidence)
    if not merged:
        merged["N"] = {
            "label": "N", "score": 1.0, "sources": ["fallback"], "source_scores": {"fallback": 1.0}
        }
        selected_label = "N"
    ordered = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    if selected_label not in merged:
        selected_label = ordered[0]["label"]
    total = sum(max(item["score"], 1e-6) for item in ordered)
    for item in ordered:
        item["probability"] = max(item["score"], 1e-6) / total
        item["selected"] = item["label"] == selected_label
    return ordered


def _remove_existing_harmony(graph: dict[str, Any]) -> None:
    removed = {node["id"] for node in graph.get("nodes", []) if node.get("type") == "chord"}
    graph["nodes"] = [node for node in graph.get("nodes", []) if node.get("id") not in removed]
    productions = []
    for production in graph.get("productions", []):
        if production.get("parent") in removed:
            continue
        updated = copy.deepcopy(production)
        updated["children"] = [child for child in updated.get("children", []) if child not in removed]
        if updated["children"]:
            productions.append(updated)
    graph["productions"] = productions
    graph["relations"] = [
        relation for relation in graph.get("relations", [])
        if relation.get("source") not in removed and relation.get("target") not in removed
    ]


def fuse_moss_mir(moss_artifact: dict[str, Any], mir_artifact: dict[str, Any]) -> dict[str, Any]:
    """Build a valid candidate AOG using MOSS semantics and MIR timing.

    MIR's top candidate is selected by a fixed, pre-registered policy. MOSS
    harmony remains as an alternative when it overlaps a beat interval; its
    self-reported confidence does not override the selection on this test set.
    """

    moss_audio = moss_artifact.get("audio", {})
    mir_audio = mir_artifact.get("audio", {})
    if moss_audio.get("sha256") and mir_audio.get("sha256") and moss_audio["sha256"] != mir_audio["sha256"]:
        raise ValueError("MOSS and MIR artifacts refer to different audio")
    if not isinstance(moss_artifact.get("graph"), dict):
        raise ValueError("MOSS artifact has no candidate graph")

    graph = copy.deepcopy(moss_artifact["graph"])
    _remove_existing_harmony(graph)
    root_id = graph["root"]
    root = next(node for node in graph["nodes"] if node["id"] == root_id)
    root_span = root["span"]
    root_children: list[str] = []
    boundary_by_time = {
        round(_number(item.get("time_seconds")), 6): _confidence(item.get("confidence"))
        for item in mir_artifact.get("structure_boundary_candidates", []) if isinstance(item, dict)
    }

    beat_ids: list[str] = []
    for index, beat in enumerate(mir_artifact.get("metrical_grid", {}).get("beats", []), start=1):
        time = round(_number(beat.get("time_seconds")), 6)
        beat_id = f"fused_beat_{index:04d}"
        beat_ids.append(beat_id)
        graph["nodes"].append(
            {
                "id": beat_id,
                "kind": "terminal",
                "level": "metrical",
                "type": "beat",
                "span": {"start_seconds": time, "end_seconds": time},
                "attributes": {
                    "index": beat.get("index", index - 1),
                    "assumed_bar_position": beat.get("assumed_bar_position"),
                    "structure_boundary_confidence": boundary_by_time.get(time),
                },
                "confidence": _confidence(mir_artifact.get("tempo", {}).get("confidence")),
                "evidence": [{
                    "start_seconds": time, "end_seconds": time,
                    "description": "Beat detected from onset strength and dynamic-programming tracking.",
                    "source": "librosa.beat.beat_track",
                }],
            }
        )
        root_children.append(beat_id)

    moss_global = moss_artifact.get("tasks", {}).get("global", {}).get("parsed") or {}
    mir_keys = mir_artifact.get("key_candidates", [])
    key_items = [
        {"label": item.get("label"), "confidence": item.get("confidence"), "source": "librosa_key_profile"}
        for item in mir_keys if isinstance(item, dict)
    ]
    if moss_global.get("global_key"):
        key_items.append({
            "label": moss_global["global_key"],
            "confidence": moss_global.get("confidence", 0.5),
            "source": "moss_music",
        })
    selected_key = str(mir_keys[0].get("label")) if mir_keys else str(moss_global.get("global_key") or "unknown")
    key_candidates = _merge_candidates(key_items, selected_key)
    key_or_id = "fused_key_choice_0001"
    graph["nodes"].append({
        "id": key_or_id, "kind": "or", "level": "song", "type": "key_event",
        "span": copy.deepcopy(root_span), "attributes": {"selection_policy": "mir_top_candidate"},
        "confidence": next(item["probability"] for item in key_candidates if item["selected"]), "evidence": [],
    })
    root_children.append(key_or_id)
    for index, candidate in enumerate(key_candidates, start=1):
        node_id = f"fused_key_candidate_{index:02d}"
        graph["nodes"].append({
            "id": node_id, "kind": "terminal", "level": "song", "type": "key_event",
            "span": copy.deepcopy(root_span),
            "attributes": {
                "label": candidate["label"],
                "sources": candidate["sources"],
                "source_scores": candidate["source_scores"],
            },
            "confidence": candidate["probability"], "evidence": [],
        })
        graph["productions"].append({
            "id": f"prod_fused_key_{index:02d}", "parent": key_or_id, "children": [node_id],
            "rule": "or_choice", "probability": candidate["probability"], "selected": candidate["selected"],
        })

    moss_chords = moss_artifact.get("tasks", {}).get("harmony", {}).get("parsed", {}).get("chords", [])
    chord_choice_ids: list[str] = []
    for index, interval in enumerate(mir_artifact.get("beat_chord_candidates", []), start=1):
        start = _number(interval.get("start_seconds"))
        end = _number(interval.get("end_seconds"), start)
        if end <= start:
            continue
        midpoint = (start + end) / 2
        mir_candidates = interval.get("candidates", [])
        selected_label = str(mir_candidates[0].get("label")) if mir_candidates else "N"
        candidates = [
            {"label": item.get("label"), "confidence": item.get("confidence"), "source": "librosa_chord_template"}
            for item in mir_candidates if isinstance(item, dict)
        ]
        for chord in moss_chords:
            if not isinstance(chord, dict):
                continue
            chord_start = _number(chord.get("start_seconds"))
            chord_end = _number(chord.get("end_seconds"), chord_start)
            if chord_start <= midpoint < chord_end:
                candidates.append({
                    "label": chord.get("label"), "confidence": chord.get("confidence"), "source": "moss_music"
                })
                break
        merged = _merge_candidates(candidates, selected_label)
        choice_id = f"fused_chord_choice_{index:04d}"
        chord_choice_ids.append(choice_id)
        span = {"start_seconds": round(start, 6), "end_seconds": round(end, 6)}
        graph["nodes"].append({
            "id": choice_id, "kind": "or", "level": "harmony", "type": "chord", "span": span,
            "attributes": {"selection_policy": "mir_top_candidate", "beat_interval_index": interval.get("index")},
            "confidence": next(item["probability"] for item in merged if item["selected"]), "evidence": [],
        })
        root_children.append(choice_id)
        for candidate_index, candidate in enumerate(merged, start=1):
            node_id = f"fused_chord_candidate_{index:04d}_{candidate_index:02d}"
            graph["nodes"].append({
                "id": node_id, "kind": "terminal", "level": "harmony", "type": "chord", "span": span,
                "attributes": {
                    "label": candidate["label"],
                    "sources": candidate["sources"],
                    "source_scores": candidate["source_scores"],
                },
                "confidence": candidate["probability"],
                "evidence": [{
                    "start_seconds": span["start_seconds"], "end_seconds": span["end_seconds"],
                    "description": "Beat-synchronous chord candidate with optional MOSS alternative.",
                    "source": "+".join(candidate["sources"]),
                }],
            })
            graph["productions"].append({
                "id": f"prod_fused_chord_{index:04d}_{candidate_index:02d}",
                "parent": choice_id, "children": [node_id], "rule": "or_choice",
                "probability": candidate["probability"], "selected": candidate["selected"],
            })

    for index, (source, target) in enumerate(zip(beat_ids, beat_ids[1:]), start=1):
        graph["relations"].append({
            "id": f"rel_fused_beat_{index:04d}", "source": source, "target": target,
            "type": "succession", "attributes": {}, "confidence": 1.0, "evidence": [],
        })
    for index, (source, target) in enumerate(zip(chord_choice_ids, chord_choice_ids[1:]), start=1):
        graph["relations"].append({
            "id": f"rel_fused_chord_{index:04d}", "source": source, "target": target,
            "type": "succession", "attributes": {}, "confidence": 1.0, "evidence": [],
        })

    root_production = next((item for item in graph["productions"] if item.get("parent") == root_id), None)
    if root_production is None:
        graph["productions"].append({
            "id": "prod_song_decomposition", "parent": root_id, "children": root_children,
            "rule": "and_decomposition", "probability": 1.0, "selected": True,
        })
    else:
        root_production["children"].extend(root_children)
    graph["provenance"].update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "MOSS-Music + librosa",
        "assembler": "audio2musicaog-fusion/0.2",
    })
    errors = validate_graph(graph)
    return {
        "artifact_version": "audio2musicaog-fusion/0.2",
        "status": "complete" if not errors else "partial",
        "selection_policy": {
            "tempo": "mir",
            "key": "mir_top_candidate",
            "chord": "mir_top_candidate",
            "section_semantics": "moss_music",
            "weights_tuned_on_test_set": False,
            "candidate_scores_preserved_by_source": True,
        },
        "source_artifacts": {"moss_sha256": _artifact_hash(moss_artifact), "mir_sha256": _artifact_hash(mir_artifact)},
        "graph": graph,
        "graph_errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

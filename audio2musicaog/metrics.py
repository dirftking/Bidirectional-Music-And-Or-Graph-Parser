"""Deterministic baseline metrics for Audio2MusicAOG experiments."""

from __future__ import annotations

import re
from typing import Any


ROOT_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}
QUALITY_ALIASES = {
    "M": "maj", "major": "maj", "m": "min", "minor": "min", "D7": "7",
    "M7": "maj7", "m7": "min7", "o": "dim", "o7": "dim7", "/o7": "hdim7",
}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _canonical_chord(label: Any) -> tuple[int, str, int | None] | None:
    if label == "N":
        return (-1, "N", None)
    if not isinstance(label, str):
        return None
    match = re.fullmatch(r"([A-Ga-g](?:#|b)?):([^/]+)(?:/([A-Ga-g](?:#|b)?))?", label.strip())
    if not match:
        return None
    root = ROOT_PC.get(match.group(1).upper())
    bass = ROOT_PC.get(match.group(3).upper()) if match.group(3) else None
    if root is None:
        return None
    return root, QUALITY_ALIASES.get(match.group(2), match.group(2)), bass


def _canonical_key(label: Any) -> tuple[int, str] | None:
    if not isinstance(label, str):
        return None
    cleaned = label.strip().replace("♯", "#").replace("♭", "b")
    match = re.fullmatch(r"([A-Ga-g](?:#|b)?)\s+(major|minor|maj|min)", cleaned, re.IGNORECASE)
    if not match:
        return None
    root = ROOT_PC.get(match.group(1).upper())
    if root is None:
        return None
    mode = "major" if match.group(2).lower() in {"major", "maj"} else "minor"
    return root, mode


def key_accuracy(predicted: Any, reference: Any) -> dict[str, Any]:
    """Compare keys while treating enharmonic spellings as equivalent."""

    ref = _canonical_key(reference)
    pred = _canonical_key(predicted)
    if ref is None:
        return {"available": False, "enharmonic_exact": None, "string_exact": None}
    return {
        "available": True,
        "enharmonic_exact": pred == ref,
        "string_exact": isinstance(predicted, str) and predicted.strip() == str(reference).strip(),
        "predicted_canonical": list(pred) if pred is not None else None,
        "reference_canonical": list(ref),
    }


def tempo_accuracy(predicted: Any, reference: Any, tolerance: float = 0.04) -> dict[str, Any]:
    pred, ref = _number(predicted), _number(reference)
    if pred is None or ref is None or ref <= 0:
        return {"available": False, "acc1": None, "acc2": None, "relative_error": None}
    relative_error = abs(pred / ref - 1.0)
    acc1 = relative_error <= tolerance
    acc2 = acc1 or abs(pred / (ref * 2.0) - 1.0) <= tolerance or abs(pred / (ref * 0.5) - 1.0) <= tolerance
    return {
        "available": True,
        "acc1": float(acc1),
        "acc2": float(acc2),
        "relative_error": round(relative_error, 6),
    }


def boundary_prf(predicted_sections: list[dict[str, Any]], reference_sections: list[dict[str, Any]], tolerance: float = 3.0) -> dict[str, Any]:
    def boundaries(sections: list[dict[str, Any]]) -> list[float]:
        values = sorted({_number(item.get("start_seconds")) for item in sections})
        return [value for value in values if value is not None and value > 1e-6]

    predicted, reference = boundaries(predicted_sections), boundaries(reference_sections)
    if not reference:
        return {"available": False, "precision": None, "recall": None, "f1": None, "tolerance_seconds": tolerance}
    unmatched = set(range(len(reference)))
    matches = 0
    for boundary in predicted:
        candidates = [(abs(boundary - reference[index]), index) for index in unmatched]
        if candidates:
            distance, index = min(candidates)
            if distance <= tolerance:
                unmatched.remove(index)
                matches += 1
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "available": True,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "matches": matches,
        "predicted_boundaries": len(predicted),
        "reference_boundaries": len(reference),
        "tolerance_seconds": tolerance,
    }


def chord_weighted_accuracy(predicted: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    def valid(items: list[dict[str, Any]]) -> list[tuple[float, float, Any]]:
        result = []
        for item in items:
            start, end = _number(item.get("start_seconds")), _number(item.get("end_seconds"))
            if start is not None and end is not None and end > start:
                result.append((start, end, item.get("label")))
        return result

    pred, ref = valid(predicted), valid(reference)
    if not ref:
        return {"available": False, "weighted_exact": None, "root_accuracy": None, "coverage": None}
    boundaries = sorted({value for start, end, _ in pred + ref for value in (start, end)})
    exact = root_match = covered = total = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint, duration = (left + right) / 2.0, right - left
        ref_label = next((label for start, end, label in ref if start <= midpoint < end), None)
        if ref_label is None:
            continue
        total += duration
        pred_label = next((label for start, end, label in pred if start <= midpoint < end), None)
        if pred_label is None:
            continue
        covered += duration
        pred_chord, ref_chord = _canonical_chord(pred_label), _canonical_chord(ref_label)
        if pred_chord is not None and ref_chord is not None:
            if pred_chord == ref_chord:
                exact += duration
            if pred_chord[0] == ref_chord[0]:
                root_match += duration
    return {
        "available": True,
        "weighted_exact": round(exact / total, 6) if total else None,
        "root_accuracy": round(root_match / total, 6) if total else None,
        "coverage": round(covered / total, 6) if total else None,
        "reference_seconds": round(total, 6),
    }


def evaluate_artifact(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    tasks = prediction.get("tasks", {})
    predicted_global = tasks.get("global", {}).get("parsed") or {}
    predicted_structure = tasks.get("structure", {}).get("parsed") or {}
    predicted_harmony = tasks.get("harmony", {}).get("parsed") or {}
    reference_global = ground_truth.get("global", {})
    key = key_accuracy(predicted_global.get("global_key"), reference_global.get("global_key"))
    return {
        "metric_version": "audio2musicaog-metrics/0.2",
        "source_id": ground_truth.get("source_id"),
        "tempo": tempo_accuracy(predicted_global.get("tempo_bpm"), reference_global.get("tempo_bpm")),
        "time_signature_exact": (
            predicted_global.get("time_signature") == reference_global.get("time_signature")
            if reference_global.get("time_signature") is not None else None
        ),
        "key": key,
        "key_exact": key["enharmonic_exact"],
        "key_string_exact": key["string_exact"],
        "chord": chord_weighted_accuracy(
            predicted_harmony.get("chords", []), ground_truth.get("harmony", {}).get("chords", [])
        ),
        "section_boundary": boundary_prf(
            predicted_structure.get("sections", []), ground_truth.get("structure", {}).get("sections", [])
        ),
        "prediction_graph_valid": not prediction.get("graph_errors", []),
        "prediction_audit_warning_count": len(prediction.get("audit_warnings", [])),
    }


def evaluate_mir_artifact(mir: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a beat-synchronous MIR candidate artifact against the same Ground Truth."""

    reference_global = ground_truth.get("global", {})
    key_candidates = mir.get("key_candidates", [])
    predicted_key = key_candidates[0].get("label") if key_candidates else None
    key = key_accuracy(predicted_key, reference_global.get("global_key"))
    predicted_sections = [
        {"start_seconds": item.get("time_seconds")}
        for item in mir.get("structure_boundary_candidates", [])
        if isinstance(item, dict)
    ]
    return {
        "metric_version": "beat-synchronous-mir-metrics/0.1",
        "source_id": ground_truth.get("source_id"),
        "tempo": tempo_accuracy(mir.get("tempo", {}).get("bpm"), reference_global.get("tempo_bpm")),
        "key": key,
        "key_exact": key["enharmonic_exact"],
        "key_string_exact": key["string_exact"],
        "chord": chord_weighted_accuracy(
            mir.get("chord_segments", []), ground_truth.get("harmony", {}).get("chords", [])
        ),
        "section_boundary": boundary_prf(predicted_sections, ground_truth.get("structure", {}).get("sections", [])),
        "beat_count": len(mir.get("metrical_grid", {}).get("beats", [])),
    }


def selected_graph_observations(graph: dict[str, Any]) -> dict[str, Any]:
    """Project selected MusicAOG branches into the common metric interface."""

    nodes = {
        node.get("id"): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    selected_child: dict[str, dict[str, Any]] = {}
    for production in graph.get("productions", []):
        if not isinstance(production, dict) or production.get("selected") is not True:
            continue
        parent = nodes.get(production.get("parent"))
        children = production.get("children", [])
        child = nodes.get(children[0]) if len(children) == 1 else None
        if parent and parent.get("kind") == "or" and child:
            selected_child[parent["id"]] = child

    keys = []
    chords = []
    sections = []
    for node in nodes.values():
        if node.get("kind") == "or" and node.get("type") == "key_event":
            child = selected_child.get(node["id"])
            if child:
                keys.append((float(node.get("span", {}).get("start_seconds", 0.0)), child))
        elif node.get("kind") == "or" and node.get("type") == "chord":
            child = selected_child.get(node["id"])
            if child:
                chords.append({
                    "start_seconds": node.get("span", {}).get("start_seconds"),
                    "end_seconds": node.get("span", {}).get("end_seconds"),
                    "label": child.get("attributes", {}).get("label"),
                })
        elif node.get("type") == "section" and node.get("kind") == "and":
            sections.append({
                "start_seconds": node.get("span", {}).get("start_seconds"),
                "end_seconds": node.get("span", {}).get("end_seconds"),
                "label": node.get("attributes", {}).get("selected_label"),
            })

    root = nodes.get(graph.get("root"), {})
    attributes = root.get("attributes", {}) if isinstance(root, dict) else {}
    keys.sort(key=lambda item: item[0])
    chords.sort(key=lambda item: float(item.get("start_seconds") or 0.0))
    sections.sort(key=lambda item: float(item.get("start_seconds") or 0.0))
    return {
        "global": {
            "tempo_bpm": attributes.get("tempo_bpm"),
            "time_signature": attributes.get("time_signature"),
            "global_key": keys[0][1].get("attributes", {}).get("label") if keys else attributes.get("global_key"),
        },
        "harmony": {"chords": chords},
        "structure": {"sections": sections},
    }


def evaluate_graph_artifact(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Evaluate selected graph branches without using non-graph side channels."""

    graph = prediction.get("graph") if isinstance(prediction.get("graph"), dict) else prediction
    observations = selected_graph_observations(graph)
    predicted_global = observations["global"]
    reference_global = ground_truth.get("global", {})
    key = key_accuracy(predicted_global.get("global_key"), reference_global.get("global_key"))
    return {
        "metric_version": "audio2musicaog-selected-graph-metrics/0.1",
        "source_id": ground_truth.get("source_id"),
        "tempo": tempo_accuracy(predicted_global.get("tempo_bpm"), reference_global.get("tempo_bpm")),
        "time_signature_exact": (
            predicted_global.get("time_signature") == reference_global.get("time_signature")
            if reference_global.get("time_signature") is not None else None
        ),
        "key": key,
        "key_exact": key["enharmonic_exact"],
        "key_string_exact": key["string_exact"],
        "chord": chord_weighted_accuracy(
            observations["harmony"]["chords"], ground_truth.get("harmony", {}).get("chords", [])
        ),
        "section_boundary": boundary_prf(
            observations["structure"]["sections"], ground_truth.get("structure", {}).get("sections", [])
        ),
        "prediction_graph_valid": not prediction.get("graph_errors", []),
    }

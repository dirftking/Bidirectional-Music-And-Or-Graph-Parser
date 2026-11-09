"""Bidirectional Music And-Or Graph Parser inference over a fused graph.

The first inference baseline deliberately uses one transparent top-down rule:
the selected global key predicts which chord candidates are harmonically
compatible.  Chords in turn provide bottom-up support for the global key.
The two directions alternate until their selected Or branches are stable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .validation import validate_graph


INFERENCE_VERSION = "bidirectional-music-and-or-graph-parser/0.1"
SEQUENCE_INFERENCE_VERSION = "bidirectional-music-and-or-graph-parser/0.2"
_EPSILON = 1e-9
_ROOT_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}
_CHORD_INTERVALS = {
    "maj": (0, 4, 7), "major": (0, 4, 7), "M": (0, 4, 7),
    "min": (0, 3, 7), "minor": (0, 3, 7), "m": (0, 3, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "M7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10), "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
    "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
}


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _softmax(log_scores: list[float]) -> list[float]:
    if not log_scores:
        return []
    maximum = max(log_scores)
    weights = [math.exp(score - maximum) for score in log_scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _canonical_key(label: Any) -> tuple[int, str] | None:
    if not isinstance(label, str):
        return None
    cleaned = label.strip().replace("♯", "#").replace("♭", "b")
    match = re.fullmatch(r"([A-Ga-g](?:#|b)?)\s+(major|minor|maj|min)", cleaned, re.IGNORECASE)
    if not match:
        return None
    root = _ROOT_PC.get(match.group(1).upper())
    if root is None:
        return None
    mode = "major" if match.group(2).lower() in {"major", "maj"} else "minor"
    return root, mode


def _chord_pitch_classes(label: Any) -> set[int] | None:
    if label == "N":
        return set()
    if not isinstance(label, str):
        return None
    cleaned = label.strip().replace("♯", "#").replace("♭", "b")
    match = re.fullmatch(r"([A-Ga-g](?:#|b)?):([^/]+)(?:/[A-Ga-g](?:#|b)?)?", cleaned)
    if not match:
        return None
    root = _ROOT_PC.get(match.group(1).upper())
    intervals = _CHORD_INTERVALS.get(match.group(2))
    if root is None or intervals is None:
        return None
    return {(root + interval) % 12 for interval in intervals}


def key_chord_compatibility(key_label: Any, chord_label: Any, *, floor: float = 0.05) -> float | None:
    """Return an interpretable [floor, 1] chord-tone inclusion score.

    `None` means the label syntax is unsupported and therefore must not be
    treated as negative evidence.  No-chord (N) is neutral with score 1.
    Minor keys use the union of natural and harmonic minor pitch classes so a
    raised leading tone is not incorrectly rejected.
    """

    key = _canonical_key(key_label)
    chord = _chord_pitch_classes(chord_label)
    if key is None or chord is None:
        return None
    if not chord:
        return 1.0
    root, mode = key
    intervals = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10, 11)
    scale = {(root + interval) % 12 for interval in intervals}
    inclusion = len(chord & scale) / len(chord)
    return max(float(floor), inclusion)


def chord_transition_compatibility(left_label: Any, right_label: Any) -> float | None:
    """Score a transparent first-order chord succession rule.

    Repetition and fourth/fifth root motion receive the highest support.  The
    score is deliberately mild and is exposed in every inference artifact;
    unsupported labels are neutral rather than guessed.
    """

    if left_label == "N" or right_label == "N":
        return 1.0
    if not isinstance(left_label, str) or not isinstance(right_label, str):
        return None
    pattern = r"([A-Ga-g](?:#|b)?):([^/]+)(?:/[A-Ga-g](?:#|b)?)?"
    left = re.fullmatch(pattern, left_label.strip().replace("♯", "#").replace("♭", "b"))
    right = re.fullmatch(pattern, right_label.strip().replace("♯", "#").replace("♭", "b"))
    if left is None or right is None:
        return None
    left_root = _ROOT_PC.get(left.group(1).upper())
    right_root = _ROOT_PC.get(right.group(1).upper())
    if left_root is None or right_root is None:
        return None
    if left_label == right_label:
        return 1.0
    distance = (right_root - left_root) % 12
    if distance == 0:
        return 0.85
    if distance in {5, 7}:
        return 0.95
    if distance in {2, 10}:
        return 0.80
    if distance in {3, 4, 8, 9}:
        return 0.75
    if distance in {1, 11}:
        return 0.60
    return 0.50


def _or_candidates(
    graph: dict[str, Any], node_type: str
) -> list[tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]]:
    nodes = {node["id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("id")}
    productions_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for production in graph.get("productions", []):
        if isinstance(production, dict):
            productions_by_parent[str(production.get("parent"))].append(production)
    result = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or node.get("kind") != "or" or node.get("type") != node_type:
            continue
        choices = []
        for production in productions_by_parent.get(node["id"], []):
            children = production.get("children", [])
            child = nodes.get(children[0]) if len(children) == 1 else None
            if child is not None:
                choices.append((production, child))
        if choices:
            result.append((node, choices))
    return sorted(
        result,
        key=lambda item: (
            float(item[0].get("span", {}).get("start_seconds", 0.0)),
            str(item[0].get("id")),
        ),
    )


def _selected_label(choices: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    selected = next((child for production, child in choices if production.get("selected") is True), choices[0][1])
    return str(selected.get("attributes", {}).get("label") or "unknown")


def _bottom_up_probability(
    production: dict[str, Any], child: dict[str, Any], *, moss_source_weight: float
) -> float:
    saved = child.get("attributes", {}).get("inference_scores", {}).get("bottom_up_probability")
    if isinstance(saved, (int, float)) and not isinstance(saved, bool):
        return max(_EPSILON, float(saved))
    source_scores = child.get("attributes", {}).get("source_scores", {})
    weighted_scores = []
    if isinstance(source_scores, dict):
        for source, score in source_scores.items():
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue
            reliability = moss_source_weight if source == "moss_music" else 1.0
            weighted_scores.append(reliability * float(score))
    value = max(weighted_scores) if weighted_scores else production.get("probability")
    return max(_EPSILON, float(value if isinstance(value, (int, float)) else _EPSILON))


def _score_chord_choices(
    choices: list[tuple[dict[str, Any], dict[str, Any]]],
    key_label: str,
    *,
    audio_weight: float,
    key_weight: float,
    compatibility_floor: float,
    moss_source_weight: float,
) -> tuple[list[float], list[float | None]]:
    scores: list[float] = []
    compatibilities: list[float | None] = []
    for production, child in choices:
        bottom_up = _bottom_up_probability(production, child, moss_source_weight=moss_source_weight)
        compatibility = key_chord_compatibility(
            key_label, child.get("attributes", {}).get("label"), floor=compatibility_floor
        )
        # Unsupported labels are neutral, not silently classified as errors.
        top_down = 1.0 if compatibility is None else compatibility
        scores.append(audio_weight * math.log(bottom_up) + key_weight * math.log(max(_EPSILON, top_down)))
        compatibilities.append(compatibility)
    return scores, compatibilities


def _score_chord_sequence(
    groups: list[tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]],
    key_label: str,
    *,
    audio_weight: float,
    key_weight: float,
    transition_weight: float,
    compatibility_floor: float,
    moss_source_weight: float,
) -> tuple[list[list[float]], list[list[float | None]], list[int], float, float]:
    """Return chain marginals and the max-product path for chord Or nodes."""

    if not groups:
        return [], [], [], 0.0, 1.0
    unary_rows: list[list[float]] = []
    compatibility_rows: list[list[float | None]] = []
    label_rows: list[list[str]] = []
    for _, choices in groups:
        unary, compatibility = _score_chord_choices(
            choices,
            key_label,
            audio_weight=audio_weight,
            key_weight=key_weight,
            compatibility_floor=compatibility_floor,
            moss_source_weight=moss_source_weight,
        )
        unary_rows.append(unary)
        compatibility_rows.append(compatibility)
        label_rows.append([str(child.get("attributes", {}).get("label") or "unknown") for _, child in choices])

    pairwise_rows: list[list[list[float]]] = []
    raw_transition_rows: list[list[list[float]]] = []
    for left_labels, right_labels in zip(label_rows, label_rows[1:]):
        pairwise: list[list[float]] = []
        raw: list[list[float]] = []
        for left_label in left_labels:
            pairwise_row = []
            raw_row = []
            for right_label in right_labels:
                compatibility = chord_transition_compatibility(left_label, right_label)
                value = 1.0 if compatibility is None else compatibility
                pairwise_row.append(transition_weight * math.log(max(_EPSILON, value)))
                raw_row.append(value)
            pairwise.append(pairwise_row)
            raw.append(raw_row)
        pairwise_rows.append(pairwise)
        raw_transition_rows.append(raw)

    # Sum-product messages provide calibrated local marginals for each Or node.
    forward: list[list[float]] = [list(unary_rows[0])]
    for index in range(1, len(unary_rows)):
        previous = forward[-1]
        forward.append([
            unary_rows[index][right] + _logsumexp([
                previous[left] + pairwise_rows[index - 1][left][right]
                for left in range(len(previous))
            ])
            for right in range(len(unary_rows[index]))
        ])
    backward: list[list[float]] = [[0.0] * len(unary_rows[-1]) for _ in unary_rows]
    for index in range(len(unary_rows) - 2, -1, -1):
        backward[index] = [
            _logsumexp([
                pairwise_rows[index][left][right]
                + unary_rows[index + 1][right]
                + backward[index + 1][right]
                for right in range(len(unary_rows[index + 1]))
            ])
            for left in range(len(unary_rows[index]))
        ]
    log_partition = _logsumexp(forward[-1])
    marginals = [
        _softmax([forward[index][choice] + backward[index][choice] - log_partition for choice in range(len(row))])
        for index, row in enumerate(unary_rows)
    ]

    # Max-product/Viterbi decoding selects one globally coherent path.
    delta: list[list[float]] = [list(unary_rows[0])]
    backpointers: list[list[int]] = []
    for index in range(1, len(unary_rows)):
        previous = delta[-1]
        pointers = []
        current = []
        for right in range(len(unary_rows[index])):
            candidates = [
                previous[left] + pairwise_rows[index - 1][left][right]
                for left in range(len(previous))
            ]
            winner = max(range(len(candidates)), key=lambda left: candidates[left])
            pointers.append(winner)
            current.append(unary_rows[index][right] + candidates[winner])
        backpointers.append(pointers)
        delta.append(current)
    last = max(range(len(delta[-1])), key=lambda choice: delta[-1][choice])
    path = [last]
    for pointers in reversed(backpointers):
        path.append(pointers[path[-1]])
    path.reverse()
    transition_values = [
        raw_transition_rows[index][path[index]][path[index + 1]]
        for index in range(len(path) - 1)
    ]
    mean_transition = sum(transition_values) / len(transition_values) if transition_values else 1.0
    return marginals, compatibility_rows, path, delta[-1][last], mean_transition


def _score_key_choices(
    choices: list[tuple[dict[str, Any], dict[str, Any]]],
    selected_chords: list[tuple[str, float]],
    *,
    audio_weight: float,
    key_weight: float,
    compatibility_floor: float,
    moss_source_weight: float,
) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    support_values: list[float] = []
    total_duration = sum(duration for _, duration in selected_chords) or 1.0
    for production, child in choices:
        key_label = child.get("attributes", {}).get("label")
        weighted_log_support = 0.0
        supported_duration = 0.0
        for chord_label, duration in selected_chords:
            compatibility = key_chord_compatibility(key_label, chord_label, floor=compatibility_floor)
            if compatibility is None:
                continue
            weighted_log_support += duration * math.log(max(_EPSILON, compatibility))
            supported_duration += duration
        denominator = supported_duration or total_duration
        mean_log_support = weighted_log_support / denominator if supported_duration else 0.0
        support = math.exp(mean_log_support)
        bottom_up = _bottom_up_probability(production, child, moss_source_weight=moss_source_weight)
        scores.append(audio_weight * math.log(bottom_up) + key_weight * mean_log_support)
        support_values.append(support)
    return scores, support_values


def _choose(
    choices: list[tuple[dict[str, Any], dict[str, Any]]],
    posteriors: list[float],
    top_down_scores: list[float | None],
    *,
    iteration: int,
    rule_name: str,
    moss_source_weight: float,
    winner_index: int | None = None,
) -> tuple[str, str]:
    before = _selected_label(choices)
    winner = winner_index if winner_index is not None else max(range(len(choices)), key=lambda index: posteriors[index])
    for index, ((production, child), posterior, top_down) in enumerate(zip(choices, posteriors, top_down_scores)):
        bottom_up = _bottom_up_probability(production, child, moss_source_weight=moss_source_weight)
        previous_scores = child.get("attributes", {}).get("inference_scores", {})
        selected_in_input = previous_scores.get(
            "selected_in_input", child.get("attributes", {}).get("label") == before
        )
        production["probability"] = posterior
        production["selected"] = index == winner
        child["confidence"] = posterior
        child.setdefault("attributes", {})["inference_scores"] = {
            "bottom_up_probability": bottom_up,
            "top_down_score": top_down,
            "posterior_probability": posterior,
            "selected_in_input": selected_in_input,
            "selected_before_iteration": child.get("attributes", {}).get("label") == before,
            "selected_after": index == winner,
            "message_rule": rule_name,
            "iteration": iteration,
        }
    parent = choices[0][0]["parent"]
    # The parent node is obtained by the caller; returning its id keeps this
    # helper independent of a second node lookup.
    return parent, _selected_label(choices)


def run_bidirectional_inference(
    fusion_artifact: dict[str, Any],
    *,
    audio_weight: float = 1.0,
    key_weight: float = 0.75,
    transition_weight: float = 0.0,
    compatibility_floor: float = 0.05,
    moss_source_weight: float = 0.1,
    max_iterations: int = 8,
) -> dict[str, Any]:
    """Alternate key-to-chord and chord-to-key messages until selections stabilize."""

    if audio_weight < 0 or key_weight < 0 or transition_weight < 0:
        raise ValueError("audio_weight, key_weight, and transition_weight must be non-negative")
    if not 0 < compatibility_floor <= 1:
        raise ValueError("compatibility_floor must be in (0, 1]")
    if not 0 <= moss_source_weight <= 1:
        raise ValueError("moss_source_weight must be in [0, 1]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    source_graph = fusion_artifact.get("graph")
    if not isinstance(source_graph, dict):
        raise ValueError("fusion artifact must contain a graph object")
    source_errors = validate_graph(source_graph)
    if source_errors:
        raise ValueError(f"input graph is invalid: {source_errors[0]}")

    graph = copy.deepcopy(source_graph)
    key_groups = _or_candidates(graph, "key_event")
    chord_groups = _or_candidates(graph, "chord")
    if len(key_groups) != 1:
        raise ValueError(f"expected exactly one global key Or node, found {len(key_groups)}")

    key_node, key_choices = key_groups[0]
    current_key = _selected_label(key_choices)
    initial_key = current_key
    initial_chords = [_selected_label(choices) for _, choices in chord_groups]
    trace: list[dict[str, Any]] = []
    seen_states: set[tuple[str, tuple[str, ...]]] = set()
    stop_reason = "max_iterations"
    inference_version = SEQUENCE_INFERENCE_VERSION if transition_weight > 0 else INFERENCE_VERSION

    for iteration in range(1, max_iterations + 1):
        before_key = current_key
        before_chords = [_selected_label(choices) for _, choices in chord_groups]

        sequence_path_log_score = None
        mean_transition_compatibility = None
        if transition_weight > 0:
            posterior_rows, compatibility_rows, winners, sequence_path_log_score, mean_transition_compatibility = (
                _score_chord_sequence(
                    chord_groups,
                    current_key,
                    audio_weight=audio_weight,
                    key_weight=key_weight,
                    transition_weight=transition_weight,
                    compatibility_floor=compatibility_floor,
                    moss_source_weight=moss_source_weight,
                )
            )
        else:
            posterior_rows = []
            compatibility_rows = []
            winners = []
            for _, choices in chord_groups:
                log_scores, compatibilities = _score_chord_choices(
                    choices,
                    current_key,
                    audio_weight=audio_weight,
                    key_weight=key_weight,
                    compatibility_floor=compatibility_floor,
                    moss_source_weight=moss_source_weight,
                )
                posterior_rows.append(_softmax(log_scores))
                compatibility_rows.append(compatibilities)
                winners.append(-1)

        for group_index, (chord_node, choices) in enumerate(chord_groups):
            posteriors = posterior_rows[group_index]
            compatibilities = compatibility_rows[group_index]
            winner_index = winners[group_index] if transition_weight > 0 else None
            _, selected = _choose(
                choices,
                posteriors,
                compatibilities,
                iteration=iteration,
                rule_name="key_and_succession_to_chord" if transition_weight > 0 else "key_to_chord",
                moss_source_weight=moss_source_weight,
                winner_index=winner_index,
            )
            chord_node["confidence"] = max(posteriors)
            chord_node.setdefault("attributes", {})["inference"] = {
                "rule": "key_and_succession_to_chord" if transition_weight > 0 else "key_to_chord",
                "conditioning_key": current_key,
                "transition_weight": transition_weight,
                "iteration": iteration,
                "selected_label": selected,
            }

        selected_chords: list[tuple[str, float]] = []
        for chord_node, choices in chord_groups:
            span = chord_node.get("span", {})
            duration = max(0.0, float(span.get("end_seconds", 0.0)) - float(span.get("start_seconds", 0.0)))
            selected_chords.append((_selected_label(choices), duration))
        key_log_scores, harmony_support = _score_key_choices(
            key_choices,
            selected_chords,
            audio_weight=audio_weight,
            key_weight=key_weight,
            compatibility_floor=compatibility_floor,
            moss_source_weight=moss_source_weight,
        )
        key_posteriors = _softmax(key_log_scores)
        _, current_key = _choose(
            key_choices,
            key_posteriors,
            harmony_support,
            iteration=iteration,
            rule_name="chords_to_key",
            moss_source_weight=moss_source_weight,
        )
        key_node["confidence"] = max(key_posteriors)
        key_node.setdefault("attributes", {})["inference"] = {
            "rule": "chords_to_key",
            "iteration": iteration,
            "selected_label": current_key,
        }

        after_chords = [_selected_label(choices) for _, choices in chord_groups]
        state = (current_key, tuple(after_chords))
        trace.append({
            "iteration": iteration,
            "key_before": before_key,
            "key_after": current_key,
            "changed_chord_choices": sum(a != b for a, b in zip(before_chords, after_chords)),
            "total_chord_choices": len(chord_groups),
            "state_changed": before_key != current_key or before_chords != after_chords,
            "sequence_path_log_score": sequence_path_log_score,
            "mean_selected_transition_compatibility": mean_transition_compatibility,
        })
        if not trace[-1]["state_changed"]:
            stop_reason = "fixed_point"
            break
        if state in seen_states:
            stop_reason = "cycle_detected"
            break
        seen_states.add(state)

    final_chords = [_selected_label(choices) for _, choices in chord_groups]
    graph["provenance"].update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assembler": inference_version,
    })
    graph_errors = validate_graph(graph)
    return {
        "artifact_version": inference_version,
        "status": "complete" if not graph_errors else "partial",
        "problem": "bidirectional_audio_to_musicaog_parsing",
        "inference_config": {
            "audio_weight": audio_weight,
            "key_weight": key_weight,
            "transition_weight": transition_weight,
            "compatibility_floor": compatibility_floor,
            "moss_source_weight": moss_source_weight,
            "max_iterations": max_iterations,
            "weights_tuned_on_test_set": False,
        },
        "score_definition": {
            "bottom_up": "source-specific acoustic candidate probability multiplied by a fixed source reliability",
            "top_down": "fraction of candidate chord tones contained in the selected key scale",
            "succession": (
                "first-order chord-transition compatibility with sum-product marginals and Viterbi decoding"
                if transition_weight > 0 else "disabled"
            ),
            "posterior": (
                "chain marginal under unary audio/key scores and pairwise succession scores"
                if transition_weight > 0
                else "softmax(audio_weight * log(bottom_up) + key_weight * log(top_down))"
            ),
            "unsupported_chord_labels": "neutral top-down score; retained for audit",
        },
        "initial_selection": {"global_key": initial_key, "chords": initial_chords},
        "final_selection": {"global_key": current_key, "chords": final_chords},
        "inference_trace": trace,
        "stop_reason": stop_reason,
        "selection_changes": {
            "global_key": initial_key != current_key,
            "chords": sum(a != b for a, b in zip(initial_chords, final_chords)),
            "total_chords": len(final_chords),
        },
        "source_artifact_sha256": _hash(fusion_artifact),
        "graph": graph,
        "graph_errors": graph_errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

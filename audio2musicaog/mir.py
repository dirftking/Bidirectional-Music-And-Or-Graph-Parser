"""Transparent beat-synchronous MIR candidates for Audio2MusicAOG.

This module is deliberately a classical, inspectable baseline rather than a
learned state-of-the-art system.  It consumes audio only and never reads the
POP909 annotations used for evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import sha256_file


PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
KRUMHANSL_MAJOR = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _normalise_candidates(scored: list[tuple[str, float]], top_k: int) -> list[dict[str, Any]]:
    selected = sorted(scored, key=lambda item: item[1], reverse=True)[:max(1, top_k)]
    values = np.asarray([score for _, score in selected], dtype=float)
    values = np.exp((values - values.max()) / 0.1)
    probabilities = values / values.sum() if values.sum() else np.full(len(values), 1 / len(values))
    return [
        {"label": label, "confidence": round(float(probability), 6), "score": round(float(score), 6)}
        for (label, score), probability in zip(selected, probabilities)
    ]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def estimate_key_candidates(chroma: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
    """Rank 24 major/minor keys using global chroma-profile similarity."""

    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise ValueError("chroma must have shape (12, frames)")
    observation = np.mean(np.maximum(chroma, 0.0), axis=1)
    if float(observation.sum()) <= 1e-12:
        return [{"label": "unknown", "confidence": 1.0, "score": 0.0}]
    scored: list[tuple[str, float]] = []
    for tonic, name in enumerate(PITCH_CLASSES):
        scored.append((f"{name} major", _cosine(observation, np.roll(KRUMHANSL_MAJOR, tonic))))
        scored.append((f"{name} minor", _cosine(observation, np.roll(KRUMHANSL_MINOR, tonic))))
    return _normalise_candidates(scored, top_k)


def estimate_chord_candidates(chroma_vector: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
    """Rank major/minor triads for one beat interval."""

    observation = np.maximum(np.asarray(chroma_vector, dtype=float).reshape(-1), 0.0)
    if observation.size != 12:
        raise ValueError("chroma_vector must contain 12 pitch classes")
    if float(observation.sum()) <= 1e-12:
        return [{"label": "N", "confidence": 1.0, "score": 0.0}]
    scored: list[tuple[str, float]] = []
    for root, name in enumerate(PITCH_CLASSES):
        for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            template = np.zeros(12, dtype=float)
            template[[(root + interval) % 12 for interval in intervals]] = (1.0, 0.8, 0.8)
            scored.append((f"{name}:{quality}", _cosine(observation, template)))
    return _normalise_candidates(scored, top_k)


def aggregate_chroma_to_beats(
    chroma: np.ndarray,
    frame_times: np.ndarray,
    beat_times: np.ndarray,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """Aggregate frame-level chroma into intervals bounded by detected beats."""

    internal = sorted({float(value) for value in beat_times if 1e-6 < float(value) < duration_seconds - 1e-6})
    boundaries = [0.0, *internal, float(duration_seconds)]
    intervals: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        mask = (frame_times >= start) & (frame_times < end)
        vector = np.median(chroma[:, mask], axis=1) if np.any(mask) else np.zeros(12, dtype=float)
        intervals.append(
            {
                "index": index,
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "chroma": [round(float(value), 8) for value in vector],
            }
        )
    return intervals


def beat_synchronous_boundaries(
    beat_intervals: list[dict[str, Any]],
    *,
    window_beats: int = 4,
    min_distance_beats: int = 8,
    max_boundaries: int | None = None,
) -> list[dict[str, Any]]:
    """Propose structural boundaries from local chroma change on the beat grid."""

    vectors = np.asarray([item["chroma"] for item in beat_intervals], dtype=float)
    count = len(vectors)
    if count < window_beats * 2 + 1:
        return []
    candidates: list[tuple[int, float]] = []
    for index in range(window_beats, count - window_beats):
        before = np.mean(vectors[index - window_beats:index], axis=0)
        after = np.mean(vectors[index:index + window_beats], axis=0)
        score = max(0.0, 1.0 - _cosine(before, after))
        candidates.append((index, score))
    local_maxima = [
        item for position, item in enumerate(candidates)
        if item[1] > 0 and item[1] >= (candidates[position - 1][1] if position else -1)
        and item[1] >= (candidates[position + 1][1] if position + 1 < len(candidates) else -1)
    ]
    target = max_boundaries if max_boundaries is not None else max(1, min(12, count // 32))
    selected: list[tuple[int, float]] = []
    for candidate in sorted(local_maxima, key=lambda item: item[1], reverse=True):
        if all(abs(candidate[0] - existing[0]) >= min_distance_beats for existing in selected):
            selected.append(candidate)
        if len(selected) >= target:
            break
    peak = max((score for _, score in selected), default=1.0)
    return [
        {
            "beat_interval_index": index,
            "time_seconds": beat_intervals[index]["start_seconds"],
            "confidence": round(score / peak, 6) if peak else 0.0,
            "novelty_score": round(score, 6),
        }
        for index, score in sorted(selected)
    ]


def _merge_selected_chords(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in intervals:
        selected = item["candidates"][0]
        if merged and merged[-1]["label"] == selected["label"]:
            merged[-1]["end_seconds"] = item["end_seconds"]
            merged[-1]["confidence"] = round(min(merged[-1]["confidence"], selected["confidence"]), 6)
        else:
            merged.append(
                {
                    "start_seconds": item["start_seconds"],
                    "end_seconds": item["end_seconds"],
                    "label": selected["label"],
                    "confidence": selected["confidence"],
                    "source": "librosa_chroma_template",
                }
            )
    return merged


def analyze_mir(
    audio_path: Path,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    top_k: int = 3,
    assumed_beats_per_bar: int = 4,
) -> dict[str, Any]:
    """Extract reproducible beat-synchronous candidates from an audio file."""

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("MIR analysis requires: pip install -r research/requirements-mir.txt") from exc

    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    harmonic = librosa.effects.harmonic(y, margin=8.0)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_value, beat_times = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=hop_length, units="time", trim=False
    )
    tempo = float(np.asarray(tempo_value).reshape(-1)[0])
    beat_times = np.asarray(beat_times, dtype=float)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    beat_intervals = aggregate_chroma_to_beats(chroma, frame_times, beat_times, duration)
    boundaries = beat_synchronous_boundaries(beat_intervals)

    for item in beat_intervals:
        item["candidates"] = estimate_chord_candidates(np.asarray(item.pop("chroma")), top_k=top_k)
    differences = np.diff(beat_times)
    tempo_confidence = (
        max(0.0, min(1.0, 1.0 - float(np.std(differences) / np.mean(differences))))
        if len(differences) > 1 and float(np.mean(differences)) > 0 else 0.0
    )
    return {
        "artifact_version": "beat-synchronous-mir/0.1",
        "audio": {
            "path": str(audio_path),
            "filename": audio_path.name,
            "sha256": sha256_file(audio_path),
            "duration_seconds": round(duration, 6),
        },
        "configuration": {
            "backend": "librosa",
            "sample_rate": sr,
            "hop_length": hop_length,
            "top_k": top_k,
            "assumed_beats_per_bar": assumed_beats_per_bar,
        },
        "tempo": {"bpm": round(tempo, 6), "confidence": round(tempo_confidence, 6)},
        "metrical_grid": {
            "beats": [
                {
                    "index": index,
                    "time_seconds": round(float(value), 6),
                    "assumed_bar_position": index % assumed_beats_per_bar,
                }
                for index, value in enumerate(beat_times)
            ],
            "downbeats": [],
            "assumption": (
                f"Beat positions are detected; {assumed_beats_per_bar} beats per bar is a configuration "
                "assumption, not a downbeat estimate."
            ),
        },
        "key_candidates": estimate_key_candidates(chroma, top_k=top_k),
        "beat_chord_candidates": beat_intervals,
        "chord_segments": _merge_selected_chords(beat_intervals),
        "structure_boundary_candidates": boundaries,
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "methods": {
                "beats": "librosa.beat.beat_track",
                "features": "harmonic chroma_cqt",
                "keys": "Krumhansl-Schmuckler profile matching",
                "chords": "major/minor triad cosine templates",
                "structure": "local beat-synchronous chroma novelty",
            },
            "ground_truth_used_as_input": False,
        },
    }

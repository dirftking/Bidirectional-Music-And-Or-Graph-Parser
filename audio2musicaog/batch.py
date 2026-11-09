"""Resumable POP909-CL benchmark orchestration."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import requests

from .groundtruth import load_pop909cl_ground_truth, render_pop909cl_score
from .fusion import fuse_moss_mir
from .inference import run_bidirectional_inference
from .metrics import evaluate_artifact, evaluate_graph_artifact, evaluate_mir_artifact
from .mir import analyze_mir
from .pipeline import atomic_write_json, run_pipeline
from .validation import validate_graph


POP909_CL_MIDI_URL = (
    "https://raw.githubusercontent.com/AndyWeasley2004/"
    "POP909-CL-Dataset/main/POP909_processed/{song_id}.mid"
)
POP909_STRUCTURE_URL = (
    "https://raw.githubusercontent.com/Dsqvival/"
    "hierarchical-structure-analysis/master/POP909/{song_id}/human_label1.txt"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def prediction_is_reusable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        artifact = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    tasks = artifact.get("tasks", {})
    return (
        artifact.get("status") in {"complete", "partial"}
        and all(isinstance(tasks.get(name), dict) for name in ("global", "structure", "harmony"))
        and isinstance(artifact.get("graph"), dict)
        and "completed_at_utc" in artifact
    )


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    """Download atomically; an existing non-empty file is left untouched."""

    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=(15, 120))
            response.raise_for_status()
            temporary.write_bytes(response.content)
            if not temporary.stat().st_size:
                raise RuntimeError(f"empty download: {url}")
            temporary.replace(destination)
            return
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average available per-song metrics without hiding missing values."""

    successful = [item for item in items if isinstance(item.get("metrics"), dict)]

    def values(*keys: str) -> list[float]:
        result: list[float] = []
        for item in successful:
            value: Any = item["metrics"]
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, bool):
                result.append(float(value))
            elif isinstance(value, (int, float)):
                result.append(float(value))
        return result

    def summary(*keys: str) -> dict[str, Any]:
        observed = values(*keys)
        return {
            "mean": round(fmean(observed), 6) if observed else None,
            "songs_available": len(observed),
        }

    return {
        "summary_version": "audio2musicaog-benchmark/0.1",
        "songs_requested": len(items),
        "songs_evaluated": len(successful),
        "songs_failed": len(items) - len(successful),
        "prediction_complete_rate": round(
            sum(item.get("prediction_status") == "complete" for item in successful) / len(successful), 6
        ) if successful else None,
        "prediction_graph_valid_rate": summary("prediction_graph_valid"),
        "tempo_acc1": summary("tempo", "acc1"),
        "tempo_acc2": summary("tempo", "acc2"),
        "time_signature_accuracy": summary("time_signature_exact"),
        "key_accuracy": summary("key_exact"),
        "chord_weighted_exact": summary("chord", "weighted_exact"),
        "chord_root_accuracy": summary("chord", "root_accuracy"),
        "chord_coverage": summary("chord", "coverage"),
        "section_boundary_f1": summary("section_boundary", "f1"),
        "audit_warning_count": summary("prediction_audit_warning_count"),
    }


def aggregate_mir_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average the MIR-only condition using the same content metrics."""

    successful = [item for item in items if isinstance(item.get("metrics"), dict)]

    def summary(*keys: str) -> dict[str, Any]:
        observed: list[float] = []
        for item in successful:
            value: Any = item["metrics"]
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, bool):
                observed.append(float(value))
            elif isinstance(value, (int, float)):
                observed.append(float(value))
        return {
            "mean": round(fmean(observed), 6) if observed else None,
            "songs_available": len(observed),
        }

    return {
        "summary_version": "beat-synchronous-mir-benchmark/0.1",
        "songs_requested": len(items),
        "songs_evaluated": len(successful),
        "songs_failed": len(items) - len(successful),
        "tempo_acc1": summary("tempo", "acc1"),
        "tempo_acc2": summary("tempo", "acc2"),
        "key_accuracy": summary("key_exact"),
        "key_string_accuracy": summary("key_string_exact"),
        "chord_weighted_exact": summary("chord", "weighted_exact"),
        "chord_root_accuracy": summary("chord", "root_accuracy"),
        "chord_coverage": summary("chord", "coverage"),
        "section_boundary_f1": summary("section_boundary", "f1"),
        "beat_count": summary("beat_count"),
    }


def aggregate_bidirectional_metrics(
    items: list[dict[str, Any]], *, summary_version: str = "bidirectional-music-and-or-graph-parser-pop909/0.1"
) -> dict[str, Any]:
    """Compare fixed one-shot fusion and bidirectional inference song by song."""

    successful = [
        item for item in items
        if isinstance(item.get("before"), dict) and isinstance(item.get("after"), dict)
    ]

    def observed(side: str, *keys: str) -> list[float]:
        values: list[float] = []
        for item in successful:
            value: Any = item[side]
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def comparison(*keys: str) -> dict[str, Any]:
        before = observed("before", *keys)
        after = observed("after", *keys)
        before_mean = fmean(before) if before else None
        after_mean = fmean(after) if after else None
        return {
            "before": round(before_mean, 6) if before_mean is not None else None,
            "after": round(after_mean, 6) if after_mean is not None else None,
            "delta": round(after_mean - before_mean, 6)
            if before_mean is not None and after_mean is not None else None,
            "songs_available": min(len(before), len(after)),
        }

    return {
        "summary_version": summary_version,
        "songs_requested": len(items),
        "songs_evaluated": len(successful),
        "songs_failed": len(items) - len(successful),
        "key_accuracy": comparison("key_exact"),
        "chord_weighted_exact": comparison("chord", "weighted_exact"),
        "chord_root_accuracy": comparison("chord", "root_accuracy"),
        "chord_coverage": comparison("chord", "coverage"),
        "selected_chords_changed": sum(int(item.get("selected_chords_changed", 0)) for item in successful),
        "fixed_point_rate": round(
            sum(item.get("stop_reason") == "fixed_point" for item in successful) / len(successful), 6
        ) if successful else None,
    }


def run_pop909_bidirectional_batch(
    *,
    song_ids: list[str],
    output_root: Path,
    audio_weight: float = 1.0,
    key_weight: float = 0.75,
    transition_weight: float = 0.0,
    compatibility_floor: float = 0.05,
    moss_source_weight: float = 0.1,
    max_iterations: int = 8,
) -> dict[str, Any]:
    """Replay the fixed bidirectional rule on saved predictions and MIR candidates."""

    output_root = output_root.expanduser().resolve()
    version_tag = "v02" if transition_weight > 0 else "v01"
    version_number = "0.2" if transition_weight > 0 else "0.1"
    manifest = {
        "batch_version": f"bidirectional-music-and-or-graph-parser-pop909/{version_number}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "song_ids": song_ids,
        "output_root": str(output_root),
        "audio_weight": audio_weight,
        "key_weight": key_weight,
        "transition_weight": transition_weight,
        "compatibility_floor": compatibility_floor,
        "moss_source_weight": moss_source_weight,
        "max_iterations": max_iterations,
        "ground_truth_used_for_inference": False,
        "weights_tuned_on_test_set": False,
        "development_examples_seen": ["001"],
    }
    items: list[dict[str, Any]] = []
    for song_id in song_ids:
        started = time.monotonic()
        song_root = output_root / song_id
        record: dict[str, Any] = {"song_id": song_id, "status": "running"}
        items.append(record)
        try:
            prediction = _read_json(song_root / "prediction.json")
            mir = _read_json(song_root / "mir.json")
            ground_truth = _read_json(song_root / "ground-truth.json")
            fusion = fuse_moss_mir(prediction, mir)
            atomic_write_json(song_root / "fusion-v02.json", fusion)
            inference = run_bidirectional_inference(
                fusion,
                audio_weight=audio_weight,
                key_weight=key_weight,
                transition_weight=transition_weight,
                compatibility_floor=compatibility_floor,
                moss_source_weight=moss_source_weight,
                max_iterations=max_iterations,
            )
            atomic_write_json(song_root / f"bidirectional-{version_tag}.json", inference)
            before = evaluate_graph_artifact(fusion, ground_truth)
            after = evaluate_graph_artifact(inference, ground_truth)
            atomic_write_json(song_root / "fusion-v02-metrics.json", before)
            atomic_write_json(song_root / f"bidirectional-{version_tag}-metrics.json", after)
            record.update(
                status="evaluated",
                before=before,
                after=after,
                selected_chords_changed=inference["selection_changes"]["chords"],
                global_key_changed=inference["selection_changes"]["global_key"],
                inference_iterations=len(inference["inference_trace"]),
                stop_reason=inference["stop_reason"],
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
        except Exception as exc:
            record.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
    aggregate = aggregate_bidirectional_metrics(
        items, summary_version=f"bidirectional-music-and-or-graph-parser-pop909/{version_number}"
    )
    result = {"manifest": manifest, "aggregate": aggregate, "items": items}
    atomic_write_json(output_root / f"bidirectional-summary-{version_tag}.json", result)
    return result


def run_pop909_mir_batch(
    *,
    song_ids: list[str],
    output_root: Path,
    sample_rate: int = 22050,
    hop_length: int = 512,
    top_k: int = 3,
    assumed_beats_per_bar: int = 4,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the audio-only MIR condition over already rendered POP909 scores."""

    output_root = output_root.expanduser().resolve()
    progress_path = output_root / "mir-progress.json"
    manifest = {
        "batch_version": "beat-synchronous-mir-pop909/0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "song_ids": song_ids,
        "output_root": str(output_root),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "top_k": top_k,
        "assumed_beats_per_bar": assumed_beats_per_bar,
        "resume": resume,
        "ground_truth_used_as_input": False,
    }
    items: list[dict[str, Any]] = []
    for song_id in song_ids:
        started = time.monotonic()
        song_root = output_root / song_id
        wav_path = song_root / "score.wav"
        gt_path = song_root / "ground-truth.json"
        mir_path = song_root / "mir.json"
        metrics_path = song_root / "mir-metrics.json"
        record: dict[str, Any] = {"song_id": song_id, "status": "running", "stage": "analysis"}
        items.append(record)
        atomic_write_json(progress_path, {"manifest": manifest, "items": items})
        try:
            if not wav_path.is_file() or not gt_path.is_file():
                raise FileNotFoundError(f"missing rendered audio or Ground Truth for {song_id}")
            if not resume or not mir_path.is_file():
                mir = analyze_mir(
                    wav_path,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    top_k=top_k,
                    assumed_beats_per_bar=assumed_beats_per_bar,
                )
                atomic_write_json(mir_path, mir)
            else:
                mir = _read_json(mir_path)
            record["stage"] = "evaluation"
            atomic_write_json(progress_path, {"manifest": manifest, "items": items})
            metrics = evaluate_mir_artifact(mir, _read_json(gt_path))
            atomic_write_json(metrics_path, metrics)
            record.update(
                status="evaluated",
                stage="complete",
                metrics=metrics,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
        except Exception as exc:
            record.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
        atomic_write_json(progress_path, {"manifest": manifest, "items": items})
    aggregate = aggregate_mir_metrics(items)
    result = {"manifest": manifest, "aggregate": aggregate, "items": items}
    atomic_write_json(output_root / "mir-summary.json", result)
    return result


def run_pop909_batch(
    *,
    song_ids: list[str],
    dataset_root: Path,
    output_root: Path,
    base_url: str,
    max_new_tokens: int = 2048,
    harmony_chunk_seconds: float = 30.0,
    seed: int = 42,
    resume: bool = True,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    midi_root = dataset_root / "pop909-cl"
    structure_root = dataset_root / "pop909-structure"
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "batch-progress.json"
    manifest = {
        "batch_version": "audio2musicaog-pop909/0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "song_ids": song_ids,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "base_url": base_url,
        "max_new_tokens": max_new_tokens,
        "harmony_chunk_seconds": harmony_chunk_seconds,
        "seed": seed,
        "resume": resume,
        "sources": {
            "pop909_cl": POP909_CL_MIDI_URL,
            "human_structure": POP909_STRUCTURE_URL,
        },
    }
    atomic_write_json(output_root / "batch-manifest.json", manifest)
    items: list[dict[str, Any]] = []
    for index, song_id in enumerate(song_ids):
        song_started = time.monotonic()
        song_output = output_root / song_id
        midi_path = midi_root / f"{song_id}.mid"
        structure_path = structure_root / song_id / "human_label1.txt"
        gt_path = song_output / "ground-truth.json"
        wav_path = song_output / "score.wav"
        score_midi_path = song_output / "score.mid"
        render_path = song_output / "render.json"
        prediction_path = song_output / "prediction.json"
        metrics_path = song_output / "metrics.json"
        record: dict[str, Any] = {"song_id": song_id, "status": "running"}
        items.append(record)
        def checkpoint(stage: str) -> None:
            record["stage"] = stage
            atomic_write_json(progress_path, {"manifest": manifest, "items": items})

        checkpoint("download")
        try:
            download_file(POP909_CL_MIDI_URL.format(song_id=song_id), midi_path)
            download_file(POP909_STRUCTURE_URL.format(song_id=song_id), structure_path)
            checkpoint("ground_truth")
            if not resume or not gt_path.is_file():
                ground_truth = load_pop909cl_ground_truth(
                    midi_path,
                    structure_code=structure_path.read_text(encoding="utf-8").strip(),
                    source_id=song_id,
                )
                errors = validate_graph(ground_truth["graph"])
                if errors:
                    raise ValueError(f"Ground Truth graph is invalid: {errors}")
                atomic_write_json(gt_path, ground_truth)
            checkpoint("render")
            if not resume or not wav_path.is_file() or not render_path.is_file():
                render = render_pop909cl_score(
                    midi_path,
                    wav_path,
                    score_midi_path=score_midi_path,
                )
                atomic_write_json(render_path, render)
            checkpoint("inference")
            if not resume or not prediction_is_reusable(prediction_path):
                run_pipeline(
                    audio_path=wav_path,
                    output_path=prediction_path,
                    base_url=base_url,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    seed=seed + index * 10000,
                    harmony_chunk_seconds=harmony_chunk_seconds,
                )
            checkpoint("evaluation")
            prediction = _read_json(prediction_path)
            metrics = evaluate_artifact(prediction, _read_json(gt_path))
            atomic_write_json(metrics_path, metrics)
            record.update(
                status="evaluated",
                stage="complete",
                prediction_status=prediction.get("status"),
                metrics=metrics,
                elapsed_seconds=round(time.monotonic() - song_started, 3),
            )
        except Exception as exc:
            record.update(
                status="failed",
                stage=record.get("stage"),
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=round(time.monotonic() - song_started, 3),
            )
        atomic_write_json(progress_path, {"manifest": manifest, "items": items})

    aggregate = aggregate_metrics(items)
    result = {"manifest": manifest, "aggregate": aggregate, "items": items}
    atomic_write_json(output_root / "summary.json", result)
    return result

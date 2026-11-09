"""Command-line entry point for Audio2MusicAOG research experiments."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline import DEFAULT_PROMPTS, atomic_write_json, rebuild_artifact, retry_failed_tasks, run_pipeline
from .groundtruth import load_pop909cl_ground_truth, render_pop909cl_score
from .fusion import fuse_moss_mir
from .inference import run_bidirectional_inference
from .metrics import evaluate_artifact, evaluate_graph_artifact, evaluate_mir_artifact
from .mir import analyze_mir
from .batch import run_pop909_batch, run_pop909_bidirectional_batch, run_pop909_mir_batch
from .validation import validate_graph


def _default_output(audio: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("research_outputs") / "audio2musicaog" / f"{audio.stem}-{stamp}.json"


def _load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("graph"), dict):
        return payload["graph"]
    if not isinstance(payload, dict):
        raise ValueError("JSON file must contain an object")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse music audio into a candidate MusicAOG graph")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Run MOSS-Music tasks and assemble a graph")
    analyze.add_argument("--audio", required=True, type=Path, help="Audio path visible to the SGLang server")
    analyze.add_argument(
        "--base-url",
        default=os.environ.get("MOSS_MUSIC_SGLANG_BASE_URL", "http://127.0.0.1:30000"),
    )
    analyze.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--max-new-tokens", type=int, default=1024)
    analyze.add_argument("--temperature", type=float, default=0.0)
    analyze.add_argument("--top-p", type=float, default=0.95)
    analyze.add_argument("--top-k", type=int, default=50)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument(
        "--harmony-chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk long audio for harmony transcription; use 0 to disable",
    )

    validate = commands.add_parser("validate", help="Validate a graph or pipeline artifact")
    validate.add_argument("json_file", type=Path)

    ground_truth = commands.add_parser("ground-truth", help="Convert POP909-CL MIDI into evaluation JSON")
    ground_truth.add_argument("--midi", required=True, type=Path)
    ground_truth.add_argument("--structure-label", type=Path, help="Optional human_label1.txt or human_label2.txt")
    ground_truth.add_argument("--source-id")
    ground_truth.add_argument("--output", required=True, type=Path)

    evaluate = commands.add_parser("evaluate", help="Compare one prediction artifact with Ground Truth")
    evaluate.add_argument("--prediction", required=True, type=Path)
    evaluate.add_argument("--ground-truth", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)

    render = commands.add_parser("render-score", help="Remove chord-label track and render POP909-CL MIDI")
    render.add_argument("--midi", required=True, type=Path)
    render.add_argument("--output-wav", required=True, type=Path)
    render.add_argument("--score-midi", type=Path)
    render.add_argument("--soundfont", type=Path, default=Path("/usr/share/sounds/sf2/default-GM.sf2"))
    render.add_argument("--sample-rate", type=int, default=44100)
    render.add_argument("--manifest", type=Path)

    rebuild = commands.add_parser("rebuild", help="Reassemble and revalidate a saved artifact")
    rebuild.add_argument("artifact", type=Path)

    repair = commands.add_parser("repair", help="Retry only tasks with failed structured output")
    repair.add_argument("artifact", type=Path)
    repair.add_argument("--base-url")
    repair.add_argument("--max-new-tokens", type=int, default=4096)
    repair.add_argument("--seed", type=int)

    mir = commands.add_parser("mir-analyze", help="Extract beat-synchronous MIR candidates from audio")
    mir.add_argument("--audio", required=True, type=Path)
    mir.add_argument("--output", required=True, type=Path)
    mir.add_argument("--sample-rate", type=int, default=22050)
    mir.add_argument("--hop-length", type=int, default=512)
    mir.add_argument("--top-k", type=int, default=3)
    mir.add_argument("--assumed-beats-per-bar", type=int, default=4)

    evaluate_mir = commands.add_parser("evaluate-mir", help="Compare MIR candidates with Ground Truth")
    evaluate_mir.add_argument("--mir", required=True, type=Path)
    evaluate_mir.add_argument("--ground-truth", required=True, type=Path)
    evaluate_mir.add_argument("--output", type=Path)

    evaluate_graph = commands.add_parser("evaluate-graph", help="Evaluate selected branches of a MusicAOG graph")
    evaluate_graph.add_argument("--prediction", required=True, type=Path)
    evaluate_graph.add_argument("--ground-truth", required=True, type=Path)
    evaluate_graph.add_argument("--output", type=Path)

    mir_batch = commands.add_parser("mir-batch-pop909", help="Run the MIR-only POP909 benchmark")
    mir_batch.add_argument("--ids", default="001,002,003,004,005,006,007,008,009,010")
    mir_batch.add_argument("--output-root", required=True, type=Path)
    mir_batch.add_argument("--sample-rate", type=int, default=22050)
    mir_batch.add_argument("--hop-length", type=int, default=512)
    mir_batch.add_argument("--top-k", type=int, default=3)
    mir_batch.add_argument("--assumed-beats-per-bar", type=int, default=4)
    mir_batch.add_argument("--no-resume", action="store_true")

    fuse = commands.add_parser("fuse", help="Fuse one MOSS prediction with MIR candidates")
    fuse.add_argument("--prediction", required=True, type=Path)
    fuse.add_argument("--mir", required=True, type=Path)
    fuse.add_argument("--output", required=True, type=Path)

    infer = commands.add_parser("infer", help="Run auditable bottom-up/top-down inference on a fused graph")
    infer.add_argument("--fusion", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--audio-weight", type=float, default=1.0)
    infer.add_argument("--key-weight", type=float, default=0.75)
    infer.add_argument("--transition-weight", type=float, default=0.0)
    infer.add_argument("--compatibility-floor", type=float, default=0.05)
    infer.add_argument("--moss-source-weight", type=float, default=0.1)
    infer.add_argument("--max-iterations", type=int, default=8)

    infer_batch = commands.add_parser(
        "infer-batch-pop909", help="Run fixed bidirectional inference on saved POP909 artifacts"
    )
    infer_batch.add_argument("--ids", default="001,002,003,004,005,006,007,008,009,010")
    infer_batch.add_argument("--output-root", required=True, type=Path)
    infer_batch.add_argument("--audio-weight", type=float, default=1.0)
    infer_batch.add_argument("--key-weight", type=float, default=0.75)
    infer_batch.add_argument("--transition-weight", type=float, default=0.0)
    infer_batch.add_argument("--compatibility-floor", type=float, default=0.05)
    infer_batch.add_argument("--moss-source-weight", type=float, default=0.1)
    infer_batch.add_argument("--max-iterations", type=int, default=8)

    batch = commands.add_parser("batch-pop909", help="Run a resumable POP909-CL benchmark")
    batch.add_argument("--ids", default="001,002,003,004,005,006,007,008,009,010")
    batch.add_argument("--dataset-root", required=True, type=Path)
    batch.add_argument("--output-root", required=True, type=Path)
    batch.add_argument(
        "--base-url",
        default=os.environ.get("MOSS_MUSIC_SGLANG_BASE_URL", "http://127.0.0.1:30000"),
    )
    batch.add_argument("--max-new-tokens", type=int, default=2048)
    batch.add_argument("--harmony-chunk-seconds", type=float, default=30.0)
    batch.add_argument("--seed", type=int, default=42)
    batch.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        errors = validate_graph(_load_graph(args.json_file))
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print("VALID")
        return 0

    if args.command == "ground-truth":
        structure_code = args.structure_label.read_text(encoding="utf-8").strip() if args.structure_label else None
        artifact = load_pop909cl_ground_truth(
            args.midi,
            structure_code=structure_code,
            source_id=args.source_id,
        )
        _write_json(args.output.expanduser().resolve(), artifact)
        print(args.output.expanduser().resolve())
        return 0

    if args.command == "evaluate":
        metrics = evaluate_artifact(_load_json(args.prediction), _load_json(args.ground_truth))
        if args.output:
            _write_json(args.output.expanduser().resolve(), metrics)
            print(args.output.expanduser().resolve())
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    if args.command == "render-score":
        manifest = render_pop909cl_score(
            args.midi,
            args.output_wav,
            score_midi_path=args.score_midi,
            soundfont=args.soundfont,
            sample_rate=args.sample_rate,
        )
        if args.manifest:
            _write_json(args.manifest.expanduser().resolve(), manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "rebuild":
        path = args.artifact.expanduser().resolve()
        artifact = rebuild_artifact(_load_json(path))
        atomic_write_json(path, artifact)
        print(f"status={artifact['status']} graph_errors={len(artifact['graph_errors'])}")
        return 0 if artifact["status"] == "complete" else 2

    if args.command == "repair":
        path = args.artifact.expanduser().resolve()
        artifact = retry_failed_tasks(
            _load_json(path),
            output_path=path,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        remaining = artifact.get("repair_runs", [{}])[-1].get("remaining_parse_errors", [])
        print(f"status={artifact['status']} remaining_parse_errors={','.join(remaining) or 'none'}")
        return 0 if not remaining else 2

    if args.command == "mir-analyze":
        result = analyze_mir(
            args.audio,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            top_k=args.top_k,
            assumed_beats_per_bar=args.assumed_beats_per_bar,
        )
        output = args.output.expanduser().resolve()
        atomic_write_json(output, result)
        print(output)
        print(
            f"beats={len(result['metrical_grid']['beats'])} "
            f"chord_segments={len(result['chord_segments'])} "
            f"boundaries={len(result['structure_boundary_candidates'])}"
        )
        return 0

    if args.command == "evaluate-mir":
        metrics = evaluate_mir_artifact(_load_json(args.mir), _load_json(args.ground_truth))
        if args.output:
            _write_json(args.output.expanduser().resolve(), metrics)
            print(args.output.expanduser().resolve())
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-graph":
        metrics = evaluate_graph_artifact(_load_json(args.prediction), _load_json(args.ground_truth))
        if args.output:
            _write_json(args.output.expanduser().resolve(), metrics)
            print(args.output.expanduser().resolve())
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    if args.command == "mir-batch-pop909":
        song_ids = [item.strip().zfill(3) for item in args.ids.split(",") if item.strip()]
        result = run_pop909_mir_batch(
            song_ids=song_ids,
            output_root=args.output_root,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            top_k=args.top_k,
            assumed_beats_per_bar=args.assumed_beats_per_bar,
            resume=not args.no_resume,
        )
        print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
        return 0 if result["aggregate"]["songs_failed"] == 0 else 2

    if args.command == "fuse":
        result = fuse_moss_mir(_load_json(args.prediction), _load_json(args.mir))
        output = args.output.expanduser().resolve()
        atomic_write_json(output, result)
        print(output)
        print(f"status={result['status']} graph_errors={len(result['graph_errors'])}")
        return 0 if result["status"] == "complete" else 2

    if args.command == "infer":
        result = run_bidirectional_inference(
            _load_json(args.fusion),
            audio_weight=args.audio_weight,
            key_weight=args.key_weight,
            transition_weight=args.transition_weight,
            compatibility_floor=args.compatibility_floor,
            moss_source_weight=args.moss_source_weight,
            max_iterations=args.max_iterations,
        )
        output = args.output.expanduser().resolve()
        atomic_write_json(output, result)
        print(output)
        print(
            f"status={result['status']} stop_reason={result['stop_reason']} "
            f"iterations={len(result['inference_trace'])} "
            f"key_changed={result['selection_changes']['global_key']} "
            f"chords_changed={result['selection_changes']['chords']}"
        )
        return 0 if result["status"] == "complete" else 2

    if args.command == "infer-batch-pop909":
        song_ids = [item.strip().zfill(3) for item in args.ids.split(",") if item.strip()]
        result = run_pop909_bidirectional_batch(
            song_ids=song_ids,
            output_root=args.output_root,
            audio_weight=args.audio_weight,
            key_weight=args.key_weight,
            transition_weight=args.transition_weight,
            compatibility_floor=args.compatibility_floor,
            moss_source_weight=args.moss_source_weight,
            max_iterations=args.max_iterations,
        )
        print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
        return 0 if result["aggregate"]["songs_failed"] == 0 else 2

    if args.command == "batch-pop909":
        song_ids = [item.strip().zfill(3) for item in args.ids.split(",") if item.strip()]
        result = run_pop909_batch(
            song_ids=song_ids,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            harmony_chunk_seconds=args.harmony_chunk_seconds,
            seed=args.seed,
            resume=not args.no_resume,
        )
        print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
        return 0 if result["aggregate"]["songs_failed"] == 0 else 2

    audio = args.audio.expanduser().resolve()
    output = (args.output or _default_output(audio)).expanduser().resolve()
    artifact = run_pipeline(
        audio_path=audio,
        output_path=output,
        base_url=args.base_url,
        prompt_path=args.prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        harmony_chunk_seconds=args.harmony_chunk_seconds,
    )
    print(output)
    print(f"status={artifact['status']} graph_errors={len(artifact['graph_errors'])}")
    return 0 if artifact["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

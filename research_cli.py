"""Reproducible research CLI for MOSS-Music.

This keeps the upstream demo untouched and records every analysis as a JSON
artifact with prompts, generation parameters, audio hashes, and environment
metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Music-8B-Instruct"
PROMPT_FILE = ROOT / "research" / "prompts.json"
FULL_TASKS = (
    "overview",
    "structure",
    "harmony",
    "rhythm",
    "timbre",
    "lyrics",
    "interpretation",
    "uncertainty",
    "musicaog",
)


def load_prompt_catalog(path: Path = PROMPT_FILE) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not catalog or any("prompt" not in item or "title" not in item for item in catalog.values()):
        raise ValueError(f"Invalid prompt catalog: {path}")
    return catalog


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_model_label(model_id: str) -> str:
    return model_id.rstrip("/\\").replace("\\", "_").replace("/", "_").replace(":", "_")


def parse_json_candidate(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model response")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("MusicAOG response must be a JSON object")
    return parsed


def validate_musicaog_candidate(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("schema_version", "representation", "root", "nodes", "productions", "relations", "uncertainties")
    for key in required:
        if key not in candidate:
            errors.append(f"missing top-level field: {key}")
    nodes = candidate.get("nodes")
    if not isinstance(nodes, list):
        return [*errors, "nodes must be an array"]

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            errors.append(f"nodes[{index}] has no string id")
            continue
        node_id = node["id"]
        if node_id in node_ids:
            errors.append(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        onset, offset = node.get("onset_seconds"), node.get("offset_seconds")
        if not isinstance(onset, (int, float)) or not isinstance(offset, (int, float)) or onset < 0 or offset < onset:
            errors.append(f"invalid time span for node: {node_id}")

    root = candidate.get("root")
    if isinstance(root, str) and root not in node_ids:
        errors.append(f"root references missing node: {root}")
    for index, production in enumerate(candidate.get("productions", [])):
        if not isinstance(production, dict):
            errors.append(f"productions[{index}] must be an object")
            continue
        references = [production.get("parent"), *production.get("children", [])]
        for reference in references:
            if reference not in node_ids:
                errors.append(f"productions[{index}] references missing node: {reference}")
    for index, relation in enumerate(candidate.get("relations", [])):
        if not isinstance(relation, dict):
            errors.append(f"relations[{index}] must be an object")
            continue
        for field in ("source", "target"):
            if relation.get(field) not in node_ids:
                errors.append(f"relations[{index}].{field} references missing node: {relation.get(field)}")
    return errors


def collect_audio_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        metadata.update(
            sample_rate=info.sample_rate,
            num_frames=info.num_frames,
            num_channels=info.num_channels,
            duration_seconds=(info.num_frames / info.sample_rate if info.sample_rate else None),
        )
    except Exception as exc:  # Metadata is useful but must not block inference.
        metadata["probe_warning"] = f"{type(exc).__name__}: {exc}"
    return metadata


def resolve_tasks(args: argparse.Namespace, catalog: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    if args.question:
        return [{"id": "custom", "title": "自定义问题", "prompt": args.question}]
    task_ids = list(FULL_TASKS) if args.task == "full" else [args.task]
    return [{"id": task_id, **catalog[task_id]} for task_id in task_ids]


def load_runtime(args: argparse.Namespace):
    import torch

    from src.modeling_moss_music import MossMusicModel
    from src.processing_moss_music import MossMusicProcessor

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")

    device_map: str = "cuda:0" if args.device == "cuda" else args.device
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": args.dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if args.device == "auto":
        load_kwargs["max_memory"] = {
            0: f"{args.gpu_memory_gib}GiB",
            "cpu": f"{args.cpu_memory_gib}GiB",
        }
        load_kwargs["offload_folder"] = str(Path(args.offload_dir).resolve())

    model = MossMusicModel.from_pretrained(args.model, **load_kwargs)
    model.eval()
    processor = MossMusicProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        enable_time_marker=not args.no_time_markers,
    )
    return torch, model, processor


def prepare_inputs(torch, model, processor, audio_path: Path, prompt: str):
    from src.audio_io import load_audio

    raw_audio = load_audio(str(audio_path), sample_rate=processor.config.mel_sr)
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")

    # The supported baseline uses one execution device.  For Accelerate's
    # automatic placement, put token tensors with the embedding layer and the
    # audio tensor with the audio encoder; module hooks handle later transfers.
    text_device = model.get_input_embeddings().weight.device
    audio_device = next(model.audio_encoder.parameters()).device
    for key, value in list(inputs.items()):
        if not hasattr(value, "to"):
            continue
        target = audio_device if key in {"audio_data", "audio_data_seqlens"} else text_device
        inputs[key] = value.to(target)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(dtype=model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    return inputs


def generate_one(torch, model, processor, audio_path: Path, prompt: str, args: argparse.Namespace) -> str:
    inputs = prepare_inputs(torch, model, processor, audio_path, prompt)
    generation: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "use_cache": True,
        "do_sample": args.sample,
    }
    if args.sample:
        generation.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def run(args: argparse.Namespace) -> Path:
    catalog = load_prompt_catalog()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    random.seed(args.seed)
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}-{run_id}.json"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "loading_model",
        "run_id": run_id,
        "started_at_utc": started.isoformat(),
        "model": {"id_or_path": args.model, "label": safe_model_label(args.model)},
        "audio": collect_audio_metadata(audio_path),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": args.device,
            "dtype": args.dtype,
            "time_markers": not args.no_time_markers,
            "seed": args.seed,
        },
        "generation": {
            "do_sample": args.sample,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature if args.sample else None,
            "top_p": args.top_p if args.sample else None,
            "top_k": args.top_k if args.sample else None,
        },
        "analyses": [],
    }
    atomic_write_json(output_path, payload)

    torch, model, processor = load_runtime(args)
    payload["runtime"].update(
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_runtime=torch.version.cuda,
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payload["status"] = "running"
    atomic_write_json(output_path, payload)
    for task in resolve_tasks(args, catalog):
        task_started = time.perf_counter()
        response = generate_one(torch, model, processor, audio_path, task["prompt"], args)
        analysis = {
            "task": task["id"],
            "title": task["title"],
            "prompt": task["prompt"],
            "response": response,
            "elapsed_seconds": round(time.perf_counter() - task_started, 3),
        }
        if task["id"] == "musicaog":
            try:
                structured = parse_json_candidate(response)
                analysis["structured_response"] = structured
                analysis["schema_errors"] = validate_musicaog_candidate(structured)
            except (ValueError, json.JSONDecodeError) as exc:
                analysis["structured_parse_error"] = f"{type(exc).__name__}: {exc}"
        payload["analyses"].append(analysis)
        atomic_write_json(output_path, payload)

    payload["status"] = "complete"
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["total_elapsed_seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
    atomic_write_json(output_path, payload)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    catalog = load_prompt_catalog()
    parser = argparse.ArgumentParser(description="Reproducible MOSS-Music analysis runner")
    parser.add_argument("--audio", help="Audio path for analysis")
    parser.add_argument("--task", choices=[*catalog.keys(), "full"], default="overview")
    parser.add_argument("--question", help="Use a custom question instead of --task")
    parser.add_argument("--list-tasks", action="store_true", help="Print the prompt catalog and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local path")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--gpu-memory-gib", type=int, default=20, help="GPU cap for --device auto")
    parser.add_argument("--cpu-memory-gib", type=int, default=48, help="CPU RAM cap for --device auto")
    parser.add_argument("--offload-dir", default=str(ROOT / "offload"))
    parser.add_argument("--output-dir", default=str(ROOT / "research_outputs"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", action="store_true", help="Enable sampling; greedy decoding is the reproducible default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-time-markers", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_tasks:
        for task_id, item in load_prompt_catalog().items():
            print(f"{task_id:14} {item['title']}")
        return 0
    if not args.audio:
        parser.error("--audio is required unless --list-tasks is used")
    try:
        output_path = run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

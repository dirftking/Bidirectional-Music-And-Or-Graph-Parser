"""Reproducible multi-prompt Audio2MusicAOG inference pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import audit_task_results
from .assembler import assemble_candidate_graph
from .client import SGLangClient
from .parsing import JsonExtractionError, extract_json_object
from .validation import validate_graph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "research" / "audio2musicaog_prompts.json"
TASK_KEYS = {
    "global": {"tempo_bpm", "time_signature", "global_key"},
    "structure": {"sections"},
    "harmony": {"chords"},
}
REPAIR_SUFFIX = (
    " REPAIR RUN: The previous answer could not be parsed. Be concise and finish exactly one "
    "JSON object before the token limit. Do not emit Markdown or prose after the JSON. Use decimal "
    "seconds, never minute:second notation. Prefer a coarse complete answer over a long truncated one."
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration_seconds(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def load_prompts(path: Path = DEFAULT_PROMPTS) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    missing = set(TASK_KEYS) - set(prompts)
    if missing:
        raise ValueError(f"Prompt catalog is missing tasks: {sorted(missing)}")
    return prompts


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _canonicalize_chord_label(value: Any) -> tuple[Any, Any | None]:
    if not isinstance(value, str) or value == "N" or re.fullmatch(r"[A-G](?:#|b)?:[^\s]+", value):
        return value, None
    cleaned = value.strip().replace("♯", "#").replace("♭", "b")
    match = re.fullmatch(
        r"([A-Ga-g](?:#|b)?)(?:\s*)(major|minor|maj|min|m|M|7|maj7|min7|m7)?",
        cleaned,
    )
    if not match:
        return value, None
    aliases = {
        None: "maj", "major": "maj", "maj": "maj", "M": "maj",
        "minor": "min", "min": "min", "m": "min", "7": "7",
        "maj7": "maj7", "min7": "min7", "m7": "min7",
    }
    label = f"{match.group(1)[0].upper() + match.group(1)[1:]}:{aliases[match.group(2)]}"
    return label, value if label != value else None


def _coerce_timestamp(value: Any) -> tuple[Any, Any | None]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), None
    if isinstance(value, str):
        try:
            return float(value.strip()), value
        except ValueError:
            pass
    return value, None


def _merge_chord_chunks(chunks: list[tuple[float, dict[str, Any]]]) -> dict[str, Any]:
    chords: list[dict[str, Any]] = []
    uncertainties: list[str] = []
    for offset, parsed in chunks:
        for item in parsed.get("chords", []):
            if not isinstance(item, dict):
                continue
            shifted = dict(item)
            shifted["label"], original_label = _canonicalize_chord_label(shifted.get("label"))
            if original_label is not None:
                shifted["original_label"] = original_label
            for field in ("start_seconds", "end_seconds"):
                value, original_value = _coerce_timestamp(shifted.get(field))
                if original_value is not None:
                    shifted.setdefault("original_timestamps", {})[field] = original_value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    shifted[field] = round(float(value) + offset, 6)
            if (
                chords
                and chords[-1].get("label") == shifted.get("label")
                and isinstance(chords[-1].get("end_seconds"), (int, float))
                and isinstance(shifted.get("start_seconds"), (int, float))
                and abs(float(chords[-1]["end_seconds"]) - float(shifted["start_seconds"])) <= 0.25
            ):
                chords[-1]["end_seconds"] = shifted.get("end_seconds")
                chords[-1]["confidence"] = min(
                    float(chords[-1].get("confidence", 0.5)), float(shifted.get("confidence", 0.5))
                )
            else:
                chords.append(shifted)
        uncertainties.extend(str(item) for item in parsed.get("uncertainties", []) or [])
    return {"chords": chords, "uncertainties": uncertainties}


def _render_chunk(source: Path, destination: Path, offset: float, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(offset), "-t", str(duration), "-i", str(source),
            "-vn", "-acodec", "pcm_s16le", str(destination),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def _run_chunked_harmony(
    *,
    client: SGLangClient,
    audio_path: Path,
    duration: float,
    task: dict[str, Any],
    chunk_seconds: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    chunk_artifacts: list[dict[str, Any]] = []
    parsed_chunks: list[tuple[float, dict[str, Any]]] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="audio2musicaog-harmony-") as temporary:
        temporary_path = Path(temporary)
        offset = 0.0
        index = 0
        while offset < duration - 1e-6:
            index += 1
            chunk_duration = min(chunk_seconds, duration - offset)
            chunk_path = temporary_path / f"chunk-{index:04d}.wav"
            _render_chunk(audio_path, chunk_path, offset, chunk_duration)
            prompt = (
                task["prompt"]
                + f" This file is chunk {index} of a longer recording. "
                + f"All timestamps MUST be relative to this chunk, between 0 and {chunk_duration:.3f} seconds. "
                + "Do not continue beyond the supplied audio and do not use minute:second notation."
            )
            item: dict[str, Any] = {
                "index": index,
                "offset_seconds": round(offset, 6),
                "duration_seconds": round(chunk_duration, 6),
                "prompt": prompt,
                "attempts": [],
            }
            parsed = None
            last_error = None
            for attempt_index in range(2):
                attempt_prompt = prompt
                if attempt_index:
                    attempt_prompt += (
                        " RETRY: The previous answer was not valid JSON. Be concise. "
                        "Finish one JSON object before any token limit and do not emit analysis or prose."
                    )
                result = client.generate(
                    audio_path=str(chunk_path),
                    prompt=attempt_prompt,
                    max_new_tokens=max_new_tokens if attempt_index == 0 else max(4096, max_new_tokens),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed + index + attempt_index * 100000,
                )
                attempt: dict[str, Any] = {
                    "attempt": attempt_index + 1,
                    "prompt": attempt_prompt,
                    "raw_response": result.text,
                    "meta_info": result.meta_info,
                }
                try:
                    parsed = extract_json_object(result.text, TASK_KEYS["harmony"])
                    attempt["parse_error"] = None
                    item["attempts"].append(attempt)
                    break
                except (JsonExtractionError, json.JSONDecodeError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempt["parse_error"] = last_error
                    item["attempts"].append(attempt)
            final_attempt = item["attempts"][-1]
            item["raw_response"] = final_attempt["raw_response"]
            item["meta_info"] = final_attempt["meta_info"]
            if parsed is not None:
                item["parsed"] = parsed
                item["parse_error"] = None
                parsed_chunks.append((offset, parsed))
            else:
                message = f"chunk {index}: {last_error}"
                item["parsed"] = None
                item["parse_error"] = message
                errors.append(message)
            chunk_artifacts.append(item)
            offset += chunk_duration
    parsed = _merge_chord_chunks(parsed_chunks)
    artifact = {
        "title": task["title"],
        "prompt": task["prompt"],
        "mode": "chunked",
        "chunk_seconds": chunk_seconds,
        "chunks": chunk_artifacts,
        "parsed": parsed,
        "parse_error": "; ".join(errors) if errors else None,
    }
    return artifact, parsed, artifact["parse_error"]


def rebuild_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Reassemble and revalidate saved model outputs without repeating inference."""

    harmony_task = artifact.get("tasks", {}).get("harmony", {})
    if isinstance(harmony_task, dict) and harmony_task.get("mode") == "chunked":
        saved_chunks = []
        for chunk in harmony_task.get("chunks", []):
            if isinstance(chunk, dict) and isinstance(chunk.get("parsed"), dict):
                saved_chunks.append((float(chunk.get("offset_seconds", 0.0)), chunk["parsed"]))
        harmony_task["parsed"] = _merge_chord_chunks(saved_chunks)

    parsed_results = {
        name: task.get("parsed")
        for name, task in artifact.get("tasks", {}).items()
        if isinstance(task, dict) and isinstance(task.get("parsed"), dict)
    }
    audio = artifact.get("audio", {})
    server = artifact.get("server", {})
    model_info = server.get("model_info", {}) if isinstance(server, dict) else {}
    graph = assemble_candidate_graph(
        parsed_results.get("global"),
        parsed_results.get("structure"),
        parsed_results.get("harmony"),
        audio_sha256=str(audio.get("sha256") or ""),
        audio_filename=str(audio.get("filename") or "unknown"),
        duration_seconds=audio.get("duration_seconds"),
        model=str(model_info.get("model_path") or "MOSS-Music"),
    )
    graph_errors = validate_graph(graph)
    audit_warnings = audit_task_results(
        parsed_results.get("global"),
        parsed_results.get("structure"),
        parsed_results.get("harmony"),
        duration_seconds=audio.get("duration_seconds"),
    )
    parse_errors = [
        name for name, task in artifact.get("tasks", {}).items()
        if isinstance(task, dict) and task.get("parse_error")
    ]
    artifact.update(
        graph=graph,
        graph_errors=graph_errors,
        audit_warnings=audit_warnings,
        status="complete" if not parse_errors and not graph_errors else "partial",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return artifact


def _preserve_initial_attempt(task: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = task.setdefault("attempts", [])
    if not attempts and task.get("raw_response") is not None:
        attempts.append(
            {
                "attempt": 1,
                "prompt": task.get("prompt"),
                "raw_response": task.get("raw_response"),
                "meta_info": task.get("meta_info", {}),
                "parse_error": task.get("parse_error"),
            }
        )
    return attempts


def retry_failed_tasks(
    artifact: dict[str, Any],
    *,
    output_path: Path,
    base_url: str | None = None,
    max_new_tokens: int = 4096,
    seed: int | None = None,
) -> dict[str, Any]:
    """Retry only task outputs that failed JSON parsing, then rebuild the graph.

    Original responses remain in each task's ``attempts`` list, making the
    structured-output repair condition auditable as a separate experiment.
    """

    tasks = artifact.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError("artifact.tasks must be an object")
    audio = artifact.get("audio", {})
    audio_path = Path(str(audio.get("path") or "")).expanduser()
    failed = [
        name for name, task in tasks.items()
        if name in TASK_KEYS and isinstance(task, dict) and task.get("parse_error")
    ]
    if failed and not audio_path.is_file():
        raise FileNotFoundError(f"artifact audio path is unavailable: {audio_path}")

    generation = artifact.get("generation", {})
    repair_seed = int(seed if seed is not None else generation.get("seed", 42))
    repair_base_url = base_url or str(artifact.get("server", {}).get("base_url") or "http://127.0.0.1:30000")
    client = SGLangClient(repair_base_url)

    def generate(prompt: str, offset: int, source: Path) -> Any:
        return client.generate(
            audio_path=str(source),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=float(generation.get("temperature", 0.0)),
            top_p=float(generation.get("top_p", 0.95)),
            top_k=int(generation.get("top_k", 50)),
            seed=repair_seed + 200000 + offset,
        )

    for task_index, task_name in enumerate(("global", "structure", "harmony")):
        task = tasks.get(task_name)
        if not isinstance(task, dict) or not task.get("parse_error"):
            continue
        if task_name == "harmony" and task.get("mode") == "chunked":
            with tempfile.TemporaryDirectory(prefix="audio2musicaog-repair-") as temporary:
                temporary_path = Path(temporary)
                for chunk in task.get("chunks", []):
                    if not isinstance(chunk, dict) or not chunk.get("parse_error"):
                        continue
                    index = int(chunk.get("index", 0))
                    offset = float(chunk.get("offset_seconds", 0.0))
                    duration = float(chunk.get("duration_seconds", 0.0))
                    chunk_path = temporary_path / f"chunk-{index:04d}.wav"
                    _render_chunk(audio_path, chunk_path, offset, duration)
                    prompt = str(chunk.get("prompt") or task.get("prompt") or "") + REPAIR_SUFFIX
                    result = generate(prompt, task_index * 10000 + index, chunk_path)
                    attempts = _preserve_initial_attempt(chunk)
                    attempt: dict[str, Any] = {
                        "attempt": len(attempts) + 1,
                        "prompt": prompt,
                        "raw_response": result.text,
                        "meta_info": result.meta_info,
                    }
                    try:
                        parsed = extract_json_object(result.text, TASK_KEYS[task_name])
                        attempt["parse_error"] = None
                        chunk.update(
                            raw_response=result.text,
                            meta_info=result.meta_info,
                            parsed=parsed,
                            parse_error=None,
                        )
                    except (JsonExtractionError, json.JSONDecodeError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        attempt["parse_error"] = error
                        chunk.update(
                            raw_response=result.text,
                            meta_info=result.meta_info,
                            parsed=None,
                            parse_error=f"chunk {index}: {error}",
                        )
                    attempts.append(attempt)
                    atomic_write_json(output_path, artifact)
            remaining = [
                str(chunk["parse_error"])
                for chunk in task.get("chunks", [])
                if isinstance(chunk, dict) and chunk.get("parse_error")
            ]
            task["parse_error"] = "; ".join(remaining) if remaining else None
            continue

        prompt = str(task.get("prompt") or "") + REPAIR_SUFFIX
        result = generate(prompt, task_index * 10000, audio_path)
        attempts = _preserve_initial_attempt(task)
        attempt = {
            "attempt": len(attempts) + 1,
            "prompt": prompt,
            "raw_response": result.text,
            "meta_info": result.meta_info,
        }
        try:
            parsed = extract_json_object(result.text, TASK_KEYS[task_name])
            attempt["parse_error"] = None
            task.update(
                raw_response=result.text,
                meta_info=result.meta_info,
                parsed=parsed,
                parse_error=None,
            )
        except (JsonExtractionError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempt["parse_error"] = error
            task.update(
                raw_response=result.text,
                meta_info=result.meta_info,
                parsed=None,
                parse_error=error,
            )
        attempts.append(attempt)
        atomic_write_json(output_path, artifact)

    rebuild_artifact(artifact)
    artifact.setdefault("repair_runs", []).append(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": repair_base_url,
            "max_new_tokens": max_new_tokens,
            "seed": repair_seed,
            "targeted_tasks": failed,
            "remaining_parse_errors": [
                name for name, task in tasks.items()
                if name in TASK_KEYS and isinstance(task, dict) and task.get("parse_error")
            ],
        }
    )
    atomic_write_json(output_path, artifact)
    return artifact


def run_pipeline(
    *,
    audio_path: Path,
    output_path: Path,
    base_url: str,
    prompt_path: Path = DEFAULT_PROMPTS,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int = 42,
    harmony_chunk_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run global, structure, and harmony extraction, then assemble one graph."""

    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    prompts = load_prompts(prompt_path)
    client = SGLangClient(base_url)
    model_info = client.model_info()
    audio_hash = sha256_file(audio_path)
    duration = probe_duration_seconds(audio_path)
    started = datetime.now(timezone.utc)
    artifact: dict[str, Any] = {
        "artifact_version": "0.1",
        "status": "running",
        "started_at_utc": started.isoformat(),
        "audio": {
            "path": str(audio_path),
            "filename": audio_path.name,
            "sha256": audio_hash,
            "size_bytes": audio_path.stat().st_size,
            "duration_seconds": duration,
        },
        "server": {"base_url": base_url, "model_info": model_info},
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "harmony_chunk_seconds": harmony_chunk_seconds,
        },
        "tasks": {},
        "graph": None,
        "graph_errors": [],
        "audit_warnings": [],
    }
    atomic_write_json(output_path, artifact)

    parsed_results: dict[str, dict[str, Any]] = {}
    for task_index, task_name in enumerate(("global", "structure", "harmony")):
        task = prompts[task_name]
        if (
            task_name == "harmony"
            and duration is not None
            and harmony_chunk_seconds > 0
            and duration > harmony_chunk_seconds
        ):
            task_artifact, parsed, _ = _run_chunked_harmony(
                client=client,
                audio_path=audio_path,
                duration=duration,
                task=task,
                chunk_seconds=harmony_chunk_seconds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed + task_index * 1000,
            )
            parsed_results[task_name] = parsed
            artifact["tasks"][task_name] = task_artifact
            atomic_write_json(output_path, artifact)
            continue
        result = client.generate(
            audio_path=str(audio_path),
            prompt=task["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed + task_index,
        )
        task_artifact: dict[str, Any] = {
            "title": task["title"],
            "prompt": task["prompt"],
            "raw_response": result.text,
            "meta_info": result.meta_info,
        }
        try:
            parsed = extract_json_object(result.text, TASK_KEYS[task_name])
            parsed_results[task_name] = parsed
            task_artifact["parsed"] = parsed
            task_artifact["parse_error"] = None
        except (JsonExtractionError, json.JSONDecodeError) as exc:
            task_artifact["parsed"] = None
            task_artifact["parse_error"] = f"{type(exc).__name__}: {exc}"
        artifact["tasks"][task_name] = task_artifact
        atomic_write_json(output_path, artifact)

    rebuild_artifact(artifact)
    atomic_write_json(output_path, artifact)
    return artifact

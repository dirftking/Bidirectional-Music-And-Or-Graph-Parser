"""Local-only web service for interactive Audio -> MusicAOG parsing on AutoDL."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "webapp"
JOBS_ROOT = Path(
    os.environ.get("MUSIC_AOG_JOBS_ROOT", ROOT / "research_outputs" / "web_jobs")
).expanduser().resolve()
SGLANG_BASE_URL = os.environ.get(
    "MOSS_MUSIC_SGLANG_BASE_URL", "http://127.0.0.1:30000"
).rstrip("/")
HOST = os.environ.get("MUSIC_AOG_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("MUSIC_AOG_WEB_PORT", "7861"))
MAX_UPLOAD_BYTES = int(os.environ.get("MUSIC_AOG_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MOSS_MAX_NEW_TOKENS = int(os.environ.get("MUSIC_AOG_MOSS_MAX_NEW_TOKENS", "2048"))
MOSS_REPAIR_MAX_NEW_TOKENS = int(
    os.environ.get("MUSIC_AOG_MOSS_REPAIR_MAX_NEW_TOKENS", "4096")
)
MOSS_REPAIR_ATTEMPTS = int(os.environ.get("MUSIC_AOG_MOSS_REPAIR_ATTEMPTS", "2"))
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac"}
PYTHON = os.environ.get("MUSIC_AOG_PYTHON", sys.executable)

app = FastAPI(title="Bidirectional Music And-Or Graph Parser", version="0.1.0")
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_gpu_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _job_dir(job_id: str) -> Path:
    try:
        parsed = uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    path = (JOBS_ROOT / str(parsed)).resolve()
    if path.parent != JOBS_ROOT:
        raise HTTPException(status_code=404, detail="Job not found")
    return path


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "logs": job["logs"][-120:],
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "result_url": f"/api/jobs/{job['id']}/result" if job["status"] == "complete" else None,
        "audio_url": f"/api/jobs/{job['id']}/audio",
        "artifact_url": f"/api/jobs/{job['id']}/artifact" if job["status"] == "complete" else None,
    }


def _persist(job: dict[str, Any]) -> None:
    target = _job_dir(job["id"]) / "status.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(_public_job(job), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


def _update(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(changes)
        job["updated_at"] = _now()
        _persist(job)


def _log(job_id: str, message: str) -> None:
    clean = message.rstrip()
    if not clean:
        return
    stamp = time.strftime("%H:%M:%S")
    with _jobs_lock:
        job = _jobs[job_id]
        job["logs"].append(f"[{stamp}] {clean}")
        job["logs"] = job["logs"][-500:]
        job["updated_at"] = _now()
        _persist(job)


def _sglang_ready(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{SGLANG_BASE_URL}/health", timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def _run_command(
    job_id: str,
    label: str,
    args: list[str],
    *,
    allowed_return_codes: set[int] | None = None,
) -> int:
    _log(job_id, f"开始：{label}")
    env = os.environ.copy()
    env["MOSS_MUSIC_SGLANG_BASE_URL"] = SGLANG_BASE_URL
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _log(job_id, line)
    return_code = process.wait()
    allowed = allowed_return_codes or {0}
    if return_code not in allowed:
        raise RuntimeError(f"{label}失败，退出码 {return_code}")
    if return_code == 0:
        _log(job_id, f"完成：{label}")
    else:
        _log(job_id, f"{label}产生可恢复的部分结果，退出码 {return_code}")
    return return_code


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_result(job_id: str, work: Path) -> dict[str, Any]:
    inference = _load(work / "inference.json")
    mir = _load(work / "mir.json")
    moss = _load(work / "moss.json")
    moss_parse_errors = {
        name: str(task.get("parse_error"))
        for name, task in (moss.get("tasks") or {}).items()
        if isinstance(task, dict) and task.get("parse_error")
    }
    graph = inference.get("graph") or {}
    nodes = graph.get("nodes") or []
    productions = graph.get("productions") or []
    node_by_id = {node.get("id"): node for node in nodes}
    productions_by_parent: dict[str, list[dict[str, Any]]] = {}
    for production in productions:
        productions_by_parent.setdefault(str(production.get("parent")), []).append(production)

    chord_nodes = [
        node for node in nodes
        if node.get("kind") == "or" and node.get("type") == "chord"
    ]
    chord_nodes.sort(key=lambda node: int((node.get("attributes") or {}).get("beat_interval_index", 0)))
    initial_chords = (inference.get("initial_selection") or {}).get("chords") or []
    final_chords = (inference.get("final_selection") or {}).get("chords") or []
    choices: list[dict[str, Any]] = []
    for fallback_index, parent in enumerate(chord_nodes):
        attributes = parent.get("attributes") or {}
        index = int(attributes.get("beat_interval_index", fallback_index))
        span = parent.get("span") or {}
        candidate_rows: list[dict[str, Any]] = []
        for production in productions_by_parent.get(str(parent.get("id")), []):
            child_ids = production.get("children") or []
            child = node_by_id.get(child_ids[0]) if child_ids else None
            if not child:
                continue
            child_attributes = child.get("attributes") or {}
            scores = child_attributes.get("inference_scores") or {}
            candidate_rows.append({
                "label": child_attributes.get("label", "unknown"),
                "sources": child_attributes.get("sources") or [],
                "bottom_up": round(_float(scores.get("bottom_up_probability")), 6),
                "top_down": round(_float(scores.get("top_down_score"), 1.0), 6),
                "posterior": round(
                    _float(scores.get("posterior_probability"), production.get("probability")), 6
                ),
                "selected": bool(scores.get("selected_after", production.get("selected", False))),
            })
        candidate_rows.sort(key=lambda row: row["posterior"], reverse=True)
        before = initial_chords[index] if index < len(initial_chords) else None
        after = final_chords[index] if index < len(final_chords) else None
        choices.append({
            "index": index,
            "start": round(_float(span.get("start_seconds")), 6),
            "end": round(_float(span.get("end_seconds")), 6),
            "bottom_up": before,
            "final": after,
            "changed": before != after,
            "candidates": candidate_rows,
        })

    sections = []
    for node in nodes:
        if node.get("level") != "section" or node.get("kind") != "and":
            continue
        span = node.get("span") or {}
        attributes = node.get("attributes") or {}
        sections.append({
            "start": round(_float(span.get("start_seconds")), 6),
            "end": round(_float(span.get("end_seconds")), 6),
            "label": attributes.get("label") or attributes.get("symbol") or "section",
        })

    audio = mir.get("audio") or {}
    trace = inference.get("inference_trace") or []
    return {
        "job_id": job_id,
        "filename": _jobs[job_id]["filename"],
        "audio_url": f"/api/jobs/{job_id}/audio",
        "has_ground_truth": False,
        "duration": round(_float(audio.get("duration_seconds")), 6),
        "tempo": mir.get("tempo") or {},
        "key_candidates": mir.get("key_candidates") or [],
        "initial_key": (inference.get("initial_selection") or {}).get("global_key"),
        "final_key": (inference.get("final_selection") or {}).get("global_key"),
        "choices": choices,
        "beats": (mir.get("metrical_grid") or {}).get("beats") or [],
        "structure_boundaries": mir.get("structure_boundary_candidates") or [],
        "sections": sections,
        "trace": trace,
        "stop_reason": inference.get("stop_reason"),
        "selection_changes": inference.get("selection_changes") or {},
        "moss_status": moss.get("status"),
        "moss_parse_errors": moss_parse_errors,
        "graph_stats": {
            "nodes": len(nodes),
            "productions": len(productions),
            "relations": len(graph.get("relations") or []),
            "valid": not bool(inference.get("graph_errors")),
        },
    }


def _run_job(job_id: str) -> None:
    work = _job_dir(job_id)
    upload = next(work.glob("input.*"))
    moss = work / "moss.json"
    mir = work / "mir.json"
    fusion = work / "fusion.json"
    inference = work / "inference.json"
    try:
        _update(job_id, status="queued", stage="等待 GPU", progress=0.08)
        _log(job_id, "任务已进入队列；同一时间只运行一个 GPU 解析任务")
        with _gpu_lock:
            _update(job_id, status="running", stage="检查模型服务", progress=0.10)
            if not _sglang_ready():
                raise RuntimeError(
                    f"MOSS-Music 模型服务未就绪：{SGLANG_BASE_URL}/health"
                )
            _log(job_id, "MOSS-Music 模型服务正常")

            _update(job_id, stage="MOSS-Music 结构分析", progress=0.16)
            moss_return_code = _run_command(job_id, "MOSS-Music 结构分析", [
                PYTHON, "-m", "audio2musicaog.cli", "analyze",
                "--audio", str(upload), "--output", str(moss),
                "--base-url", SGLANG_BASE_URL,
                "--max-new-tokens", str(MOSS_MAX_NEW_TOKENS),
                "--temperature", "0",
                "--harmony-chunk-seconds", "30",
            ], allowed_return_codes={0, 2})

            if moss_return_code == 2:
                _update(job_id, stage="修复 MOSS-Music 结构化输出", progress=0.50)
                for repair_attempt in range(1, MOSS_REPAIR_ATTEMPTS + 1):
                    _log(
                        job_id,
                        f"检测到 JSON 截断或格式错误；仅重试失败任务（{repair_attempt}/{MOSS_REPAIR_ATTEMPTS}）",
                    )
                    moss_return_code = _run_command(
                        job_id,
                        "MOSS-Music 结构化输出修复",
                        [
                            PYTHON, "-m", "audio2musicaog.cli", "repair", str(moss),
                            "--base-url", SGLANG_BASE_URL,
                            "--max-new-tokens", str(MOSS_REPAIR_MAX_NEW_TOKENS),
                            "--seed", str(142 + repair_attempt),
                        ],
                        allowed_return_codes={0, 2},
                    )
                    if moss_return_code == 0:
                        break
                if moss_return_code == 2:
                    _log(
                        job_id,
                        "警告：少量 MOSS 任务仍为部分结果；继续使用 MIR 候选完成后续解析",
                    )

            _update(job_id, stage="MIR Bottom-up 候选", progress=0.62)
            _run_command(job_id, "MIR Bottom-up 候选", [
                PYTHON, "-m", "audio2musicaog.cli", "mir-analyze",
                "--audio", str(upload), "--output", str(mir), "--top-k", "3",
            ])

            _update(job_id, stage="融合 MOSS 与 MIR", progress=0.78)
            _run_command(job_id, "融合候选图", [
                PYTHON, "-m", "audio2musicaog.cli", "fuse",
                "--prediction", str(moss), "--mir", str(mir), "--output", str(fusion),
            ])

            _update(job_id, stage="Bottom-up / Top-down 双向推理", progress=0.88)
            _run_command(job_id, "双向 MusicAOG 推理", [
                PYTHON, "-m", "audio2musicaog.cli", "infer",
                "--fusion", str(fusion), "--output", str(inference),
                "--audio-weight", "1.0", "--key-weight", "0.75",
                "--transition-weight", "0.25", "--compatibility-floor", "0.05",
                "--moss-source-weight", "0.1", "--max-iterations", "8",
            ])

            _update(job_id, stage="生成可视化数据", progress=0.97)
            result = _build_result(job_id, work)
            (work / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
            _log(job_id, "可视化数据已生成")
            _update(job_id, status="complete", stage="完成", progress=1.0, error=None)
    except Exception as exc:  # pragma: no cover - exercised in deployment
        _log(job_id, f"错误：{exc}")
        _update(job_id, status="failed", stage="失败", error=str(exc))


def _get_job(job_id: str) -> dict[str, Any]:
    _job_dir(job_id)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


def _restore_jobs() -> None:
    for status_file in JOBS_ROOT.glob("*/status.json"):
        try:
            payload = _load(status_file)
            job_id = str(uuid.UUID(str(payload["id"])))
            job = {
                "id": job_id,
                "filename": str(payload.get("filename") or "audio"),
                "status": str(payload.get("status") or "failed"),
                "stage": str(payload.get("stage") or "未知"),
                "progress": _float(payload.get("progress")),
                "logs": list(payload.get("logs") or []),
                "error": payload.get("error"),
                "created_at": _float(payload.get("created_at"), _now()),
                "updated_at": _float(payload.get("updated_at"), _now()),
            }
            if job["status"] in {"queued", "running"}:
                job["status"] = "failed"
                job["stage"] = "服务重启，任务已中断"
                job["error"] = "解析服务在任务执行期间重启，请重新上传音频。"
            _jobs[job_id] = job
            _persist(job)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue


_restore_jobs()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "music-aog-live.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "sglang_ready": _sglang_ready(), "sglang_base_url": SGLANG_BASE_URL}


@app.post("/api/jobs", status_code=202)
async def create_job(audio: UploadFile = File(...)) -> dict[str, Any]:
    original = Path(audio.filename or "audio")
    extension = original.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only MP3, WAV and FLAC are supported")

    job_id = str(uuid.uuid4())
    work = _job_dir(job_id)
    work.mkdir(parents=True, exist_ok=False)
    destination = work / f"input{extension}"
    size = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Audio file is too large")
                handle.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink()
        work.rmdir()
        raise
    finally:
        await audio.close()

    job = {
        "id": job_id,
        "filename": original.name,
        "status": "queued",
        "stage": "已上传",
        "progress": 0.05,
        "logs": [f"[{time.strftime('%H:%M:%S')}] 已接收 {original.name}（{size / 1024 / 1024:.2f} MB）"],
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
        _persist(job)
    threading.Thread(target=_run_job, args=(job_id,), daemon=True, name=f"musicaog-{job_id[:8]}").start()
    return _public_job(job)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    return _public_job(_get_job(job_id))


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="Job is not complete")
    return FileResponse(_job_dir(job_id) / "result.json", media_type="application/json")


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str) -> FileResponse:
    _get_job(job_id)
    work = _job_dir(job_id)
    matches = list(work.glob("input.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(matches[0], filename=_jobs[job_id]["filename"])


@app.get("/api/jobs/{job_id}/artifact")
def job_artifact(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="Job is not complete")
    return FileResponse(
        _job_dir(job_id) / "inference.json",
        media_type="application/json",
        filename=f"{Path(job['filename']).stem}-musicaog.json",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

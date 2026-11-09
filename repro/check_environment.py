"""Diagnose the local MOSS-Music runtime without loading model weights."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_version(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if not executable:
        return {"ok": False, "error": "not found"}
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    first_line = (result.stdout or result.stderr).splitlines()
    return {
        "ok": result.returncode == 0,
        "executable": executable,
        "version": first_line[0] if first_line else "",
    }


def check_weights(path: Path) -> dict[str, object]:
    required = ("config.json", "model.safetensors.index.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    return {
        "ok": not missing and bool(shards),
        "path": str(path.resolve()),
        "missing": missing,
        "shard_count": len(shards),
        "size_gib": round(sum(item.stat().st_size for item in shards) / 1024**3, 3),
    }


def collect(model_path: Path | None) -> tuple[dict[str, object], bool]:
    report: dict[str, object] = {
        "python": {"ok": sys.version_info[:2] == (3, 12), "version": sys.version},
        "platform": platform.platform(),
        "ffmpeg": command_version(["ffmpeg", "-version"]),
        "nvidia_smi": command_version(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"]),
    }
    success = bool(report["python"]["ok"] and report["ffmpeg"]["ok"])
    try:
        import accelerate
        import soundfile
        import torch
        import torchaudio
        import transformers

        report["packages"] = {
            "ok": True,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "soundfile": soundfile.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        success = success and torch.cuda.is_available()
        sample = ROOT / "test" / "tonghua.mp3"
        waveform, sample_rate = torchaudio.load(str(sample))
        report["audio_smoke_test"] = {
            "ok": waveform.numel() > 0 and sample_rate > 0,
            "path": str(sample),
            "shape": list(waveform.shape),
            "sample_rate": sample_rate,
        }
        success = success and bool(report["audio_smoke_test"]["ok"])
    except Exception as exc:
        report["packages"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        success = False

    if model_path is not None:
        report["weights"] = check_weights(model_path)
        success = success and bool(report["weights"]["ok"])
    return report, success


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report, success = collect(args.model_path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for key, value in report.items():
            print(f"[{key}] {value}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

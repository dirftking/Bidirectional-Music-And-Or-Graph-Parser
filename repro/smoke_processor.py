"""Exercise local weights' processor, audio decoding, mel extraction, and time markers."""

from pathlib import Path

from src.audio_io import load_audio
from src.processing_moss_music import MossMusicProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "MOSS-Music-8B-Instruct"


def main() -> None:
    processor = MossMusicProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        enable_time_marker=True,
    )
    audio = load_audio(
        str(ROOT / "test" / "tonghua.mp3"),
        sample_rate=processor.config.mel_sr,
    )
    clip = audio[: processor.config.mel_sr * 5]
    inputs = processor(text="描述这段音乐", audios=[clip], return_tensors="pt")
    print(
        {
            "mel_sr": processor.config.mel_sr,
            "clip_samples": len(clip),
            "input_features": tuple(inputs["audio_data"].shape),
            "input_ids": tuple(inputs["input_ids"].shape),
            "audio_tokens": int(
                (inputs["input_ids"] == processor.audio_token_id).sum()
            ),
            "time_markers": processor.enable_time_marker,
        }
    )


if __name__ == "__main__":
    main()

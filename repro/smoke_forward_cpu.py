"""Run one end-to-end generated token on CPU to validate the full model path."""

from pathlib import Path

import torch

from src.audio_io import load_audio
from src.modeling_moss_music import MossMusicModel
from src.processing_moss_music import MossMusicProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "MOSS-Music-8B-Instruct"


def main() -> None:
    model = MossMusicModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = MossMusicProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        enable_time_marker=True,
    )
    audio = load_audio(
        str(ROOT / "test" / "tonghua.mp3"),
        sample_rate=processor.config.mel_sr,
    )[: processor.config.mel_sr]
    inputs = processor(text="描述音乐", audios=[audio], return_tensors="pt")
    inputs = inputs.to("cpu")
    inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated[0, input_length:]
    print(
        {
            "input_tokens": input_length,
            "audio_tokens": int(inputs["audio_input_mask"].sum()),
            "generated_tokens": int(new_tokens.numel()),
            "decoded": processor.decode(new_tokens, skip_special_tokens=True),
        }
    )


if __name__ == "__main__":
    main()

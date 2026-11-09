"""Deserialize all local model shards on CPU without running generation."""

from pathlib import Path

from src.modeling_moss_music import MossMusicModel


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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        {
            "model_type": model.config.model_type,
            "parameters": parameter_count,
            "parameters_billions": round(parameter_count / 1_000_000_000, 3),
            "dtype": str(model.dtype),
            "device": str(model.device),
            "memory_footprint_gib": round(model.get_memory_footprint() / 1024**3, 3),
            "deepstack_encoder_layers": list(
                model.config.audio_config.deepstack_encoder_layer_indexes
            ),
            "deepstack_inject_layers": model.config.deepstack_num_inject_layers,
        }
    )


if __name__ == "__main__":
    main()

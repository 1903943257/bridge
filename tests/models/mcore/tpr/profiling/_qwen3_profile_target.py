"""Shared real-Qwen3 target selection for TPR NPU profiling.

Profiling/capacity experiments intentionally exclude the old synthetic ~0.6B
proxy. Only real Qwen3-1.7B and Qwen3-4B checkpoints are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from transformers import AutoConfig


_ALLOWED_QWEN_PROFILE_SIZES = {
    "1.7B": (1_500_000_000, 1_900_000_000, "TPR_QWEN_1_7B_PATH"),
    "4B": (3_500_000_000, 4_500_000_000, "TPR_QWEN_4B_PATH"),
}


@dataclass(frozen=True, slots=True)
class Qwen3ProfileTarget:
    size: str
    path: Path
    hf_config: object
    min_parameters: int
    max_parameters: int

    @property
    def label(self) -> str:
        return f"Qwen3-{self.size}"

    def assert_model_scale(self, model) -> int:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if not self.min_parameters <= parameter_count <= self.max_parameters:
            raise AssertionError(
                f"{self.label} expected {self.min_parameters / 1e9:.1f}B to "
                f"{self.max_parameters / 1e9:.1f}B parameters, got "
                f"{parameter_count / 1e9:.3f}B; checkpoint={self.path}"
            )
        return parameter_count


def resolve_qwen3_profile_target() -> Qwen3ProfileTarget:
    """Resolve one supported real checkpoint; default to /workspace Qwen3-1.7B."""
    size = os.getenv("TPR_QWEN_PROFILE_SIZE", "1.7B")
    if size not in _ALLOWED_QWEN_PROFILE_SIZES:
        raise ValueError(
            f"unsupported TPR_QWEN_PROFILE_SIZE={size!r}; "
            f"profiling only supports {tuple(_ALLOWED_QWEN_PROFILE_SIZES)}"
        )
    minimum, maximum, size_path_env = _ALLOWED_QWEN_PROFILE_SIZES[size]
    default_path = f"/workspace/hf_models/Qwen3-{size}"
    model_path = Path(
        os.getenv(size_path_env, os.getenv("TPR_QWEN_MODEL_PATH", default_path))
    )
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing Qwen config: {config_path}")
    if not tuple(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"no safetensors weights found under {model_path}")
    hf_config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if hf_config.model_type != "qwen3":
        raise ValueError(
            f"expected dense Qwen3 checkpoint at {model_path}, got {hf_config.model_type!r}"
        )
    return Qwen3ProfileTarget(
        size=size,
        path=model_path,
        hf_config=hf_config,
        min_parameters=minimum,
        max_parameters=maximum,
    )

"""Minimal client for a MOSS-Music model served by SGLang."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class GenerationResult:
    text: str
    meta_info: dict[str, Any]


class SGLangClient:
    def __init__(self, base_url: str, timeout_seconds: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def model_info(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/model_info", timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("SGLang /model_info returned a non-object response")
        return payload

    def generate(
        self,
        *,
        audio_path: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: int | None = None,
    ) -> GenerationResult:
        sampling_params: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        if seed is not None:
            # SGLang's native SamplingParams name.  The OpenAI-compatible API
            # uses other conventions, but this client targets /generate.
            sampling_params["sampling_seed"] = seed
        response = self.session.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "audio_data": audio_path,
                "sampling_params": sampling_params,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("SGLang /generate returned a non-object response")
        if payload.get("object") == "error":
            raise RuntimeError(str(payload.get("message") or payload))
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError("SGLang response does not contain a string 'text' field")
        return GenerationResult(text=text, meta_info=payload.get("meta_info") or {})

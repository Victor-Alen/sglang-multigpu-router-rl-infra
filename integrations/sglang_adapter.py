from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class SGLangAdapterError(RuntimeError):
    pass


@dataclass
class SGLangWorkerAdapter:
    endpoint: str
    timeout_seconds: float = 120.0

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.endpoint.rstrip("/") + path,
            data=data,
            method=method,
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                return json.loads(raw) if raw else {"status": response.status}
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise SGLangAdapterError(f"SGLang request failed: {method} {path}: {exc}") from exc

    def health(self) -> bool:
        try:
            self._request("GET", "/health")
            return True
        except SGLangAdapterError:
            return False

    def model_info(self) -> dict:
        return self._request("GET", "/model_info")

    def update_weights_from_disk(
        self,
        model_path: str,
        policy_version: int,
        load_format: Optional[str] = None,
        abort_all_requests: bool = False,
        keep_pause: bool = False,
        flush_cache: bool = True,
    ) -> dict:
        """Invoke the API schema in SGLang commit ``3d1b54e6...``."""
        result = self._request(
            "POST",
            "/update_weights_from_disk",
            {
                "model_path": model_path,
                "load_format": load_format,
                "abort_all_requests": abort_all_requests,
                "weight_version": str(policy_version),
                "keep_pause": keep_pause,
                "flush_cache": flush_cache,
            },
        )
        if not result.get("success", False):
            raise SGLangAdapterError(f"weight update was rejected: {result}")
        model_info = self.model_info()
        if str(model_info.get("weight_version")) != str(policy_version):
            raise SGLangAdapterError(
                f"weight version probe returned {model_info.get('weight_version')}, "
                f"expected {policy_version}"
            )
        return {**result, "model_info": model_info}

    def fixed_input_probe(self, prompt: str, max_new_tokens: int = 1) -> dict:
        return self._request(
            "POST",
            "/v1/completions",
            {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "logprobs": 1,
            },
        )

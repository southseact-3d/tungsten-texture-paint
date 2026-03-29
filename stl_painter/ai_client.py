from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request


@dataclass(slots=True)
class AIClientConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0


class OpenAICompatibleClient:
    def __init__(self, config: AIClientConfig) -> None:
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + path

    def _image_part(self, image_path: str) -> dict[str, Any] | None:
        if not image_path:
            return None
        path = Path(image_path)
        if not path.exists():
            return None
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        }

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        image_path: str = "",
    ) -> dict[str, Any]:
        prepared_messages = []
        image_part = self._image_part(image_path)
        for message in messages:
            if image_part is not None and message["role"] == "user" and isinstance(message["content"], str):
                prepared_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message["content"]},
                            image_part,
                        ],
                    }
                )
                image_part = None
            else:
                prepared_messages.append(message)

        payload = {
            "model": self.config.model,
            "messages": prepared_messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._url("/chat/completions"),
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

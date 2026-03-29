from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .ai_client import AIClientConfig, OpenAICompatibleClient
from .ai_tools import AIToolContext, run_tool, tool_specs
from .interaction_state import ToolCallLogEntry


SYSTEM_PROMPT = """
You are the STL Texture Painter assistant.
You can inspect state, create sketch geometry on a face-aligned plane, bake sketches, paint faces, and export.
Always inspect mesh state first when the task depends on the current model.
Favor incremental edits and tool calls over long explanations.
Never assume a mesh or sketch plane exists without checking.
""".strip()


@dataclass(slots=True)
class AgentResponse:
    text: str
    tool_logs: list[ToolCallLogEntry]


class AIAgent:
    def __init__(self, context: AIToolContext) -> None:
        self.context = context

    def run(self, prompt: str) -> AgentResponse:
        settings = self.context.interaction_state.ai_settings
        client = OpenAICompatibleClient(
            AIClientConfig(
                base_url=settings.provider_base_url,
                api_key=settings.api_key,
                model=settings.model,
                timeout=settings.timeout,
            )
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_logs: list[ToolCallLogEntry] = []
        for _ in range(4):
            response = client.create_chat_completion(
                messages,
                tool_specs(),
                image_path=settings.image_path,
            )
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return AgentResponse(text=message.get("content", ""), tool_logs=tool_logs)
            messages.append(message)
            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = tool_call["function"].get("arguments", "{}")
                try:
                    result = run_tool(self.context, name, arguments)
                    log = ToolCallLogEntry(
                        name=name,
                        arguments=arguments,
                        ok=True,
                        summary=result,
                    )
                except Exception as exc:
                    result = json.dumps({"ok": False, "error": str(exc)})
                    log = ToolCallLogEntry(
                        name=name,
                        arguments=arguments,
                        ok=False,
                        summary=str(exc),
                    )
                tool_logs.append(log)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result,
                    }
                )
        return AgentResponse(
            text="The assistant hit the tool-call iteration limit before finishing.",
            tool_logs=tool_logs,
        )

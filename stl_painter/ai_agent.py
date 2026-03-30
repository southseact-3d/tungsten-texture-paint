from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ai_client import AIClientConfig, OpenAICompatibleClient
from .ai_tools import AIToolContext, run_tool, tool_specs
from .interaction_state import ToolCallLogEntry


SYSTEM_PROMPT = """
You are the STL Texture Painter assistant.
You have access to tools to manipulate a 3D mesh: get_mesh_state, set_mode, set_active_color, create_sketch_plane, add_rect_sketch, bake_sketch, paint_faces, export_project_snapshot, export_3mf_file, undo.

**IMPORTANT**: You MUST use the provided tools to complete tasks. NEVER explain what steps to take - instead, USE THE TOOLS directly to accomplish the task.
- To paint, first use set_active_color to pick a color, then use paint_faces to paint faces.
- To create textures from images, use create_sketch_plane on a face, add_rect_sketch to draw rectangles, then bake_sketch.
- ALWAYS inspect mesh state first with get_mesh_state to see current faces/vertices.

When given a reference image, you MUST try to recreate the texture by:
1. Call get_mesh_state to understand the mesh
2. Pick colors from the image and use set_active_color
3. Create sketch planes and draw rectangles/lines
4. Use bake_sketch to apply the sketch to faces
5. Use paint_faces to directly paint faces with colors

Start by inspecting the mesh state. Use tools to actually DO the work, not just describe it.
""".strip()

IMAGE_ONLY_PROMPT = (
    "Use the attached reference image to recreate the texture on the current model "
    "as closely as practical. Infer the important colors, shapes, and layout from "
    "the image even if no extra text instructions are provided."
)


@dataclass(slots=True)
class AgentResponse:
    text: str
    tool_logs: list[ToolCallLogEntry]


class AIAgent:
    def __init__(
        self,
        context: AIToolContext,
        on_status: Callable[[str], None] | None = None,
        check_stop: Callable[[], bool] | None = None,
        save_progress: Callable[[], None] | None = None,
    ) -> None:
        self.context = context
        self.on_status = on_status
        self.check_stop = check_stop
        self.save_progress = save_progress

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

        stop_flag_path = Path("agent_stop.flag")
        if stop_flag_path.exists():
            stop_flag_path.unlink()

        for i in range(50):
            if self.on_status:
                self.on_status(f"=== Iteration {i + 1}/50 ===")

            if stop_flag_path.exists():
                if self.on_status:
                    self.on_status("STOP FLAG DETECTED - stopping agent")
                stop_flag_path.unlink()
                break

            if self.check_stop and self.check_stop():
                if self.on_status:
                    self.on_status("Stop requested via callback")
                break

            response = client.create_chat_completion(
                messages,
                tool_specs(),
                image_path=settings.image_path,
            )
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                if self.on_status:
                    self.on_status("AI finished (no tool calls needed)")
                Path("agent_completed.flag").write_text(
                    f"Completed at iteration {i + 1}"
                )
                if self.save_progress:
                    try:
                        self.save_progress()
                    except Exception as e:
                        if self.on_status:
                            self.on_status(f"Final save failed: {e}")
                return AgentResponse(
                    text=message.get("content", ""), tool_logs=tool_logs
                )
            messages.append(message)

            if self.on_status:
                self.on_status(
                    f"AI returned {len(tool_calls)} tool call(s): {[tc['function']['name'] for tc in tool_calls]}"
                )

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = tool_call["function"].get("arguments", "{}")
                if self.on_status:
                    self.on_status(f"Running tool: {name}...")
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
                if self.on_status:
                    self.on_status(
                        f"Tool {name} completed: {'OK' if log.ok else 'FAIL'}"
                    )
                if self.save_progress:
                    try:
                        self.save_progress()
                    except Exception as e:
                        if self.on_status:
                            self.on_status(f"Save progress failed: {e}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result,
                    }
                )
        if self.on_status:
            self.on_status("Hit iteration limit")
        return AgentResponse(
            text="The assistant hit the tool-call iteration limit before finishing.",
            tool_logs=tool_logs,
        )


def build_user_prompt(prompt: str, image_path: str = "") -> str:
    normalized_prompt = prompt.strip()
    if normalized_prompt:
        return normalized_prompt
    if image_path.strip():
        return IMAGE_ONLY_PROMPT
    return ""

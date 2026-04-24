#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from stl_painter.ai_agent import AIAgent, build_user_prompt
from stl_painter.ai_client import AIClientConfig, OpenAICompatibleClient
from stl_painter.ai_tools import AIToolContext, run_tool, tool_specs
from stl_painter.commands import AppCommands
from stl_painter.importer import import_stl
from stl_painter.interaction_state import AISettings, InteractionState
from stl_painter.paint_tool import PaintTool
from stl_painter.sketch_tool import SketchTool


def create_context(stl_path: str) -> AIToolContext:
    mesh = import_stl(stl_path)
    state = InteractionState()
    paint_tool = PaintTool(mesh)
    sketch_tool = SketchTool()
    commands = AppCommands(mesh, paint_tool)
    return AIToolContext(
        mesh_model=mesh,
        interaction_state=state,
        commands=commands,
        paint_tool=paint_tool,
        sketch_tool=sketch_tool,
        capture_viewport=lambda: Path("snapshot.png"),
    )


def run_ai_agent(
    stl_path: str,
    base_url: str,
    api_key: str,
    model: str,
    image_path: str = "",
    prompt: str = "",
    timeout: float = 120.0,
) -> dict[str, Any]:
    context = create_context(stl_path)
    context.interaction_state.ai_settings = AISettings(
        provider_base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=240.0,
        image_path=image_path,
    )

    save_count = [0]

    def on_status(msg: str) -> None:
        print(f"[STATUS] {msg}", flush=True)

    def check_stop() -> bool:
        return Path("agent_stop.flag").exists()

    def save_progress() -> None:
        save_count[0] += 1
        save_path = f"ai_progress_{save_count[0]:03d}.tg3d"
        from stl_painter.project_io import save_project

        mesh = context.mesh_model
        if mesh:
            save_project(Path(save_path), mesh)
            print(f"[SAVE] Progress saved to {save_path}", flush=True)

    full_prompt = build_user_prompt(prompt, image_path)
    if not full_prompt:
        raise ValueError("Either prompt or image_path must be provided")

    agent = AIAgent(
        context, on_status=on_status, check_stop=check_stop, save_progress=save_progress
    )
    response = agent.run(full_prompt)

    # Final save
    final_path = "ai_final_output.tg3d"
    mesh = context.mesh_model
    if mesh:
        from stl_painter.project_io import save_project

        save_project(Path(final_path), mesh)
        print(f"[SAVE] Final output saved to {final_path}", flush=True)

    return {
        "response_text": response.text,
        "tool_logs": [
            {
                "name": log.name,
                "arguments": log.arguments,
                "ok": log.ok,
                "summary": log.summary,
            }
            for log in response.tool_logs
        ],
    }


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python test_ai_agent_cli.py <stl_path> <base_url> <api_key> <model> [image_path] [prompt]"
        )
        print("Example:")
        print(
            '  python test_ai_agent_cli.py "dart.stl" "https://opencode.ai/zen/v1" "sk-xxx" "model/name" "image.jpg" "paint blue"'
        )
        sys.exit(1)

    stl_path = sys.argv[1]
    base_url = sys.argv[2]
    api_key = sys.argv[3]
    model = sys.argv[4]
    image_path = sys.argv[5] if len(sys.argv) > 5 else ""
    prompt = sys.argv[6] if len(sys.argv) > 6 else ""

    print(f"Loading mesh: {stl_path}", flush=True)
    result = run_ai_agent(stl_path, base_url, api_key, model, image_path, prompt)

    print("\n=== RESPONSE ===")
    print(result["response_text"])
    print("\n=== TOOL LOGS ===")
    for log in result["tool_logs"]:
        status = "OK" if log["ok"] else "FAIL"
        print(f"{log['name']} | {status}")
        print(f"  Args: {log['arguments']}")
        print(
            f"  Result: {log['summary'][:200]}{'...' if len(log['summary']) > 200 else ''}"
        )

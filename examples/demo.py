#!/usr/bin/env python3
"""Minimal demo: run RS-Agent on a single query."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rs_agent.config import load_config, resolve_path  # noqa: E402
from rs_agent.controller.agent import RSAgent  # noqa: E402
from rs_agent.trace import export_trace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RS-Agent single-query demo")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument(
        "--responses-api",
        action="store_true",
        help="Use the Responses API for backends that do not serve Chat Completions",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Set an application User-Agent header for the LLM API client",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Save intermediate_steps as trace.json, trace.html, and trace.svg in this directory",
    )
    parser.add_argument(
        "--trace-layout",
        choices=("compact", "detailed"),
        default="compact",
        help="Trace diagram layout: compact for slides (default), detailed for full previews",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config.setdefault("agent", {})["mode"] = args.mode
    if args.responses_api:
        config["llm"]["use_responses_api"] = True
    if args.user_agent:
        config["llm"].setdefault("default_headers", {})["User-Agent"] = args.user_agent

    started_at = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    agent = RSAgent.from_config(config)
    image_path = args.image or str(resolve_path(config["paths"]["default_image"]))

    result = agent.run(question=args.question, image_path=image_path)
    duration = perf_counter() - start

    print("Predicted task type:", result.get("predicted_task_type"))
    print("Guidance:", result.get("guidance"))
    print("Output:", result.get("output"))
    if result.get("intermediate_steps"):
        print("Tools invoked:", [s[0].tool for s in result["intermediate_steps"]])
    if args.trace_dir is not None:
        paths = export_trace(
            result,
            question=args.question,
            image_path=image_path,
            output_dir=args.trace_dir,
            layout=args.trace_layout,
            metadata={
                "model": config["llm"]["model"],
                "mode": config.get("agent", {}).get("mode", "full"),
                "use_responses_api": config["llm"].get("use_responses_api"),
                "embedding_model": config["embedding"]["model"],
                "embedding_device": config["embedding"]["device"],
                "prompt": "local_structured_chat",
                "stub_tools": True,
                "started_at": started_at,
                "duration_seconds": round(duration, 3),
            },
        )
        print("Tool implementation: stub (planning only; no image processing)")
        for format_name, path in paths.items():
            print(f"Trace {format_name.upper()}: {path.resolve()}")


if __name__ == "__main__":
    main()

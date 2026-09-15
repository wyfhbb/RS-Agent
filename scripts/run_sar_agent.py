"""Run RS-Agent with configured real SAR detectors and complete execution evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_agent.config import load_config  # noqa: E402
from rs_agent.controller.agent import RSAgent  # noqa: E402
from rs_agent.toolkit.sar_detection import SARDetectorBackend, sha256  # noqa: E402
from rs_agent.trace import export_trace  # noqa: E402


class LLMRecorder(BaseCallbackHandler):
    """Persist public prompt/response messages and usage, without client credentials."""

    def __init__(self, path: Path):
        self.path = path
        self.calls = 0
        self.usage: list[dict[str, Any]] = []

    def _append(self, kind: str, **payload: Any) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "time": datetime.now(timezone.utc).isoformat(),
                                     **payload}, ensure_ascii=False, default=str) + "\n")

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.calls += 1
        self._append("request", run_id=str(run_id), messages=[
            [{"type": message.type, "content": message.content} for message in batch]
            for batch in messages])

    def on_llm_end(self, response, *, run_id, **kwargs):
        generations = []
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                row = {"text": generation.text}
                if message is not None:
                    row["content"] = message.content
                    row["response_metadata"] = message.response_metadata
                    row["usage_metadata"] = message.usage_metadata
                    if message.usage_metadata:
                        self.usage.append(message.usage_metadata)
                generations.append(row)
        self._append("response", run_id=str(run_id), generations=generations)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._append("error", run_id=str(run_id), error_type=type(error).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--detectors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("full", "baseline", "task_inference_only",
                                           "solution_retrieval_only"), default="full")
    parser.add_argument("--responses-api", action="store_true")
    parser.add_argument("--user-agent")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config["agent"]["mode"] = args.mode
    # Leave .env intact. This local override avoids occupying the training GPU.
    config["embedding"]["device"] = "cpu"
    if args.responses_api:
        config["llm"]["use_responses_api"] = True
    if args.user_agent:
        config["llm"].setdefault("default_headers", {})["User-Agent"] = args.user_agent
    backend = SARDetectorBackend.from_json(args.detectors, output / "detector_calls")
    recorder = LLMRecorder(output / "llm_messages.jsonl")
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provider": config["llm"].get("provider"),
        "model": config["llm"]["model"],
        "endpoint_host": urlparse(config["llm"].get("api_base", "")).hostname,
        "mode": args.mode,
        "use_responses_api": config["llm"].get("use_responses_api", False),
        "embedding_model": config["embedding"]["model"],
        "embedding_device": "cpu",
        "tool_backend": "real_cli",
        "stub_tools": False,
        "ground_truth_provided_to_agent": False,
        "real_forward_count_definition": (
            "Detector tool calls producing new predictions; internal warmup forwards excluded"
        ),
        "image_sha256": sha256(args.image),
        "detectors_config": str(args.detectors.resolve()),
        "detectors_config_sha256": sha256(args.detectors),
        "registered_tools": [tool.name for tool in backend.tools()],
        "llm_timeout_seconds": 120,
        "llm_max_retries": 0,
        "llm_max_output_tokens": 2048,
    }
    source_root = Path(__file__).resolve().parents[1]
    source_files = ["scripts/run_sar_agent.py", "rs_agent/toolkit/sar_detection.py",
                    "rs_agent/controller/agent.py", "rs_agent/controller/prompts.py",
                    "rs_agent/solution_space/retriever.py", "rs_agent/trace.py"]
    metadata["source_snapshot_sha256"] = {}
    for relative in source_files:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
        metadata["source_snapshot_sha256"][relative] = sha256(destination)
    metadata["source_snapshot_method"] = "captured_before_agent_execution"
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.perf_counter()
    try:
        agent = RSAgent.from_config(config, tools=backend.tools())
        # Configure the SDK at construction; changing these attributes after
        # construction does not update the already-created HTTP client.
        llm_config = config["llm"]
        agent.llm = ChatOpenAI(
            model=llm_config["model"], temperature=llm_config.get("temperature", 0),
            api_key=llm_config["api_key"], base_url=llm_config.get("api_base"),
            use_responses_api=llm_config.get("use_responses_api", False),
            default_headers=llm_config.get("default_headers"),
            timeout=120, max_retries=0, max_tokens=2048, callbacks=[recorder],
        )
        result = agent.run(args.question, str(args.image.resolve()))
        metadata.update(duration_seconds=time.perf_counter() - start,
                        llm_request_count=recorder.calls, llm_usage=recorder.usage,
                        tool_call_count=len(backend.events),
                        real_forward_count=sum(e["execution_kind"] == "real_forward"
                                               for e in backend.events),
                        cache_hit_count=sum(e["execution_kind"] == "cache_hit"
                                            for e in backend.events))
        success = bool(metadata["real_forward_count"] and result.get("output") and
                       "Agent stopped" not in result["output"] and
                       all(e.get("success") for e in backend.events))
        metadata["success"] = success
        paths = export_trace(result, question=args.question, image_path=str(args.image.resolve()),
                             output_dir=output, metadata=metadata, layout="compact")
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"success": success, "predicted_task_type": result["predicted_task_type"],
                          "output": result["output"], "metadata": metadata,
                          "artifacts": {key: str(value) for key, value in paths.items()}},
                         indent=2, ensure_ascii=False))
        if not success:
            raise SystemExit(2)
    except (Exception, KeyboardInterrupt) as exc:
        metadata.update(success=False, duration_seconds=time.perf_counter() - start,
                        error_type=type(exc).__name__, llm_request_count=recorder.calls,
                        llm_usage=recorder.usage, tool_call_count=len(backend.events))
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()

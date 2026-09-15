"""Real SAR tools backed by isolated, explicitly configured inference CLIs.

The JSON configuration is trusted application configuration, never model input.
The language model supplies only an existing image path. No shell is invoked.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.tools import Tool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    display_name: str
    command: tuple[str, ...]
    device: str = "cpu"
    timeout_seconds: int = 600


class SARDetectorBackend:
    """Run each image/model once, preserving actual subprocess evidence on disk."""

    def __init__(self, specs: list[DetectorSpec], output_dir: Path, default_model: str,
                 image_ids: dict[str, int] | None = None):
        self.specs = {spec.name: spec for spec in specs}
        if default_model not in self.specs:
            raise ValueError("The configured default SAR model is not registered")
        self.default_model = default_model
        self.image_ids = image_ids or {}
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self._cache: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    @classmethod
    def from_json(cls, config_path: Path, output_dir: Path) -> SARDetectorBackend:
        config = json.loads(config_path.read_text())
        specs = [DetectorSpec(name=name, display_name=value["display_name"],
                              command=tuple(value["command"]),
                              device=value.get("device", "cpu"),
                              timeout_seconds=value.get("timeout_seconds", 600))
                 for name, value in config["models"].items()]
        return cls(specs, output_dir, config["default_model"], config.get("image_ids"))

    def _record(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        with (self.output_dir / "tool_calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def detect(self, image_path: str, model: str | None = None) -> str:
        model = model or self.default_model
        start = time.perf_counter()
        event: dict[str, Any] = {
            "call_id": uuid4().hex,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "tool_input": image_path,
            "execution_kind": "failed_before_forward",
        }
        try:
            path = Path(image_path.strip().strip('"')).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError("The image path must identify a file")
            spec = self.specs[model]
            input_hash = sha256(path)
            image_id = int(self.image_ids.get(str(path), -1))
            event.update(image=str(path), image_sha256=input_hash, image_id=image_id)
            # Identical pixels in different dataset records are not the same ID.
            key = (model, str(path), input_hash, image_id)
            if key in self._cache:
                observation = {**self._cache[key], "execution_kind": "cache_hit",
                               "cached_from_call_id": self._cache[key]["call_id"]}
                event.update(execution_kind="cache_hit", success=True,
                             cached_from_call_id=observation["cached_from_call_id"])
            else:
                call_name = f"{len(self.events) + 1:02d}_{model}_{event['call_id']}"
                call_dir = self.output_dir / call_name
                call_dir.mkdir()
                command = [*spec.command, "--image", str(path), "--output", str(call_dir),
                           "--device", spec.device, "--image-id", str(image_id)]
                event.update(command=command, device=spec.device, output_dir=str(call_dir))
                # Vision subprocesses do not need LLM credentials.
                child_env = {key: value for key, value in os.environ.items()
                             if not any(part in key.upper() for part in
                                        ("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "SECRET"))}
                with (call_dir / "stdout.log").open("w") as out, \
                        (call_dir / "stderr.log").open("w") as err:
                    event["execution_kind"] = "detector_call_failed"
                    process = subprocess.Popen(command, stdout=out, stderr=err, env=child_env,
                                               start_new_session=True)
                    event["pid"] = process.pid
                    try:
                        returncode = process.wait(timeout=spec.timeout_seconds)
                    except BaseException:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                event["returncode"] = returncode
                if returncode:
                    raise RuntimeError(f"Detector exited {returncode}; see {call_dir}")
                event["execution_kind"] = "detector_result_unverified"
                result_path = call_dir / "result.json"
                prediction = json.loads(result_path.read_text())
                if prediction.get("actual_forward") is not True or prediction.get("cache_hit"):
                    raise ValueError("New detector subprocess did not report a real forward")
                if Path(prediction["image"]).resolve() != path:
                    raise ValueError("Detector returned a different input image")
                if prediction.get("image_id") != image_id:
                    raise ValueError("Detector returned a different image ID")
                if prediction.get("display_score") != 0.30:
                    raise ValueError("Detector visualization threshold differs from fixed 0.30")
                if not isinstance(prediction.get("detections"), list):
                    raise ValueError("Detector result has no detections list")
                if not prediction.get("checkpoint"):
                    raise ValueError("Detector result has no checkpoint provenance")
                for field in ("coco_json", "visualization"):
                    if not Path(prediction[field]).is_file():
                        raise ValueError(f"Detector did not produce {field}")
                shown = [row for row in prediction["detections"] if row["score"] >= 0.30]
                counts: dict[str, int] = {}
                for row in shown:
                    category = row.get("class_name", str(row["category_id"]))
                    counts[category] = counts.get(category, 0) + 1
                observation = {
                    "status": "success",
                    "execution_kind": "real_forward",
                    "call_id": event["call_id"],
                    "model": prediction.get("model", spec.display_name),
                    "checkpoint": prediction["checkpoint"],
                    "checkpoint_sha256": prediction.get("checkpoint_sha256"),
                    "image": str(path),
                    "image_id": prediction.get("image_id"),
                    "identity_scope": (
                        "dataset_image_id" if image_id >= 0 else "standalone_unindexed"
                    ),
                    "visualization_score_threshold": 0.30,
                    "counts_at_visualization_threshold": counts,
                    "raw_prediction_count": len(prediction["detections"]),
                    "detections_at_visualization_threshold": shown,
                    "prediction_json": prediction["coco_json"],
                    "visualization": prediction["visualization"],
                    "result_json": str(result_path),
                    "pure_forward_seconds": prediction.get("pure_forward_seconds"),
                    "cold_start_seconds": prediction.get("cold_start_seconds"),
                }
                self._cache[key] = observation
                event.update(execution_kind="real_forward", success=True,
                             result_json=str(result_path), result_sha256=sha256(result_path),
                             prediction_sha256=sha256(Path(prediction["coco_json"])),
                             raw_prediction_count=len(prediction["detections"]))
            event["observation"] = observation
        except Exception as exc:
            observation = {"status": "error", "execution_kind": event["execution_kind"],
                           "model": model, "error_type": type(exc).__name__, "error": str(exc)}
            event.update(success=False, observation=observation)
        event["elapsed_seconds"] = time.perf_counter() - start
        self._record(event)
        return json.dumps(observation, ensure_ascii=False, allow_nan=False)

    def tools(self) -> list[Tool]:
        """Register real tools only; the historical planning stubs stay opt-in."""
        result = [Tool(
            name="sar_detection", func=self.detect,
            description=("Detect and count ship, aircraft, car, tank, bridge and harbor in a "
                         "SAR image using a real trained horizontal-box detector. "
                         "Input is the image file path. Returns prediction JSON and an "
                         "annotated image with counts at score >= 0.30."))]
        for name, spec in self.specs.items():
            result.append(Tool(
                name=f"sar_detection_{name}",
                func=lambda image_path, selected=name: self.detect(image_path, selected),
                description=(f"Detect objects in SAR images with {spec.display_name}. "
                             "Input is an image path. Use when this detector is specifically "
                             "needed for a requested comparison. Returns JSON and an image.")))
        return result

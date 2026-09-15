#!/usr/bin/env python3
"""Run a real COCO horizontal-box baseline, independently of RS-Agent's stub tools.

Usage: uv run --locked --extra vision scripts/probe_horizontal_detection.py

This tests torchvision Faster R-CNN, not the unspecified MMDetection model from
the project README. COCO weights are not remote-sensing task weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL_URL = (
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
)
MODEL_DOCS = (
    "https://docs.pytorch.org/vision/stable/models/generated/"
    "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html"
)


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(url: str, path: Path, expected_hash_prefix: str | None = None) -> None:
    """Download with verified TLS and publish the file only after completion."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        command = [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "900",
            "--output",
            str(temporary),
        ]
        system_ca = Path("/etc/ssl/certs/ca-certificates.crt")
        if system_ca.exists():
            command.extend(["--cacert", str(system_ca)])
        command.append(url)
        print(f"Downloading {url}", flush=True)
        subprocess.run(command, check=True)
        if expected_hash_prefix and not sha256(temporary).startswith(expected_hash_prefix):
            raise ValueError(f"Weight checksum mismatch: {temporary}")
        temporary.replace(path)
    if expected_hash_prefix and not sha256(path).startswith(expected_hash_prefix):
        raise ValueError(f"Weight checksum mismatch: {path}")


def annotate(image: Image.Image, detections: list[dict], path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, min(canvas.size) // 35))
    for index, item in enumerate(detections):
        color = ("#e63946", "#00a896", "#ffad00", "#4361ee")[index % 4]
        box = item["box_xyxy"]
        draw.rectangle(box, outline=color, width=3)
        text = f"{item['class_name']} {item['confidence']:.3f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        x = max(0, min(box[0], canvas.width - (right - left)))
        y = max(0, box[1] - (bottom - top) - 5)
        draw.rectangle((x, y, x + right - left + 4, y + bottom - top + 4), fill=color)
        draw.text((x + 2, y - top + 2), text, fill="white", font=font)
    if not detections:
        draw.rectangle((0, 0, canvas.width, 27), fill="black")
        draw.text((5, 5), "No detections above the display threshold", fill="white", font=font)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "examples/sample.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/model-smoke/horizontal")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "outputs/model-smoke/weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    weight_path = args.weights_dir.resolve() / Path(weight.url).name
    download(weight.url, weight_path, expected_hash_prefix="dd69338a")
    control_path = output_dir / "inputs/bus.jpg"
    download(CONTROL_URL, control_path)

    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(0)
    start = time.perf_counter()
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - start
    preprocess = weight.transforms()
    categories = weight.meta["categories"]
    inputs = [
        ("project_sample", args.image.resolve(), "Local project example, no ground truth supplied"),
        ("bus_control", control_path, CONTROL_URL),
    ]
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": "torchvision.fasterrcnn_resnet50_fpn_v2",
        "weights_enum": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "training_dataset": "COCO",
        "role": "Alternative horizontal-box baseline; not original RS-Agent/MMDetection model",
        "remote_sensing_finetuned": False,
        "agent_tool_integrated": False,
        "model_docs": MODEL_DOCS,
        "weights_url": weight.url,
        "weights_path": str(weight_path),
        "weights_sha256": sha256(weight_path),
        "weights_sha256_prefix_verified": "dd69338a",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_load_seconds": model_load_seconds,
        "parameters": {
            "display_confidence": args.confidence,
            "model_score_threshold": 0.05,
            "nms_iou_threshold": 0.5,
            "max_detections": 100,
            "min_image_side": 800,
            "max_image_side": 1333,
            "precision": "float32",
        },
        "timing_scope": (
            "One warmup forward per image, then one synchronized forward; includes internal "
            "resize/normalization and NMS, excludes file reading and host-to-device copy. "
            "Smoke timing under shared GPU load, not a throughput benchmark."
        ),
        "limitations": [
            "Successful inference does not establish remote-sensing accuracy.",
            "COCO everyday object classes do not cover a remote-sensing taxonomy.",
            "No labeled remote-sensing validation set; no mAP, recall, or accuracy measured.",
            "The bus image is a visible-content control, not a labeled benchmark.",
        ],
        "results": [],
    }
    for name, input_path, source in inputs:
        with Image.open(input_path) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
        tensor = preprocess(rgb).to(device)
        with torch.inference_mode():
            model([tensor])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            prediction = model([tensor])[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        prediction = {key: value.detach().cpu() for key, value in prediction.items()}
        if not all(torch.isfinite(prediction[key]).all() for key in ("boxes", "scores")):
            raise ValueError(f"Non-finite model output for {input_path}")
        detections = [
            {
                "class_id": int(label),
                "class_name": categories[int(label)],
                "confidence": float(score),
                "box_xyxy": [float(value) for value in box],
            }
            for box, label, score in zip(
                prediction["boxes"], prediction["labels"], prediction["scores"], strict=True
            )
        ]
        kept = [item for item in detections if item["confidence"] >= args.confidence]
        annotated_path = output_dir / f"{name}_annotated.png"
        annotate(rgb, kept, annotated_path)
        result = {
            "name": name,
            "input_path": str(input_path),
            "input_source": source,
            "input_sha256": sha256(input_path),
            "input_size_wh": list(rgb.size),
            "inference_seconds": elapsed,
            "peak_gpu_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "raw_detection_count": len(detections),
            "displayed_detection_count": len(kept),
            "counts_by_class": dict(Counter(item["class_name"] for item in kept)),
            "annotated_path": str(annotated_path),
            "detections": detections,
        }
        if name == "bus_control":
            result["control_expected_classes"] = ["bus", "person"]
            result["control_classes_detected"] = all(
                cls in result["counts_by_class"] for cls in result["control_expected_classes"]
            )
        report["results"].append(result)
        (output_dir / f"{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{name}: {len(kept)} detections >= {args.confidence}; "
            f"{result['counts_by_class']}; {elapsed:.3f} s",
            flush=True,
        )

    report["runtime_status"] = "passed"
    report["control_status"] = (
        "passed" if report["results"][-1]["control_classes_detected"] else "failed"
    )
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_path}", flush=True)
    if report["control_status"] != "passed":
        raise SystemExit("Model ran, but expected control classes were not detected")


if __name__ == "__main__":
    main()

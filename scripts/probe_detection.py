#!/usr/bin/env python3
"""Run the README's YOLOv8x-OBB model on real aerial images, without an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-obb.pt"
WEIGHTS_SHA256 = "f74198806bf94ab9f067b1c69f0ae4dadcadabe604f670af818bed8fa2a357b0"
DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8.zip"
DATA_SHA256 = "6f901955660fc5c613a0389f44240db359f07741effb48afc99d17155ee3e7ec"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(url: str, destination: Path, expected_hash: str) -> None:
    """Pin the downloaded artifact, including when reusing an existing file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "3",
                "--connect-timeout",
                "20",
                "--max-time",
                "600",
                url,
                "-o",
                str(temporary),
            ],
            check=True,
        )
        if sha256(temporary) != expected_hash:
            raise ValueError(f"Downloaded artifact checksum mismatch: {url}")
        temporary.replace(destination)
    if sha256(destination) != expected_hash:
        raise ValueError(f"Existing artifact checksum mismatch: {destination}")


def draw_result(result, detections: list[dict], destination: Path) -> None:
    """Save numbered rotated boxes and a readable legend without font downloads."""
    import cv2
    import numpy as np

    source = result.orig_img
    height, width = source.shape[:2]
    rows = max(height, 55 + 22 * len(detections))
    canvas = np.full((rows, width + 310, 3), 245, dtype=np.uint8)
    canvas[:height, :width] = source
    cv2.putText(
        canvas,
        "YOLOv8x-OBB / DOTA",
        (width + 12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    for index, detection in enumerate(detections, 1):
        polygon = np.asarray(detection["polygon_xy"], dtype=np.int32)
        color = (60, 160, 30) if detection["class_id"] % 2 else (220, 90, 20)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        x, y = polygon.mean(axis=0).astype(int)
        cv2.putText(
            canvas,
            str(index),
            (max(0, x), max(12, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(index),
            (max(0, x), max(12, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        label = f"{index:02d} {detection['class_name']} {detection['confidence']:.3f}"
        cv2.putText(
            canvas,
            label,
            (width + 12, 48 + (index - 1) * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model-smoke/detection")
    parser.add_argument("--source", type=Path, action="append", help="Repeat for custom images")
    parser.add_argument("--device", default="0", help="CUDA device index, or cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.imgsz < 1:
        parser.error("--imgsz must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / "ultralytics-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"

    import numpy as np
    import torch
    from ultralytics import YOLO

    artifacts = ROOT / "outputs/model-smoke"
    weights = artifacts / "weights/yolov8x-obb.pt"
    download(WEIGHTS_URL, weights, WEIGHTS_SHA256)
    sources = args.source
    dataset = None
    if not sources:
        archive_path = artifacts / "data/dota8.zip"
        download(DATA_URL, archive_path, DATA_SHA256)
        extraction_root = archive_path.parent.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not (extraction_root / name).resolve().is_relative_to(extraction_root):
                    raise ValueError(f"Unsafe archive member: {name}")
            archive.extractall(extraction_root)
        dataset = {"url": DATA_URL, "sha256": DATA_SHA256, "split": "val (4 images)"}
        sources = [ROOT / "examples/sample.png"] + sorted(
            (extraction_root / "dota8/images/val").glob("*.jpg")
        )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    model = YOLO(str(weights), task="obb")
    options = dict(
        device=args.device, imgsz=args.imgsz, conf=args.confidence, verbose=False, save=False
    )
    # Warm up with the same image size; subsequent timings exclude loading/downloads.
    model.predict(str(sources[0]), **options)
    cuda = args.device != "cpu"
    torch_device = torch.device(f"cuda:{args.device}") if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.reset_peak_memory_stats(torch_device)
    rows = []
    for index, source in enumerate(sources):
        if cuda:
            torch.cuda.synchronize(torch_device)
        started = time.perf_counter()
        result = model.predict(str(source), **options)[0]
        if cuda:
            torch.cuda.synchronize(torch_device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if result.obb is None:
            raise RuntimeError("OBB model did not return oriented boxes")
        polygons = result.obb.xyxyxyxy.cpu().numpy()
        scores = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)
        assert np.isfinite(polygons).all() and np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()
        detections = [
            {
                "class_id": int(label),
                "class_name": result.names[int(label)],
                "confidence": float(score),
                "polygon_xy": polygon.tolist(),
            }
            for label, score, polygon in zip(classes, scores, polygons)
        ]
        counts = dict(Counter(d["class_name"] for d in detections))
        annotation = output / f"{index:02d}_{source.stem}_annotated.png"
        draw_result(result, detections, annotation)
        label_file = (
            source.parent.parent.parent / "labels" / source.parent.name / (source.stem + ".txt")
        )
        reference_counts = None
        if dataset and label_file.is_file():
            reference_counts = dict(
                Counter(
                    result.names[int(line.split()[0])]
                    for line in label_file.read_text().splitlines()
                    if line.strip()
                )
            )
        rows.append(
            {
                "image": str(source.resolve()),
                "image_sha256": sha256(source),
                "image_shape_hw": list(result.orig_shape),
                "elapsed_ms_after_warmup": elapsed_ms,
                "model_speed_ms": result.speed,
                "class_counts": counts,
                "reference_class_counts": reference_counts,
                "detections": detections,
                "annotated_image": str(annotation),
            }
        )
        print(f"{source.name}: {counts}, {elapsed_ms:.1f} ms", flush=True)

    report = {
        "status": "passed",
        "model": "YOLOv8x-OBB (official DOTA checkpoint)",
        "scope": "README optical_detection model family; exact paper checkpoint unknown.",
        "weights": {"url": WEIGHTS_URL, "path": str(weights), "sha256": WEIGHTS_SHA256},
        "dataset": dataset,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in [
                "torch",
                "torchvision",
                "ultralytics-opencv-headless",
                "opencv-python-headless",
                "numpy",
            ]
        },
        "device": str(torch_device),
        "gpu": torch.cuda.get_device_name(torch_device) if cuda else None,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(torch_device) / 2**20
        if cuda
        else None,
        "inference_parameters": {"imgsz": args.imgsz, "confidence": args.confidence},
        "limitations": [
            "Small inference smoke test, not a held-out accuracy benchmark.",
            "Predicted counts and reference counts are not an object-level matching metric.",
            "Timings are single warm runs and can be affected by concurrent GPU workloads.",
            "The DOTA classes identify planes, not aircraft subtypes; SAR is not tested.",
            "This probe does not replace the Agent's stub tools.",
        ],
        "results": rows,
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {output / 'results.json'}")


if __name__ == "__main__":
    main()

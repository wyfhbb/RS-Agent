"""Run public EuroSAT classifiers; these are substitutes, not RS-Agent's custom weights.

Examples:
    uv run --locked --extra vision scripts/probe_classification.py
    uv run --locked --extra vision scripts/probe_classification.py --image satellite.jpg
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import ssl
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

MODELS = {
    "resnet18": {
        "repo": "cm93/resnet18-eurosat",
        "revision": "f416c040bab2a4ce61704eb8c8e07016722446f1",
        "weights_sha256": "73f574447eeaff3f3c179bcf106b94bdc89ef7b3def2b64873a1494b4c3d15a9",
        "architecture": "resnet18",
    },
    "vit": {
        "repo": "cm93/vit-base-patch16-224-eurosat",
        "revision": "0aafda0d3c84c3f44e9c2e9ee111d692444dc55a",
        "weights_sha256": "8e2925f0b872b61c0115427bd5283a977c26d66f5d6d526add3edde15a169922",
        "architecture": "vit_base_patch16_224",
    },
}
DATASET = {
    "repo": "cm93/eurosat",
    "revision": "00095a028f821f0d1c8330808063718b6a71649f",
    "file": "data/test-00000-of-00001.parquet",
    "sha256": "89eabd58d97835df25ceffc32036a1192c87f199aa1be0de33090c787e88c9e8",
    "split": "test",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, destination: Path, expected_hash: str | None = None) -> Path:
    """Cache a pinned public artifact and verify its hash, keeping TLS enabled."""
    if destination.exists() and (expected_hash is None or sha256(destination) == expected_hash):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    ca_file = os.environ.get("SSL_CERT_FILE")
    if ca_file is None and Path("/etc/ssl/certs/ca-certificates.crt").exists():
        ca_file = "/etc/ssl/certs/ca-certificates.crt"
    context = ssl.create_default_context(cafile=ca_file)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    try:
        # Honor an existing curl network setup when present, with a standard-library fallback.
        # The download query also avoids stale resolve redirects in intermediary caches.
        download_url = url + "?download=true"
        if shutil.which("curl"):
            command = [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "2",
                "--connect-timeout",
                "15",
                "--max-time",
                "600",
            ]
            if ca_file:
                command.extend(["--cacert", ca_file])
            subprocess.run([*command, download_url, "--output", str(temporary)], check=True)
        else:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(
                        download_url,
                        timeout=45,
                        context=context,
                    ) as response:
                        with temporary.open("wb") as stream:
                            while chunk := response.read(1024 * 1024):
                                stream.write(chunk)
                    break
                except (TimeoutError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    print(f"Retrying download ({attempt + 2}/3): {destination.name}", flush=True)
        if expected_hash is not None and sha256(temporary) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_samples(output: Path, images: list[Path], per_class: int) -> tuple[list[dict], dict]:
    from PIL import Image, ImageOps

    samples = []
    if images:
        for path in images:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            samples.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "expected_label": None,
                    "image": image,
                }
            )
        return samples, {"kind": "user_images", "labels_available": False}

    import pyarrow.parquet as pq

    url = (
        f"https://huggingface.co/datasets/{DATASET['repo']}/resolve/"
        f"{DATASET['revision']}/{DATASET['file']}"
    )
    path = download(url, output / "downloads" / "eurosat-test.parquet", DATASET["sha256"])
    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    labels = metadata["info"]["features"]["label"]["names"]
    selected = {name: 0 for name in labels}
    image_dir = output / "inputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(table.to_pylist()):
        label = labels[row["label"]]
        if selected[label] >= per_class:
            continue
        image_bytes = row["image"]["bytes"]
        image_path = image_dir / f"test-{index:04d}-{label}.jpg"
        image_path.write_bytes(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
        samples.append(
            {
                "path": str(image_path.resolve()),
                "sha256": sha256(image_path),
                "dataset_row_index": index,
                "expected_label": label,
                "image": image,
            }
        )
        selected[label] += 1
        if all(count >= per_class for count in selected.values()):
            break
    return samples, {
        **DATASET,
        "source_url": url,
        "selection": "first N rows of each class, before viewing model predictions",
        "requested_samples_per_class": per_class,
        "selected_per_class": selected,
        "label_names": labels,
    }


def run_model(name: str, samples: list[dict], output: Path, device: str) -> dict:
    import timm
    import torch
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_data_config

    specification = MODELS[name]
    model_dir = output / "downloads" / name
    base_url = f"https://huggingface.co/{specification['repo']}/resolve/{specification['revision']}"
    config_path = download(f"{base_url}/config.json", model_dir / "config.json")
    weights_path = download(
        f"{base_url}/model.safetensors",
        model_dir / "model.safetensors",
        specification["weights_sha256"],
    )
    config = json.loads(config_path.read_text())
    if config["architecture"] != specification["architecture"]:
        raise ValueError("Unexpected model architecture")
    labels = config.get("label_names", config["pretrained_cfg"].get("label_names"))
    if labels is None or len(labels) != config["num_classes"]:
        raise ValueError("Missing or inconsistent model labels")
    started = time.perf_counter()
    # No remote model code or pickle checkpoint is executed.
    model = timm.create_model(
        specification["architecture"],
        pretrained=False,
        num_classes=config["num_classes"],
    )
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    model = model.eval().to(device)
    data_config = resolve_data_config(config["pretrained_cfg"], model=model)
    transform = create_transform(**data_config, is_training=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    transformed = [transform(sample["image"]).unsqueeze(0).to(device) for sample in samples]
    predictions = []
    with torch.inference_mode():
        # Warmup is excluded from the batch-one forward-pass timings.
        model(transformed[0])
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        for sample, tensor in zip(samples, transformed, strict=True):
            started = time.perf_counter()
            logits = model(tensor)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if logits.shape != (1, len(labels)) or not torch.isfinite(logits).all():
                raise ValueError("Invalid classification logits")
            probabilities = logits.softmax(dim=-1)[0].cpu()
            values, indices = probabilities.topk(min(3, len(labels)))
            top_k = [
                {"label": labels[index], "score": float(score)}
                for score, index in zip(values.tolist(), indices.tolist(), strict=True)
            ]
            expected = sample["expected_label"]
            predictions.append(
                {
                    **{key: value for key, value in sample.items() if key != "image"},
                    "predictions": top_k,
                    "top1_correct": None if expected is None else top_k[0]["label"] == expected,
                    "forward_ms": elapsed_ms,
                }
            )
    labeled = [row for row in predictions if row["top1_correct"] is not None]
    result = {
        "name": name,
        "status": "passed",
        **specification,
        "source_url": f"https://huggingface.co/{specification['repo']}",
        "author_repo": "https://github.com/chathumal93/EuroSat-RGB-Classifiers",
        "relationship_to_rs_agent": "public EuroSAT substitute; not the original custom classifier",
        "weights_path": str(weights_path.resolve()),
        "config_sha256": sha256(config_path),
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "label_names": labels,
        "preprocessing": data_config,
        "model_load_seconds_excluding_download": load_seconds,
        "sample_count": len(samples),
        "labeled_count": len(labeled),
        "top1_matches": sum(row["top1_correct"] for row in labeled),
        "median_forward_ms_batch1_excluding_preprocessing": statistics.median(
            row["forward_ms"] for row in predictions
        ),
        "predictions": predictions,
    }
    del model, transformed
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def save_preview(report: dict, output: Path) -> None:
    """Plot the real samples and predictions as a compact, inspectable result sheet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    models = [model for model in report["models"] if model["status"] == "passed"]
    if not models:
        return
    rows = models[0]["predictions"]
    columns = min(5, len(rows))
    figure, axes = plt.subplots(
        (len(rows) + columns - 1) // columns,
        columns,
        figsize=(columns * 3.2, ((len(rows) + columns - 1) // columns) * 4.0),
        squeeze=False,
        layout="constrained",
    )
    for axis in axes.flat:
        axis.axis("off")
    for index, row in enumerate(rows):
        axis = list(axes.flat)[index]
        with Image.open(row["path"]) as image:
            axis.imshow(image)
        lines = [f"Label: {row['expected_label'] or 'unknown'}"]
        for model in models:
            prediction = model["predictions"][index]["predictions"][0]
            lines.append(f"{model['name']}: {prediction['label']} ({prediction['score']:.1%})")
        axis.set_title("\n".join(lines), fontsize=9)
    figure.suptitle("EuroSAT public substitutes: inference smoke test (not a benchmark)")
    figure.savefig(output / "classification-preview.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/model-smoke/classification"))
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")
    import torch

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    samples, dataset = prepare_samples(args.output, args.image, args.samples_per_class)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "real pretrained inference smoke test; no training and no LLM calls",
        "limitations": [
            "These EuroSAT models are substitutes, not RS-Agent's RSSDIVCS or aircraft weights.",
            "The small selected test sample is not an accuracy benchmark.",
            "Land-cover classes do not identify aircraft models or classify SAR imagery.",
            "Reported softmax scores are not calibrated confidence estimates.",
            "RS-Agent's toolkit stubs are unchanged; this runs models independently.",
        ],
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "torchvision", "timm", "pillow", "safetensors", "pyarrow"]
            },
            "cuda_runtime": torch.version.cuda,
            "device": device,
            "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        },
        "dataset": dataset,
        "models": [],
    }
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        try:
            result = run_model(name, samples, args.output, device)
            print(
                f"{name}: {result['top1_matches']}/{result['labeled_count']} labeled matches; "
                f"{result['sample_count']} images on {device}",
                flush=True,
            )
        except Exception as error:
            result = {"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            print(f"{name}: {result['error']}", flush=True)
        report["models"].append(result)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    save_preview(report, args.output)
    print(f"Results: {args.output / 'results.json'}", flush=True)
    if any(model["status"] != "passed" for model in report["models"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

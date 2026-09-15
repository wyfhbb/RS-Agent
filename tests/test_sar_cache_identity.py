"""Cache identity regression tests using temporary unit fixtures, not detections."""

import json
from pathlib import Path

from rs_agent.toolkit.sar_detection import DetectorSpec, SARDetectorBackend, sha256


def prepared_backend(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first.write_bytes(b"identical-unit-test-image-bytes")
    second.write_bytes(first.read_bytes())
    backend = SARDetectorBackend(
        [DetectorSpec("cascade", "unit-test", ("/nonexistent-unit-test-detector",))],
        tmp_path / "calls", "cascade", {str(first): 10, str(second): 11},
    )
    # Seed only the cache contract. No inference/API is simulated as evidence.
    backend._cache[("cascade", str(first), sha256(first), 10)] = {
        "status": "success", "call_id": "unit-test-cached-call", "image": str(first),
        "image_id": 10, "model": "cascade",
    }
    return backend, first, second


def test_same_path_pixels_and_id_can_reuse_existing_result(tmp_path):
    backend, first, _ = prepared_backend(tmp_path)
    observation = json.loads(backend.detect(str(first)))
    assert observation["execution_kind"] == "cache_hit"
    assert observation["image_id"] == 10
    assert observation["image"] == str(first)


def test_identical_pixels_at_different_paths_do_not_reuse_another_id(tmp_path):
    backend, _, second = prepared_backend(tmp_path)
    observation = json.loads(backend.detect(str(second)))
    assert observation["status"] == "error"  # Real executable unavailable in this unit fixture.
    assert observation["execution_kind"] != "cache_hit"
    assert backend.events[-1]["image_id"] == 11
    assert backend.events[-1]["command"][-2:] == ["--image-id", "11"]


def test_changed_id_at_same_path_invalidates_cache(tmp_path):
    backend, first, _ = prepared_backend(tmp_path)
    backend.image_ids[str(first)] = 12
    observation = json.loads(backend.detect(str(first)))
    assert observation["execution_kind"] != "cache_hit"
    assert backend.events[-1]["command"][-2:] == ["--image-id", "12"]


def test_unindexed_image_passes_minus_one_without_guessing_filename(tmp_path):
    backend, _, _ = prepared_backend(tmp_path)
    image = tmp_path / "12345.png"
    image.write_bytes(b"new-unit-test-image")
    observation = json.loads(backend.detect(str(image)))
    assert observation["execution_kind"] != "cache_hit"
    assert backend.events[-1]["command"][-2:] == ["--image-id", "-1"]
    assert backend.events[-1]["image_id"] == -1


def test_canonical_symlink_path_preserves_same_identity(tmp_path):
    backend, first, _ = prepared_backend(tmp_path)
    alias = Path(tmp_path) / "alias.png"
    alias.symlink_to(first)
    observation = json.loads(backend.detect(str(alias)))
    assert observation["execution_kind"] == "cache_hit"
    assert observation["image_id"] == 10

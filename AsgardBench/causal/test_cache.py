from pathlib import Path

from AsgardBench.causal.cache import InferenceCache, cache_key, image_sha256


def test_cache_key_covers_prompt_image_model_and_decoding(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    common = {
        "provider": "openai_compatible",
        "model": "m1",
        "prompt": "prompt",
        "image_hash": image_sha256(image),
        "decoding": {"temperature": 0},
    }
    baseline = cache_key(**common)
    for field, changed in (
        ("provider", "other"),
        ("model", "m2"),
        ("prompt", "other prompt"),
        ("image_hash", None),
        ("decoding", {"temperature": 1}),
    ):
        candidate = dict(common)
        candidate[field] = changed
        assert cache_key(**candidate) != baseline


def test_inference_cache_round_trip(tmp_path: Path) -> None:
    cache = InferenceCache(tmp_path / "cache")
    key = "a" * 64
    assert cache.get(key) is None
    cache.put(key, {"raw_output": "ok", "response": {"usage": {}}})
    record = cache.get(key)
    assert record is not None
    assert record["raw_output"] == "ok"
    assert record["key"] == key

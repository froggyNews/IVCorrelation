import pandas as pd

import analysis.model_params_logger as mpl


def _setup_cache(monkeypatch, tmp_path):
    cache_file = tmp_path / "cache.parquet"
    monkeypatch.setattr(mpl, "CACHE_PATH", cache_file)
    monkeypatch.setattr(mpl, "_ARTIFACT_VERSION", {})
    return cache_file


def test_compute_or_load_ttl_and_version(monkeypatch, tmp_path):
    _setup_cache(monkeypatch, tmp_path)

    base = pd.Timestamp("2024-01-01 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "utcnow", lambda: base)

    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return calls["n"]

    payload = {"a": 1}

    # initial compute
    result1 = mpl.compute_or_load("kind", payload, builder, ttl_sec=10)
    assert result1 == 1
    assert calls["n"] == 1

    # within TTL -> cache hit
    monkeypatch.setattr(pd.Timestamp, "utcnow", lambda: base + pd.Timedelta(seconds=5))
    result2 = mpl.compute_or_load("kind", payload, builder, ttl_sec=10)
    assert result2 == 1
    assert calls["n"] == 1

    # after TTL -> recompute
    monkeypatch.setattr(pd.Timestamp, "utcnow", lambda: base + pd.Timedelta(seconds=15))
    result3 = mpl.compute_or_load("kind", payload, builder, ttl_sec=10)
    assert result3 == 2
    assert calls["n"] == 2

    # version bump forces recompute
    monkeypatch.setattr(pd.Timestamp, "utcnow", lambda: base + pd.Timedelta(seconds=16))
    mpl.set_artifact_version("kind", "2")
    result4 = mpl.compute_or_load("kind", payload, builder, ttl_sec=10)
    assert result4 == 3
    assert calls["n"] == 3


def test_cache_prune_and_clear(monkeypatch, tmp_path):
    _setup_cache(monkeypatch, tmp_path)

    base = pd.Timestamp("2024-01-01 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "utcnow", lambda: base)

    calls = {"a": 0, "b": 0}

    def builder_a():
        calls["a"] += 1
        return "A"

    def builder_b():
        calls["b"] += 1
        return "B"

    payload = {}
    mpl.compute_or_load("k1", payload, builder_a, ttl_sec=10)
    mpl.compute_or_load("k2", payload, builder_b, ttl_sec=100)
    stats = mpl.cache_stats()
    assert stats["entries"] == 2
    assert set(stats["kinds"]) == {"k1", "k2"}

    # Advance time beyond k1 expiry
    monkeypatch.setattr(pd.Timestamp, "utcnow", lambda: base + pd.Timedelta(seconds=20))
    removed = mpl.prune_expired()
    assert removed == 1
    stats = mpl.cache_stats()
    assert stats["entries"] == 1
    assert stats["kinds"] == ["k2"]

    cleared = mpl.clear_cache("k2")
    assert cleared == 1
    assert mpl.cache_stats()["entries"] == 0

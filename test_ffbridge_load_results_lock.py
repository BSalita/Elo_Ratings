import json
import threading
import time

import polars as pl

import ffbridge_report_service as reports


def _write_elo_cache(cache_dir, api_key: str = "FFBridge_Lancelot_API") -> str:
    key = reports.elo_cache_key(api_key, True)
    results_path, players_path, meta_path = reports.elo_cache_paths(key)
    pair_path = reports.elo_pair_cache_path(key)
    pl.DataFrame({"player_id": ["1"], "scratch_elo": [1400.0]}).write_parquet(
        results_path
    )
    pl.DataFrame({"player_id": ["1"]}).write_parquet(players_path)
    pl.DataFrame({"pair_id": ["1_2"]}).write_parquet(pair_path)
    meta_path.write_text(
        json.dumps({"built_at": "2026-09-08T00:00:00Z"}), encoding="utf-8"
    )
    return key


def test_load_results_serializes_concurrent_parquet_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "ELO_CACHE_DIR", tmp_path)
    reports._RESULTS_CACHE.clear()
    _write_elo_cache(tmp_path)

    original = reports.pl.read_parquet
    calls: list[str] = []
    inflight = 0
    max_inflight = 0
    counter_lock = threading.Lock()

    def slow_read(path, *args, **kwargs):
        nonlocal inflight, max_inflight
        with counter_lock:
            calls.append(str(path))
            inflight += 1
            max_inflight = max(max_inflight, inflight)
        time.sleep(0.2)
        try:
            return original(path, *args, **kwargs)
        finally:
            with counter_lock:
                inflight -= 1

    monkeypatch.setattr(reports.pl, "read_parquet", slow_read)

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            reports.load_results()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert max_inflight == 1
    assert len(calls) == 1

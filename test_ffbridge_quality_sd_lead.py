import json
from datetime import date, datetime, timezone

import polars as pl

import build_ffbridge_quality_parquets as builder
import ffbridge_quality_pipeline as pipeline


def _fragment(dd_cost: list, sd_cost: list) -> pl.DataFrame:
    rows = len(dd_cost)
    frame = pl.DataFrame({column: [None] * rows for column in pipeline.QUALITY_BOARD_COLUMNS})
    return frame.with_columns(
        pl.Series("DD_Lead_Cost", dd_cost, dtype=pl.UInt8),
        pl.Series("SD_Lead_Cost", sd_cost, dtype=pl.Float32),
    )


def test_fragment_without_sd_lead_is_rebuilt_only_when_asked(tmp_path):
    old = tmp_path / "old.parquet"
    scored = tmp_path / "scored.parquet"
    no_leads = tmp_path / "no_leads.parquet"
    _fragment([0, 1], [None, None]).write_parquet(old)
    _fragment([0, 1], [0.0, 0.4]).write_parquet(scored)
    _fragment([None, None], [None, None]).write_parquet(no_leads)
    assert pipeline._read_current_fragment(old) is not None
    assert pipeline._read_current_fragment(old, need_sd_lead=True) is None
    assert pipeline._read_current_fragment(scored, need_sd_lead=True) is not None
    # Nothing to score: a session without recorded leads is not rebuilt forever.
    assert pipeline._read_current_fragment(no_leads, need_sd_lead=True) is not None


def test_changing_sd_lead_samples_makes_the_cache_stale(tmp_path):
    (tmp_path / "ffbridge_quality_metadata.json").write_text(
        json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cutoff": "2026-11-03",
            "schema_version": pipeline.SCHEMA_VERSION,
        }),
        encoding="utf-8",
    )
    cutoff = date(2026, 11, 3)
    assert builder._quality_cache_is_fresh(tmp_path, cutoff, 24)[0]
    fresh, reason = builder._quality_cache_is_fresh(tmp_path, cutoff, 24, sd_lead_samples=20)
    assert not fresh and "SD lead samples were 0, now 20" in reason

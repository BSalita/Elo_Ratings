"""Session-level summaries for board-by-board Elo detail tables."""

from __future__ import annotations

import polars as pl


def summarize_acbl_sessions(detail: pl.DataFrame) -> pl.DataFrame:
    """Collapse ACBL board detail to one chronologically ordered row per session."""
    if "Session" not in detail.columns:
        raise ValueError("ACBL detail is missing required 'Session' column")
    if detail.is_empty():
        return detail

    sort_columns = [
        column
        for column in ("Date", "Session", "Round", "Board")
        if column in detail.columns
    ]
    descending = [
        column in ("Date", "Session")
        for column in sort_columns
    ]
    ordered = detail.sort(sort_columns, descending=descending)

    aggregations: list[pl.Expr] = []
    if "Date" in ordered.columns:
        aggregations.append(pl.col("Date").first().alias("Date"))
    if "Partner" in ordered.columns:
        aggregations.append(pl.col("Partner").first().alias("Partner"))
    aggregations.append(pl.len().alias("Boards"))
    if "Opponents" in ordered.columns:
        aggregations.append(
            pl.col("Opponents").drop_nulls().n_unique().alias("Opponent_Pairs")
        )
    if "Pct" in ordered.columns:
        aggregations.append(
            pl.col("Pct").mean().cast(pl.Float64).round(1).alias("Avg_Pct")
        )
    if "Elo_Before" in ordered.columns:
        aggregations.append(
            pl.col("Elo_Before")
            .drop_nulls()
            .first()
            .cast(pl.Int32, strict=False)
            .alias("Elo_Start")
        )
    if "Elo_After" in ordered.columns:
        aggregations.append(
            pl.col("Elo_After")
            .drop_nulls()
            .last()
            .cast(pl.Int32, strict=False)
            .alias("Elo_End")
        )

    summary = ordered.group_by("Session", maintain_order=True).agg(aggregations)
    if "Elo_Start" in summary.columns and "Elo_End" in summary.columns:
        summary = summary.with_columns(
            (pl.col("Elo_End") - pl.col("Elo_Start")).alias("Elo_Delta")
        )

    preferred_order = [
        "Date",
        "Session",
        "Partner",
        "Boards",
        "Opponent_Pairs",
        "Avg_Pct",
        "Elo_Start",
        "Elo_End",
        "Elo_Delta",
    ]
    return summary.select(
        [column for column in preferred_order if column in summary.columns]
    )

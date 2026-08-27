"""Session-level summaries for board-by-board Elo detail tables."""

from __future__ import annotations

import polars as pl


def acbl_results_url_expr(
    club_or_tournament: str,
    *,
    event_id_column: str = "Event_ID",
    session_column: str = "Session",
) -> pl.Expr:
    """Build the published ACBL results URL for a detail row."""
    source = club_or_tournament.strip().lower()
    if source == "club":
        event_id = (
            pl.col(event_id_column).cast(pl.Utf8).fill_null("").str.strip_chars()
        )
        return (
            pl.when(event_id != "")
            .then(
                pl.concat_str(
                    [
                        pl.lit("https://my.acbl.org/club-results/details/"),
                        event_id,
                    ]
                )
            )
            .otherwise(None)
            .alias("Results_URL")
        )
    if source == "tournament":
        session = (
            pl.col(session_column).cast(pl.Utf8).fill_null("").str.strip_chars()
        )
        return (
            pl.when(session != "")
            .then(
                pl.concat_str(
                    [
                        pl.lit("https://live.acbl.org/event/"),
                        session.str.replace_all("-", "/"),
                        pl.lit("/summary"),
                    ]
                )
            )
            .otherwise(None)
            .alias("Results_URL")
        )
    raise ValueError("club_or_tournament must be either 'Club' or 'Tournament'")


def summarize_acbl_sessions(
    detail: pl.DataFrame,
    club_or_tournament: str | None = None,
) -> pl.DataFrame:
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
    if "Event_ID" in ordered.columns:
        event_id = pl.col("Event_ID").cast(pl.Utf8).str.strip_chars()
        aggregations.append(
            event_id
            .filter(event_id.is_not_null() & (event_id != ""))
            .first()
            .alias("Event_ID")
        )
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
    if club_or_tournament is not None:
        source = club_or_tournament.strip().lower()
        if source == "club":
            results_url = (
                acbl_results_url_expr(source)
                if "Event_ID" in summary.columns
                else pl.lit(None, dtype=pl.Utf8).alias("Results_URL")
            )
        elif source == "tournament":
            results_url = acbl_results_url_expr(source)
        else:
            raise ValueError(
                "club_or_tournament must be either 'Club' or 'Tournament'"
            )
        summary = summary.with_columns(results_url)

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
        "Results_URL",
    ]
    return summary.select(
        [column for column in preferred_order if column in summary.columns]
    )

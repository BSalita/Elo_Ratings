"""Foresight metrics: decisions that fit the unseen cards better than the odds explain.

Three per-player (or per-pair) statistics, each a z-score against peers of
similar Elo, and their combination:

- ``Lead_Foresight_Z``: opening leads. Per lead, single-dummy cost minus
  double-dummy cost (``SD_Lead_Cost - DD_Lead_Cost``). For a leader who cannot
  see the layout its expectation depends on the deal, not on the card chosen;
  a leader who knows the layout scores higher.
- ``Guess_Foresight_Z``: declarer play. Slope of ``DD_Tricks_Diff`` on
  ``SC_Guess_Declarer`` (tricks a double-dummy declarer gains over percentage
  play). A declarer who cannot see the cards loses those tricks (negative
  slope); one who can does not.
- ``Home_Edge_Z``: session results at the entity's most-played club minus
  results everywhere else.
- ``Foresight_Z``: Stouffer combination of the components that exist.

Each z is ``(value - peer mean) / sqrt(tau2 + sampling variance)`` within an
Elo band, where ``tau2`` is the between-entity variance of honest play
(DerSimonian-Laird). So a z stays near N(0, 1) among peers whatever the
sample size. These are screening statistics, not proof of anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import polars as pl

FORESIGHT_COLUMNS = (
    "Foresight_Z",
    "Lead_Foresight_Z",
    "Guess_Foresight_Z",
    "Home_Edge_Z",
)
MIN_LEADS = 50
MIN_GUESS_BOARDS = 50
MIN_HOME_SESSIONS = 10
MIN_AWAY_SESSIONS = 5
PEER_BANDS = 10
MIN_BAND_SIZE = 50

FORESIGHT_METRIC_DEFINITIONS = {
    "Foresight_Z": {
        "label": "Foresight",
        "calculation": (
            "Sum of the available component z-scores divided by the square "
            "root of their count."
        ),
    },
    "Lead_Foresight_Z": {
        "label": "Lead foresight",
        "population": f"Opening leader (defending pair in pair reports); at least {MIN_LEADS} scored leads.",
        "calculation": (
            "Mean of SD_Lead_Cost - DD_Lead_Cost per lead, as a z-score against "
            "entities of similar Elo. Higher: leads fit the actual layout better "
            "than their odds."
        ),
    },
    "Guess_Foresight_Z": {
        "label": "Guess foresight",
        "population": f"Declarer (declaring pair in pair reports); at least {MIN_GUESS_BOARDS} declarations.",
        "calculation": (
            "Slope of DD_Tricks_Diff on SC_Guess_Declarer, as a z-score against "
            "entities of similar Elo. Higher: fewer tricks lost on boards where "
            "the percentage play differs from the double-dummy play."
        ),
    },
    "Home_Edge_Z": {
        "label": "Home edge",
        "population": (
            f"At least {MIN_HOME_SESSIONS} sessions at the most-played club and "
            f"{MIN_AWAY_SESSIONS} elsewhere."
        ),
        "calculation": (
            "Mean session result at the most-played club minus the mean "
            "elsewhere, as a z-score against entities of similar Elo. ACBL uses "
            "percentage minus Elo-expected percentage; FFBridge uses the "
            "scratch percentage."
        ),
    },
}

FORESIGHT_CHECKBOX_HELP = (
    "Add Foresight_Z, Lead_Foresight_Z, Guess_Foresight_Z and Home_Edge_Z: "
    "z-scores against players or pairs of similar Elo. They screen for results "
    "that fit the unseen cards better than the odds explain; they prove nothing."
)


def foresight_caption(status: Mapping[str, str]) -> str:
    """Markdown caption for the leaderboard: what the columns are and which are missing."""
    text = (
        "🔎 **Foresight columns** are z-scores against players or pairs of similar "
        "Elo (about 0 ± 1 for ordinary play; blank below the minimum sample). "
        f"**Lead_Foresight_Z**: opening leads fit the actual layout better than "
        f"their single-dummy odds (≥{MIN_LEADS} leads). **Guess_Foresight_Z**: "
        f"declarer loses fewer tricks where the percentage play differs from the "
        f"double-dummy play (≥{MIN_GUESS_BOARDS} declarations). **Home_Edge_Z**: "
        f"results at the most-played club beat results elsewhere "
        f"(≥{MIN_HOME_SESSIONS} home and ≥{MIN_AWAY_SESSIONS} away sessions). "
        "**Foresight_Z** combines the ones available. A high value is a reason "
        "to look closer, not evidence of anything by itself."
    )
    problems = [f"{metric}: {state}" for metric, state in status.items() if state != "ok"]
    if problems:
        text += " Not computed here — " + "; ".join(problems) + "."
    return text


_STAT_SCHEMA: dict[str, Any] = {
    "entity_id": pl.Utf8,
    "elo": pl.Float64,
    "lead_n": pl.Int64,
    "lead_mean": pl.Float64,
    "lead_var": pl.Float64,
    "guess_n": pl.Int64,
    "guess_sxx": pl.Float64,
    "guess_syy": pl.Float64,
    "guess_sxy": pl.Float64,
    "home_n": pl.Int64,
    "home_mean": pl.Float64,
    "home_var": pl.Float64,
    "away_n": pl.Int64,
    "away_mean": pl.Float64,
    "away_var": pl.Float64,
}
_METRIC_STAT_COLUMNS = {
    "lead": ("lead_n", "lead_mean", "lead_var"),
    "guess": ("guess_n", "guess_sxx", "guess_syy", "guess_sxy"),
    "home": ("home_n", "home_mean", "home_var", "away_n", "away_mean", "away_var"),
}


def empty_foresight_stats() -> pl.DataFrame:
    """Zero-row sufficient-statistics frame."""
    return pl.DataFrame(schema=_STAT_SCHEMA)


def combine_stats(frames: Mapping[str, Optional[pl.DataFrame]]) -> pl.DataFrame:
    """One row per entity from the per-metric frames (``lead``, ``guess``, ``home``).

    Each frame has ``entity_id``, that metric's statistic columns and optionally
    ``elo``. A metric that is None leaves its columns null. ``elo`` is the mean
    of the metrics' values.
    """
    unknown = set(frames) - set(_METRIC_STAT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown foresight metrics {sorted(unknown)}")
    out = pl.DataFrame(schema={"entity_id": pl.Utf8})
    elo_columns: list[str] = []
    for metric, columns in _METRIC_STAT_COLUMNS.items():
        frame = frames.get(metric)
        if frame is None:
            continue
        missing = [c for c in ("entity_id", *columns) if c not in frame.columns]
        if missing:
            raise ValueError(f"Foresight {metric} frame lacks columns {missing}")
        selected = [pl.col("entity_id").cast(pl.Utf8), *[pl.col(c) for c in columns]]
        if "elo" in frame.columns:
            selected.append(pl.col("elo").cast(pl.Float64).alias(f"_elo_{metric}"))
            elo_columns.append(f"_elo_{metric}")
        part = frame.select(selected).filter(pl.col("entity_id").is_not_null())
        if part.get_column("entity_id").n_unique() != part.height:
            raise ValueError(f"Foresight {metric} frame has duplicate entity_id")
        out = out.join(part, on="entity_id", how="full", coalesce=True)
    out = out.with_columns(
        pl.mean_horizontal(elo_columns).alias("elo")
        if elo_columns
        else pl.lit(None, dtype=pl.Float64).alias("elo")
    )
    return out.select(
        *[
            (pl.col(name) if name in out.columns else pl.lit(None)).cast(dtype).alias(name)
            for name, dtype in _STAT_SCHEMA.items()
        ]
    )


def _metric_values(stats: pl.DataFrame) -> pl.DataFrame:
    """Per-entity value and sampling variance of each metric (null below the minimums)."""
    slope = pl.col("guess_sxy") / pl.col("guess_sxx")
    residual = (pl.col("guess_syy") - slope * pl.col("guess_sxy")) / (pl.col("guess_n") - 2)
    lead_ok = pl.col("lead_n") >= MIN_LEADS
    guess_ok = (pl.col("guess_n") >= MIN_GUESS_BOARDS) & (pl.col("guess_sxx") > 0)
    home_ok = (pl.col("home_n") >= MIN_HOME_SESSIONS) & (pl.col("away_n") >= MIN_AWAY_SESSIONS)
    return stats.select(
        "entity_id",
        "elo",
        pl.when(lead_ok).then(pl.col("lead_mean")).alias("lead_value"),
        pl.when(lead_ok).then(pl.col("lead_var") / pl.col("lead_n")).alias("lead_variance"),
        pl.when(guess_ok).then(slope).alias("guess_value"),
        pl.when(guess_ok).then(residual / pl.col("guess_sxx")).alias("guess_variance"),
        pl.when(home_ok).then(pl.col("home_mean") - pl.col("away_mean")).alias("home_value"),
        pl.when(home_ok)
        .then(pl.col("home_var") / pl.col("home_n") + pl.col("away_var") / pl.col("away_n"))
        .alias("home_variance"),
    )


def peer_z(frame: pl.DataFrame, output: str) -> pl.DataFrame:
    """Random-effects z-score of ``value`` within Elo bands.

    ``frame``: ``entity_id``, ``elo``, ``value``, ``variance`` (sampling variance
    of ``value``). Rows with a null or non-finite field, or zero variance, get
    no score. Returns ``entity_id`` and ``output``.
    """
    valid = frame.filter(
        pl.col("elo").is_finite()
        & pl.col("value").is_finite()
        & pl.col("variance").is_finite()
        & (pl.col("variance") > 0)
    )
    empty = pl.DataFrame(schema={"entity_id": pl.Utf8, output: pl.Float64})
    if valid.height < 2:
        return empty
    bands = max(1, min(PEER_BANDS, valid.height // MIN_BAND_SIZE))
    weight = 1.0 / pl.col("variance")
    valid = valid.with_columns(
        ((pl.col("elo").rank("ordinal") - 1) * bands // valid.height).alias("_band")
    ).with_columns(
        ((pl.col("value") * weight).sum().over("_band") / weight.sum().over("_band")).alias("_fixed_mean")
    )
    # DerSimonian-Laird between-entity variance per band.
    q = (weight * (pl.col("value") - pl.col("_fixed_mean")) ** 2).sum().over("_band")
    scale = weight.sum().over("_band") - (weight**2).sum().over("_band") / weight.sum().over("_band")
    valid = valid.with_columns(
        pl.when(scale > 0)
        .then(((q - (pl.len().over("_band") - 1)) / scale).clip(lower_bound=0.0))
        .otherwise(None)
        .alias("_tau2")
    )
    total = pl.col("_tau2") + pl.col("variance")
    valid = valid.with_columns(
        ((pl.col("value") / total).sum().over("_band") / (1.0 / total).sum().over("_band")).alias("_mean")
    )
    return valid.select(
        "entity_id",
        pl.when(total > 0)
        .then((pl.col("value") - pl.col("_mean")) / total.sqrt())
        .otherwise(None)
        .alias(output),
    )


def score_foresight(stats: pl.DataFrame, elo: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """``entity_id`` plus FORESIGHT_COLUMNS from a sufficient-statistics frame.

    ``elo`` (``entity_id``, ``elo``) replaces the frame's own ``elo`` column and
    limits scoring to the entities it lists.
    """
    missing = [c for c in _STAT_SCHEMA if c not in stats.columns]
    if missing:
        raise ValueError(f"Foresight stats lack columns {missing}")
    if elo is not None:
        stats = stats.drop("elo").join(
            elo.select(pl.col("entity_id").cast(pl.Utf8), pl.col("elo").cast(pl.Float64)),
            on="entity_id",
            how="inner",
        )
    values = _metric_values(stats)
    out = values.select("entity_id")
    for metric, column in (
        ("lead", "Lead_Foresight_Z"),
        ("guess", "Guess_Foresight_Z"),
        ("home", "Home_Edge_Z"),
    ):
        scored = peer_z(
            values.select(
                "entity_id",
                "elo",
                pl.col(f"{metric}_value").alias("value"),
                pl.col(f"{metric}_variance").alias("variance"),
            ),
            column,
        )
        out = out.join(scored, on="entity_id", how="left")
    components = [pl.col(c) for c in FORESIGHT_COLUMNS[1:]]
    count = pl.sum_horizontal([c.is_not_null().cast(pl.Int64) for c in components])
    return out.with_columns(
        pl.when(count > 0)
        .then(pl.sum_horizontal(components) / count.cast(pl.Float64).sqrt())
        .otherwise(None)
        .alias("Foresight_Z")
    ).select("entity_id", *FORESIGHT_COLUMNS)


def attach_foresight(
    leaderboard: pl.DataFrame,
    stats: pl.DataFrame,
    *,
    id_column: str,
    elo_column: Optional[str] = None,
    after: Optional[str] = None,
) -> pl.DataFrame:
    """Left-join the rounded FORESIGHT_COLUMNS onto a leaderboard.

    ``elo_column``: leaderboard column to band peers by. The leaderboard must
    then hold the whole ranked population, not a top-N cut. Omit it to use the
    statistics frame's own ``elo``. ``after``: column the new ones follow
    (default: appended).
    """
    if id_column not in leaderboard.columns:
        raise ValueError(f"Leaderboard lacks {id_column}")
    present = [c for c in FORESIGHT_COLUMNS if c in leaderboard.columns]
    if present:
        raise ValueError(f"Leaderboard already has {present}")
    elo = None
    if elo_column is not None:
        if elo_column not in leaderboard.columns:
            raise ValueError(f"Leaderboard lacks {elo_column}")
        elo = leaderboard.select(
            pl.col(id_column).cast(pl.Utf8).alias("entity_id"),
            pl.col(elo_column).cast(pl.Float64).alias("elo"),
        )
    scored = score_foresight(stats, elo).with_columns(
        [pl.col(c).round(2) for c in FORESIGHT_COLUMNS]
    )
    joined = leaderboard.with_columns(pl.col(id_column).cast(pl.Utf8)).join(
        scored, left_on=id_column, right_on="entity_id", how="left"
    )
    if after is None or after not in leaderboard.columns:
        return joined
    position = leaderboard.columns.index(after) + 1
    order = [
        *leaderboard.columns[:position],
        *FORESIGHT_COLUMNS,
        *leaderboard.columns[position:],
    ]
    return joined.select(order)


# ---------------------------------------------------------------------------
# ACBL: board-level DuckDB view ``self``
# ---------------------------------------------------------------------------

_ACBL_SEATS = ("N", "E", "S", "W")
# Opening leader for each declarer (the seat to declarer's left).
_ACBL_LEADER_FOR_DECLARER = {"W": "N", "N": "E", "E": "S", "S": "W"}
_ACBL_PAIR_ID_SQL = {
    "NS": (
        "CASE WHEN Player_ID_N < Player_ID_S THEN Player_ID_N || '-' || Player_ID_S "
        "ELSE Player_ID_S || '-' || Player_ID_N END"
    ),
    "EW": (
        "CASE WHEN Player_ID_E < Player_ID_W THEN Player_ID_E || '-' || Player_ID_W "
        "ELSE Player_ID_W || '-' || Player_ID_E END"
    ),
}
_ACBL_METRIC_REQUIREMENTS = {
    "lead": (
        ("SD_Lead_Cost", "DD_Lead_Cost"),
        "the single-dummy lead score (ACBL_SD_LEAD_SAMPLES in acbl_all.bat step 3c; club only)",
    ),
    "guess": (
        ("SC_Guess_Declarer", "DD_Tricks_Diff"),
        "the SuitC columns (ACBL_SUITC=1 in acbl_all.bat step 3a)",
    ),
    "home": (
        ("Club", "Pct_NS", "Elo_E_NS", "Elo_E_EW"),
        "the Club column (rebuild the Elo parquet with acbl_elo_ratings_create.py; club only)",
    ),
}


def _finite_sql(column: Optional[str]) -> str:
    if not column:
        return "CAST(NULL AS DOUBLE)"
    return f"CASE WHEN {column} IS NULL OR isnan({column}) THEN NULL ELSE CAST({column} AS DOUBLE) END"


def _acbl_seat_case(selector: str, by_seat: Mapping[str, str]) -> str:
    whens = " ".join(f"WHEN '{seat}' THEN {expr}" for seat, expr in by_seat.items())
    return f"CASE {selector} {whens} END"


def _acbl_entity_sql(rating_type: str, role: str, elo_columns: Mapping[str, Optional[str]]) -> tuple[str, str]:
    """(entity id SQL, entity Elo SQL) for the opening leader or the declarer."""
    if role not in ("leader", "declarer"):
        raise ValueError(f"Unknown role {role!r}")
    pattern = elo_columns.get("player_pattern")
    if rating_type == "Players":
        seat_of = _ACBL_LEADER_FOR_DECLARER if role == "leader" else {s: s for s in _ACBL_SEATS}
        ids = {d: f"Player_ID_{seat}" for d, seat in seat_of.items()}
        elos = {
            d: _finite_sql(pattern.format(pos=seat) if pattern else None)
            for d, seat in seat_of.items()
        }
        return (
            f"CAST({_acbl_seat_case('Declarer_Direction', ids)} AS VARCHAR)",
            _acbl_seat_case("Declarer_Direction", elos),
        )
    if rating_type != "Pairs":
        raise ValueError(f"Unknown rating_type {rating_type!r}")
    side_of = {"NS": "EW", "EW": "NS"} if role == "leader" else {"NS": "NS", "EW": "EW"}
    ids = {d: _ACBL_PAIR_ID_SQL[side] for d, side in side_of.items()}
    elos = {
        d: _finite_sql(elo_columns.get("pair_ns" if side == "NS" else "pair_ew"))
        for d, side in side_of.items()
    }
    return (
        _acbl_seat_case("Declarer_Pair_Direction", ids),
        _acbl_seat_case("Declarer_Pair_Direction", elos),
    )


def acbl_lead_sql(rating_type: str, elo_columns: Mapping[str, Optional[str]]) -> str:
    entity, elo = _acbl_entity_sql(rating_type, "leader", elo_columns)
    return f"""
SELECT entity_id, AVG(elo) AS elo, COUNT(*) AS lead_n, AVG(diff) AS lead_mean, VAR_SAMP(diff) AS lead_var
FROM (
  SELECT {entity} AS entity_id, {elo} AS elo,
         CAST(SD_Lead_Cost AS DOUBLE) - CAST(DD_Lead_Cost AS DOUBLE) AS diff
  FROM self
  WHERE SD_Lead_Cost IS NOT NULL AND DD_Lead_Cost IS NOT NULL
)
WHERE entity_id IS NOT NULL
GROUP BY entity_id
"""


def acbl_guess_sql(rating_type: str, elo_columns: Mapping[str, Optional[str]]) -> str:
    entity, elo = _acbl_entity_sql(rating_type, "declarer", elo_columns)
    return f"""
SELECT entity_id, AVG(elo) AS elo,
       REGR_COUNT(y, x) AS guess_n, REGR_SXX(y, x) AS guess_sxx,
       REGR_SYY(y, x) AS guess_syy, REGR_SXY(y, x) AS guess_sxy
FROM (
  SELECT {entity} AS entity_id, {elo} AS elo,
         CAST(SC_Guess_Declarer AS DOUBLE) AS x, CAST(DD_Tricks_Diff AS DOUBLE) AS y
  FROM self
  WHERE SC_Guess_Declarer IS NOT NULL AND DD_Tricks_Diff IS NOT NULL
)
WHERE entity_id IS NOT NULL
GROUP BY entity_id
"""


def acbl_home_sql(rating_type: str, elo_columns: Mapping[str, Optional[str]]) -> str:
    pattern = elo_columns.get("player_pattern")

    def player_elo(seat: str) -> str:
        return _finite_sql(pattern.format(pos=seat) if pattern else None)

    sides = []
    for side, first, second, pct, expected, pair_elo in (
        ("NS", "N", "S", "Pct_NS", "Elo_E_NS", elo_columns.get("pair_ns")),
        ("EW", "E", "W", "1 - Pct_NS", "Elo_E_EW", elo_columns.get("pair_ew")),
    ):
        sides.append(
            f"""
  SELECT session_id, ANY_VALUE(Club) AS club,
         CAST(Player_ID_{first} AS VARCHAR) AS player_a, CAST(Player_ID_{second} AS VARCHAR) AS player_b,
         ANY_VALUE({_ACBL_PAIR_ID_SQL[side]}) AS pair_id,
         AVG({pct}) - AVG({expected}) AS result,
         AVG({player_elo(first)}) AS elo_a, AVG({player_elo(second)}) AS elo_b,
         AVG({_finite_sql(pair_elo)}) AS elo_pair
  FROM self
  WHERE Pct_NS IS NOT NULL AND {expected} IS NOT NULL AND Club IS NOT NULL
  GROUP BY session_id, Player_ID_{first}, Player_ID_{second}"""
        )
    if rating_type == "Players":
        entity_sessions = """
  SELECT player_a AS entity_id, club, result, elo_a AS elo FROM pair_sessions
  UNION ALL
  SELECT player_b AS entity_id, club, result, elo_b AS elo FROM pair_sessions"""
    elif rating_type == "Pairs":
        entity_sessions = """
  SELECT pair_id AS entity_id, club, result, elo_pair AS elo FROM pair_sessions"""
    else:
        raise ValueError(f"Unknown rating_type {rating_type!r}")
    return f"""
WITH pair_sessions AS ({' UNION ALL '.join(sides)}
),
entity_sessions AS ({entity_sessions}
),
club_counts AS (
  SELECT entity_id, club, COUNT(*) AS sessions FROM entity_sessions
  WHERE entity_id IS NOT NULL AND result IS NOT NULL
  GROUP BY entity_id, club
),
home_club AS (
  SELECT entity_id, ARG_MAX(club, sessions) AS club FROM club_counts GROUP BY entity_id
)
SELECT es.entity_id, AVG(es.elo) AS elo,
       COUNT(*) FILTER (WHERE es.club = hc.club) AS home_n,
       AVG(es.result) FILTER (WHERE es.club = hc.club) AS home_mean,
       VAR_SAMP(es.result) FILTER (WHERE es.club = hc.club) AS home_var,
       COUNT(*) FILTER (WHERE es.club <> hc.club) AS away_n,
       AVG(es.result) FILTER (WHERE es.club <> hc.club) AS away_mean,
       VAR_SAMP(es.result) FILTER (WHERE es.club <> hc.club) AS away_var
FROM entity_sessions es JOIN home_club hc USING (entity_id)
WHERE es.result IS NOT NULL
GROUP BY es.entity_id
"""


_ACBL_SQL = {"lead": acbl_lead_sql, "guess": acbl_guess_sql, "home": acbl_home_sql}


def acbl_foresight_stats(
    con: Any,
    *,
    columns: set[str],
    rating_type: str,
    elo_columns: Mapping[str, Optional[str]],
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Sufficient statistics from DuckDB view ``self`` and a per-metric status.

    ``columns``: columns the Elo parquet really has (``self`` may add typed-null
    stand-ins). A metric whose inputs are absent is reported in the status and
    left null.
    """
    frames: dict[str, Optional[pl.DataFrame]] = {}
    status: dict[str, str] = {}
    for metric, (required, source) in _ACBL_METRIC_REQUIREMENTS.items():
        missing = [c for c in required if c not in columns]
        if missing:
            frames[metric] = None
            status[metric] = f"unavailable: Elo parquet lacks {missing}; needs {source}"
            continue
        frames[metric] = con.sql(_ACBL_SQL[metric](rating_type, elo_columns)).pl()
        status[metric] = "ok"
    return combine_stats(frames), status


# ---------------------------------------------------------------------------
# FFBridge: quality boards, club board results and session results (polars)
# ---------------------------------------------------------------------------

# Opening leader's seat for each declarer.
_FF_LEADER_FOR_DECLARER = {"W": "N", "N": "E", "E": "S", "S": "W"}


def _require_columns(frame_columns: Any, required: tuple[str, ...], label: str) -> None:
    missing = [c for c in required if c not in frame_columns]
    if missing:
        raise ValueError(f"{label} lacks columns {missing}")


def _ff_pair_id(first: str, second: str) -> pl.Expr:
    """FFBridge Elo pair key: the two ids, lexically sorted, joined by ``_``."""
    a = pl.col(first).cast(pl.Utf8)
    b = pl.col(second).cast(pl.Utf8)
    return (
        pl.when(a.is_null() | b.is_null())
        .then(None)
        .when(a <= b)
        .then(a + "_" + b)
        .otherwise(b + "_" + a)
    )


def ffbridge_lead_stats(boards: pl.LazyFrame, rating_type: str) -> pl.DataFrame:
    """Lead statistics from the FFBridge quality boards (``ffbridge_quality_boards.parquet``)."""
    columns = boards.collect_schema().names()
    _require_columns(
        columns,
        ("Declarer_Direction", "DD_Lead_Cost", "SD_Lead_Cost"),
        "FFBridge quality boards",
    )
    declarer = pl.col("Declarer_Direction")
    if rating_type == "Players":
        _require_columns(columns, tuple(f"Player_ID_{s}" for s in "NESW"), "FFBridge quality boards")
        entity = pl.coalesce(
            *[
                pl.when(declarer == d).then(pl.col(f"Player_ID_{seat}").cast(pl.Utf8))
                for d, seat in _FF_LEADER_FOR_DECLARER.items()
            ]
        )
    elif rating_type == "Pairs":
        _require_columns(columns, ("Pair_ID_NS", "Pair_ID_EW"), "FFBridge quality boards")
        entity = (
            pl.when(declarer.is_in(["E", "W"]))
            .then(pl.col("Pair_ID_NS").cast(pl.Utf8))
            .when(declarer.is_in(["N", "S"]))
            .then(pl.col("Pair_ID_EW").cast(pl.Utf8))
        )
    else:
        raise ValueError(f"Unknown rating_type {rating_type!r}")
    diff = pl.col("SD_Lead_Cost").cast(pl.Float64) - pl.col("DD_Lead_Cost").cast(pl.Float64)
    return (
        boards.select(entity.alias("entity_id"), diff.alias("_diff"))
        .filter(pl.col("entity_id").is_not_null() & pl.col("_diff").is_not_null())
        .group_by("entity_id")
        .agg(
            pl.len().cast(pl.Int64).alias("lead_n"),
            pl.col("_diff").mean().alias("lead_mean"),
            pl.col("_diff").var().alias("lead_var"),
        )
        .collect()
    )


def ffbridge_guess_stats(club_boards: pl.LazyFrame, rating_type: str) -> pl.DataFrame:
    """Guess statistics from the FFBridge club board results (``ffbridge_club_board_results_augmented.parquet``)."""
    columns = club_boards.collect_schema().names()
    _require_columns(
        columns,
        ("Declarer", "Dummy", "Tricks", "DD_Tricks", "SC_Guess_Declarer"),
        "FFBridge club board results",
    )
    if rating_type == "Players":
        entity = pl.col("Declarer").cast(pl.Utf8)
    elif rating_type == "Pairs":
        entity = _ff_pair_id("Declarer", "Dummy")
    else:
        raise ValueError(f"Unknown rating_type {rating_type!r}")
    x = pl.col("SC_Guess_Declarer").cast(pl.Float64)
    y = pl.col("Tricks").cast(pl.Float64) - pl.col("DD_Tricks").cast(pl.Float64)
    return (
        club_boards.select(entity.alias("entity_id"), x.alias("_x"), y.alias("_y"))
        .filter(
            pl.col("entity_id").is_not_null()
            & pl.col("_x").is_not_null()
            & pl.col("_y").is_not_null()
        )
        .group_by("entity_id")
        .agg(
            pl.len().cast(pl.Int64).alias("guess_n"),
            ((pl.col("_x") - pl.col("_x").mean()) ** 2).sum().alias("guess_sxx"),
            ((pl.col("_y") - pl.col("_y").mean()) ** 2).sum().alias("guess_syy"),
            ((pl.col("_x") - pl.col("_x").mean()) * (pl.col("_y") - pl.col("_y").mean()))
            .sum()
            .alias("guess_sxy"),
        )
        .collect()
    )


def ffbridge_home_stats(results: pl.DataFrame, rating_type: str) -> pl.DataFrame:
    """Home-versus-away statistics from the FFBridge Elo session results (table ``self``)."""
    _require_columns(
        results.columns,
        ("club_code", "National_Scratch_Pct", "Club_Scratch_Pct"),
        "FFBridge results",
    )
    result = pl.coalesce(
        pl.col("National_Scratch_Pct").cast(pl.Float64),
        pl.col("Club_Scratch_Pct").cast(pl.Float64),
    ).alias("result")
    club = pl.col("club_code").cast(pl.Utf8).alias("club")
    if rating_type == "Players":
        _require_columns(results.columns, ("player1_id", "player2_id"), "FFBridge results")
        sessions = pl.concat(
            [
                results.select(pl.col(c).cast(pl.Utf8).alias("entity_id"), club, result)
                for c in ("player1_id", "player2_id")
            ]
        )
    elif rating_type == "Pairs":
        _require_columns(results.columns, ("pair_id",), "FFBridge results")
        sessions = results.select(pl.col("pair_id").cast(pl.Utf8).alias("entity_id"), club, result)
    else:
        raise ValueError(f"Unknown rating_type {rating_type!r}")
    sessions = sessions.drop_nulls()
    home = (
        sessions.group_by("entity_id", "club")
        .agg(pl.len().alias("_sessions"))
        .sort("entity_id", "_sessions", "club", descending=[False, True, False])
        .unique(subset="entity_id", keep="first", maintain_order=True)
        .select("entity_id", pl.col("club").alias("_home"))
    )
    at_home = pl.col("club") == pl.col("_home")
    return (
        sessions.join(home, on="entity_id", how="inner")
        .group_by("entity_id")
        .agg(
            at_home.sum().cast(pl.Int64).alias("home_n"),
            pl.col("result").filter(at_home).mean().alias("home_mean"),
            pl.col("result").filter(at_home).var().alias("home_var"),
            (~at_home).sum().cast(pl.Int64).alias("away_n"),
            pl.col("result").filter(~at_home).mean().alias("away_mean"),
            pl.col("result").filter(~at_home).var().alias("away_var"),
        )
    )


def ffbridge_foresight_stats(
    *,
    results: pl.DataFrame,
    rating_type: str,
    quality_boards_path: Optional[Path],
    club_boards_path: Optional[Path],
    session_ids: Optional[list[str]] = None,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Sufficient statistics for FFBridge and a per-metric status.

    ``session_ids`` limits the board-level metrics to those sessions (the
    leaderboard's filtered population); None uses every session.
    """
    frames: dict[str, Optional[pl.DataFrame]] = {}
    status: dict[str, str] = {}

    def scan(path: Path) -> pl.LazyFrame:
        frame = pl.scan_parquet(path)
        if session_ids is None:
            return frame
        return frame.filter(pl.col("session_id").cast(pl.Utf8).is_in(session_ids))

    if quality_boards_path is None or not Path(quality_boards_path).is_file():
        frames["lead"] = None
        status["lead"] = "unavailable: FFBridge quality boards parquet not found"
    else:
        frames["lead"] = ffbridge_lead_stats(scan(Path(quality_boards_path)), rating_type)
        status["lead"] = "ok"
    if club_boards_path is None or not Path(club_boards_path).is_file():
        frames["guess"] = None
        status["guess"] = (
            "unavailable: ffbridge_club_board_results_augmented.parquet not found "
            "(set FFBRIDGE_STATS_CLUB_BOARD_RESULTS)"
        )
    else:
        frames["guess"] = ffbridge_guess_stats(scan(Path(club_boards_path)), rating_type)
        status["guess"] = "ok"
    frames["home"] = ffbridge_home_stats(results, rating_type)
    status["home"] = "ok"
    return combine_stats(frames), status

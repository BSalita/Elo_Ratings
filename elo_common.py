# elo_common.py
"""
Common utilities shared between ACBL and FFBridge Elo rating apps.

Contains:
- Elo rating constants and calculation functions
- Chess-range scaling (so ~2800 is approx top Elo)
- Chess-style title mapping (SGM, GM, IM, FM, CM, etc.)
- Polars expression helpers for scaling and titles
- Common theme CSS for Streamlit apps
- Common UI helpers (AgGrid sizing, header/footer)
"""

import os
import math
import pathlib
import sys
import threading
import html
from datetime import datetime
import polars as pl


# -------------------------------
# Elo Rating Constants
# -------------------------------
DEFAULT_ELO = 1200.0
K_FACTOR = 32.0
PERFORMANCE_SCALING = 400  # Standard Elo scaling factor

# -------------------------------
# Field-strength K-dampening (the "weak-field inflation" fix)
# -------------------------------
# FFBridge Elo is already field-relative: the expected score is computed
# against the field's *current* mean rating, so a player who rises above a
# weak field gains progressively less. What that does NOT close is the
# closed-weak-club channel: when every pair in an isolated club starts at
# DEFAULT_ELO, the field mean tracks the local population (not the global
# one), so a consistently high-percentage pair can inflate the whole club
# together. These anchors damp the K-factor when a session's field mean is
# below the GLOBAL population mean (DEFAULT_ELO), capping per-session gains
# in absolutely-weak fields.
#
# Anchors are in z-score units: z = (field_mean - population_mean) / sigma.
# Kept deliberately conservative vs. ACBL because FFBridge's IV_Bonus already
# nudges percentages for field difficulty (avoid double-counting).
FIELD_STRENGTH_Z_FULL = 0.0     # full K at/above population mean (z >= 0)
FIELD_STRENGTH_Z_FLOOR = -2.0   # k_min at this many sigmas below population mean
FIELD_STRENGTH_K_MIN = 0.7      # floor on the K multiplier (conservative)
# Spread of per-session field-mean ratings around DEFAULT_ELO. Field means are
# averages of pair ratings (each pair = mean of two player ratings), so their
# spread is far tighter than individual ratings. Calibrated empirically via
# calibrate_ffbridge_field_strength.py over the full classic cache (517 sessions,
# 2026-05): scratch sigma=5.6, handicap sigma=6.0, field means span ~1192-1229.
# Using the larger (~6) keeps dampening conservative; at this sigma a field mean
# of ~1188 hits the K floor and ~1194 gives K x0.85 (z_floor=-2). Re-run the
# calibration script after large data refreshes to confirm.
FFBRIDGE_FIELD_STRENGTH_SIGMA = 6.0


def field_strength_scale(
    z: float,
    z_full: float = FIELD_STRENGTH_Z_FULL,
    z_floor: float = FIELD_STRENGTH_Z_FLOOR,
    k_min: float = FIELD_STRENGTH_K_MIN,
) -> float:
    """K-dampening multiplier in ``[k_min, 1.0]`` given a field-mean z-score.

    ``z = (field_mean_rating - population_mean) / sigma``.

    Fields at or above the population mean (``z >= z_full``) get full K. As the
    field mean drifts below the population mean the multiplier decays linearly
    to ``k_min`` by ``z_floor``, capping the Elo a weak-field player can
    accumulate per session even when routinely scoring above their own (weak)
    field. NaN input (degenerate field) returns full K.
    """
    if math.isnan(z):
        return 1.0
    if z >= z_full:
        return 1.0
    if z <= z_floor:
        return k_min
    return k_min + (1.0 - k_min) * (z - z_floor) / (z_full - z_floor)


def field_strength_scale_from_mean(
    field_mean_rating: float,
    population_mean: float = DEFAULT_ELO,
    sigma: float = FFBRIDGE_FIELD_STRENGTH_SIGMA,
) -> float:
    """Convenience wrapper: convert a field mean rating to a K multiplier."""
    if sigma <= 0 or field_mean_rating is None or math.isnan(field_mean_rating):
        return 1.0
    z = (field_mean_rating - population_mean) / sigma
    return field_strength_scale(z)

# Chess federation scaling parameters
# Target: Top players at ~2800 (Magnus Carlsen level), beginners at ~1200
CHESS_SCALING_ENABLED = True
CHESS_TARGET_MIN = 1200.0
CHESS_TARGET_MAX = 2800.0
CURRENT_REFERENCE_MIN = 1200.0
CURRENT_REFERENCE_MAX = 1600.0

# UI Constants
AGGRID_ROW_HEIGHT = 42
AGGRID_HEADER_HEIGHT = 50
AGGRID_FOOTER_HEIGHT = 20
AGGRID_MAX_DISPLAY_ROWS = 10

# Branding
ASSISTANT_LOGO_URL = 'https://github.com/BSalita/Elo_Ratings/blob/master/assets/logo_assistant.gif?raw=true'


# -------------------------------
# Elo Rating Calculation
# -------------------------------
def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for player A against player B.

    Uses the standard Elo formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Includes overflow protection for extreme rating differences.
    """
    rating_diff = (rating_b - rating_a) / PERFORMANCE_SCALING
    rating_diff = max(-10.0, min(10.0, rating_diff))
    try:
        return 1.0 / (1.0 + 10 ** rating_diff)
    except OverflowError:
        return 0.0 if rating_diff > 0 else 1.0


def calculate_elo_from_percentage(
    current_rating: float,
    percentage: float,
    field_average_rating: float,
    k_factor: float = K_FACTOR,
    field_strength_scale: float = 1.0,
) -> float:
    """
    Calculate new Elo rating based on percentage score.

    Uses percentage as the actual performance and compares to expected
    performance based on rating difference from field average.

    ``field_strength_scale`` (in ``(0, 1]``) damps the effective K-factor when
    the session field is weak relative to the global population mean; see
    :func:`field_strength_scale`. Defaults to ``1.0`` (no damping).
    """
    actual_score = percentage / 100.0
    expected_score = calculate_expected_score(current_rating, field_average_rating)
    return current_rating + k_factor * field_strength_scale * (actual_score - expected_score)


def scale_to_chess_range(rating: float) -> float:
    """
    Scale Elo rating to chess federation range.

    Maps ratings so that top players reach Magnus Carlsen level and beyond.
    Uses a simple linear scaling: rating * 2.0 (e.g., 1200 -> 2400, 1400 -> 2800).

    Includes bounds checking to prevent extreme values that could cause overflow errors.

    Args:
        rating: Current Elo rating

    Returns:
        Scaled Elo rating (bounded to reasonable range to prevent overflow)
    """
    if not CHESS_SCALING_ENABLED:
        return rating

    max_input_rating = 10000.0
    min_input_rating = -1000.0
    rating = max(min_input_rating, min(max_input_rating, rating))

    scaled = rating * 2.0

    max_output_rating = 3500.0
    min_output_rating = 0.0
    scaled = max(min_output_rating, min(max_output_rating, scaled))

    return round(scaled, 1)


# -------------------------------
# Chess-Style Titles
# -------------------------------
def get_elo_title(rating: float) -> str:
    """
    Get chess-style title based on Elo rating.

    Based on FIDE/chess federation standards:
    - Super Grandmaster (SGM): 2600+
    - Grandmaster (GM): 2500+
    - International Master (IM): 2400-2499
    - FIDE Master (FM): 2300-2399
    - Candidate Master (CM): 2200-2299
    - Expert: 2000-2199
    - Advanced: 1800-1999
    - Intermediate: 1600-1799
    - Novice: 1400-1599
    - Beginner: Below 1400
    """
    if rating >= 2600:
        return "SGM"
    elif rating >= 2500:
        return "GM"
    elif rating >= 2400:
        return "IM"
    elif rating >= 2300:
        return "FM"
    elif rating >= 2200:
        return "CM"
    elif rating >= 2000:
        return "Expert"
    elif rating >= 1800:
        return "Advanced"
    elif rating >= 1600:
        return "Intermediate"
    elif rating >= 1400:
        return "Novice"
    else:
        return "Beginner"


# SQL fragment for computing title from an Elo column.
# Usage: ELO_TITLE_SQL_CASE.format(elo_col="Player_Elo_Score")
ELO_TITLE_SQL_CASE = """CASE
    WHEN {elo_col} >= 2600 THEN 'SGM'
    WHEN {elo_col} >= 2500 THEN 'GM'
    WHEN {elo_col} >= 2400 THEN 'IM'
    WHEN {elo_col} >= 2300 THEN 'FM'
    WHEN {elo_col} >= 2200 THEN 'CM'
    WHEN {elo_col} >= 2000 THEN 'Expert'
    WHEN {elo_col} >= 1800 THEN 'Advanced'
    WHEN {elo_col} >= 1600 THEN 'Intermediate'
    WHEN {elo_col} >= 1400 THEN 'Novice'
    ELSE 'Beginner'
END"""

# SQL fragment for scaling an Elo column to chess range.
# Usage: ELO_SCALE_SQL.format(elo_col="Elo_R_Player")
ELO_SCALE_SQL = "CAST(LEAST(GREATEST(ROUND({elo_col} * 2.0), 0), 3500) AS INTEGER)"


# -------------------------------
# Unified z-score -> chess scale (aligns ACBL, FFBridge, and chess titles)
# -------------------------------
# Each system's native Elo lives on a different center (FFBridge ~1200,
# ACBL ~1500) with a different spread, so a fixed multiplier cannot make titles
# mean the same thing across systems. Instead we standardize each leaderboard's
# ratings to a common chess-anchored normal:  display = MEAN + z * SD, where
# z = (rating - pop_mean) / pop_sd is computed over that leaderboard's own
# population. The shared title ladder (ELO_TITLE_SQL_CASE / get_elo_title) then
# labels identical *percentiles* in every system. SD controls title rarity:
# at SD=400 with MEAN=1500, IM(2400)~+2.25sd (~top 1%), SGM(2600)~+2.75sd
# (~top 0.3%). See get_elo_title for the bands.
CHESS_DISPLAY_MEAN = 1500.0
CHESS_DISPLAY_SD = 400.0


def zscore_to_chess(
    rating: float,
    pop_mean: float,
    pop_sd: float,
    mean: float = CHESS_DISPLAY_MEAN,
    sd: float = CHESS_DISPLAY_SD,
) -> float:
    """Standardize a native Elo to the common chess-anchored scale, clamped 0-3500.

    ``z = (rating - pop_mean) / pop_sd`` then ``mean + z * sd``. A degenerate
    population (``pop_sd <= 0``) maps everything to ``mean``.
    """
    if rating is None or math.isnan(rating):
        return mean
    if pop_sd is None or pop_sd <= 0 or math.isnan(pop_sd):
        return mean
    z = (rating - pop_mean) / pop_sd
    scaled = mean + z * sd
    return round(max(0.0, min(3500.0, scaled)), 1)


def zscore_chess_sql(
    elo_col: str,
    pop_mean_sql: str,
    pop_sd_sql: str,
    mean: float = CHESS_DISPLAY_MEAN,
    sd: float = CHESS_DISPLAY_SD,
) -> str:
    """SQL expression standardizing ``elo_col`` to the chess scale (integer, 0-3500).

    ``pop_mean_sql`` / ``pop_sd_sql`` are SQL scalar expressions (e.g. references
    to a CROSS JOINed stats CTE) giving the population mean and population stdev.
    A non-positive/NULL stdev collapses to ``mean`` so the column never errors.
    """
    return (
        f"CAST(LEAST(GREATEST(ROUND("
        f"CASE WHEN COALESCE({pop_sd_sql}, 0) <= 0 THEN {mean} "
        f"ELSE {mean} + ((CAST({elo_col} AS DOUBLE) - ({pop_mean_sql})) "
        f"/ ({pop_sd_sql})) * {sd} END"
        f"), 0), 3500) AS INTEGER)"
    )


# -------------------------------
# Polars Expression Helpers
# -------------------------------
def scale_elo_expr(col_name: str, scale_factor: float = 2.0) -> pl.Expr:
    """Return a Polars expression that scales an Elo column to chess range, clamped 0–3500."""
    return (
        pl.col(col_name)
        .cast(pl.Float64)
        .clip(-1000.0, 10000.0)
        .mul(scale_factor)
        .clip(0.0, 3500.0)
        .round(0)
        .cast(pl.Int32)
        .alias(col_name)
    )


def title_from_elo_expr(elo_col: str, title_col: str = "Title") -> pl.Expr:
    """Return a Polars expression that maps a (scaled) Elo column to a chess-style title."""
    return (
        pl.when(pl.col(elo_col).cast(pl.Float64) >= 2600).then(pl.lit("SGM"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 2500).then(pl.lit("GM"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 2400).then(pl.lit("IM"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 2300).then(pl.lit("FM"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 2200).then(pl.lit("CM"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 2000).then(pl.lit("Expert"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 1800).then(pl.lit("Advanced"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 1600).then(pl.lit("Intermediate"))
        .when(pl.col(elo_col).cast(pl.Float64) >= 1400).then(pl.lit("Novice"))
        .otherwise(pl.lit("Beginner"))
        .alias(title_col)
    )


def post_process_elo_table(table_df: pl.DataFrame, elo_col: str, scale_factor: float = 2.0) -> pl.DataFrame:
    """
    Post-process a ranking table:
      1. Scale the Elo column to chess range (clamped 0–3500).
      2. Add a 'Title' column right after the Elo column.

    Args:
        table_df: Polars DataFrame with ranking results.
        elo_col: Name of the Elo score column to scale.
        scale_factor: Multiplier for scaling (default 2.0 for FFBridge, 1.6 for ACBL).

    Works on the output of both Polars and SQL engines.
    """
    if not isinstance(table_df, pl.DataFrame) or table_df.is_empty() or elo_col not in table_df.columns:
        return table_df

    cols = list(table_df.columns)
    elo_idx = cols.index(elo_col)

    # Scale Elo
    table_df = table_df.with_columns(scale_elo_expr(elo_col, scale_factor))

    # Add Title
    table_df = table_df.with_columns(title_from_elo_expr(elo_col, "Title"))

    # Reorder: insert Title right after the Elo column
    new_cols = cols[: elo_idx + 1] + ["Title"] + cols[elo_idx + 1 :]
    table_df = table_df.select(new_cols)

    return table_df


# -------------------------------
# AgGrid Height Helper
# -------------------------------
def calculate_aggrid_height(row_count: int) -> int:
    """Calculate AgGrid height based on row count."""
    display_rows = min(AGGRID_MAX_DISPLAY_ROWS, row_count)
    return AGGRID_HEADER_HEIGHT + (AGGRID_ROW_HEIGHT * display_rows) + AGGRID_FOOTER_HEIGHT


# -------------------------------
# AgGrid Numeric Column Helper
# -------------------------------

# Column-name conventions used across the ACBL + FFBridge reports. A column
# matching one of these patterns is assumed to hold numeric data (integer or
# float) and must sort numerically in the AgGrid, not alphabetically (which
# would put "10" before "2"). Defensive against JSON round-trips that leave
# nullable INTEGER columns as pandas object dtype.
_NUMERIC_NAME_SUFFIXES: tuple[str, ...] = (
    "_Rank", "_Score", "_Raw", "_Published",
    "_Pct", "_Rate", "_Rate_Pct", "_Avg", "_Bonus", "_Stdev",
    "_Points", "_Played", "_Diff", "_MPs",
)
_NUMERIC_NAME_EXACT: frozenset[str] = frozenset({
    "Rank", "Sessions", "Sessions_Played", "Games",
    "MasterPoints", "MasterPoint_Rank",
    "Avg_MPs", "Geo_MPs",
    "Quality_Rank", "Quality_Score",
    "Player_Elo", "HC_Player_Elo", "Pair_Elo", "HC_Pair_Elo",
    "Elo", "Elo_Before", "Elo_After", "Elo_Delta",
    "DD_Tricks_Diff_Avg",
})


def is_numeric_column_name(col_name: str) -> bool:
    """True if a column name suggests numeric values (suffix or exact match).

    Used by :func:`coerce_numeric_columns` so we can force numeric dtype before
    the AgGrid is built. Any column ending in ``_Rank``, ``_Score``, ``_Raw``,
    ``_Published``, ``_Pct``, ``_Rate``, ``_Avg``, ``_Bonus``, ``_Stdev``,
    ``_Points``, ``_Played``, ``_Diff``, or ``_MPs`` qualifies, as does the
    explicit exact-match set above. Column names like ``Player_ID`` /
    ``Pair_IDs`` / ``Player_Name`` are deliberately excluded — those are
    sortable as strings.
    """
    if col_name in _NUMERIC_NAME_EXACT:
        return True
    return col_name.endswith(_NUMERIC_NAME_SUFFIXES)


def coerce_numeric_columns(pdf):
    """Coerce columns whose names suggest numeric values to numeric dtype.

    Mutates ``pdf`` in place AND returns it for chaining. Non-convertible
    values become NaN. Skips columns that are already numeric. ``pdf`` must
    be a pandas DataFrame (kept untyped here to avoid forcing pandas as a
    top-level import for callers that don't need it).
    """
    import pandas as _pd  # local import: keep elo_common's top-level deps minimal
    for col in pdf.columns:
        if not is_numeric_column_name(col):
            continue
        if _pd.api.types.is_numeric_dtype(pdf[col]):
            continue
        pdf[col] = _pd.to_numeric(pdf[col], errors="coerce")
    return pdf


# -------------------------------
# URL Query Parameter Sync Helpers
# -------------------------------
def coerce_int(min_value: int | None = None, max_value: int | None = None, step: int | None = None):
    """Build a parser that coerces a string to int, optionally clamped and snapped to step.

    Snapping is anchored at ``min_value`` (or 0 if not provided).
    """
    def _parse(raw: str) -> int:
        v = int(raw)
        if min_value is not None:
            v = max(min_value, v)
        if max_value is not None:
            v = min(max_value, v)
        if step is not None and step > 0:
            anchor = min_value if min_value is not None else 0
            v = anchor + round((v - anchor) / step) * step
            if min_value is not None:
                v = max(min_value, v)
            if max_value is not None:
                v = min(max_value, v)
        return v
    return _parse


def init_url_params_to_state(
    st_module,
    params_config: dict,
    init_flag_key: str = "_url_params_initialized",
) -> None:
    """Read URL query params into ``st.session_state`` on first script run only.

    On subsequent runs, widgets are the source of truth and
    ``sync_state_to_url_params`` pushes their values back into the URL.

    ``params_config`` maps URL keys to dicts with:
      - ``session_key`` (required): the ``st.session_state`` key used by the widget.
      - ``parser`` (optional): callable to convert the URL string (default: ``str``).
      - ``valid_values`` (optional): iterable; the parsed value must be in it.
    """
    if st_module.session_state.get(init_flag_key, False):
        return
    qp = st_module.query_params
    for url_key, cfg in params_config.items():
        if url_key not in qp:
            continue
        session_key = cfg["session_key"]
        parser = cfg.get("parser", str)
        try:
            parsed = parser(qp[url_key])
        except (ValueError, TypeError):
            continue
        valid_values = cfg.get("valid_values")
        if valid_values is not None and parsed not in valid_values:
            continue
        st_module.session_state[session_key] = parsed
    st_module.session_state[init_flag_key] = True


def sync_state_to_url_params(
    st_module,
    params_config: dict,
) -> None:
    """Write current session_state values to URL query params after widgets render.

    A param is omitted from the URL when its session_state value equals the
    configured ``default`` (or is empty/None), keeping shareable URLs concise.
    """
    qp = st_module.query_params
    for url_key, cfg in params_config.items():
        session_key = cfg["session_key"]
        default = cfg.get("default")
        if session_key not in st_module.session_state:
            if url_key in qp:
                del qp[url_key]
            continue
        val = st_module.session_state[session_key]
        is_empty = (val is None) or (isinstance(val, str) and val == "")
        is_default = (default is not None) and (val == default)
        if is_empty or is_default:
            if url_key in qp:
                del qp[url_key]
            continue
        str_val = str(val)
        if qp.get(url_key) != str_val:
            qp[url_key] = str_val


# -------------------------------
# Common Theme CSS
# -------------------------------
APP_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background-color: #004d40;
    color: #f5f5f5;
}

/* Hide/style the Streamlit header bar */
header[data-testid="stHeader"] {
    background-color: #004d40 !important;
}

/* Style header toolbar buttons consistently */
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] [data-testid="stStatusWidget"],
header[data-testid="stHeader"] span,
header[data-testid="stHeader"] a {
    color: #f5f5f5 !important;
}

/* Style the running man icon white */
header[data-testid="stHeader"] svg,
header[data-testid="stHeader"] [data-testid="stStatusWidget"] svg {
    fill: #f5f5f5 !important;
    color: #f5f5f5 !important;
}

h1, h2, h3 {
    color: #ffc107 !important;
    font-weight: 700;
    letter-spacing: 1px;
}

.stSidebar {
    background-color: #00332e !important;
    border-right: 1px solid #00695c;
}

.stSidebar .stMarkdown, 
.stSidebar label:not(.stRadio label):not(.stSelectbox label):not(.stTextInput label):not(.stSlider label):not(.stNumberInput label) {
    color: #e0e0e0 !important;
}

.metric-card {
    background: rgba(0, 105, 92, 0.4);
    border: 1px solid #00796b;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.stDataFrame {
    background-color: white;
    border-radius: 8px;
}

.stButton > button {
    background-color: #ffc107;
    color: #004d40;
    border: none;
    border-radius: 5px;
    font-weight: 700;
    padding: 0.5rem 1rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #ffca28;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(255,193,7,0.3);
}

.stDownloadButton > button {
    background-color: #00695c !important;
    color: #ffc107 !important;
    border: 2px solid #ffc107 !important;
    border-radius: 5px;
    font-weight: 700;
}

.stDownloadButton > button:hover {
    background-color: #00796b !important;
    color: #ffca28 !important;
    border-color: #ffca28 !important;
}

/* Override sidebar label color for specific widget types */
.stSidebar .stRadio label,
.stSidebar .stRadio > label,
.stSidebar div[data-testid="stRadio"] > label,
.stSidebar .stRadio label p,
.stSidebar .stRadio label span { 
    color: #ffc107 !important; 
    font-weight: 600 !important; 
}
/* Radio button option text */
.stRadio [data-testid="stMarkdownContainer"] p { 
    color: #ffffff !important; 
    font-size: 1rem !important; 
    font-weight: 500 !important; 
}
.stSidebar .stSelectbox label,
.stSidebar .stSelectbox > label { 
    color: #ffc107 !important; 
    font-weight: 600 !important; 
}
.stSidebar .stTextInput label,
.stSidebar .stTextInput > label { 
    color: #ffc107 !important; 
    font-weight: 600 !important; 
}
.stSidebar .stSlider label,
.stSidebar .stSlider > label { 
    color: #ffc107 !important; 
    font-weight: 600 !important; 
}
.stSidebar .stNumberInput label,
.stSidebar .stNumberInput > label { 
    color: #ffc107 !important; 
    font-weight: 600 !important; 
}
.stCheckbox label span, .stCheckbox label p, .stCheckbox [data-testid="stMarkdownContainer"] p { 
    color: #ffffff !important; 
    font-weight: 500 !important; 
}

/* Style st.info / st.success / st.warning / st.error to match theme */
.stAlert [data-testid="stMarkdownContainer"] p,
.stAlert [data-testid="stMarkdownContainer"] {
    color: #ffc107 !important;
}
.stAlert {
    background-color: rgba(0, 105, 92, 0.4) !important;
    border: 1px solid #00796b !important;
    color: #ffc107 !important;
}
</style>
"""


def apply_app_theme(st_module) -> None:
    """Apply the common dark green/gold theme CSS to a Streamlit app."""
    st_module.markdown(APP_THEME_CSS, unsafe_allow_html=True)


def get_memory_usage_line() -> str:
    """
    Return a formatted memory usage line for footer display.

    Prefers Linux cgroup limits/usage (useful in containers like Railway),
    then falls back to psutil host metrics when cgroup values are unavailable.
    """
    try:
        import psutil

        def _gb(v: int) -> float:
            return v / (1024 ** 3)

        ram_used = ram_total = None
        swap_used = swap_total = None

        # Linux cgroup v2 paths (container-aware)
        try:
            if os.path.exists("/sys/fs/cgroup/memory.current") and os.path.exists("/sys/fs/cgroup/memory.max"):
                with open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8") as f:
                    ram_used = int(f.read().strip())
                with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as f:
                    max_raw = f.read().strip()
                ram_total = None if max_raw == "max" else int(max_raw)

                if os.path.exists("/sys/fs/cgroup/memory.swap.current"):
                    with open("/sys/fs/cgroup/memory.swap.current", "r", encoding="utf-8") as f:
                        swap_used = int(f.read().strip())
                if os.path.exists("/sys/fs/cgroup/memory.swap.max"):
                    with open("/sys/fs/cgroup/memory.swap.max", "r", encoding="utf-8") as f:
                        smax_raw = f.read().strip()
                    swap_total = None if smax_raw == "max" else int(smax_raw)
        except Exception:
            ram_used = ram_total = swap_used = swap_total = None

        if ram_used is None or ram_total is None or ram_total <= 0:
            vm = psutil.virtual_memory()
            ram_used, ram_total = vm.used, vm.total
        ram_pct = (ram_used / ram_total * 100.0) if ram_total else 0.0

        if swap_used is None or swap_total is None:
            sm = psutil.swap_memory()
            swap_used, swap_total = sm.used, sm.total

        if swap_total and swap_total > 0:
            swap_pct = (swap_used / swap_total * 100.0)
            swap_text = f"Virtual/Pagefile {_gb(swap_used):.2f}/{_gb(swap_total):.2f} GB ({swap_pct:.1f}%)"
        else:
            swap_text = "Virtual/Pagefile N/A (swap disabled)"

        cpu_count = os.cpu_count() or 0
        thread_count = threading.active_count()
        return (
            f"Memory: RAM {_gb(ram_used):.2f}/{_gb(ram_total):.2f} GB ({ram_pct:.1f}%) • "
            f"{swap_text} • CPU/Threads {cpu_count}/{thread_count}"
        )
    except Exception:
        return "Memory: RAM/Virtual usage unavailable"


def get_cache_diagnostic_line(cache_dir_env_var: str = "FFBRIDGE_CACHE_DIR") -> str:
    """Return cache diagnostics: resolved dir, exists/writable, and file count."""
    cache_dir_raw = os.getenv(cache_dir_env_var, "").strip()
    if not cache_dir_raw:
        return f"Cache ({cache_dir_env_var}): dir=not set • exists=False • writable=False • files=0"

    cache_dir = pathlib.Path(cache_dir_raw).expanduser()
    cache_dir_resolved = cache_dir.resolve(strict=False)
    exists = cache_dir_resolved.exists() and cache_dir_resolved.is_dir()
    writable = os.access(cache_dir_resolved, os.W_OK) if exists else False
    file_count = 0
    if exists:
        try:
            for _root, _dirs, files in os.walk(cache_dir_resolved):
                file_count += len(files)
        except OSError:
            file_count = -1

    file_count_text = str(file_count) if file_count >= 0 else "unavailable"
    return (
        f"Cache ({cache_dir_env_var}): dir={cache_dir_resolved} • "
        f"exists={exists} • writable={writable} • files={file_count_text}"
    )


def render_app_footer(
    st_module,
    endplay_version: str,
    source_line: str | None = None,
    dependency_versions: dict[str, str] | None = None,
) -> None:
    """Render the common footer used by both Streamlit apps."""
    dependency_versions = dependency_versions or {}
    pandas_version = dependency_versions.get("pandas", "N/A")
    polars_version = dependency_versions.get("polars", pl.__version__)
    duckdb_version = dependency_versions.get("duckdb", "N/A")
    memory_line = get_memory_usage_line()
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Build the centred info block: project credit + versions + memory + date all in one div.
    source_html = f"{html.escape(source_line)}<br>" if source_line else ""
    st_module.markdown(
        f"""
        <div style="text-align: center; color: #80cbc4; font-size: 0.8rem; opacity: 0.7;">
            Project lead is Robert Salita research@AiPolice.org. Code written in Python by Cursor AI. UI written in streamlit. Data engine is polars. Repo: <a href="https://github.com/BSalita/Elo_Ratings" target="_blank" style="color: #80cbc4;">github.com/BSalita/Elo_Ratings</a><br>
            Query Params:{st_module.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}<br>
            Streamlit:{st_module.__version__} Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pandas_version} polars:{polars_version} duckdb:{duckdb_version} endplay:{endplay_version}<br>
            {source_html}{html.escape(memory_line)}<br>
            System Current Date: {html.escape(date_str)}
        </div>
        """,
        unsafe_allow_html=True,
    )

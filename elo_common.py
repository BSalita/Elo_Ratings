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
import sys
from datetime import datetime
import polars as pl


# -------------------------------
# Elo Rating Constants
# -------------------------------
DEFAULT_ELO = 1200.0
K_FACTOR = 32.0
PERFORMANCE_SCALING = 400  # Standard Elo scaling factor

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
) -> float:
    """
    Calculate new Elo rating based on percentage score.

    Uses percentage as the actual performance and compares to expected
    performance based on rating difference from field average.
    """
    actual_score = percentage / 100.0
    expected_score = calculate_expected_score(current_rating, field_average_rating)
    return current_rating + k_factor * (actual_score - expected_score)


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

        return (
            f"Memory: RAM {_gb(ram_used):.2f}/{_gb(ram_total):.2f} GB ({ram_pct:.1f}%) • "
            f"{swap_text}"
        )
    except Exception:
        return "Memory: RAM/Virtual usage unavailable"


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
    st_module.markdown(
        f"""
        <div style="text-align: center; color: #80cbc4; font-size: 0.8rem; opacity: 0.7;">
            Project lead is Robert Salita research@AiPolice.org. Code written in Python by Cursor AI. UI written in streamlit. Data engine is polars. Repo: <a href="https://github.com/BSalita/Elo_Ratings" target="_blank" style="color: #80cbc4;">github.com/BSalita/Elo_Ratings</a><br>
            Query Params:{st_module.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}<br>
            Streamlit:{st_module.__version__} Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pandas_version} polars:{polars_version} duckdb:{duckdb_version} endplay:{endplay_version}<br>
            {memory_line}<br>
            System Current Date: {datetime.now().strftime('%Y-%m-%d')}
        </div>
    """,
        unsafe_allow_html=True,
    )

    if source_line:
        st_module.caption(source_line)

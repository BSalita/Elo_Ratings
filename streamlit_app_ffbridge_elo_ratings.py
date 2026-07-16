# streamlit_app_ffbridge_elo.py
#
# streamlit==1.59.2 (see requirements.txt). Older 1.53.x segfaulted on Linux
# remounts; confirm AgGrid still paints (historical blank-frame risk >=1.56).
#
"""
FFBridge Elo Ratings - Unified Streamlit Application

This app fetches duplicate bridge tournament results from FFBridge and calculates
Elo ratings based on percentage scores.

Supports both:
- Classic API (api.ffbridge.fr) - requires authentication
- Lancelot API (api-lancelot.ffbridge.fr) - public access
"""

import json
import math
import os

# Prevent Intel Fortran runtime (libifcoremd.dll / MKL) from installing its own
# Ctrl+C handler that crashes with "forrtl: error (200)".
# Must be set before any numpy/scipy/MKL imports.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import pathlib
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

# On Windows, install a process-level console control handler via ctypes
# for immediate clean exit on Ctrl+C. Works from any thread (unlike signal.signal).
if sys.platform == "win32":
    try:
        import ctypes
        _kernel32 = ctypes.windll.kernel32
        _CTRL_C_EVENT = 0
        _CTRL_BREAK_EVENT = 1
        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def _console_ctrl_handler(event):
            if event in (_CTRL_C_EVENT, _CTRL_BREAK_EVENT):
                os._exit(0)
            return False
        _kernel32.SetConsoleCtrlHandler(_console_ctrl_handler, True)
    except (AttributeError, OSError) as exc:
        sys.stderr.write(f"[ffbridge] console ctrl handler setup skipped: {exc}\n")

import pandas as pd
import polars as pl
import requests
import streamlit as st
import duckdb

from streamlitlib.streamlitlib import (
    ShowDataFrameTable,
    create_pdf,
    widen_scrollbars,
)
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    ColumnsAutoSizeMode,
    AgGridTheme,
    DataReturnMode,
    JsCode,
)

# Import common Elo utilities (shared with ACBL app)
from elo_common import (
    DEFAULT_ELO,
    K_FACTOR,
    AGGRID_ROW_HEIGHT,
    AGGRID_HEADER_HEIGHT,
    AGGRID_FOOTER_HEIGHT,
    AGGRID_MAX_DISPLAY_ROWS,
    ASSISTANT_LOGO_URL,
    calculate_expected_score,
    calculate_elo_from_percentage,
    field_strength_scale_from_mean,
    scale_to_chess_range,
    get_elo_title,
    CHESS_DISPLAY_MEAN,
    CHESS_DISPLAY_SD,
    apply_app_theme,
    calculate_aggrid_height,
    coerce_int,
    coerce_numeric_columns,
    init_url_params_to_state,
    leaderboard_aggrid_viewport_height,
    LEADERBOARD_PAGE_SIZE,
    LEADERBOARD_ROW_HEIGHT,
    render_app_footer,
    footer_streamlit_app_diagnostics_line,
    get_cache_diagnostic_line,
    sync_state_to_url_params,
)

# Import FFBridge-specific utilities
from elo_ffbridge_common import normalize_series_id

# Import API adapters
import elo_ffbridge_classic as classic_api
import elo_ffbridge_lancelot as lancelot_api

# Import for version display only
try:
    import endplay
    ENDPLAY_VERSION = endplay.__version__
except (ImportError, AttributeError):
    ENDPLAY_VERSION = "N/A"

# Available API backends
API_BACKENDS = {
    "FFBridge Classic API": classic_api,
    "FFBridge Lancelot API": lancelot_api,
}

# Directory for the persisted (precomputed) Elo dataset parquets. Building the
# full multi-tournament Elo history is expensive and memory-heavy; we compute it
# once, persist it here, and reload on subsequent (cold) starts so a restarted
# container does not rebuild everything in RAM (the OOM/restart-loop cause).
#
# In production set FFBRIDGE_CACHE_DIR to a persistent mount (e.g. /data/ffbridge)
# so the raw tournament cache and elo_cache parquet survive redeploys. Locally
# (FFBRIDGE_CACHE_DIR unset) we fall back to the app dir.
_FFBRIDGE_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_FFBRIDGE_CACHE_DIR_ENV = os.environ.get("FFBRIDGE_CACHE_DIR", "").strip()
if _FFBRIDGE_CACHE_DIR_ENV:
    _FFBRIDGE_ELO_CACHE_DIR = pathlib.Path(_FFBRIDGE_CACHE_DIR_ENV).resolve() / "elo_cache"
else:
    _FFBRIDGE_ELO_CACHE_DIR = _FFBRIDGE_SCRIPT_DIR / "data" / "ffbridge" / "elo_cache"

# Console-logging throttle for long tournament replays. Log a heartbeat every N
# tournaments plus a final summary instead of per-tournament spam.
_PROCESSING_LOG_EVERY = max(1, int(os.environ.get("FFBRIDGE_LOG_EVERY", "50")))
_FFBRIDGE_DEBUG = os.environ.get("FFBRIDGE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
_FFBRIDGE_LOAD_DEBUG = (
    os.environ.get("FFBRIDGE_LOAD_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    or _FFBRIDGE_DEBUG
    or os.environ.get("STREAMLIT_ENV", "").strip().lower() == "production"
)


def _load_debug_cgroup_summary() -> str:
    """Short cgroup memory snippet for load-phase console logs."""
    try:
        used = int(pathlib.Path("/sys/fs/cgroup/memory.current").read_text(encoding="utf-8"))
        limit_raw = pathlib.Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        if limit_raw.isdigit():
            limit = int(limit_raw)
            if limit > 0:
                return f"mem {used / 1024 ** 3:.2f}/{limit / 1024 ** 3:.2f} GB ({100 * used / limit:.0f}%)"
        return f"mem {used / 1024 ** 3:.2f} GB"
    except Exception:
        return "mem n/a"


def _load_debug_log(message: str, *, reset: bool = False) -> None:
    """Timestamped load-phase log line (enabled in production via STREAMLIT_ENV)."""
    if not _FFBRIDGE_LOAD_DEBUG:
        return
    ss = st.session_state
    now = time.perf_counter()
    if reset or "_load_debug_t0" not in ss:
        ss["_load_debug_t0"] = now
        ss["_load_debug_phase_t0"] = now
    phase_s = now - ss["_load_debug_phase_t0"]
    total_s = now - ss["_load_debug_t0"]
    print(
        f"[ffbridge][load] {message} "
        f"(+{phase_s:.1f}s, total {total_s:.1f}s, {_load_debug_cgroup_summary()})",
        flush=True,
    )
    ss["_load_debug_phase_t0"] = now


def _load_debug_log_standalone(message: str, t0: float) -> None:
    """Load log outside Streamlit session state (cache_resource / parquet read)."""
    if not _FFBRIDGE_LOAD_DEBUG:
        return
    elapsed = time.perf_counter() - t0
    print(
        f"[ffbridge][load] {message} "
        f"(+{elapsed:.1f}s, {_load_debug_cgroup_summary()})",
        flush=True,
    )


# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def _bridgeinter_fetch_text(url: str) -> str:
    """
    Fetch a BridgeInterNet page and return a whitespace-normalized text version.
    Cached because this is used only for small, occasional reconciliations.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text
    # Strip scripts/styles then tags; keep numbers/%/names.
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _bridgeinter_octopus_url(date_yyyy_mm_dd: str, classement_type: str) -> Optional[str]:
    """
    Build BridgeInterNet Octopus URL for a given date.
    Thursday sessions use octopus_j + code joYYMMDD.
    Monday sessions use octopus_l + code loYYMMDD.
    """
    try:
        d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").date()
    except Exception:
        return None

    yymmdd = d.strftime("%y%m%d")
    # Monday=0, Thursday=3
    if d.weekday() == 3:
        base_path = "octopus_j/resseance_j.php"
        session_code = f"jo{yymmdd}"
    elif d.weekday() == 0:
        base_path = "octopus_l/resseance_l.php"
        session_code = f"lo{yymmdd}"
    else:
        return None

    classement_type = str(classement_type).lower().strip()
    if classement_type not in {"s", "h"}:
        return None

    return f"http://www.bridgeinter.net/{base_path}?v_codeseance={session_code}&v_type_classement={classement_type}"


def _bridgeinter_find_pair_pct(page_text: str, surname1: str, surname2: str) -> Optional[float]:
    """
    Heuristic: find the percentage whose nearby context contains both surnames.
    BridgeInterNet format: "RANK LINE EO/NS SURNAME1 First1 SURNAME2 First2 XX.XX% ..."
    Names appear BEFORE the percentage, so we search primarily in the preceding text.
    """
    t = (page_text or "").upper()
    s1 = (surname1 or "").strip().upper()
    s2 = (surname2 or "").strip().upper()
    if not t or not s1 or not s2:
        return None

    best_match = None
    best_distance = float('inf')
    
    for m in re.finditer(r"(\d{1,2}\.\d{2})%", t):
        # Look primarily BEFORE the percentage (names come before %)
        # Use smaller window: 100 chars before, 20 after
        start = max(0, m.start() - 100)
        end = min(len(t), m.end() + 20)
        ctx = t[start:end]
        
        if s1 in ctx and s2 in ctx:
            # Find how close the surnames are to the percentage
            s1_pos = ctx.rfind(s1)  # Last occurrence before %
            s2_pos = ctx.rfind(s2)
            if s1_pos >= 0 and s2_pos >= 0:
                # Distance from surnames to the percentage
                pct_pos_in_ctx = m.start() - start
                distance = max(pct_pos_in_ctx - s1_pos, pct_pos_in_ctx - s2_pos)
                if distance < best_distance:
                    best_distance = distance
                    try:
                        best_match = float(m.group(1))
                    except (TypeError, ValueError):
                        pass

    return best_match


def _pair_surnames(pair_name: str) -> Tuple[str, str]:
    """
    Extract rough surnames from "First LAST - First LAST".
    Falls back gracefully if format is unexpected.
    """
    if not pair_name:
        return "", ""

    left, right = "", ""
    if " - " in pair_name:
        left, right = pair_name.split(" - ", 1)
    elif "-" in pair_name:
        left, right = pair_name.split("-", 1)
    else:
        parts = pair_name.split()
        if len(parts) >= 2:
            return parts[-1], parts[-1]
        return "", ""

    l = left.strip().split()
    r = right.strip().split()
    return (l[-1] if l else ""), (r[-1] if r else "")


def _ffbridge_webpage_url(tournament_id: str, team_id: str, club_id: str) -> str:
    """Build the public FFBridge "Espace Licenci├⌐" results page URL for a row.

    Pattern:
        https://licencie.ffbridge.fr/#/resultats/simultane/{tournament_id}/details/{team_id}?orgId={club_id}

    Requires all three IDs; returns "" if any is missing.
    """
    tid = str(tournament_id or "").strip()
    teamid = str(team_id or "").strip()
    cid = str(club_id or "").strip()
    if not (tid and teamid and cid):
        return ""
    return f"https://licencie.ffbridge.fr/#/resultats/simultane/{tid}/details/{teamid}?orgId={cid}"


def _ffbridge_api_url(tournament_id: str) -> str:
    """Build the FFBridge Classic API URL for a tournament (dev reference; needs JWT)."""
    tid = str(tournament_id or "").strip()
    if not tid:
        return ""
    return f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{tid}"


def _maybe_override_octopus_pct_rows(detail_df: pl.DataFrame, pair_name: str, use_handicap: bool = True) -> pl.DataFrame:
    """
    For Octopus sessions, try to override Scratch_% / Handicap_% with BridgeInterNet values.
    Adds three URL columns:
      - 'Organization URL'     : BridgeInterNet results page when reconciliation succeeds (Octopus only).
      - 'ffbridge Result'      : Public FFBridge "Espace Licenci├⌐" results page (requires being signed in).
      - 'ffbridge API Endpoint': Raw FFBridge JSON API URL (devs only; returns 401 without JWT).

    Args:
        detail_df: DataFrame with tournament results. Should include hidden helper
                   columns '_team_id' and '_club_id' so the Webpage URL can be built.
        pair_name: Pair name or player name. If "Partner" column exists, uses that for each row.
        use_handicap: Whether to use handicap percentage for Pct_Used (else scratch)
    """
    if detail_df.is_empty():
        return detail_df

    # Get default surnames from passed pair_name
    default_s1, default_s2 = _pair_surnames(pair_name)
    
    rows = []
    for r in detail_df.to_dicts():
        bi_url = ""  # BridgeInterNet URL (only when reconciliation succeeds)
        date_str = str(r.get("Date", "") or "")[:10]
        tournament_label = str(r.get("Tournament", "") or "")
        event_id = str(r.get("Event_ID", "") or "")
        team_id = str(r.get("_team_id", "") or "")
        club_id = str(r.get("_club_id", "") or "")
        
        # Use Partner column if available (contains full pair name), otherwise use default
        partner_col = r.get("Partner", "")
        if partner_col:
            s1, s2 = _pair_surnames(str(partner_col))
            print(f"[BI Reconcile] Partner='{partner_col}' -> s1='{s1}', s2='{s2}'", flush=True)
        else:
            s1, s2 = default_s1, default_s2
            print(f"[BI Reconcile] No Partner col, using default s1='{s1}', s2='{s2}'", flush=True)

        scratch = r.get("Scratch_%")
        handicap = r.get("Handicap_%")

        # Attempt BridgeInterNet reconciliation for Octopus days (Monday/Thursday)
        # Check both by name and by date (Mon=0, Thu=3 are Octopus days)
        is_octopus_name = "octopus" in tournament_label.lower()
        is_octopus_day = False
        try:
            from datetime import datetime as dt
            d = dt.strptime(date_str, "%Y-%m-%d").date()
            is_octopus_day = d.weekday() in (0, 3)  # Monday or Thursday
        except (TypeError, ValueError):
            pass
        
        if (is_octopus_name or is_octopus_day) and s1 and s2:
            url_s = _bridgeinter_octopus_url(date_str, "s")
            url_h = _bridgeinter_octopus_url(date_str, "h")
            print(f"[BI Reconcile] date={date_str}, s1={s1}, s2={s2}, url_s={url_s}", flush=True)
            if url_s and url_h:
                try:
                    txt_s = _bridgeinter_fetch_text(url_s)
                    txt_h = _bridgeinter_fetch_text(url_h)
                    s_pct = _bridgeinter_find_pair_pct(txt_s, s1, s2)
                    h_pct = _bridgeinter_find_pair_pct(txt_h, s1, s2)
                    print(f"[BI Reconcile] Found: scratch={s_pct}, handicap={h_pct}", flush=True)
                    if s_pct is not None and h_pct is not None:
                        scratch = round(float(s_pct), 2)
                        handicap = round(float(h_pct), 2)
                        bi_url = url_s  # Link to scratch results page
                except Exception as e:
                    # Fail fast-ish: just don't override if BridgeInterNet fetch/parsing fails.
                    print(f"[BI Reconcile] Error: {e}", flush=True)

        webpage_url = _ffbridge_webpage_url(event_id, team_id, club_id)
        api_url = _ffbridge_api_url(event_id)

        r["Scratch_%"] = scratch
        r["Handicap_%"] = handicap
        # Update Pct_Used to match the updated values
        # Use handicap if requested AND available, otherwise use scratch
        if use_handicap and handicap is not None:
            r["Pct_Used"] = handicap
        else:
            r["Pct_Used"] = scratch
        # User-facing URL columns: cell renderer turns http(s) values into anchors.
        r["Organization URL"] = bi_url if bi_url else None
        r["ffbridge Result"] = webpage_url if webpage_url else None
        r["ffbridge API Endpoint"] = api_url if api_url else None
        rows.append(r)

    # The URL columns are None for most (non-Octopus) rows and only get a string
    # on reconciled Octopus days, which can appear after the default 100-row
    # schema-inference window -> Polars would lock the column to Null and then
    # fail to append the late URL string. Force them to Utf8 and scan all rows.
    return pl.DataFrame(
        rows,
        schema_overrides={
            "Organization URL": pl.Utf8,
            "ffbridge Result": pl.Utf8,
            "ffbridge API Endpoint": pl.Utf8,
        },
        infer_schema_length=None,
    )


_URL_CELL_RENDERER = JsCode("""
class UrlCellRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        var v = (params.value === null || params.value === undefined)
            ? '' : String(params.value).trim();
        if (v === '') {
            return;
        }
        if (!/^https?:\\/\\//i.test(v)) {
            this.eGui.textContent = v;
            return;
        }
        var a = document.createElement('a');
        a.href = v;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.title = v;
        a.style.color = '#0066cc';
        a.style.textDecoration = 'underline';
        a.textContent = v;
        this.eGui.appendChild(a);
    }
    getGui() {
        return this.eGui;
    }
    refresh(params) {
        return false;
    }
}
""")


def _url_columns(display_df: pd.DataFrame) -> List[str]:
    """Return column names whose non-null values look like http(s) URLs."""
    url_cols: List[str] = []
    for col in display_df.columns:
        try:
            series = display_df[col].dropna()
        except Exception:
            continue
        if series.empty:
            continue
        # Only inspect string-like columns to avoid scanning huge numeric arrays.
        if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            continue
        sample = series.astype(str).str.strip()
        if sample.str.match(r"^https?://", case=False).any():
            url_cols.append(col)
    return url_cols


def _aggrid_key_part(value: str) -> str:
    return re.sub(r"[^\w\-]", "_", (value or "").strip())[:48]


def _leaderboard_aggrid_key(
    entity: str,
    rating_type: str,
    simultaneous_type: str,
    club_filter: str,
    top_n: int,
    min_games: int,
    name_filter: str,
    prior_sessions: int,
) -> str:
    # Omit Scratch/Handicap so H/S toggles update the same AgGrid instance.
    return (
        f"ff_{entity}_table_{rating_type}_{simultaneous_type}_"
        f"club_{_aggrid_key_part(club_filter)}_top{top_n}_min{min_games}_"
        f"name_{_aggrid_key_part(name_filter)}_prior{prior_sessions}"
    )


def build_selectable_aggrid(df: pl.DataFrame, key: str, *, render_links: bool = True) -> Dict[str, Any]:
    """Build an AgGrid with single-click row selection (streamlit-aggrid)."""
    display_df = df.to_pandas(use_pyarrow_extension_array=False)
    coerce_numeric_columns(display_df)

    page_size = LEADERBOARD_PAGE_SIZE
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False, suppressRowClickSelection=False)
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=page_size)
    gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True)

    numeric_comparator = JsCode("""
        function(valueA, valueB, nodeA, nodeB, isDescending) {
            return Number(valueA) - Number(valueB);
        }
    """)
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            gb.configure_column(
                col,
                type=['numericColumn', 'numberColumnFilter'],
                comparator=numeric_comparator,
            )
    sort_col = 'Quality_Rank' if 'Quality_Rank' in display_df.columns else (
        'Rank' if 'Rank' in display_df.columns else None
    )
    if sort_col is not None:
        gb.configure_column(sort_col, sort='asc')
    if 'Games' in display_df.columns:
        gb.configure_column('Games', width=100)
    if render_links:
        for col in _url_columns(display_df):
            gb.configure_column(col, cellRenderer=_URL_CELL_RENDERER, minWidth=240, width=360)
    grid_options = gb.build()
    grid_options['rowHeight'] = LEADERBOARD_ROW_HEIGHT
    grid_options['headerHeight'] = 50
    grid_options['domLayout'] = 'normal'
    height = leaderboard_aggrid_viewport_height(len(display_df), page_size, pagination=True)

    return AgGrid(
        display_df,
        gridOptions=grid_options,
        columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
        theme=AgGridTheme.BALHAM,
        height=height,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        key=key,
        allow_unsafe_jscode=True,
        update_on=["selectionChanged"],
    )


def _render_detail_aggrid_ff(detail_df: pl.DataFrame, key: str, selectable: bool = False):
    """Render a detail DataFrame as a selectable AgGrid. Returns grid response if selectable."""
    display_df = detail_df.to_pandas(use_pyarrow_extension_array=False)
    coerce_numeric_columns(display_df)
    gb = GridOptionsBuilder.from_dataframe(display_df)
    if selectable:
        gb.configure_selection(selection_mode='single', use_checkbox=False, suppressRowClickSelection=False)
    gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True)
    numeric_comparator = JsCode("""
        function(valueA, valueB, nodeA, nodeB, isDescending) {
            return Number(valueA) - Number(valueB);
        }
    """)
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            gb.configure_column(
                col,
                type=['numericColumn', 'numberColumnFilter'],
                comparator=numeric_comparator,
            )
    for col in _url_columns(display_df):
        gb.configure_column(col, cellRenderer=_URL_CELL_RENDERER, minWidth=240, width=360)
    grid_options = gb.build()
    grid_options['rowHeight'] = 28
    grid_options['domLayout'] = 'normal'
    header_height = 50
    row_height = grid_options['rowHeight']
    max_visible = 25
    n_rows = len(display_df)
    visible = max(1, min(n_rows, max_visible))
    if n_rows <= max_visible:
        grid_options['alwaysShowVerticalScroll'] = False
        height = header_height + visible * row_height + 20
    else:
        grid_options['alwaysShowVerticalScroll'] = True
        height = header_height + max_visible * row_height + 20
    response = AgGrid(
        display_df,
        gridOptions=grid_options,
        height=height,
        theme=AgGridTheme.BALHAM,
        key=key,
        allow_unsafe_jscode=True,
        update_on=["selectionChanged"] if selectable else [],
    )
    return response if selectable else None


def _show_tournament_opponents(results_df: pl.DataFrame, tournament_id: str, exclude_pair_id: str = None) -> None:
    """3rd df: Show all pairs who played in a given tournament (the 'opponents')."""
    if results_df.is_empty():
        return
    tourney_results = results_df.filter(pl.col('tournament_id') == tournament_id)
    if exclude_pair_id:
        tourney_results = tourney_results.filter(pl.col('pair_id') != exclude_pair_id)
    if tourney_results.is_empty():
        st.info("No other pairs found in this tournament.")
        return

    cols = [pl.col('pair_name').alias('Pair')]
    if 'rank' in tourney_results.columns:
        cols.append(pl.col('rank').cast(pl.Int32, strict=False).alias('Rank'))
    if 'scratch_percentage' in tourney_results.columns:
        cols.append(pl.col('scratch_percentage').cast(pl.Float64, strict=False).round(2).alias('Scratch_%'))
    if 'handicap_percentage' in tourney_results.columns:
        cols.append(pl.col('handicap_percentage').cast(pl.Float64, strict=False).round(2).alias('Handicap_%'))
    if 'pair_elo' in tourney_results.columns:
        cols.append(pl.col('pair_elo').cast(pl.Float64, strict=False).round(0).cast(pl.Int32, strict=False).alias('Pair_Elo'))

    opp_df = tourney_results.select(cols)
    if 'Rank' in opp_df.columns:
        opp_df = opp_df.sort('Rank')

    t_name = tourney_results.select('tournament_name').row(0)[0] if 'tournament_name' in tourney_results.columns else tournament_id
    t_date = tourney_results.select(pl.col('date').str.slice(0, 10)).row(0)[0] if 'date' in tourney_results.columns else ''
    st.markdown(f"#### Tournament Opponents — {t_name} ({t_date})")
    st.caption(f"{len(opp_df)} pairs in this tournament")
    _render_detail_aggrid_ff(opp_df, key=f"ff_topp_{tournament_id}")


def _show_partner_aggregation(detail_df: pl.DataFrame, key_suffix: str) -> None:
    """4th df for Players: Aggregate tournament results by Partner across all tournaments."""
    if detail_df.is_empty() or 'Partner' not in detail_df.columns:
        return
    st.markdown("#### Partner Summary — All Tournaments")
    agg_cols = [
        pl.len().alias('Tournaments'),
    ]
    if 'Pct_Used' in detail_df.columns:
        agg_cols.append(pl.col('Pct_Used').mean().cast(pl.Float64).round(2).alias('Avg_Pct'))
    if 'Rank' in detail_df.columns:
        agg_cols.append(pl.col('Rank').cast(pl.Float64, strict=False).mean().round(1).alias('Avg_Rank'))
    if 'Elo_After' in detail_df.columns:
        agg_cols.append(pl.col('Elo_After').last().round(0).cast(pl.Int32, strict=False).alias('Last_Elo'))

    partner_agg = (
        detail_df
        .group_by('Partner')
        .agg(agg_cols)
        .sort('Tournaments', descending=True)
    )
    st.caption(f"{len(partner_agg)} unique partners")
    _render_detail_aggrid_ff(partner_agg, key=f"ff_partner_{key_suffix}")


def _show_club_aggregation(detail_df: pl.DataFrame, results_df: pl.DataFrame, pair_id: str, key_suffix: str) -> None:
    """4th df for Pairs: Aggregate tournament results by Club across all tournaments."""
    if results_df.is_empty():
        return
    # Get pair's results with club info
    pair_data = results_df.filter(pl.col('pair_id') == pair_id)
    if pair_data.is_empty() or 'club_name' not in pair_data.columns:
        return
    st.markdown("#### Club Summary — All Tournaments")
    agg_cols = [
        pl.len().alias('Tournaments'),
    ]
    if 'scratch_percentage' in pair_data.columns:
        agg_cols.append(pl.col('scratch_percentage').cast(pl.Float64, strict=False).mean().round(2).alias('Avg_Scratch_%'))
    if 'handicap_percentage' in pair_data.columns:
        agg_cols.append(pl.col('handicap_percentage').cast(pl.Float64, strict=False).mean().round(2).alias('Avg_Handicap_%'))
    if 'rank' in pair_data.columns:
        agg_cols.append(pl.col('rank').cast(pl.Float64, strict=False).mean().round(1).alias('Avg_Rank'))

    club_agg = (
        pair_data
        .group_by('club_name')
        .agg(agg_cols)
        .sort('Tournaments', descending=True)
        .rename({'club_name': 'Club'})
    )
    st.caption(f"{len(club_agg)} clubs")
    _render_detail_aggrid_ff(club_agg, key=f"ff_club_{key_suffix}")


def _build_opponent_data(results_df: pl.DataFrame, entity_tournaments: pl.DataFrame, exclude_id: str, exclude_mode: str = 'pair') -> pl.DataFrame:
    """Build a DataFrame of all opponents faced across tournaments.
    
    entity_tournaments: the tournaments the selected player/pair played in (with tournament_id).
    exclude_id: the ID value to exclude from the opponents list.
    exclude_mode: 'pair' excludes by pair_id, 'player' excludes rows where player appears as player1 or player2.
    """
    if entity_tournaments.is_empty() or results_df.is_empty():
        return pl.DataFrame()

    # Get all tournament IDs this entity played in
    tourney_ids = entity_tournaments.select('tournament_id').unique().to_series().to_list()

    # Get all results from those tournaments, excluding the entity's own rows
    tourney_filter = pl.col('tournament_id').is_in(tourney_ids)
    if exclude_mode == 'player':
        # Exclude any row where the player appears as either player1 or player2
        exclude_filter = (pl.col('player1_id') != exclude_id) & (pl.col('player2_id') != exclude_id)
    else:
        # Exclude by pair_id
        exclude_filter = pl.col('pair_id') != exclude_id
    all_opp = results_df.filter(tourney_filter & exclude_filter)
    if all_opp.is_empty():
        return pl.DataFrame()

    cols = [
        pl.col('tournament_id').alias('Event_ID'),
        pl.col('date').str.slice(0, 10).alias('Date'),
        pl.col('pair_name').alias('Opponent'),
    ]
    if 'rank' in all_opp.columns:
        cols.append(pl.col('rank').cast(pl.Int32, strict=False).alias('Rank'))
    if 'scratch_percentage' in all_opp.columns:
        cols.append(pl.col('scratch_percentage').cast(pl.Float64, strict=False).round(2).alias('Scratch_%'))
    if 'handicap_percentage' in all_opp.columns:
        cols.append(pl.col('handicap_percentage').cast(pl.Float64, strict=False).round(2).alias('Handicap_%'))
    if 'pair_elo' in all_opp.columns:
        cols.append(pl.col('pair_elo').cast(pl.Float64, strict=False).round(0).cast(pl.Int32, strict=False).alias('Pair_Elo'))

    return all_opp.select(cols).sort(['Date', 'Event_ID', 'Rank'], descending=[True, False, False])


def _show_opponent_history(results_df: pl.DataFrame, entity_tournaments: pl.DataFrame, exclude_id: str, exclude_mode: str = 'pair', key_suffix: str = '') -> None:
    """Show all opponents faced across all tournaments (Opponent History Details)."""
    opp_detail = _build_opponent_data(results_df, entity_tournaments, exclude_id, exclude_mode)
    if opp_detail.is_empty():
        return
    n_events = opp_detail.select('Event_ID').n_unique()
    st.markdown("#### Opponent History Details — All Tournaments")
    st.caption(f"{len(opp_detail)} opponent results across {n_events} tournaments")
    _render_detail_aggrid_ff(opp_detail, key=f"ff_opp_hist_{key_suffix}")


def _show_opponent_summary(results_df: pl.DataFrame, entity_tournaments: pl.DataFrame, exclude_id: str, exclude_mode: str = 'pair', key_suffix: str = '') -> None:
    """Show aggregation of opponents across all tournaments (Opponent Summary).

    Includes both the selected entity's averages and the opponent's averages
    for each metric, so the user can compare side by side.
    """
    opp_detail = _build_opponent_data(results_df, entity_tournaments, exclude_id, exclude_mode)
    if opp_detail.is_empty():
        return

    # Build entity's per-tournament stats for joining
    entity_cols = [pl.col('tournament_id').alias('Event_ID')]
    has_rank = 'rank' in entity_tournaments.columns
    has_scratch = 'scratch_percentage' in entity_tournaments.columns
    has_handicap = 'handicap_percentage' in entity_tournaments.columns
    has_elo = 'pair_elo' in entity_tournaments.columns
    if has_rank:
        entity_cols.append(pl.col('rank').cast(pl.Float64, strict=False).alias('My_Rank'))
    if has_scratch:
        entity_cols.append(pl.col('scratch_percentage').cast(pl.Float64, strict=False).alias('My_Scratch_%'))
    if has_handicap:
        entity_cols.append(pl.col('handicap_percentage').cast(pl.Float64, strict=False).alias('My_Handicap_%'))
    if has_elo:
        entity_cols.append(pl.col('pair_elo').cast(pl.Float64, strict=False).alias('My_Pair_Elo'))

    entity_per_tourney = entity_tournaments.select(entity_cols)
    # A player may appear multiple times per tournament (different partners);
    # take the first result per tournament for a simple join
    entity_per_tourney = entity_per_tourney.group_by('Event_ID').first()

    # Join entity's stats onto each opponent row by Event_ID
    opp_joined = opp_detail.join(entity_per_tourney, on='Event_ID', how='left')

    st.markdown("#### Opponent Summary — All Tournaments")

    agg_cols = [pl.len().alias('Events')]

    # Opponent averages
    if 'Rank' in opp_joined.columns:
        agg_cols.append(pl.col('Rank').mean().round(1).alias('Opp_Avg_Rank'))
    if 'Scratch_%' in opp_joined.columns:
        agg_cols.append(pl.col('Scratch_%').mean().round(2).alias('Opp_Avg_Scratch_%'))
    if 'Handicap_%' in opp_joined.columns:
        agg_cols.append(pl.col('Handicap_%').mean().round(2).alias('Opp_Avg_Handicap_%'))
    if 'Pair_Elo' in opp_joined.columns:
        agg_cols.append(pl.col('Pair_Elo').mean().round(0).cast(pl.Int32, strict=False).alias('Opp_Avg_Pair_Elo'))

    # Entity's averages (in the same tournaments as this opponent)
    if 'My_Rank' in opp_joined.columns:
        agg_cols.append(pl.col('My_Rank').mean().round(1).alias('My_Avg_Rank'))
    if 'My_Scratch_%' in opp_joined.columns:
        agg_cols.append(pl.col('My_Scratch_%').mean().round(2).alias('My_Avg_Scratch_%'))
    if 'My_Handicap_%' in opp_joined.columns:
        agg_cols.append(pl.col('My_Handicap_%').mean().round(2).alias('My_Avg_Handicap_%'))
    if 'My_Pair_Elo' in opp_joined.columns:
        agg_cols.append(pl.col('My_Pair_Elo').mean().round(0).cast(pl.Int32, strict=False).alias('My_Avg_Pair_Elo'))

    opp_agg = (
        opp_joined
        .group_by('Opponent')
        .agg(agg_cols)
        .sort('Events', descending=True)
    )
    st.caption(f"{len(opp_agg)} unique opponents")
    _render_detail_aggrid_ff(opp_agg, key=f"ff_opp_summ_{key_suffix}")


def _filter_valid_percentages_ffbridge(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows with invalid percentage values (<0 or >100)."""
    if df.is_empty():
        return df

    pct_cols = [c for c in ("percentage", "scratch_percentage", "handicap_percentage", "club_percentage") if c in df.columns]
    if not pct_cols:
        return df

    valid_expr = pl.lit(True)
    for col_name in pct_cols:
        col = pl.col(col_name).cast(pl.Float64, strict=False)
        valid_expr = valid_expr & (col.is_null() | ((col >= 0.0) & (col <= 100.0)))

    return df.filter(valid_expr)


# -------------------------------
# Data Processing (common for both APIs)
# -------------------------------
# Activity floor for computing the standardization population. Players below it
# (one-off / very-low-game accounts clustered near the seed) would compress the
# stdev and distort titles, so they are excluded from the mean/sd estimate
# (but still standardized and shown).
_STANDARDIZE_MIN_GAMES = 5


def _standardize_elo_frames(
    results_df: pl.DataFrame,
    players_df: pl.DataFrame,
    use_handicap: bool,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Affine-map all Elo columns onto the common chess scale (mean/sd anchored).

    Computes the player population mean/sd separately for scratch and handicap,
    then applies ``CHESS_DISPLAY_MEAN + z * CHESS_DISPLAY_SD`` (clamped 0-3500)
    to every stored Elo column. Pair and selected columns reuse the matching
    scratch/handicap transform so the whole frame stays on one coherent scale.
    """
    if players_df.is_empty():
        return results_df, players_df

    def _stats(col: str) -> Tuple[float, float]:
        sub = players_df.filter(pl.col('games_played') >= _STANDARDIZE_MIN_GAMES)
        if sub.height < 10:
            sub = players_df
        m = sub.select(pl.col(col).mean()).item()
        s = sub.select(pl.col(col).std(ddof=0)).item()
        return (float(m) if m is not None else CHESS_DISPLAY_MEAN,
                float(s) if s is not None else 0.0)

    def _aff(col: str, mean: float, sd: float) -> pl.Expr:
        if sd and sd > 0:
            return (
                (pl.lit(CHESS_DISPLAY_MEAN) + (pl.col(col) - mean) / sd * CHESS_DISPLAY_SD)
                .clip(0.0, 3500.0)
                .round(1)
                .alias(col)
            )
        return pl.lit(CHESS_DISPLAY_MEAN).alias(col)

    s_mean, s_sd = _stats('scratch_elo')
    h_mean, h_sd = _stats('handicap_elo')

    players_df = players_df.with_columns([
        _aff('scratch_elo', s_mean, s_sd),
        _aff('handicap_elo', h_mean, h_sd),
    ]).with_columns([
        (pl.col('handicap_elo') if use_handicap else pl.col('scratch_elo')).alias('elo_rating'),
    ])

    if not results_df.is_empty():
        scratch_cols = [
            'player1_scratch_elo_before', 'player1_scratch_elo_after',
            'player2_scratch_elo_before', 'player2_scratch_elo_after',
            'scratch_pair_elo',
        ]
        handicap_cols = [
            'player1_handicap_elo_before', 'player1_handicap_elo_after',
            'player2_handicap_elo_before', 'player2_handicap_elo_after',
            'handicap_pair_elo',
        ]
        results_df = results_df.with_columns(
            [_aff(c, s_mean, s_sd) for c in scratch_cols if c in results_df.columns]
            + [_aff(c, h_mean, h_sd) for c in handicap_cols if c in results_df.columns]
        )
        # Rebuild the use_handicap-selected display columns from the standardized values.
        sel = {
            'player1_elo_before': ('player1_handicap_elo_before', 'player1_scratch_elo_before'),
            'player1_elo_after': ('player1_handicap_elo_after', 'player1_scratch_elo_after'),
            'player2_elo_before': ('player2_handicap_elo_before', 'player2_scratch_elo_before'),
            'player2_elo_after': ('player2_handicap_elo_after', 'player2_scratch_elo_after'),
            'pair_elo': ('handicap_pair_elo', 'scratch_pair_elo'),
        }
        results_df = results_df.with_columns([
            pl.col(h if use_handicap else s).alias(dst)
            for dst, (h, s) in sel.items()
            if (h if use_handicap else s) in results_df.columns
        ])

    return results_df, players_df


def process_tournaments_to_elo(
    tournaments: List[Dict[str, Any]],
    api_module,
    initial_players: Optional[Dict[str, Dict]] = None,
    use_handicap: bool = False,
    fetch_iv: bool = False,
    sort_ascending: bool = True,
    show_progress: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float], Dict[str, int]]:
    """
    Process tournament list and calculate Elo ratings.
    Works with both Classic and Lancelot API data.
    Computes BOTH scratch and handicap Elo in one pass for efficient caching.
    
    Args:
        fetch_iv: If True, fetch current IV values for each player (slower)
        use_handicap: Determines which Elo is returned as current_ratings (both are stored in DataFrame)
        sort_ascending: If True, sort tournaments by date ascending (oldest first), else descending (newest first)
    
    Returns:
        Tuple of (results_df, players_df, current_ratings, cache_stats)
        cache_stats contains: {"cached": int, "fetched": int, "missing": int}
    """
    all_results = []
    # Track ratings for BOTH scratch and handicap
    scratch_ratings: Dict[str, float] = {}
    handicap_ratings: Dict[str, float] = {}
    player_names: Dict[str, str] = {}
    player_games: Dict[str, int] = {}
    player_pct_n: Dict[str, int] = {}
    player_pct_mean: Dict[str, float] = {}
    player_pct_m2: Dict[str, float] = {}
    
    if initial_players:
        for pid, pinfo in initial_players.items():
            initial_elo = pinfo.get('elo', DEFAULT_ELO)
            # Don't scale during calculation - scale only for display
            scratch_ratings[pid] = initial_elo
            handicap_ratings[pid] = initial_elo
            player_names[pid] = pinfo.get('name', pid)
            player_games[pid] = pinfo.get('games_played', 0)
    
    # Sort tournaments chronologically
    sorted_tournaments = sorted(tournaments, key=lambda x: x.get('date', ''), reverse=not sort_ascending)
    
    # Filter out future tournaments
    today = datetime.now().strftime('%Y-%m-%d')
    sorted_tournaments = [t for t in sorted_tournaments if t.get('date', '')[:10] <= today]
    
    # Progress tracking. UI widgets only when show_progress (avoids Streamlit
    # widget calls inside cached/offline contexts).
    progress_bar = st.progress(0) if show_progress else None
    status_text = st.empty() if show_progress else None
    cache_stats = {"cached": 0, "fetched": 0, "missing_ids": []}

    total_t = len(sorted_tournaments)
    _t_start = datetime.now()
    print(f"[Processing] start: {total_t} tournaments (fetch_iv={fetch_iv})", flush=True)
    for i, tournament in enumerate(sorted_tournaments):
        t_id = str(tournament.get('id', ''))
        t_series = normalize_series_id(tournament.get('series_id'))
        t_name = tournament.get('name') or tournament.get('label') or tournament.get('moment_label') or f"Tournament {t_id}"
        t_date = tournament.get('date', '')

        if show_progress:
            missing_count = len(cache_stats['missing_ids'])
            cache_info = f"[Cached: {cache_stats['cached']}, Fetched: {cache_stats['fetched']}, Missing results: {missing_count}]"
            status_text.markdown(f"<span style='color: white;'>Processing {i+1}/{total_t}: {t_name[:35]} (ID:{t_id})... {cache_info}</span>", unsafe_allow_html=True)
            progress_bar.progress((i + 1) / total_t)

        # Throttled console heartbeat during long replays.
        if _FFBRIDGE_DEBUG or i == 0 or (i + 1) % _PROCESSING_LOG_EVERY == 0 or (i + 1) == total_t:
            print(f"[Processing] {i+1}/{total_t}: cached={cache_stats['cached']} "
                  f"fetched={cache_stats['fetched']} missing={len(cache_stats['missing_ids'])}", flush=True)

        # Fetch results using the appropriate API module
        results, was_cached = api_module.fetch_tournament_results(t_id, tournament_date=t_date, series_id=t_series, fetch_iv=fetch_iv)

        if was_cached:
            cache_stats["cached"] += 1
        else:
            cache_stats["fetched"] += 1
            # Live network fetches are rare and worth surfacing individually.
            print(f"[Processing] {i+1}/{total_t}: ID={t_id} fetched from API (count={len(results) if results else 0})", flush=True)

        if not results:
            cache_stats["missing_ids"].append(t_id)
            if _FFBRIDGE_DEBUG:
                print(f"[Processing] {i+1}/{total_t}: ID={t_id} - no results, skipping", flush=True)
            continue
        
        # Calculate field average rating for both scratch and handicap
        scratch_field_ratings = []
        handicap_field_ratings = []
        for result in results:
            p1_id = result.get('player1_id')
            p2_id = result.get('player2_id')
            if p1_id and p2_id:
                scratch_field_ratings.append((scratch_ratings.get(p1_id, DEFAULT_ELO) + scratch_ratings.get(p2_id, DEFAULT_ELO)) / 2)
                handicap_field_ratings.append((handicap_ratings.get(p1_id, DEFAULT_ELO) + handicap_ratings.get(p2_id, DEFAULT_ELO)) / 2)
        
        scratch_field_avg = sum(scratch_field_ratings) / len(scratch_field_ratings) if scratch_field_ratings else DEFAULT_ELO
        handicap_field_avg = sum(handicap_field_ratings) / len(handicap_field_ratings) if handicap_field_ratings else DEFAULT_ELO

        # Field-strength K-dampening: damp Elo gains when this session's field is
        # weak relative to the global population mean (closed-weak-club channel).
        # Conservative anchors live in elo_common; IV_Bonus already handles part
        # of this so we avoid over-correcting.
        scratch_field_strength = field_strength_scale_from_mean(scratch_field_avg)
        handicap_field_strength = field_strength_scale_from_mean(handicap_field_avg)
        
        # Update ratings for each result
        for result in results:
            p1_id = result.get('player1_id')
            p2_id = result.get('player2_id')
            p1_name = result.get('player1_name', '')
            p2_name = result.get('player2_name', '')
            
            # Get percentages
            raw_pct = float(result.get('percentage', 50.0) or 50.0)
            handicap_pct_raw = result.get('handicap_percentage')
            # handicap_pct_raw may be None for scratch-only events
            try:
                handicap_pct = float(handicap_pct_raw) if handicap_pct_raw is not None else None
            except (ValueError, TypeError):
                handicap_pct = None
            
            pe_bonus_raw = result.get('pe_bonus', 0)
            try:
                pe_bonus = float(pe_bonus_raw or 0)
            except (ValueError, TypeError):
                pe_bonus = 0.0
            
            club_pct_raw = result.get('club_percentage')
            try:
                club_pct = float(club_pct_raw) if club_pct_raw is not None else (raw_pct - pe_bonus / 10.0)
            except (ValueError, TypeError):
                club_pct = raw_pct - pe_bonus / 10.0
            
            # Get scratch (unhandicapped) percentage and IV bonus
            scratch_pct_raw = result.get('scratch_percentage')
            try:
                scratch_pct = float(scratch_pct_raw) if scratch_pct_raw is not None else club_pct
            except (ValueError, TypeError):
                scratch_pct = club_pct
            
            iv_bonus_raw = result.get('iv_bonus')
            try:
                iv_bonus = float(iv_bonus_raw) if iv_bonus_raw is not None else (pe_bonus / 10.0)
            except (ValueError, TypeError):
                iv_bonus = pe_bonus / 10.0
            
            # Get individual IV values (if available from API)
            player1_iv = result.get('player1_iv')
            player2_iv = result.get('player2_iv')
            pair_iv = result.get('pair_iv')
            
            # We'll compute both, but use_handicap determines which is used for 'percentage' column
            # For scratch-only events (handicap_pct is None), always use scratch
            percentage = handicap_pct if (use_handicap and handicap_pct is not None) else scratch_pct
            
            rank_club = result.get('rank') or 0
            rank_handicap = result.get('theoretical_rank') or 0
            rank = rank_handicap if use_handicap and rank_handicap is not None else rank_club
            
            team_id = result.get('team_id') or ''
            pe = result.get('pe') or 0
            club_id = result.get('club_id', '')
            club_name = result.get('club_name', '')
            club_code = result.get('club_code', '')
            
            # Create stable pair identification
            if p1_id and p2_id:
                p1_str, p2_str = str(p1_id), str(p2_id)
                if p1_str < p2_str:
                    pair_id = f"{p1_str}_{p2_str}"
                    stable_p1_name, stable_p2_name = p1_name, p2_name
                else:
                    pair_id = f"{p2_str}_{p1_str}"
                    stable_p1_name, stable_p2_name = p2_name, p1_name
                pair_name = f"{stable_p1_name} - {stable_p2_name}"
            else:
                pair_id = result.get('pair_id') or f"{p1_id}-{p2_id}"
                pair_name = f"{p1_name} - {p2_name}"
            
            # Use tournament_id or session_id depending on API
            event_id = str(t_id)
            
            result_record = {
                'tournament_id': event_id,
                'tournament_name': str(t_name),
                'date': str(t_date),
                'series_id': int(t_series) if t_series is not None else 0,
                'team_id': str(team_id),
                'pair_id': str(pair_id),
                'player1_id': str(p1_id),
                'player2_id': str(p2_id),
                'player1_name': str(p1_name),
                'player2_name': str(p2_name),
                'pair_name': str(pair_name),
                'percentage': float(percentage),
                'handicap_percentage': float(handicap_pct) if handicap_pct is not None else None,
                'scratch_percentage': float(scratch_pct),
                'iv_bonus': float(iv_bonus),
                'club_percentage': float(club_pct),
                'rank': int(rank) if rank is not None else 0,
                'rank_without_handicap': int(rank_club) if rank_club is not None else 0,
                'theoretical_rank': int(rank_handicap) if rank_handicap is not None else 0,
                'pe': float(pe) if pe is not None else 0.0,
                'pe_bonus': str(pe_bonus) if pe_bonus is not None else '',
                'scratch_field_avg': float(scratch_field_avg),
                'handicap_field_avg': float(handicap_field_avg),
                'scratch_field_strength': float(scratch_field_strength),
                'handicap_field_strength': float(handicap_field_strength),
                'club_id': str(club_id),
                'club_name': str(club_name),
                'club_code': str(club_code),
                'player1_current_iv': float(player1_iv) if player1_iv is not None else None,
                'player2_current_iv': float(player2_iv) if player2_iv is not None else None,
                'pair_iv': float(pair_iv) if pair_iv is not None else None,  # Current pair IV (sum of current player IVs)
            }
            
            # Update player 1 ratings (BOTH scratch and handicap)
            if p1_id:
                # Scratch Elo - calculate in original range, don't scale during calculation
                scratch_r1_before = scratch_ratings.get(p1_id, DEFAULT_ELO)
                scratch_r1_after = calculate_elo_from_percentage(scratch_r1_before, scratch_pct, scratch_field_avg, field_strength_scale=scratch_field_strength)
                scratch_ratings[p1_id] = scratch_r1_after
                
                # Handicap Elo (use scratch if handicap not available)
                handicap_r1_before = handicap_ratings.get(p1_id, DEFAULT_ELO)
                h_pct_for_elo = handicap_pct if handicap_pct is not None else scratch_pct
                h_field_for_elo = handicap_field_avg if handicap_pct is not None else scratch_field_avg
                h_strength_for_elo = handicap_field_strength if handicap_pct is not None else scratch_field_strength
                handicap_r1_after = calculate_elo_from_percentage(handicap_r1_before, h_pct_for_elo, h_field_for_elo, field_strength_scale=h_strength_for_elo)
                handicap_ratings[p1_id] = handicap_r1_after
                
                # Scale Elo values for display (calculations done in original range)
                result_record['player1_scratch_elo_before'] = scale_to_chess_range(scratch_r1_before)
                result_record['player1_scratch_elo_after'] = scale_to_chess_range(scratch_r1_after)
                result_record['player1_handicap_elo_before'] = scale_to_chess_range(handicap_r1_before)
                result_record['player1_handicap_elo_after'] = scale_to_chess_range(handicap_r1_after)
                # For backward compatibility, use selected type
                result_record['player1_elo_before'] = scale_to_chess_range(handicap_r1_before if use_handicap else scratch_r1_before)
                result_record['player1_elo_after'] = scale_to_chess_range(handicap_r1_after if use_handicap else scratch_r1_after)
                
                player_names[p1_id] = p1_name
                player_games[p1_id] = player_games.get(p1_id, 0) + 1
                
                n = player_pct_n.get(p1_id, 0) + 1
                mean = player_pct_mean.get(p1_id, 0.0)
                m2 = player_pct_m2.get(p1_id, 0.0)
                x = float(percentage)
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2
                player_pct_n[p1_id] = n
                player_pct_mean[p1_id] = mean
                player_pct_m2[p1_id] = m2
            
            # Update player 2 ratings (BOTH scratch and handicap)
            if p2_id:
                # Scratch Elo - calculate in original range, don't scale during calculation
                scratch_r2_before = scratch_ratings.get(p2_id, DEFAULT_ELO)
                scratch_r2_after = calculate_elo_from_percentage(scratch_r2_before, scratch_pct, scratch_field_avg, field_strength_scale=scratch_field_strength)
                scratch_ratings[p2_id] = scratch_r2_after
                
                # Handicap Elo (use scratch if handicap not available)
                handicap_r2_before = handicap_ratings.get(p2_id, DEFAULT_ELO)
                h_pct_for_elo = handicap_pct if handicap_pct is not None else scratch_pct
                h_field_for_elo = handicap_field_avg if handicap_pct is not None else scratch_field_avg
                h_strength_for_elo = handicap_field_strength if handicap_pct is not None else scratch_field_strength
                handicap_r2_after = calculate_elo_from_percentage(handicap_r2_before, h_pct_for_elo, h_field_for_elo, field_strength_scale=h_strength_for_elo)
                handicap_ratings[p2_id] = handicap_r2_after
                
                # Scale Elo values for display (calculations done in original range)
                result_record['player2_scratch_elo_before'] = scale_to_chess_range(scratch_r2_before)
                result_record['player2_scratch_elo_after'] = scale_to_chess_range(scratch_r2_after)
                result_record['player2_handicap_elo_before'] = scale_to_chess_range(handicap_r2_before)
                result_record['player2_handicap_elo_after'] = scale_to_chess_range(handicap_r2_after)
                # For backward compatibility, use selected type (scaled for display)
                result_record['player2_elo_before'] = scale_to_chess_range(handicap_r2_before if use_handicap else scratch_r2_before)
                result_record['player2_elo_after'] = scale_to_chess_range(handicap_r2_after if use_handicap else scratch_r2_after)
                
                player_names[p2_id] = p2_name
                player_games[p2_id] = player_games.get(p2_id, 0) + 1
                
                n = player_pct_n.get(p2_id, 0) + 1
                mean = player_pct_mean.get(p2_id, 0.0)
                m2 = player_pct_m2.get(p2_id, 0.0)
                x = float(percentage)
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2
                player_pct_n[p2_id] = n
                player_pct_mean[p2_id] = mean
                player_pct_m2[p2_id] = m2
            
            # Calculate pair Elo (both types)
            if p1_id and p2_id:
                # Calculate pair Elo from unscaled player ratings, then scale for display
                scratch_pair_elo_raw = (scratch_ratings[p1_id] + scratch_ratings[p2_id]) / 2
                handicap_pair_elo_raw = (handicap_ratings[p1_id] + handicap_ratings[p2_id]) / 2
                result_record['scratch_pair_elo'] = scale_to_chess_range(scratch_pair_elo_raw)
                result_record['handicap_pair_elo'] = scale_to_chess_range(handicap_pair_elo_raw)
                result_record['pair_elo'] = result_record['handicap_pair_elo'] if use_handicap else result_record['scratch_pair_elo']
            
            all_results.append(result_record)

    _elapsed = (datetime.now() - _t_start).total_seconds()
    cache_stats["processed_tournament_ids"] = [
        str(t.get("id")) for t in sorted_tournaments if t.get("id") is not None
    ]
    print(f"[Processing] done: {total_t} tournaments in {_elapsed:.1f}s "
          f"(cached={cache_stats['cached']} fetched={cache_stats['fetched']} "
          f"missing={len(cache_stats['missing_ids'])}, rows={len(all_results)})", flush=True)

    if progress_bar is not None:
        progress_bar.progress(1.0)
    if status_text is not None:
        status_text.markdown(
            "<span style='color: white;'>Tournaments loaded — building tables "
            f"({len(all_results):,} rows)…</span>",
            unsafe_allow_html=True,
        )

    # Convert to DataFrames with explicit schema to handle None values
    if all_results:
        results_schema = {
            'tournament_id': pl.Utf8,
            'tournament_name': pl.Utf8,
            'date': pl.Utf8,
            'series_id': pl.Int64,
            'team_id': pl.Utf8,
            'pair_id': pl.Utf8,
            'player1_id': pl.Utf8,
            'player2_id': pl.Utf8,
            'player1_name': pl.Utf8,
            'player2_name': pl.Utf8,
            'pair_name': pl.Utf8,
            'percentage': pl.Float64,
            'handicap_percentage': pl.Float64,
            'scratch_percentage': pl.Float64,
            'iv_bonus': pl.Float64,
            'club_percentage': pl.Float64,
            'rank': pl.Int64,
            'rank_without_handicap': pl.Int64,
            'theoretical_rank': pl.Int64,
            'pe': pl.Float64,
            'pe_bonus': pl.Utf8,
            'scratch_field_avg': pl.Float64,
            'handicap_field_avg': pl.Float64,
            'scratch_field_strength': pl.Float64,
            'handicap_field_strength': pl.Float64,
            'club_id': pl.Utf8,
            'club_name': pl.Utf8,
            'club_code': pl.Utf8,
            'player1_current_iv': pl.Float64,
            'player2_current_iv': pl.Float64,
            'pair_iv': pl.Float64,
            'player1_scratch_elo_before': pl.Float64,
            'player1_scratch_elo_after': pl.Float64,
            'player1_handicap_elo_before': pl.Float64,
            'player1_handicap_elo_after': pl.Float64,
            'player1_elo_before': pl.Float64,
            'player1_elo_after': pl.Float64,
            'player2_scratch_elo_before': pl.Float64,
            'player2_scratch_elo_after': pl.Float64,
            'player2_handicap_elo_before': pl.Float64,
            'player2_handicap_elo_after': pl.Float64,
            'player2_elo_before': pl.Float64,
            'player2_elo_after': pl.Float64,
            'scratch_pair_elo': pl.Float64,
            'handicap_pair_elo': pl.Float64,
            'pair_elo': pl.Float64,
        }
        results_df = pl.DataFrame(all_results, schema=results_schema)
    else:
        results_df = pl.DataFrame()

    if status_text is not None:
        status_text.markdown(
            "<span style='color: white;'>Building player summary "
            f"({len(scratch_ratings):,} players)…</span>",
            unsafe_allow_html=True,
        )

    # Create player ratings summary with BOTH scratch and handicap Elo
    # Scale ratings for display (calculations done in original range)
    player_summary = []
    for pid in set(scratch_ratings.keys()) | set(handicap_ratings.keys()):
        scratch_elo_raw = scratch_ratings.get(pid, DEFAULT_ELO)
        handicap_elo_raw = handicap_ratings.get(pid, DEFAULT_ELO)
        # Scale for display
        scratch_elo = scale_to_chess_range(scratch_elo_raw)
        handicap_elo = scale_to_chess_range(handicap_elo_raw)
        # Use the selected type for the main 'elo_rating' column
        rating = handicap_elo if use_handicap else scratch_elo
        
        n = player_pct_n.get(pid, 0)
        avg_pct = float(player_pct_mean.get(pid, 0.0)) if n > 0 else None
        stdev_pct = float((player_pct_m2.get(pid, 0.0) / (n - 1)) ** 0.5) if n > 1 else None
        player_summary.append({
            'player_id': str(pid),
            'player_name': str(player_names.get(pid, pid)),
            'scratch_elo': float(round(scratch_elo, 1)),
            'handicap_elo': float(round(handicap_elo, 1)),
            'elo_rating': float(round(rating, 1)),
            'games_played': int(player_games.get(pid, 0)),
            'avg_percentage': avg_pct,
            'stdev_percentage': stdev_pct,
        })
    
    if player_summary:
        players_df = pl.DataFrame(player_summary, schema={
            'player_id': pl.Utf8,
            'player_name': pl.Utf8,
            'scratch_elo': pl.Float64,
            'handicap_elo': pl.Float64,
            'elo_rating': pl.Float64,
            'games_played': pl.Int64,
            'avg_percentage': pl.Float64,
            'stdev_percentage': pl.Float64,
        })
    else:
        players_df = pl.DataFrame()

    if status_text is not None:
        status_text.markdown(
            "<span style='color: white;'>Standardizing Elo ratings…</span>",
            unsafe_allow_html=True,
        )

    # --- Unified z-score -> chess scale (aligns FFBridge with ACBL + chess) ---
    # Standardize every stored Elo column to mean CHESS_DISPLAY_MEAN, sd
    # CHESS_DISPLAY_SD using the player population's own mean/sd. z-score is
    # scale-invariant, so the prior x2 chess scaling cancels out and the result
    # is identical to standardizing the raw ratings. Scratch and handicap are
    # standardized against their own populations; pair / selected columns reuse
    # the player-level transform so an average pair lands near the anchor and
    # the chess-title ladder (>=2400 IM, >=2500 GM, >=2600 SGM) labels the same
    # percentile in FFBridge as in ACBL.
    results_df, players_df = _standardize_elo_frames(results_df, players_df, use_handicap)

    # Return selected type ratings dict for backward compatibility
    current_ratings = handicap_ratings if use_handicap else scratch_ratings
    return results_df, players_df, current_ratings, cache_stats


def _apply_club_name_mapping(
    results_df: pl.DataFrame, api_module, all_tournaments: List[Dict[str, Any]]
) -> pl.DataFrame:
    """Backfill club names for APIs (e.g. Lancelot) that only return club codes."""
    if results_df.is_empty() or 'club_code' not in results_df.columns:
        return results_df
    non_empty_names = results_df.filter(pl.col('club_name') != '').height
    if non_empty_names >= results_df.height * 0.1:  # already mostly populated
        return results_df
    unique_codes = [str(c) for c in results_df.select('club_code').unique().to_series().to_list() if c]
    if not hasattr(api_module, 'build_club_name_mapping'):
        return results_df
    club_mapping = api_module.build_club_name_mapping(unique_codes, all_tournaments, results_df)
    if not club_mapping:
        return results_df
    return results_df.with_columns(
        pl.col('club_code')
        .cast(pl.Utf8)
        .str.strip_chars_start('0')
        .replace(club_mapping, default=None)
        .fill_null(pl.col('club_name'))
        .alias('club_name')
    )


def _elo_cache_key(api_key: str, fetch_iv: bool, n_tournaments: int = 0) -> str:
    """Stable parquet identity per backend and IV mode.

    Tournament-list length (including scheduled future sessions) is intentionally
    excluded: embedding it forced a full rebuild whenever the API added future
    sessions even when no new past events existed.
    """
    del n_tournaments
    return f"elo_full_v3_{api_key}_iv_{int(fetch_iv)}"


def _legacy_elo_cache_keys(api_key: str, fetch_iv: bool) -> List[str]:
    """Parquet keys from before the stable-key change (middle segment = list length)."""
    prefix = f"elo_full_v3_{api_key}_"
    suffix = f"_iv_{int(fetch_iv)}"
    keys: List[str] = []
    for meta_path in _FFBRIDGE_ELO_CACHE_DIR.glob(f"{prefix}*{suffix}.meta.json"):
        key = meta_path.name[: -len(".meta.json")]
        middle = key[len(prefix): -len(suffix)]
        if middle.isdigit():
            keys.append(key)
    return keys


def _resolve_elo_cache_key(api_key: str, fetch_iv: bool) -> Optional[str]:
    """Best on-disk parquet set: stable key first, else newest legacy count-keyed set."""
    stable = _elo_cache_key(api_key, fetch_iv)
    results_path, players_path, meta_path = _elo_cache_paths(stable)
    if results_path.exists() and players_path.exists() and meta_path.exists():
        return stable

    newest_key: Optional[str] = None
    newest_dt: Optional[datetime] = None
    for key in _legacy_elo_cache_keys(api_key, fetch_iv):
        results_path, players_path, meta_path = _elo_cache_paths(key)
        if not (results_path.exists() and players_path.exists() and meta_path.exists()):
            continue
        try:
            built_at = json.loads(meta_path.read_text(encoding="utf-8")).get("built_at")
            if not built_at:
                continue
            dt = datetime.fromisoformat(built_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if newest_dt is None or dt > newest_dt:
                newest_dt = dt
                newest_key = key
        except Exception:
            continue
    return newest_key


def _elo_cache_meta_paths(api_key: str, fetch_iv: bool) -> List[pathlib.Path]:
    """All meta.json paths for a backend (stable + legacy count-keyed sets)."""
    iv = int(fetch_iv)
    seen: set[pathlib.Path] = set()
    paths: List[pathlib.Path] = []
    for pattern in (
        f"elo_full_v3_{api_key}_iv_{iv}.meta.json",
        f"elo_full_v3_{api_key}_*_iv_{iv}.meta.json",
    ):
        for meta_path in _FFBRIDGE_ELO_CACHE_DIR.glob(pattern):
            if meta_path not in seen:
                seen.add(meta_path)
                paths.append(meta_path)
    return paths


def _elo_cache_paths(key: str) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    return (
        _FFBRIDGE_ELO_CACHE_DIR / f"{key}.results.parquet",
        _FFBRIDGE_ELO_CACHE_DIR / f"{key}.players.parquet",
        _FFBRIDGE_ELO_CACHE_DIR / f"{key}.meta.json",
    )


def _prune_old_elo_cache(api_key: str, fetch_iv: bool, keep_key: str) -> None:
    """Delete superseded parquet sets for this backend (legacy count-keyed files).

    After a rebuild we keep only ``keep_key`` (the stable key). Best-effort:
    pruning failures never block the deploy.
    """
    try:
        for key in _legacy_elo_cache_keys(api_key, fetch_iv):
            if key == keep_key:
                continue
            for path in _elo_cache_paths(key):
                path.unlink(missing_ok=True)
            print(f"[ffbridge] pruned stale Elo cache '{key}'", flush=True)
    except Exception as exc:
        print(f"[ffbridge] elo cache prune skipped ({exc})", flush=True)


def _read_persisted_elo_dataset(api_key: str, fetch_iv: bool) -> Optional[Dict[str, Any]]:
    """Return the persisted dataset for this backend if present and readable."""
    key = _resolve_elo_cache_key(api_key, fetch_iv)
    if key is None:
        return None
    results_path, players_path, meta_path = _elo_cache_paths(key)
    if not (results_path.exists() and players_path.exists() and meta_path.exists()):
        return None
    try:
        t0 = time.perf_counter()
        _load_debug_log_standalone(f"reading results parquet {results_path.name}", t0)
        results_df = pl.read_parquet(results_path)
        _load_debug_log_standalone(
            f"results parquet loaded ({results_df.height} rows, {results_df.width} cols)",
            t0,
        )
        players_df = pl.read_parquet(players_path)
        _load_debug_log_standalone(
            f"players parquet loaded ({players_df.height} rows)",
            t0,
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[ffbridge] loaded persisted Elo dataset '{key}' "
              f"({results_df.height} result rows) from parquet", flush=True)
        return {
            "results_df": results_df,
            "players_df": players_df,
            "scratch_ratings": meta.get("scratch_ratings", {}),
            "handicap_ratings": meta.get("handicap_ratings", {}),
            "processing_stats": meta.get(
                "processing_stats", {"cached": 0, "fetched": 0, "missing_ids": []}
            ),
            "built_at": meta.get("built_at"),
        }
    except Exception as exc:  # corrupt/partial cache -> caller rebuilds
        print(f"[ffbridge] persisted Elo dataset load failed ({exc}); rebuilding", flush=True)
        return None


def _newest_persisted_age_hours(api_key: str, fetch_iv: bool) -> Optional[float]:
    """Age in hours of the newest persisted parquet set for this backend, or None."""
    newest: Optional[datetime] = None
    for meta_path in _elo_cache_meta_paths(api_key, fetch_iv):
        results_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".results.parquet"))
        players_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".players.parquet"))
        if not (results_path.exists() and players_path.exists()):
            continue
        try:
            built_at = json.loads(meta_path.read_text(encoding="utf-8")).get("built_at")
            if not built_at:
                continue
            dt = datetime.fromisoformat(built_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if newest is None or dt > newest:
                newest = dt
        except Exception:
            continue
    if newest is None:
        return None
    return (datetime.now(timezone.utc) - newest).total_seconds() / 3600.0


def _past_tournament_ids(all_tournaments: List[Dict[str, Any]]) -> set[str]:
    """Session IDs that should have been processed (date on or before today).

    Must match the future-tournament filter in ``process_tournaments_to_elo``.
    The Lancelot list includes hundreds of scheduled future sessions that are
    intentionally skipped during Elo replay — they must not count as "missing".
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        str(t.get("id"))
        for t in all_tournaments
        if t.get("id") is not None and str(t.get("date", ""))[:10] <= today
    }


def _fetch_tournament_list_resilient(
    api_module,
    force_refresh: bool,
) -> Tuple[List[Dict[str, Any]], str]:
    """Load the tournament list, falling back between disk cache and live API."""
    tournaments = api_module.fetch_tournament_list(
        series_id="all", limit=None, force_refresh=force_refresh,
    )
    if tournaments:
        return tournaments, "api" if force_refresh else "disk"

    fallback_refresh = not force_refresh
    print(
        f"[ffbridge] tournament list empty (force_refresh={force_refresh}); "
        f"retrying with force_refresh={fallback_refresh}",
        flush=True,
    )
    tournaments = api_module.fetch_tournament_list(
        series_id="all", limit=None, force_refresh=fallback_refresh,
    )
    if tournaments:
        return tournaments, "api" if fallback_refresh else "disk"
    return [], "none"


def _needs_elo_rebuild(
    api_key: str,
    fetch_iv: bool,
    n_tournaments: int,
    all_tournaments: List[Dict[str, Any]],
    max_age_hours: float,
) -> Tuple[bool, str]:
    """Return (True, reason) when the persisted parquet must be recomputed."""
    del n_tournaments
    key = _resolve_elo_cache_key(api_key, fetch_iv)
    if key is None:
        return True, "missing or unreadable parquet"
    results_path, players_path, meta_path = _elo_cache_paths(key)
    if not (results_path.exists() and players_path.exists() and meta_path.exists()):
        return True, "missing or unreadable parquet"

    age = _newest_persisted_age_hours(api_key, fetch_iv)
    if age is None:
        return True, "incomplete parquet metadata"
    if age >= max_age_hours:
        return True, f"cache stale ({age:.1f}h >= {max_age_hours:g}h)"

    eligible_ids = _past_tournament_ids(all_tournaments)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True, "incomplete parquet metadata"
    built_ids = {str(x) for x in meta.get("processed_tournament_ids", [])}

    if built_ids:
        missing = eligible_ids - built_ids
    else:
        # Backward compat: read only tournament_id column (not the full 700k+ row frame).
        try:
            parquet_ids = set(
                pl.read_parquet(results_path, columns=["tournament_id"])
                .select(pl.col("tournament_id").cast(pl.Utf8))
                .unique()
                .to_series()
                .to_list()
            )
        except Exception:
            return True, "unreadable parquet"
        skipped_ids = {
            str(x)
            for x in (meta.get("processing_stats") or {}).get("missing_ids", [])
            if x is not None
        }
        missing = eligible_ids - parquet_ids - skipped_ids
    if missing:
        return True, f"{len(missing)} new past tournament(s) not in parquet"
    return False, ""


def compute_and_persist_elo_dataset(
    api_module,
    all_tournaments: List[Dict[str, Any]],
    api_key: str,
    fetch_iv: bool,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Compute the full FFBridge Elo dataset and persist it to parquet.

    Not cached — used both by the Streamlit boot loader (on a cold cache) and by
    the offline builder script. Both scratch and handicap Elo columns are stored
    so the ``use_handicap`` toggle is derived downstream.
    """
    key = _elo_cache_key(api_key, fetch_iv)

    results_path, players_path, meta_path = _elo_cache_paths(key)

    results_df, players_df, _ratings, stats = process_tournaments_to_elo(
        all_tournaments, api_module, initial_players=None,
        use_handicap=False, fetch_iv=fetch_iv, sort_ascending=True, show_progress=show_progress,
    )
    finalize_status = st.empty() if show_progress else None
    if finalize_status is not None:
        finalize_status.markdown(
            "<span style='color: white;'>Mapping club names…</span>",
            unsafe_allow_html=True,
        )
    results_df = _apply_club_name_mapping(results_df, api_module, all_tournaments)

    scratch_ratings_dict: Dict[str, float] = {}
    handicap_ratings_dict: Dict[str, float] = {}
    if not players_df.is_empty():
        for row in players_df.iter_rows(named=True):
            scratch_ratings_dict[row['player_id']] = row['scratch_elo']
            handicap_ratings_dict[row['player_id']] = row['handicap_elo']

    # Persist for the next cold start (best-effort; ephemeral FS is fine).
    try:
        if finalize_status is not None:
            finalize_status.markdown(
                "<span style='color: white;'>Writing parquet cache…</span>",
                unsafe_allow_html=True,
            )
        _FFBRIDGE_ELO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        results_df.write_parquet(results_path)
        players_df.write_parquet(players_path)
        meta_path.write_text(json.dumps({
            "scratch_ratings": scratch_ratings_dict,
            "handicap_ratings": handicap_ratings_dict,
            "processing_stats": stats,
            "processed_tournament_ids": stats.get("processed_tournament_ids", []),
            "built_at": datetime.now(timezone.utc).isoformat(),
        }), encoding="utf-8")
        print(f"[ffbridge] persisted Elo dataset '{key}' to parquet", flush=True)
        _prune_old_elo_cache(api_key, fetch_iv, key)
    except Exception as exc:
        print(f"[ffbridge] parquet persist failed ({exc}); continuing without persistence", flush=True)
    finally:
        if finalize_status is not None:
            finalize_status.empty()

    return {
        "results_df": results_df,
        "players_df": players_df,
        "scratch_ratings": scratch_ratings_dict,
        "handicap_ratings": handicap_ratings_dict,
        "processing_stats": stats,
    }


@st.cache_resource(show_spinner="Building FFBridge Elo ratings (first load may take ~1 minute)…")
def load_ffbridge_elo_dataset(
    _api_module,
    _all_tournaments: List[Dict[str, Any]],
    api_key: str,
    fetch_iv: bool,
    n_tournaments: int,
) -> Dict[str, Any]:
    """Load the full FFBridge Elo dataset once per process, shared across sessions.

    ``@st.cache_resource`` keeps a single copy in the process (not per browser
    session, unlike ``st.session_state``) — this removes the per-session /
    concurrent-session memory multiplication that was OOM-killing the container.

    On a cold cache it prefers a parquet persisted by an earlier run or the
    offline builder (``build_ffbridge_elo_parquets.py``); only if none exists
    does it rebuild the history in RAM and persist it.

    Underscore-prefixed args (``_api_module``, ``_all_tournaments``) are excluded
    from Streamlit's cache key; identity is ``(api_key, fetch_iv)``.
    """
    t0 = time.perf_counter()
    _load_debug_log_standalone(
        f"load_ffbridge_elo_dataset cache lookup api_key={api_key} fetch_iv={fetch_iv}",
        t0,
    )
    persisted = _read_persisted_elo_dataset(api_key, fetch_iv)
    if persisted is not None:
        _load_debug_log_standalone("load_ffbridge_elo_dataset returning persisted parquet", t0)
        return persisted
    _load_debug_log_standalone("load_ffbridge_elo_dataset no parquet; inline compute", t0)
    return compute_and_persist_elo_dataset(
        _api_module, _all_tournaments, api_key, fetch_iv, show_progress=False
    )


def show_top_players(
    players_df: pl.DataFrame,
    top_n: int,
    min_games: int = 5,
    use_handicap: bool = False,
    prior_sessions: int = 0,
) -> Tuple[pl.DataFrame, str, Optional[float]]:
    """Get top players sorted by Elo rating using SQL.

    When ``prior_sessions > 0`` and a prior anchor (median Elo of the
    qualifying subset) is available, the headline Elo is the Bayesian-shrunk
    "Published" Elo:

        Published = (games * Raw + prior_sessions * prior_anchor)
                    / (games + prior_sessions)

    Both the Published and Raw values are returned in the table; the
    leaderboard is ordered by Published. Returns ``(df, sql, prior_anchor)``.
    """
    if players_df.is_empty():
        return players_df, "", None

    elo_col_name = "HC_Player_Elo" if use_handicap else "Player_Elo"
    title_col_name = "Scratch_Title" if use_handicap else "Title"

    anchor_query = f"""
        SELECT
            MEDIAN(elo_rating) AS anchor,
            MEDIAN(scratch_elo) AS scratch_anchor
        FROM players_df
        WHERE games_played >= {min_games}
    """
    anchor_df = duckdb.sql(anchor_query).pl()
    if anchor_df.is_empty():
        prior_anchor: Optional[float] = None
        scratch_anchor: Optional[float] = None
    else:
        val = anchor_df.item(0, 0)
        prior_anchor = float(val) if val is not None else None
        sval = anchor_df.item(0, 1)
        scratch_anchor = float(sval) if sval is not None else None

    if prior_sessions > 0 and prior_anchor is not None:
        ps_lit = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
        anchor_lit = f"CAST({float(prior_anchor)!r} AS DOUBLE)"
        published_expr = (
            f"CAST(ROUND(LEAST(GREATEST("
            f"(CAST(games_played AS DOUBLE) * CAST(elo_rating AS DOUBLE) "
            f"+ {ps_lit} * {anchor_lit}) "
            f"/ NULLIF(CAST(games_played AS DOUBLE) + {ps_lit}, 0)"
            f", 0), 3500), 0) AS INTEGER)"
        )
    else:
        published_expr = "CAST(ROUND(LEAST(GREATEST(elo_rating, 0), 3500), 0) AS INTEGER)"

    # Titles derive from the *published* (Bayesian-shrunk) SCRATCH Elo so they
    # always agree with the shrunk headline and never show an inflated title for
    # a low-sample player. In scratch view this equals the headline; in handicap
    # view it is the shrunk scratch rating (the "Scratch_Title" skill indicator).
    if prior_sessions > 0 and scratch_anchor is not None:
        ps_lit_s = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
        sanchor_lit = f"CAST({float(scratch_anchor)!r} AS DOUBLE)"
        published_scratch_expr = (
            f"CAST(ROUND(LEAST(GREATEST("
            f"(CAST(games_played AS DOUBLE) * CAST(scratch_elo AS DOUBLE) "
            f"+ {ps_lit_s} * {sanchor_lit}) "
            f"/ NULLIF(CAST(games_played AS DOUBLE) + {ps_lit_s}, 0)"
            f", 0), 3500), 0) AS INTEGER)"
        )
    else:
        published_scratch_expr = "CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER)"

    title_col = f""",
            CASE 
                WHEN published_scratch_int >= 2600 THEN 'SGM'
                WHEN published_scratch_int >= 2500 THEN 'GM'
                WHEN published_scratch_int >= 2400 THEN 'IM'
                WHEN published_scratch_int >= 2300 THEN 'FM'
                WHEN published_scratch_int >= 2200 THEN 'CM'
                WHEN published_scratch_int >= 2000 THEN 'Expert'
                WHEN published_scratch_int >= 1800 THEN 'Advanced'
                WHEN published_scratch_int >= 1600 THEN 'Intermediate'
                WHEN published_scratch_int >= 1400 THEN 'Novice'
                ELSE 'Beginner'
            END AS {title_col_name}"""

    query = f"""
        WITH filtered AS (
            SELECT *
            FROM players_df
            WHERE games_played >= {min_games}
        ),
        ranked AS (
            SELECT
                *,
                CAST(ROUND(LEAST(GREATEST(elo_rating, 0), 3500), 0) AS INTEGER) AS raw_elo_int,
                {published_expr} AS published_elo_int,
                {published_scratch_expr} AS published_scratch_int
            FROM filtered
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY published_elo_int DESC, games_played DESC, player_name ASC, player_id ASC) AS INTEGER) AS Rank,
            published_elo_int AS {elo_col_name},
            raw_elo_int AS {elo_col_name}_Raw{title_col},
            player_id AS Player_ID,
            player_name AS Player_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games
        FROM ranked
        ORDER BY Rank ASC
        LIMIT {top_n}
    """

    result = duckdb.sql(query).pl()
    return result, query, prior_anchor


def show_top_pairs(
    results_df: pl.DataFrame,
    top_n: int,
    min_games: int = 5,
    use_handicap: bool = False,
    players_df: Optional[pl.DataFrame] = None,
    prior_sessions: int = 0,
) -> Tuple[pl.DataFrame, str, Optional[float]]:
    """Get top pairs sorted by Elo rating using SQL.

    When ``prior_sessions > 0`` and a prior anchor (median pair Elo of the
    qualifying subset) is available, the headline pair Elo is the
    Bayesian-shrunk "Published" Elo, computed the same way as for players in
    :func:`show_top_players`. Both Published and Raw are returned in the
    output; the leaderboard is ordered by Published. Returns
    ``(df, sql, prior_anchor)``.
    """
    if results_df.is_empty():
        return results_df, "", None

    elo_col = "handicap_pair_elo" if use_handicap else "scratch_pair_elo"
    pct_col = "COALESCE(handicap_percentage, scratch_percentage)" if use_handicap else "scratch_percentage"

    pair_elo_col_name = "HC_Pair_Elo" if use_handicap else "Pair_Elo"
    title_col_name = "Scratch_Title" if use_handicap else "Title"

    anchor_query = f"""
        WITH pair_anchor AS (
            SELECT
                pair_id,
                ARG_MAX({elo_col}, date) AS avg_pair_elo,
                COUNT(*) AS games_played
            FROM results_df
            GROUP BY pair_id
        )
        SELECT MEDIAN(avg_pair_elo) AS anchor
        FROM pair_anchor
        WHERE games_played >= {min_games}
    """
    anchor_df = duckdb.sql(anchor_query).pl()
    if anchor_df.is_empty():
        prior_anchor: Optional[float] = None
    else:
        val = anchor_df.item(0, 0)
        prior_anchor = float(val) if val is not None else None

    if prior_sessions > 0 and prior_anchor is not None:
        ps_lit = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
        anchor_lit = f"CAST({float(prior_anchor)!r} AS DOUBLE)"
        published_expr = (
            f"CAST(ROUND(LEAST(GREATEST("
            f"(CAST(games_played AS DOUBLE) * CAST(avg_pair_elo AS DOUBLE) "
            f"+ {ps_lit} * {anchor_lit}) "
            f"/ NULLIF(CAST(games_played AS DOUBLE) + {ps_lit}, 0)"
            f", 0), 3500), 0) AS INTEGER)"
        )
    else:
        published_expr = "CAST(ROUND(LEAST(GREATEST(avg_pair_elo, 0), 3500), 0) AS INTEGER)"
    
    # Build Title column - use lower title of the two players based on their
    # *published* (Bayesian-shrunk) scratch Elo, so a pair never inherits an
    # inflated title from a low-sample partner. Each player's scratch Elo is
    # shrunk toward the population scratch median using their own games count,
    # mirroring the headline shrinkage.
    scratch_anchor: Optional[float] = None
    if players_df is not None and not players_df.is_empty():
        sa_df = duckdb.sql(
            f"""
            SELECT MEDIAN(scratch_elo) AS scratch_anchor
            FROM players_df
            WHERE games_played >= {min_games}
            """
        ).pl()
        if not sa_df.is_empty():
            sv = sa_df.item(0, 0)
            scratch_anchor = float(sv) if sv is not None else None

    def _pub_scratch_sql(alias: str) -> str:
        """Shrunk, chess-clamped scratch Elo for a joined player alias."""
        if prior_sessions > 0 and scratch_anchor is not None:
            ps_lit_s = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
            sanchor_lit = f"CAST({float(scratch_anchor)!r} AS DOUBLE)"
            return (
                f"CAST(ROUND(LEAST(GREATEST("
                f"(CAST(COALESCE({alias}.games_played, 0) AS DOUBLE) "
                f"* CAST(COALESCE({alias}.scratch_elo, 0) AS DOUBLE) "
                f"+ {ps_lit_s} * {sanchor_lit}) "
                f"/ NULLIF(CAST(COALESCE({alias}.games_played, 0) AS DOUBLE) + {ps_lit_s}, 0)"
                f", 0), 3500), 0) AS INTEGER)"
            )
        return f"CAST(ROUND(LEAST(GREATEST(COALESCE({alias}.scratch_elo, 0), 0), 3500), 0) AS INTEGER)"

    if players_df is not None and not players_df.is_empty():
        p1_pub = _pub_scratch_sql("p1")
        p2_pub = _pub_scratch_sql("p2")
        # Join with players_df to get individual player scratch Elo and calculate lower title
        title_col = """,
            CASE 
                -- Calculate title rank for player1 (1=SGM, 10=Beginner)
                WHEN p1_pub_scratch >= 2600 THEN 1
                WHEN p1_pub_scratch >= 2500 THEN 2
                WHEN p1_pub_scratch >= 2400 THEN 3
                WHEN p1_pub_scratch >= 2300 THEN 4
                WHEN p1_pub_scratch >= 2200 THEN 5
                WHEN p1_pub_scratch >= 2000 THEN 6
                WHEN p1_pub_scratch >= 1800 THEN 7
                WHEN p1_pub_scratch >= 1600 THEN 8
                WHEN p1_pub_scratch >= 1400 THEN 9
                ELSE 10
            END AS p1_title_rank,
            CASE 
                -- Calculate title rank for player2 (1=SGM, 10=Beginner)
                WHEN p2_pub_scratch >= 2600 THEN 1
                WHEN p2_pub_scratch >= 2500 THEN 2
                WHEN p2_pub_scratch >= 2400 THEN 3
                WHEN p2_pub_scratch >= 2300 THEN 4
                WHEN p2_pub_scratch >= 2200 THEN 5
                WHEN p2_pub_scratch >= 2000 THEN 6
                WHEN p2_pub_scratch >= 1800 THEN 7
                WHEN p2_pub_scratch >= 1600 THEN 8
                WHEN p2_pub_scratch >= 1400 THEN 9
                ELSE 10
            END AS p2_title_rank"""
        
        title_select = f""",
            CASE 
                -- Use GREATEST to get the higher rank number, which corresponds to the lower title
                -- (Higher rank number = lower title: 1=SGM, 2=GM, ..., 10=Beginner)
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 1 THEN 'SGM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 2 THEN 'GM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 3 THEN 'IM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 4 THEN 'FM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 5 THEN 'CM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 6 THEN 'Expert'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 7 THEN 'Advanced'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 8 THEN 'Intermediate'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 9 THEN 'Novice'
                ELSE 'Beginner'
            END AS {title_col_name}"""
    else:
        # Fallback: use pair scratch Elo if players_df not available
        title_col = f""",
            CASE 
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2600 THEN 'SGM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2500 THEN 'GM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2400 THEN 'IM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2300 THEN 'FM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2200 THEN 'CM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2000 THEN 'Expert'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 1800 THEN 'Advanced'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 1600 THEN 'Intermediate'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 1400 THEN 'Novice'
                ELSE 'Beginner'
            END AS {title_col_name}"""
        title_select = ""
    
    query = f"""
        WITH pair_stats AS (
            SELECT 
                pair_id,
                ARG_MAX(pair_name, date) AS pair_name,
                ARG_MAX(player1_id, date) AS player1_id,
                ARG_MAX(player2_id, date) AS player2_id,
                -- Headline Elo uses Latest semantics (ARG_MAX over date) to
                -- match show_top_players, which already uses ARG_MAX. Avoids
                -- the early-tournament-lock-in bias that ACBL's AVG method
                -- exposed: one lucky early session permanently inflating the
                -- AVG even after later results regress to the pair's real
                -- skill level. The percentage / IV / stdev aggregates below
                -- intentionally stay as AVG because users expect "average
                -- across all my tournaments" for those.
                ARG_MAX(scratch_pair_elo, date) AS avg_scratch_elo,
                ARG_MAX(COALESCE(handicap_pair_elo, scratch_pair_elo), date) AS avg_handicap_elo,
                ARG_MAX({elo_col}, date) AS avg_pair_elo,
                AVG(scratch_percentage) AS avg_scratch_pct,
                AVG(COALESCE(handicap_percentage, scratch_percentage)) AS avg_handicap_pct,
                AVG(iv_bonus) AS avg_iv_bonus,
                AVG({pct_col}) AS avg_percentage,
                STDDEV_SAMP({pct_col}) AS stdev_percentage,
                COUNT(*) AS games_played
            FROM results_df
            GROUP BY pair_id
        ),
        filtered AS (
            SELECT *
            FROM pair_stats
            WHERE games_played >= {min_games}
        ),
        ranked AS (
            SELECT
                *,
                CAST(ROUND(LEAST(GREATEST(avg_pair_elo, 0), 3500), 0) AS INTEGER) AS raw_pair_elo_int,
                {published_expr} AS published_pair_elo_int
            FROM filtered
        )"""

    if players_df is not None and not players_df.is_empty():
        query += f""",
        with_player_scratch AS (
            SELECT 
                f.*,
                {p1_pub} AS p1_pub_scratch,
                {p2_pub} AS p2_pub_scratch
            FROM ranked f
            LEFT JOIN players_df p1 ON f.player1_id = p1.player_id
            LEFT JOIN players_df p2 ON f.player2_id = p2.player_id
        ),
        with_player_titles AS (
            SELECT 
                w.*{title_col}
            FROM with_player_scratch w
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY published_pair_elo_int DESC, games_played DESC, pair_name ASC, pair_id ASC) AS INTEGER) AS Rank,
            published_pair_elo_int AS {pair_elo_col_name},
            raw_pair_elo_int AS {pair_elo_col_name}_Raw{title_select},
            pair_id AS Pair_ID,
            pair_name AS Pair_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games
        FROM with_player_titles
        ORDER BY Rank ASC
        LIMIT {top_n}
    """
    else:
        query += f"""
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY published_pair_elo_int DESC, games_played DESC, pair_name ASC, pair_id ASC) AS INTEGER) AS Rank,
            published_pair_elo_int AS {pair_elo_col_name},
            raw_pair_elo_int AS {pair_elo_col_name}_Raw{title_col},
            pair_id AS Pair_ID,
            pair_name AS Pair_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games
        FROM ranked
        ORDER BY Rank ASC
        LIMIT {top_n}
    """

    result = duckdb.sql(query).pl()
    return result, query, prior_anchor


@st.cache_data(show_spinner=False)
def _cached_top_players_both(
    players_df: pl.DataFrame,
    top_n: int,
    min_games: int,
    prior_sessions: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, str, str, Optional[float], Optional[float]]:
    if players_df.is_empty():
        empty = pl.DataFrame()
        return empty, empty, "", "", None, None
    hc, sql_h, anchor_h = show_top_players(
        players_df, top_n, min_games, use_handicap=True, prior_sessions=prior_sessions,
    )
    sc, sql_s, anchor_s = show_top_players(
        players_df, top_n, min_games, use_handicap=False, prior_sessions=prior_sessions,
    )
    return hc, sc, sql_h, sql_s, anchor_h, anchor_s


@st.cache_data(show_spinner=False)
def _cached_top_pairs_both(
    results_df: pl.DataFrame,
    top_n: int,
    min_games: int,
    prior_sessions: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, str, str, Optional[float], Optional[float]]:
    if results_df.is_empty():
        empty = pl.DataFrame()
        return empty, empty, "", "", None, None
    hc, sql_h, anchor_h = show_top_pairs(
        results_df, top_n, min_games, use_handicap=True, players_df=None,
        prior_sessions=prior_sessions,
    )
    sc, sql_s, anchor_s = show_top_pairs(
        results_df, top_n, min_games, use_handicap=False, players_df=None,
        prior_sessions=prior_sessions,
    )
    return hc, sc, sql_h, sql_s, anchor_h, anchor_s


def _aggregate_players_from_results(results_df: pl.DataFrame, use_handicap: bool) -> pl.DataFrame:
    """Aggregate per-player stats from filtered result rows (duckdb)."""
    if results_df.is_empty():
        return pl.DataFrame()
    elo_col_p1 = "player1_handicap_elo_after" if use_handicap else "player1_scratch_elo_after"
    elo_col_p2 = "player2_handicap_elo_after" if use_handicap else "player2_scratch_elo_after"
    pct_expr = "COALESCE(handicap_percentage, scratch_percentage)" if use_handicap else "scratch_percentage"
    return duckdb.sql(f"""
        WITH player_results AS (
            SELECT
                player1_id AS player_id,
                player1_name AS player_name,
                player1_scratch_elo_after AS scratch_elo,
                player1_handicap_elo_after AS handicap_elo,
                {elo_col_p1} AS elo_rating,
                scratch_percentage,
                handicap_percentage,
                iv_bonus,
                date
            FROM results_df
            UNION ALL
            SELECT
                player2_id AS player_id,
                player2_name AS player_name,
                player2_scratch_elo_after AS scratch_elo,
                player2_handicap_elo_after AS handicap_elo,
                {elo_col_p2} AS elo_rating,
                scratch_percentage,
                handicap_percentage,
                iv_bonus,
                date
            FROM results_df
        )
        SELECT
            player_id,
            ARG_MAX(player_name, date) AS player_name,
            ROUND(ARG_MAX(scratch_elo, date), 1) AS scratch_elo,
            ROUND(ARG_MAX(COALESCE(handicap_elo, scratch_elo), date), 1) AS handicap_elo,
            ROUND(ARG_MAX(elo_rating, date), 1) AS elo_rating,
            COUNT(*) AS games_played,
            ROUND(AVG(scratch_percentage), 2) AS avg_scratch_pct,
            ROUND(AVG(COALESCE(handicap_percentage, scratch_percentage)), 2) AS avg_handicap_pct,
            ROUND(AVG(iv_bonus), 1) AS avg_iv_bonus,
            ROUND(AVG({pct_expr}), 2) AS avg_percentage,
            ROUND(STDDEV_SAMP({pct_expr}), 2) AS stdev_percentage
        FROM player_results
        GROUP BY player_id
    """).pl()


@st.cache_data(show_spinner=False)
def _aggregate_players_from_results_cached(
    results_df: pl.DataFrame,
    use_handicap: bool,
) -> pl.DataFrame:
    return _aggregate_players_from_results(results_df, use_handicap)


def _player_metric_triple(players_df: pl.DataFrame, min_games: int) -> Tuple[int, float, float]:
    metric_players = players_df.filter(pl.col('games_played') >= min_games) if not players_df.is_empty() else players_df
    if metric_players.is_empty():
        return 0, 0.0, 0.0
    return (
        len(metric_players),
        metric_players.select(pl.col('games_played').mean()).item(),
        metric_players.select(pl.col('elo_rating').max()).item(),
    )


def _pair_metric_triple(results_df: pl.DataFrame, min_games: int, use_handicap: bool) -> Tuple[int, float, float]:
    if results_df.is_empty():
        return 0, 0.0, 0.0
    elo_col = "handicap_pair_elo" if use_handicap else "scratch_pair_elo"
    pair_stats = (
        results_df.group_by("pair_id")
        .agg(
            pl.len().alias("games"),
            pl.col(elo_col).mean().alias("elo"),
        )
        .filter(pl.col("games") >= min_games)
    )
    if pair_stats.is_empty():
        return 0, 0.0, 0.0
    return (
        pair_stats.height,
        pair_stats.select(pl.col("games").mean()).item(),
        pair_stats.select(pl.col("elo").max()).item(),
    )


@st.cache_data(show_spinner=False)
def _cached_pair_metric_triple(
    results_df: pl.DataFrame, min_games: int, use_handicap: bool,
) -> Tuple[int, float, float]:
    """Cached pair leaderboard metrics (expensive on full result history)."""
    return _pair_metric_triple(results_df, min_games, use_handicap)


# -------------------------------
# Session State Initialization
# -------------------------------
def initialize_session_state():
    """Initialize session state variables on first run."""
    if 'first_time' not in st.session_state:
        st.session_state.first_time = True
        st.session_state.app_datetime = datetime.fromtimestamp(
            pathlib.Path(__file__).stat().st_mtime, 
            tz=timezone.utc
        ).strftime('%Y-%m-%d %H:%M:%S %Z')
    else:
        st.session_state.first_time = False


# Default API: Lancelot is public and currently the reliable source. Classic
# requires a bearer token and has been intermittently unavailable; set
# FFBRIDGE_PREFER_CLASSIC_API=1 once Classic is healthy again to default to it
# when FFBRIDGE_BEARER_TOKEN is present.
def _default_ffbridge_api() -> str:
    prefer_classic = os.getenv("FFBRIDGE_PREFER_CLASSIC_API", "").strip().lower() in (
        "1", "true", "yes",
    )
    if prefer_classic and os.getenv("FFBRIDGE_BEARER_TOKEN", "").strip():
        return "FFBridge Classic API"
    return "FFBridge Lancelot API"


_FFBRIDGE_DEFAULT_API = _default_ffbridge_api()
_ALL_CLUBS_LABEL = "All Clubs"
# Process-level club dropdown cache (shared across Streamlit sessions in one container).
_FFBRIDGE_CLUB_OPTIONS: list[str] | None = None


def _ffbridge_club_select_options() -> list[str]:
    """Club dropdown options; always includes All Clubs (never an empty list)."""
    raw = st.session_state.get("elo_available_clubs")
    if not raw:
        return [_ALL_CLUBS_LABEL]
    names = [_ALL_CLUBS_LABEL]
    seen = {_ALL_CLUBS_LABEL}
    for club in raw:
        label = str(club).strip()
        if label and label not in seen:
            names.append(label)
            seen.add(label)
    return names


def _ffbridge_dataset_summary_line(
    *,
    selected_api_name: str,
    list_source: str,
    n_tournaments: int,
    processing_stats: dict,
    result_rows: int,
    list_fallback_note: str = "",
) -> str:
    """One-line Elo dataset load summary for the main-page caption."""
    missing_ids = processing_stats.get("missing_ids", [])
    missing_str = str(missing_ids) if missing_ids else "none"
    parts = [
        f"API {selected_api_name}",
        f"tournament list {list_source}",
        f"cached {processing_stats.get('cached', 0)}",
        f"fetched {processing_stats.get('fetched', 0)}",
        f"missing results {missing_str}",
        f"{n_tournaments} tournaments",
        f"{result_rows:,} result rows",
    ]
    if list_fallback_note:
        parts.append(list_fallback_note)
    return " • ".join(parts)


def _ffbridge_footer_diagnostics_lines(st_module) -> list[str]:
    """Streamlit container + cache stats for the page footer."""
    lines = [footer_streamlit_app_diagnostics_line(st_module)]

    cache_line = get_cache_diagnostic_line("FFBRIDGE_CACHE_DIR")
    if cache_line.startswith("Cache (FFBRIDGE_CACHE_DIR):"):
        cache_line = "Cache — " + cache_line.split(":", 1)[1].strip()
    lines.append(cache_line)
    return lines


# URL query param -> sidebar widget session state.
# Keys are short, URL-friendly names; session_key matches the widget's `key=...`.
FFBRIDGE_URL_PARAMS = {
    "api": {
        "session_key": "selected_api_widget",
        "parser": str,
        "valid_values": tuple(API_BACKENDS.keys()),
        "default": _FFBRIDGE_DEFAULT_API,
    },
    "tournament": {
        "session_key": "elo_tournament_selectbox",
        "parser": str,
        "default": "All Tournaments",
    },
    "club": {
        "session_key": "elo_club_filter",
        "parser": str,
        "default": "",
    },
    "name": {
        "session_key": "elo_name_filter",
        "parser": str,
        "default": "",
    },
    "rating": {
        "session_key": "elo_rating_type",
        "parser": str,
        "valid_values": ("Players", "Pairs"),
        "default": "Players",
    },
    "score": {
        "session_key": "elo_score_type",
        "parser": str,
        "valid_values": ("Scratch", "Handicap"),
        "default": "Scratch",
    },
    "top_n": {
        "session_key": "elo_top_n",
        "parser": coerce_int(50, 1000, 50),
        "default": 250,
    },
    "min_games": {
        "session_key": "elo_min_games",
        "parser": coerce_int(1, 100),
        "default": 10,
    },
    "prior_sessions": {
        "session_key": "elo_prior_sessions",
        "parser": coerce_int(0, 1000),
        "default": 50,
    },
}



@st.fragment
def _ffbridge_leaderboard_panel(metric_m2, metric_m3, metric_m4) -> None:
    # Fragment remounts on Scratch/Handicap (and other session-state) changes so
    # AgGrid re-renders without re-running the full dataset load path.
    _load_debug_log("leaderboard panel started")
    ctx = st.session_state.get("_ff_lb_ctx")
    if not ctx:
        _load_debug_log("leaderboard panel: no _ff_lb_ctx; skipping")
        return
    score_type = st.session_state.get("elo_score_type", "Scratch")
    use_handicap = score_type == "Handicap"
    results_df = ctx["results_df"]
    players_df = _aggregate_players_from_results_cached(results_df, use_handicap)
    _load_debug_log(
        f"leaderboard panel: aggregated players ({players_df.height} rows, "
        f"handicap={use_handicap})"
    )
    rating_type = ctx["rating_type"]
    simultaneous_type = ctx["simultaneous_type"]
    club_filter = ctx["club_filter"]
    name_filter = ctx["name_filter"]
    top_n = ctx["top_n"]
    min_games = ctx["min_games"]
    prior_sessions = ctx["prior_sessions"]
    if results_df.is_empty():
        st.info("No results found for the selected filters.")
        return
    if rating_type == "Players":
        active_count, avg_games, highest_elo = ctx["player_metrics_hc"] if use_handicap else ctx["player_metrics_sc"]
    else:
        active_count, avg_games, highest_elo = ctx["pair_metrics_hc"] if use_handicap else ctx["pair_metrics_sc"]
    metric_m2.markdown(
        f'<div class="metric-card"><small>{"Active Players" if rating_type == "Players" else "Active Pairs"}</small><br>'
        f'<span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{active_count}</span></div>',
        unsafe_allow_html=True,
    )
    metric_m3.markdown(
        f'<div class="metric-card"><small>{"Avg Games" if rating_type == "Players" else "Avg Games/Pair"}</small><br>'
        f'<span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{avg_games:.1f}</span></div>',
        unsafe_allow_html=True,
    )
    highest_elo_display = 0 if highest_elo is None or not math.isfinite(highest_elo) else round(highest_elo)
    metric_m4.markdown(
        f'<div class="metric-card"><small>{"Highest Elo" if rating_type == "Players" else "Highest Pair Elo"}</small><br>'
        f'<span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{highest_elo_display}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.full_results_df = results_df
    # Display tables
    if rating_type == "Players":
        if not players_df.is_empty():
            _load_debug_log("leaderboard panel: running top players SQL (hc+sc)")
            hc_players, sc_players, sql_h, sql_s, anchor_h, anchor_s = _cached_top_players_both(
                players_df, top_n, min_games, int(prior_sessions),
            )
            _load_debug_log(
                f"leaderboard panel: top players ready "
                f"(hc={hc_players.height}, sc={sc_players.height} rows)"
            )
            top_players = hc_players if use_handicap else sc_players
            sql_query = sql_h if use_handicap else sql_s
            prior_anchor = anchor_h if use_handicap else anchor_s

            if prior_sessions > 0 and prior_anchor is not None:
                st.caption(
                    f"Headline shows **Published Elo** (Bayesian-shrunk toward "
                    f"median of qualifying players ≈ {prior_anchor:.0f}, "
                    f"prior_sessions={prior_sessions}). `{'HC_' if use_handicap else ''}Player_Elo_Raw` "
                    f"column shown alongside."
                )
            elif prior_sessions > 0:
                st.caption("Shrinkage unavailable: no qualifying players to anchor the prior. Headline shows Raw Elo.")
            else:
                st.caption("Shrinkage disabled (prior_sessions=0). Headline shows Raw Elo.")

            st.caption(
                " Elo is standardized to a chess-style scale (mean **1500**, "
                "sd **400**) so it lines up with the ACBL app and chess; the "
                "**Title** column uses standard bands (≥2400 IM, ≥2500 GM, "
                "≥2600 SGM). A given title means the same percentile everywhere."
            )

            if sql_query:
                with st.expander("SQL Query", expanded=False):
                    st.code(sql_query, language="sql")

            if not top_players.is_empty():
                # Apply name filter if provided
                if name_filter and name_filter.strip():
                    name_filter_lower = name_filter.strip().lower()
                    top_players = top_players.filter(
                        pl.col('Player_Name').str.to_lowercase().str.contains(name_filter_lower, literal=True)
                    )

                if not top_players.is_empty():
                    st.caption("Click a row to view player's tournament history")
                    # Create dynamic key based on filter parameters to reset selection when data changes
                    dynamic_key = _leaderboard_aggrid_key(
                        "players", rating_type, simultaneous_type,
                        club_filter, top_n, min_games, name_filter, int(prior_sessions),
                    )
                    _load_debug_log(f"leaderboard panel: rendering AgGrid ({top_players.height} rows)")
                    grid_response = build_selectable_aggrid(top_players, dynamic_key, render_links=False)
                    _load_debug_log("leaderboard panel: AgGrid render returned")

                    selected_rows = grid_response.get('selected_rows', None)
                    if selected_rows is not None and len(selected_rows) > 0:
                        selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                        player_id = selected_row.get('Player_ID')
                        player_name = selected_row.get('Player_Name', 'Unknown')

                        if player_id and not results_df.is_empty():
                            # Show tournament history
                            st.markdown(f"#### Tournament History Details: **{player_name}**")
                            player_results = results_df.filter(
                                (pl.col('player1_id') == str(player_id)) | 
                                (pl.col('player2_id') == str(player_id))
                            ).sort('date', descending=True)

                            if not player_results.is_empty():
                                cols_to_select = [
                                    pl.col('date').str.slice(0, 10).alias('Date'),
                                    pl.col('tournament_id').alias('Event_ID'),
                                    pl.col('pair_name').alias('Partner'),
                                    (pl.col('scratch_percentage') if 'scratch_percentage' in player_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Scratch_%'),
                                    (pl.col('handicap_percentage') if 'handicap_percentage' in player_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Handicap_%'),
                                    (pl.col('iv_bonus') if 'iv_bonus' in player_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(1).alias('IV_Bonus'),
                                    ((pl.col('handicap_field_strength') if use_handicap else pl.col('scratch_field_strength')) if (('handicap_field_strength' if use_handicap else 'scratch_field_strength') in player_results.columns) else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(3).alias('Field_Strength'),
                                ]
                                # Pct_Used: use handicap if requested AND not null, otherwise scratch
                                if use_handicap and 'handicap_percentage' in player_results.columns:
                                    # Use handicap when available, fall back to scratch when null
                                    cols_to_select.append(
                                        pl.when(pl.col('handicap_percentage').is_not_null())
                                          .then(pl.col('handicap_percentage').cast(pl.Float64, strict=False).round(2))
                                          .otherwise(pl.col('scratch_percentage').cast(pl.Float64, strict=False).round(2))
                                          .alias('Pct_Used')
                                    )
                                else:
                                    # Use scratch
                                    cols_to_select.append(
                                        (pl.col('scratch_percentage') if 'scratch_percentage' in player_results.columns else pl.col('percentage'))
                                          .cast(pl.Float64, strict=False).round(2).alias('Pct_Used')
                                    )
                                # Dynamically select Rank based on current use_handicap setting
                                if use_handicap and 'theoretical_rank' in player_results.columns:
                                    cols_to_select.append(pl.col('theoretical_rank').alias('Rank'))
                                else:
                                    cols_to_select.append(pl.col('rank').alias('Rank'))
                                # Add current IV (note: this is current IV, not IV at tournament time)
                                if 'pair_iv' in player_results.columns:
                                    cols_to_select.append(pl.col('pair_iv').alias('Current_Pair_IV'))
                                # Dynamically select Elo based on current use_handicap setting
                                elo_col = 'player1_handicap_elo_after' if use_handicap else 'player1_scratch_elo_after'
                                if elo_col in player_results.columns:
                                    cols_to_select.append(pl.col(elo_col).round(0).alias('Elo_After'))
                                elif 'player1_elo_after' in player_results.columns:
                                    cols_to_select.append(pl.col('player1_elo_after').round(0).alias('Elo_After'))

                                # Also keep helper columns for URL construction and opponent lookup.
                                helper_cols = [pl.col('pair_id').alias('_pair_id')]
                                if 'team_id' in player_results.columns:
                                    helper_cols.append(pl.col('team_id').alias('_team_id'))
                                if 'club_id' in player_results.columns:
                                    helper_cols.append(pl.col('club_id').alias('_club_id'))
                                detail_df = player_results.select(cols_to_select + helper_cols)
                                # Optional reconciliation: for Octopus, prefer BridgeInterNet scratch/handicap when matchable.
                                detail_df = _maybe_override_octopus_pct_rows(detail_df, pair_name=player_name, use_handicap=use_handicap)

                                st.caption("Click a row to see tournament opponents")
                                # Hide helper columns from display
                                hidden = [c for c in ('_pair_id', '_team_id', '_club_id') if c in detail_df.columns]
                                display_detail = detail_df.drop(hidden) if hidden else detail_df
                                detail_key = f"detail_player_{player_id}_{rating_type}_{score_type}"
                                detail_resp = _render_detail_aggrid_ff(display_detail, key=detail_key, selectable=True)

                                # 3rd df: tournament opponents for clicked row
                                if detail_resp is not None:
                                    d_sel = detail_resp.get('selected_rows', None)
                                    if d_sel is not None and len(d_sel) > 0:
                                        d_row = d_sel.iloc[0] if hasattr(d_sel, 'iloc') else d_sel[0]
                                        event_id = str(d_row.get('Event_ID', ''))
                                        # Find pair_id for this row
                                        row_pair_id = None
                                        if event_id and '_pair_id' in detail_df.columns:
                                            match = detail_df.filter(pl.col('Event_ID') == event_id)
                                            if not match.is_empty():
                                                row_pair_id = str(match.select('_pair_id').row(0)[0])
                                        if event_id:
                                            _show_tournament_opponents(results_df, event_id, exclude_pair_id=row_pair_id)

                                # 4th df: partner aggregation across all tournaments
                                _show_partner_aggregation(display_detail, key_suffix=f"player_{player_id}_{score_type}")

                                # Opponent History Details + Opponent Summary
                                _show_opponent_history(results_df, player_results, exclude_id=str(player_id), exclude_mode='player', key_suffix=f"player_{player_id}_{score_type}")
                                _show_opponent_summary(results_df, player_results, exclude_id=str(player_id), exclude_mode='player', key_suffix=f"player_{player_id}_{score_type}")
                            else:
                                st.info("No results in selected tournaments.")

                    st.session_state.display_df = top_players
                    st.session_state.report_title = f"FFBridge Top Players - {datetime.now().strftime('%Y-%m-%d')}"
                else:
                    if name_filter and name_filter.strip():
                        st.info(f"No players match the name filter '{name_filter}'.")
                    else:
                        st.info(f"No players match the minimum requirement of {min_games} games.")
            else:
                st.info(f"No players match the minimum requirement of {min_games} games.")
    else:
        if not results_df.is_empty():
            _load_debug_log("leaderboard panel: running top pairs SQL (hc+sc)")
            hc_pairs, sc_pairs, sql_h, sql_s, anchor_h, anchor_s = _cached_top_pairs_both(
                results_df, top_n, min_games, int(prior_sessions),
            )
            _load_debug_log(
                f"leaderboard panel: top pairs ready "
                f"(hc={hc_pairs.height}, sc={sc_pairs.height} rows)"
            )
            top_pairs = hc_pairs if use_handicap else sc_pairs
            sql_query = sql_h if use_handicap else sql_s
            prior_anchor = anchor_h if use_handicap else anchor_s

            if prior_sessions > 0 and prior_anchor is not None:
                st.caption(
                    f"Headline shows **Published Elo** (Bayesian-shrunk toward "
                    f"median of qualifying pairs ≈ {prior_anchor:.0f}, "
                    f"prior_sessions={prior_sessions}). `{'HC_' if use_handicap else ''}Pair_Elo_Raw` "
                    f"column shown alongside."
                )
            elif prior_sessions > 0:
                st.caption("Shrinkage unavailable: no qualifying pairs to anchor the prior. Headline shows Raw Elo.")
            else:
                st.caption("Shrinkage disabled (prior_sessions=0). Headline shows Raw Elo.")

            st.caption(
                " Elo is standardized to a chess-style scale (mean **1500**, "
                "sd **400**) so it lines up with the ACBL app and chess; the "
                "**Title** column uses standard bands (≥2400 IM, ≥2500 GM, "
                "≥2600 SGM). A given title means the same percentile everywhere."
            )

            if sql_query:
                with st.expander("SQL Query", expanded=False):
                    st.code(sql_query, language="sql")

            if not top_pairs.is_empty():
                # Apply name filter if provided
                if name_filter and name_filter.strip():
                    name_filter_lower = name_filter.strip().lower()
                    top_pairs = top_pairs.filter(
                        pl.col('Pair_Name').str.to_lowercase().str.contains(name_filter_lower, literal=True)
                    )

                if not top_pairs.is_empty():
                    st.caption("Click a row to view pair's tournament history")
                    # Create dynamic key based on filter parameters to reset selection when data changes
                    dynamic_key = _leaderboard_aggrid_key(
                        "pairs", rating_type, simultaneous_type,
                        club_filter, top_n, min_games, name_filter, int(prior_sessions),
                    )
                    _load_debug_log(f"leaderboard panel: rendering AgGrid ({top_pairs.height} rows)")
                    grid_response = build_selectable_aggrid(top_pairs, dynamic_key, render_links=False)
                    _load_debug_log("leaderboard panel: AgGrid render returned")

                    selected_rows = grid_response.get('selected_rows', None)
                    if selected_rows is not None and len(selected_rows) > 0:
                        selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                        pair_id = selected_row.get('Pair_ID')
                        pair_name = selected_row.get('Pair_Name', 'Unknown')

                        if pair_id and not results_df.is_empty():
                            st.markdown(f"### Tournament History Details for **{pair_name}**")

                            pair_results = results_df.filter(
                                pl.col('pair_id') == str(pair_id)
                            ).sort('date', descending=True)

                            if not pair_results.is_empty():
                                cols_to_select = [
                                    pl.col('date').str.slice(0, 10).alias('Date'),
                                    pl.col('tournament_id').alias('Event_ID'),
                                    (pl.col('scratch_percentage') if 'scratch_percentage' in pair_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Scratch_%'),
                                    (pl.col('handicap_percentage') if 'handicap_percentage' in pair_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Handicap_%'),
                                    (pl.col('iv_bonus') if 'iv_bonus' in pair_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(1).alias('IV_Bonus'),
                                    ((pl.col('handicap_field_strength') if use_handicap else pl.col('scratch_field_strength')) if (('handicap_field_strength' if use_handicap else 'scratch_field_strength') in pair_results.columns) else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(3).alias('Field_Strength'),
                                ]
                                # Pct_Used: use handicap if requested AND not null, otherwise scratch
                                if use_handicap and 'handicap_percentage' in pair_results.columns:
                                    # Use handicap when available, fall back to scratch when null
                                    cols_to_select.append(
                                        pl.when(pl.col('handicap_percentage').is_not_null())
                                          .then(pl.col('handicap_percentage').cast(pl.Float64, strict=False).round(2))
                                          .otherwise(pl.col('scratch_percentage').cast(pl.Float64, strict=False).round(2))
                                          .alias('Pct_Used')
                                    )
                                else:
                                    # Use scratch
                                    cols_to_select.append(
                                        (pl.col('scratch_percentage') if 'scratch_percentage' in pair_results.columns else pl.col('percentage'))
                                          .cast(pl.Float64, strict=False).round(2).alias('Pct_Used')
                                    )
                                # Dynamically select Rank based on current use_handicap setting
                                if use_handicap and 'theoretical_rank' in pair_results.columns:
                                    cols_to_select.append(pl.col('theoretical_rank').alias('Rank'))
                                else:
                                    cols_to_select.append(pl.col('rank').alias('Rank'))
                                # Add current IV (note: this is current IV, not IV at tournament time)
                                if 'pair_iv' in pair_results.columns:
                                    cols_to_select.append(pl.col('pair_iv').alias('Current_Pair_IV'))
                                # Dynamically select Pair Elo based on current use_handicap setting
                                pair_elo_col = 'pair_handicap_elo' if use_handicap else 'pair_scratch_elo'
                                pair_elo_alias = 'HC_Pair_Elo' if use_handicap else 'Pair_Elo'
                                if pair_elo_col in pair_results.columns:
                                    cols_to_select.append(pl.col(pair_elo_col).round(0).alias(pair_elo_alias))
                                elif 'pair_elo' in pair_results.columns:
                                    cols_to_select.append(pl.col('pair_elo').round(0).alias(pair_elo_alias))

                                # Keep helper columns for URL construction; hidden from display.
                                helper_cols = []
                                if 'team_id' in pair_results.columns:
                                    helper_cols.append(pl.col('team_id').alias('_team_id'))
                                if 'club_id' in pair_results.columns:
                                    helper_cols.append(pl.col('club_id').alias('_club_id'))
                                detail_df = pair_results.select(cols_to_select + helper_cols)
                                # Optional reconciliation: for Octopus, prefer BridgeInterNet scratch/handicap when matchable.
                                detail_df = _maybe_override_octopus_pct_rows(detail_df, pair_name=pair_name, use_handicap=use_handicap)

                                st.caption("Click a row to see tournament opponents")
                                # Hide helper columns from display
                                hidden = [c for c in ('_team_id', '_club_id') if c in detail_df.columns]
                                display_detail = detail_df.drop(hidden) if hidden else detail_df
                                detail_key = f"detail_pair_{pair_id}_{rating_type}_{score_type}"
                                detail_resp = _render_detail_aggrid_ff(display_detail, key=detail_key, selectable=True)

                                # 3rd df: tournament opponents for clicked row
                                if detail_resp is not None:
                                    d_sel = detail_resp.get('selected_rows', None)
                                    if d_sel is not None and len(d_sel) > 0:
                                        d_row = d_sel.iloc[0] if hasattr(d_sel, 'iloc') else d_sel[0]
                                        event_id = str(d_row.get('Event_ID', ''))
                                        if event_id:
                                            _show_tournament_opponents(results_df, event_id, exclude_pair_id=str(pair_id))

                                # 4th df: club aggregation across all tournaments
                                _show_club_aggregation(display_detail, results_df, str(pair_id), key_suffix=f"pair_{pair_id}_{score_type}")

                                # Opponent History Details + Opponent Summary
                                _show_opponent_history(results_df, pair_results, exclude_id=str(pair_id), exclude_mode='pair', key_suffix=f"pair_{pair_id}_{score_type}")
                                _show_opponent_summary(results_df, pair_results, exclude_id=str(pair_id), exclude_mode='pair', key_suffix=f"pair_{pair_id}_{score_type}")
                            else:
                                st.info("No detailed results found for this pair.")

                    st.session_state.display_df = top_pairs
                    st.session_state.report_title = f"FFBridge Top Pairs - {datetime.now().strftime('%Y-%m-%d')}"
                else:
                    if name_filter and name_filter.strip():
                        st.info(f"No pairs match the name filter '{name_filter}'.")
                    else:
                        st.info(f"No pairs match the minimum requirement of {min_games} games.")
            else:
                st.info(f"No pairs match the minimum requirement of {min_games} games.")

    _load_debug_log("leaderboard panel complete")


# -------------------------------
# Main UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="Unofficial FFBridge Elo Ratings Playground",
        page_icon=ASSISTANT_LOGO_URL,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()

    # Apply URL query params -> session state BEFORE widgets render (first run only).
    init_url_params_to_state(st, FFBRIDGE_URL_PARAMS)

    # Apply common theme
    apply_app_theme(st)
    widen_scrollbars()
    
    # -------------------------------
    # Sidebar Controls
    # -------------------------------
    with st.sidebar:
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
        st.sidebar.markdown("[What is Elo Rating?](https://en.wikipedia.org/wiki/Elo_rating_system)")

        st.radio(
            "Elo Based On",
            ["Scratch", "Handicap"],
            index=0,
            key="elo_score_type",
            horizontal=True,
            help="Choose which percentage to use for Elo calculations (rankings always sorted by Elo)",
        )

        rating_type = st.radio(
            "Ranking Type",
            ["Players", "Pairs"],
            index=0,
            key="elo_rating_type",
            horizontal=True,
            help="Switch between individual and partnership rankings"
        )
        
        # API Backend selection
        # Keep widget state and canonical state separate so explicit reruns do not
        # unexpectedly reset the selected API.
        api_options = list(API_BACKENDS.keys())
        default_api = _default_ffbridge_api()
        if "selected_api" not in st.session_state:
            st.session_state.selected_api = default_api
        if st.session_state.selected_api not in api_options:
            st.session_state.selected_api = default_api
        if "selected_api_widget" not in st.session_state:
            st.session_state.selected_api_widget = st.session_state.selected_api
        if st.session_state.selected_api_widget not in api_options:
            st.session_state.selected_api_widget = st.session_state.selected_api
        
        selected_api_name = st.selectbox(
            "Bridge API",
            options=api_options,
            key="selected_api_widget",
            help=(
                "Lancelot is the public API and the default while Classic is "
                "unavailable. Classic requires FFBRIDGE_BEARER_TOKEN."
            ),
        )
        st.session_state.selected_api = selected_api_name
        
        # Get the appropriate API module
        api_module = API_BACKENDS[selected_api_name]
        
        # Check authentication if required; auto-fallback to Lancelot for easier deployment.
        if api_module.REQUIRES_AUTH and not api_module.is_authenticated():
            fallback_api = "FFBridge Lancelot API"
            if fallback_api in API_BACKENDS:
                st.warning("Classic API auth missing. Automatically using Lancelot API.")
                st.session_state.selected_api = fallback_api
                st.session_state.selected_api_widget = fallback_api
                selected_api_name = fallback_api
                api_module = API_BACKENDS[selected_api_name]
            else:
                st.error("**Authentication Error**")
                st.markdown(api_module.get_auth_error_message())
                return
        
        # Tournament selection
        series_names = api_module.SERIES_NAMES
        valid_series_ids = api_module.VALID_SERIES_IDS
        # Handicap tournament IDs that should have ' (H)' appended
        handicap_tournament_ids = {3, 4, 5, 384, 386}
        # Create list of (label, id) tuples for sorting
        tournament_items = [("All Tournaments", "all")]
        tournament_items.extend([
            (series_names[k] + (' (H)' if k in handicap_tournament_ids else ''), k)
            for k in valid_series_ids
        ])
        # Sort by label alphabetically, keeping "all" first
        all_item = tournament_items[0]
        other_items = sorted(tournament_items[1:], key=lambda x: x[0])
        tournament_items = [all_item] + other_items
        # Extract sorted lists
        tournament_labels = [item[0] for item in tournament_items]
        tournament_options_list = [item[1] for item in tournament_items]
        
        if "elo_tournament_selectbox" not in st.session_state:
            st.session_state.elo_tournament_selectbox = tournament_labels[0]
        
        if st.session_state.elo_tournament_selectbox not in tournament_labels:
            st.session_state.elo_tournament_selectbox = tournament_labels[0]
        
        selected_tournament_label = st.selectbox(
            "Filter by Tournament",
            options=tournament_labels,
            key="elo_tournament_selectbox",
            help="Select which simultaneous tournaments to analyze"
        )
        
        simultaneous_type = tournament_options_list[tournament_labels.index(selected_tournament_label)]
        
        # Text input avoids a 500+ option selectbox remounted on every toggle
        # (previously a SIGSEGV contributor alongside AgGrid).
        club_filter = st.text_input(
            "Filter by Club",
            key="elo_club_filter",
            help="Partial club name match (case-insensitive). Leave blank for all clubs.",
        ).strip()
        
        # Name filter
        name_filter = st.text_input(
            "Filter by Name",
            key="elo_name_filter",
            help="Filter results by player or pair name (case-insensitive)"
        )

        # IV fetching is always enabled (cached, refreshes monthly on 15th)
        fetch_iv = True
        
        # Number of results
        top_n = st.slider(
            "Show Top N",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            key="elo_top_n"
        )
        
        # Minimum games
        min_games = st.slider(
            "Minimum Games",
            min_value=1,
            max_value=100,
            value=10,
            key="elo_min_games",
            help="Minimum tournaments played to appear in rankings"
        )

        prior_sessions = st.slider(
            "Shrinkage Prior (sessions)",
            min_value=0,
            max_value=200,
            value=50,
            step=5,
            key="elo_prior_sessions",
            help=(
                "Bayesian shrinkage prior weight in 'sessions equivalent'. "
                "Higher values pull less-played players/pairs toward the "
                "global median rating, dampening leaderboard noise from "
                "small-sample outliers. 0 disables shrinkage (headline shows Raw Elo)."
            ),
        )

        # PDF Export
        generate_pdf = st.button("Export Report to PDF File", width='stretch')
        st.sidebar.markdown('<p style="color: #ffc107; font-weight: 600;">Morty\'s Automated Postmortem Apps</p>', unsafe_allow_html=True)
        st.sidebar.markdown("[ACBL Postmortem](https://acbl.postmortem.chat)")
        st.sidebar.markdown("[French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
        st.sidebar.markdown("[Calculate PBN](https://pbn.postmortem.chat)")
        #st.sidebar.markdown("[BridgeWebs Postmortem](https://bridgewebs.postmortem.chat)")

    # Persist current sidebar state to URL query params for shareable links.
    sync_state_to_url_params(st, FFBRIDGE_URL_PARAMS)

    # Header
    st.markdown(f"""
        <div style="text-align: center; padding: 0 0 1rem 0; margin-top: -2rem;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">
                <img src="{ASSISTANT_LOGO_URL}" style="height: 2.5rem; vertical-align: middle; margin-right: 0.5rem;">
                Morty's Unofficial FFBridge Elo Ratings Playground
            </h1>
            <p style="color: #ffc107; font-size: 1.2rem; font-weight: 500; opacity: 0.9;">
                 {selected_tournament_label} using {selected_api_name}
            </p>
        </div>
    """, unsafe_allow_html=True)

    stats_placeholder = st.empty()

    # -------------------------------
    # Main Content
    # -------------------------------
    _load_debug_log("main content started", reset=True)

    try:
        _load_main_content(
            stats_placeholder=stats_placeholder,
            selected_api_name=selected_api_name,
            api_module=api_module,
            selected_tournament_label=selected_tournament_label,
            simultaneous_type=simultaneous_type,
            club_filter=club_filter,
            name_filter=name_filter,
            rating_type=rating_type,
            top_n=top_n,
            min_games=min_games,
            prior_sessions=prior_sessions,
            fetch_iv=fetch_iv,
            generate_pdf=generate_pdf,
        )
        _load_debug_log("main content complete")
    except Exception as exc:
        _load_debug_log(f"FAILED: {type(exc).__name__}: {exc}")
        if _FFBRIDGE_LOAD_DEBUG:
            import traceback
            traceback.print_exc()
        raise


def _load_main_content(
    *,
    stats_placeholder,
    selected_api_name: str,
    api_module,
    selected_tournament_label: str,
    simultaneous_type,
    club_filter: str,
    name_filter: str,
    rating_type: str,
    top_n: int,
    min_games: int,
    prior_sessions: int,
    fetch_iv: bool,
    generate_pdf: bool,
) -> None:
    """Load dataset, prepare leaderboard context, and render the main panel."""

    # Cache key includes API name to separate caches
    api_key = selected_api_name.replace(" ", "_")
    
    # Fetch all tournaments
    max_age_hours = float(os.environ.get("FFBRIDGE_ELO_MAX_AGE_HOURS", "20"))
    cache_age = _newest_persisted_age_hours(api_key, fetch_iv)
    # Refresh the session list from the API only when persisted Elo parquet is
    # stale. Missing parquet must still use the on-disk Lancelot session-list cache;
    # forcing a live API fetch on cold start caused "Failed to retrieve tournament
    # data" in production when outbound API access was slow or unavailable.
    force_list_refresh = cache_age is not None and cache_age >= max_age_hours
    _load_debug_log(
        f"tournament list fetch (force_refresh={force_list_refresh}, "
        f"parquet_age={cache_age if cache_age is not None else 'none'})"
    )

    with st.spinner("Loading tournament data..."):
        all_tournaments, list_source = _fetch_tournament_list_resilient(
            api_module, force_list_refresh,
        )

        # Classic has been down for extended periods while tokens remain valid.
        # Fall back to Lancelot rather than erroring or triggering a doomed rebuild.
        if (
            not all_tournaments
            and selected_api_name == "FFBridge Classic API"
            and "FFBridge Lancelot API" in API_BACKENDS
        ):
            st.warning(
                "Classic API returned no tournaments (service may be unavailable). "
                "Falling back to Lancelot API."
            )
            selected_api_name = "FFBridge Lancelot API"
            api_module = API_BACKENDS[selected_api_name]
            api_key = selected_api_name.replace(" ", "_")
            st.session_state.selected_api = selected_api_name
            st.session_state.selected_api_widget = selected_api_name
            cache_age = _newest_persisted_age_hours(api_key, fetch_iv)
            force_list_refresh = cache_age is not None and cache_age >= max_age_hours
            all_tournaments, list_source = _fetch_tournament_list_resilient(
                api_module, force_list_refresh,
            )
        
        if not all_tournaments:
            age_label = "none" if cache_age is None else f"{cache_age:.1f}h"
            st.error(
                "Failed to retrieve tournament data. The on-disk session "
                f"cache and live API both returned empty (elo parquet age={age_label}, "
                f"force_refresh={force_list_refresh}). Check FFBRIDGE_CACHE_DIR and "
                "outbound access to the selected FFBridge API."
            )
            return
    _load_debug_log(f"tournament list ready ({len(all_tournaments)} tournaments, source={list_source})")
    list_fallback_note = ""
    if list_source == "disk" and force_list_refresh:
        list_fallback_note = "live API refresh returned empty; using disk tournament list"

    n_tournaments = len(all_tournaments)
    prev_api_key = st.session_state.get("_ffbridge_loaded_api_key")
    if prev_api_key is not None and prev_api_key != api_key:
        print(
            f"[ffbridge] API changed ({prev_api_key} -> {api_key}); "
            "clearing in-process Elo dataset cache",
            flush=True,
        )
        load_ffbridge_elo_dataset.clear()
    rebuild, rebuild_reason = _needs_elo_rebuild(
        api_key, fetch_iv, n_tournaments, all_tournaments, max_age_hours,
    )
    _load_debug_log(f"elo dataset path: rebuild={rebuild} reason={rebuild_reason or 'none'}")
    if rebuild:
        print(f"[ffbridge] rebuilding Elo dataset: {rebuild_reason}", flush=True)
        load_ffbridge_elo_dataset.clear()
        with st.spinner(f"Refreshing Elo ratings ({rebuild_reason})…"):
            dataset = compute_and_persist_elo_dataset(
                api_module, all_tournaments, api_key, fetch_iv, show_progress=True,
            )
    else:
        # Load the full dataset once per process (shared across all sessions and
        # persisted to parquet). Both scratch and handicap Elo columns are stored;
        # the use_handicap-specific view is derived downstream.
        _load_debug_log("calling load_ffbridge_elo_dataset (cache_resource)")
        dataset = load_ffbridge_elo_dataset(
            api_module, all_tournaments, api_key, fetch_iv, n_tournaments
        )
        _load_debug_log("load_ffbridge_elo_dataset returned")
    st.session_state._ffbridge_loaded_api_key = api_key
    full_results_df = dataset['results_df']
    processing_stats = dataset['processing_stats']

    stats_placeholder.caption(
        "Elo dataset — "
        + _ffbridge_dataset_summary_line(
            selected_api_name=selected_api_name,
            list_source=list_source,
            n_tournaments=n_tournaments,
            processing_stats=processing_stats,
            result_rows=full_results_df.height,
            list_fallback_note=list_fallback_note,
        )
    )

    # Exclude invalid percentage rows before any downstream filtering/aggregation
    _load_debug_log(f"filtering results ({full_results_df.height} rows before pct filter)")
    full_results_df = _filter_valid_percentages_ffbridge(full_results_df)
    _load_debug_log(f"pct filter done ({full_results_df.height} rows)")

    # Apply filters
    results_df = full_results_df
    
    if simultaneous_type != "all":
        if 'series_id' in results_df.columns:
            results_df = results_df.filter(pl.col('series_id') == simultaneous_type)
    
    if club_filter and not results_df.is_empty() and "club_name" in results_df.columns:
        club_needle = club_filter.lower()
        results_df = results_df.filter(
            pl.col("club_name").cast(pl.Utf8).str.to_lowercase().str.contains(club_needle, literal=True)
        )
    _load_debug_log(f"sidebar filters applied ({results_df.height} rows)")

    global _FFBRIDGE_CLUB_OPTIONS
    if not full_results_df.is_empty() and "club_name" in full_results_df.columns:
        unique_clubs = sorted(
            {
                str(c).strip()
                for c in full_results_df.select("club_name").to_series().to_list()
                if c is not None and str(c).strip()
            }
        )
        new_clubs = [_ALL_CLUBS_LABEL] + unique_clubs
        _FFBRIDGE_CLUB_OPTIONS = new_clubs
        st.session_state.elo_available_clubs = new_clubs
        _load_debug_log(f"club list updated ({len(unique_clubs)} clubs, no rerun)")

    _load_debug_log("aggregating players (handicap)")
    players_df_hc = (
        _aggregate_players_from_results_cached(results_df, True)
        if not results_df.is_empty() else pl.DataFrame()
    )
    _load_debug_log(f"aggregating players (scratch); hc={players_df_hc.height} rows")
    players_df_sc = (
        _aggregate_players_from_results_cached(results_df, False)
        if not results_df.is_empty() else pl.DataFrame()
    )
    _load_debug_log(f"computing metric triples; sc={players_df_sc.height} player rows")
    player_metrics_hc = _player_metric_triple(players_df_hc, min_games)
    _load_debug_log("player metrics (handicap) done")
    player_metrics_sc = _player_metric_triple(players_df_sc, min_games)
    _load_debug_log("player metrics (scratch) done")
    if rating_type == "Pairs":
        pair_metrics_hc = _cached_pair_metric_triple(results_df, min_games, True)
        _load_debug_log(f"pair metrics (handicap) done ({results_df.height} result rows scanned)")
        pair_metrics_sc = _cached_pair_metric_triple(results_df, min_games, False)
        _load_debug_log("pair metrics (scratch) done")
    else:
        pair_metrics_hc = (0, 0.0, 0.0)
        pair_metrics_sc = (0, 0.0, 0.0)
        _load_debug_log("pair metrics skipped (Players view)")
    st.session_state["_ff_lb_ctx"] = {
        "results_df": results_df,
        "rating_type": rating_type,
        "simultaneous_type": simultaneous_type,
        "club_filter": club_filter,
        "name_filter": name_filter,
        "top_n": top_n,
        "min_games": min_games,
        "prior_sessions": prior_sessions,
        "player_metrics_hc": player_metrics_hc,
        "player_metrics_sc": player_metrics_sc,
        "pair_metrics_hc": pair_metrics_hc,
        "pair_metrics_sc": pair_metrics_sc,
    }
    del players_df_hc, players_df_sc
    _load_debug_log("leaderboard context ready; rendering panel")

    
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><small>Tournaments</small><br>'
            f'<span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{len(all_tournaments)}</span></div>',
            unsafe_allow_html=True,
        )
    metric_m2 = m2.empty()
    metric_m3 = m3.empty()
    metric_m4 = m4.empty()
    lb_entity = "Players" if rating_type == "Players" else "Pairs"
    st.markdown(f"### Top {top_n} {lb_entity} (Min. {min_games} games)")
    _ffbridge_leaderboard_panel(metric_m2, metric_m3, metric_m4)

    # PDF Export
    if generate_pdf:
        if 'display_df' in st.session_state and not st.session_state.display_df.is_empty():
            with st.spinner("Preparing PDF export..."):
                title = st.session_state.get('report_title', 'Unofficial FFBridge Elo Ratings')
                pdf_bytes = create_pdf(
                    [f"# {title}\n\nProcessed on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                     st.session_state.display_df],
                    title=title,
                    shrink_to_fit=True
                )
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"FFBridge_Elo_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
    
    render_app_footer(
        st,
        ENDPLAY_VERSION,
        source_line=f"Data sourced using {selected_api_name} • {selected_tournament_label}",
        diagnostics_lines=_ffbridge_footer_diagnostics_lines(st),
        dependency_versions={
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "duckdb": duckdb.__version__,
        },
    )


if __name__ == "__main__":
    main()

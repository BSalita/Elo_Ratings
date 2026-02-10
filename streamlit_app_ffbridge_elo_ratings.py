# streamlit_app_ffbridge_elo.py
"""
FFBridge Elo Ratings - Unified Streamlit Application

This app fetches duplicate bridge tournament results from FFBridge and calculates
Elo ratings based on percentage scores.

Supports both:
- Classic API (api.ffbridge.fr) - requires authentication
- Lancelot API (api-lancelot.ffbridge.fr) - public access
"""

import math
import os

# Prevent Intel Fortran runtime (libifcoremd.dll / MKL) from installing its own
# Ctrl+C handler that crashes with "forrtl: error (200)".
# Must be set before any numpy/scipy/MKL imports.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import pathlib
import re
import sys
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
    except Exception:
        pass

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
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode, AgGridTheme

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
    scale_to_chess_range,
    get_elo_title,
    apply_app_theme,
    calculate_aggrid_height,
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
                    except Exception:
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


def _maybe_override_octopus_pct_rows(detail_df: pl.DataFrame, pair_name: str, use_handicap: bool = True) -> pl.DataFrame:
    """
    For Octopus sessions, try to override Scratch_% / Handicap_% with BridgeInterNet values.
    Adds a 'Source' column with a clickable link to verify the data source.
    
    Args:
        detail_df: DataFrame with tournament results
        pair_name: Pair name or player name. If "Partner" column exists, uses that for each row.
        use_handicap: Whether to use handicap percentage for Pct_Used (else scratch)
    """
    if detail_df.is_empty():
        return detail_df

    # Get default surnames from passed pair_name
    default_s1, default_s2 = _pair_surnames(pair_name)
    
    rows = []
    for r in detail_df.to_dicts():
        src_url = ""
        date_str = str(r.get("Date", "") or "")[:10]
        tournament_label = str(r.get("Tournament", "") or "")
        event_id = str(r.get("Event_ID", "") or "")
        
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
        except Exception:
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
                        src_url = url_s  # Link to scratch results page
                except Exception as e:
                    # Fail fast-ish: just don't override if BridgeInterNet fetch/parsing fails.
                    print(f"[BI Reconcile] Error: {e}", flush=True)
        
        # If no BridgeInterNet URL, provide FFBridge API URL for verification
        if not src_url and event_id:
            src_url = f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{event_id}"

        r["Scratch_%"] = scratch
        r["Handicap_%"] = handicap
        # Update Pct_Used to match the updated values
        # Use handicap if requested AND available, otherwise use scratch
        if use_handicap and handicap is not None:
            r["Pct_Used"] = handicap
        else:
            r["Pct_Used"] = scratch
        # Store raw URL - LinkColumn will make it clickable
        r["Source"] = src_url if src_url else None
        rows.append(r)

    return pl.DataFrame(rows)


def build_selectable_aggrid(df: pl.DataFrame, key: str) -> Dict[str, Any]:
    """Build an AgGrid with single-click row selection."""
    from st_aggrid import JsCode
    
    display_df = df.to_pandas()
    # Ensure Rank column is numeric before passing to AgGrid
    if 'Rank' in display_df.columns:
        display_df['Rank'] = pd.to_numeric(display_df['Rank'], errors='coerce').astype(int)
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False, suppressRowClickSelection=False)
    gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True)
    # Configure Rank column for numeric sorting with custom comparator
    if 'Rank' in display_df.columns:
        gb.configure_column(
            'Rank', 
            type=['numericColumn', 'numberColumnFilter'],
            sort='asc',
            comparator=JsCode("""
                function(valueA, valueB, nodeA, nodeB, isDescending) {
                    return Number(valueA) - Number(valueB);
                }
            """)
        )
    # Configure Games column width to fit column name + icon size
    if 'Games' in display_df.columns:
        gb.configure_column('Games', width=100)  # Width accommodates "Games" text + sort icon
    grid_options = gb.build()
    
    return AgGrid(
        display_df,
        gridOptions=grid_options,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        theme=AgGridTheme.BALHAM,
        height=calculate_aggrid_height(len(display_df)),
        key=key,
        allow_unsafe_jscode=True
    )


def _render_detail_aggrid_ff(detail_df: pl.DataFrame, key: str, selectable: bool = False):
    """Render a detail DataFrame as a selectable AgGrid. Returns grid response if selectable."""
    display_df = detail_df.to_pandas()
    gb = GridOptionsBuilder.from_dataframe(display_df)
    if selectable:
        gb.configure_selection(selection_mode='single', use_checkbox=False, suppressRowClickSelection=False)
    gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True)
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            gb.configure_column(col, type=['numericColumn'], filter='agNumberColumnFilter')
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
    """Show aggregation of opponents across all tournaments (Opponent Summary)."""
    opp_detail = _build_opponent_data(results_df, entity_tournaments, exclude_id, exclude_mode)
    if opp_detail.is_empty():
        return
    st.markdown("#### Opponent Summary — All Tournaments")
    agg_cols = [
        pl.len().alias('Events'),
    ]
    if 'Rank' in opp_detail.columns:
        agg_cols.append(pl.col('Rank').cast(pl.Float64, strict=False).mean().round(1).alias('Avg_Rank'))
    if 'Scratch_%' in opp_detail.columns:
        agg_cols.append(pl.col('Scratch_%').mean().round(2).alias('Avg_Scratch_%'))
    if 'Handicap_%' in opp_detail.columns:
        agg_cols.append(pl.col('Handicap_%').mean().round(2).alias('Avg_Handicap_%'))
    if 'Pair_Elo' in opp_detail.columns:
        agg_cols.append(pl.col('Pair_Elo').mean().round(0).cast(pl.Int32, strict=False).alias('Avg_Pair_Elo'))

    opp_agg = (
        opp_detail
        .group_by('Opponent')
        .agg(agg_cols)
        .sort('Events', descending=True)
    )
    st.caption(f"{len(opp_agg)} unique opponents")
    _render_detail_aggrid_ff(opp_agg, key=f"ff_opp_summ_{key_suffix}")


# -------------------------------
# Data Processing (common for both APIs)
# -------------------------------
def process_tournaments_to_elo(
    tournaments: List[Dict[str, Any]],
    api_module,
    initial_players: Optional[Dict[str, Dict]] = None,
    use_handicap: bool = False,
    fetch_iv: bool = False,
    sort_ascending: bool = True,
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
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    cache_stats = {"cached": 0, "fetched": 0, "missing_ids": []}
    
    total_t = len(sorted_tournaments)
    for i, tournament in enumerate(sorted_tournaments):
        t_id = str(tournament.get('id', ''))
        t_series = normalize_series_id(tournament.get('series_id'))
        t_name = tournament.get('name') or tournament.get('label') or tournament.get('moment_label') or f"Tournament {t_id}"
        t_date = tournament.get('date', '')
        
        missing_count = len(cache_stats['missing_ids'])
        cache_info = f"[Cached: {cache_stats['cached']}, Fetched: {cache_stats['fetched']}, Missing results: {missing_count}]"
        status_text.markdown(f"<span style='color: white;'>Processing {i+1}/{total_t}: {t_name[:35]} (ID:{t_id})... {cache_info}</span>", unsafe_allow_html=True)
        progress_bar.progress((i + 1) / total_t)
        
        # Log to console for debugging hung requests
        print(f"[Processing] {i+1}/{total_t}: ID={t_id}, date={t_date[:10] if t_date else 'N/A'}, name={t_name[:40]}", flush=True)
        
        # Fetch results using the appropriate API module
        results, was_cached = api_module.fetch_tournament_results(t_id, tournament_date=t_date, series_id=t_series, fetch_iv=fetch_iv)
        print(f"[Processing] {i+1}/{total_t}: ID={t_id} - fetch returned, cached={was_cached}, count={len(results) if results else 0}", flush=True)
        
        if was_cached:
            cache_stats["cached"] += 1
        else:
            cache_stats["fetched"] += 1
        
        if not results:
            cache_stats["missing_ids"].append(t_id)
            print(f"[Processing] {i+1}/{total_t}: ID={t_id} - no results, skipping (missing: {len(cache_stats['missing_ids'])})", flush=True)
            continue
        
        print(f"[Processing] {i+1}/{total_t}: ID={t_id} - processing {len(results)} results...", flush=True)
        
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
                scratch_r1_after = calculate_elo_from_percentage(scratch_r1_before, scratch_pct, scratch_field_avg)
                scratch_ratings[p1_id] = scratch_r1_after
                
                # Handicap Elo (use scratch if handicap not available)
                handicap_r1_before = handicap_ratings.get(p1_id, DEFAULT_ELO)
                h_pct_for_elo = handicap_pct if handicap_pct is not None else scratch_pct
                h_field_for_elo = handicap_field_avg if handicap_pct is not None else scratch_field_avg
                handicap_r1_after = calculate_elo_from_percentage(handicap_r1_before, h_pct_for_elo, h_field_for_elo)
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
                scratch_r2_after = calculate_elo_from_percentage(scratch_r2_before, scratch_pct, scratch_field_avg)
                scratch_ratings[p2_id] = scratch_r2_after
                
                # Handicap Elo (use scratch if handicap not available)
                handicap_r2_before = handicap_ratings.get(p2_id, DEFAULT_ELO)
                h_pct_for_elo = handicap_pct if handicap_pct is not None else scratch_pct
                h_field_for_elo = handicap_field_avg if handicap_pct is not None else scratch_field_avg
                handicap_r2_after = calculate_elo_from_percentage(handicap_r2_before, h_pct_for_elo, h_field_for_elo)
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
        
        print(f"[Processing] {i+1}/{total_t}: ID={t_id} - done", flush=True)
    
    progress_bar.empty()
    status_text.empty()
    
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
    
    # Return selected type ratings dict for backward compatibility
    current_ratings = handicap_ratings if use_handicap else scratch_ratings
    return results_df, players_df, current_ratings, cache_stats


def show_top_players(players_df: pl.DataFrame, top_n: int, min_games: int = 5, use_handicap: bool = False) -> Tuple[pl.DataFrame, str]:
    """Get top players sorted by Elo rating using SQL."""
    if players_df.is_empty():
        return players_df, ""
    
    # Build SELECT clause - conditionally include Title column
    # Title is always based on scratch Elo, regardless of use_handicap setting
    # Determine column names based on handicap setting
    elo_col_name = "HC_Player_Elo" if use_handicap else "Player_Elo"
    title_col_name = "Scratch_Title" if use_handicap else "Title"
    
    title_col = f""",
            CASE 
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 2600 THEN 'SGM'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 2500 THEN 'GM'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 2400 THEN 'IM'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 2300 THEN 'FM'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 2200 THEN 'CM'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 2000 THEN 'Expert'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 1800 THEN 'Advanced'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 1600 THEN 'Intermediate'
                WHEN CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER) >= 1400 THEN 'Novice'
                ELSE 'Beginner'
            END AS {title_col_name}"""
    
    query = f"""
        WITH filtered AS (
            SELECT *
            FROM players_df
            WHERE games_played >= {min_games}
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY elo_rating DESC, games_played DESC, player_name ASC, player_id ASC) AS INTEGER) AS Rank,
            CAST(ROUND(LEAST(GREATEST(elo_rating, 0), 3500), 0) AS INTEGER) AS {elo_col_name}{title_col},
            player_id AS Player_ID,
            player_name AS Player_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games
        FROM filtered
        ORDER BY Rank ASC
        LIMIT {top_n}
    """
    
    result = duckdb.sql(query).pl()
    return result, query


def show_top_pairs(results_df: pl.DataFrame, top_n: int, min_games: int = 5, use_handicap: bool = False, players_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, str]:
    """Get top pairs sorted by Elo rating using SQL."""
    if results_df.is_empty():
        return results_df, ""
    
    # Select appropriate Elo and percentage columns based on use_handicap
    elo_col = "handicap_pair_elo" if use_handicap else "scratch_pair_elo"
    # Use COALESCE for handicap to fall back to scratch when handicap is null
    pct_col = "COALESCE(handicap_percentage, scratch_percentage)" if use_handicap else "scratch_percentage"
    
    # Determine column names based on handicap setting
    pair_elo_col_name = "HC_Pair_Elo" if use_handicap else "Pair_Elo"
    title_col_name = "Scratch_Title" if use_handicap else "Title"
    
    # Build Title column - use lower title of the two players based on their scratch Elo
    if players_df is not None and not players_df.is_empty():
        # Join with players_df to get individual player scratch Elo and calculate lower title
        title_col = """,
            CASE 
                -- Calculate title rank for player1 (1=SGM, 10=Beginner)
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2600 THEN 1
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2500 THEN 2
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2400 THEN 3
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2300 THEN 4
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2200 THEN 5
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2000 THEN 6
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 1800 THEN 7
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 1600 THEN 8
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p1.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 1400 THEN 9
                ELSE 10
            END AS p1_title_rank,
            CASE 
                -- Calculate title rank for player2 (1=SGM, 10=Beginner)
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2600 THEN 1
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2500 THEN 2
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2400 THEN 3
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2300 THEN 4
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2200 THEN 5
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 2000 THEN 6
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 1800 THEN 7
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 1600 THEN 8
                WHEN CAST(ROUND(LEAST(GREATEST(COALESCE(p2.scratch_elo, 0), 0), 3500), 0) AS INTEGER) >= 1400 THEN 9
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
                AVG(scratch_pair_elo) AS avg_scratch_elo,
                AVG(COALESCE(handicap_pair_elo, scratch_pair_elo)) AS avg_handicap_elo,
                AVG({elo_col}) AS avg_pair_elo,
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
        )"""
    
    if players_df is not None and not players_df.is_empty():
        query += f""",
        with_player_titles AS (
            SELECT 
                f.*{title_col}
            FROM filtered f
            LEFT JOIN players_df p1 ON f.player1_id = p1.player_id
            LEFT JOIN players_df p2 ON f.player2_id = p2.player_id
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY avg_pair_elo DESC, games_played DESC, pair_name ASC, pair_id ASC) AS INTEGER) AS Rank,
            CAST(ROUND(LEAST(GREATEST(avg_pair_elo, 0), 3500), 0) AS INTEGER) AS {pair_elo_col_name}{title_select},
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
            CAST(ROW_NUMBER() OVER (ORDER BY avg_pair_elo DESC, games_played DESC, pair_name ASC, pair_id ASC) AS INTEGER) AS Rank,
            CAST(ROUND(LEAST(GREATEST(avg_pair_elo, 0), 3500), 0) AS INTEGER) AS {pair_elo_col_name}{title_col},
            pair_id AS Pair_ID,
            pair_name AS Pair_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games
        FROM filtered
        ORDER BY Rank ASC
        LIMIT {top_n}
    """
    
    result = duckdb.sql(query).pl()
    return result, query


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
    
    # Apply common theme
    apply_app_theme(st)
    widen_scrollbars()
    
    # -------------------------------
    # Sidebar Controls
    # -------------------------------
    with st.sidebar:
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
        st.sidebar.markdown("🔗 [What is Elo Rating?](https://en.wikipedia.org/wiki/Elo_rating_system)")
        
        # API Backend selection
        if "selected_api" not in st.session_state:
            st.session_state.selected_api = "FFBridge Classic API"
        
        selected_api_name = st.selectbox(
            "Bridge API",
            options=list(API_BACKENDS.keys()),
            key="selected_api",
            help="Select which API to use for data"
        )
        
        # Get the appropriate API module
        api_module = API_BACKENDS[selected_api_name]
        
        # Check authentication if required
        if api_module.REQUIRES_AUTH and not api_module.is_authenticated():
            st.error("❌ **Authentication Error**")
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
        
        # Club filter
        club_options = st.session_state.get('elo_available_clubs', ["All Clubs"])
        
        if "elo_club_selectbox" not in st.session_state:
            st.session_state.elo_club_selectbox = "All Clubs"
        
        if st.session_state.elo_club_selectbox not in club_options:
            st.session_state.elo_club_selectbox = "All Clubs"
        
        selected_club = st.selectbox(
            "Filter by Club",
            options=club_options,
            key="elo_club_selectbox",
            help="Filter results to show only players/pairs from a specific club"
        )
        
        # Name filter
        name_filter = st.text_input(
            "Filter by Name",
            key="elo_name_filter",
            help="Filter results by player or pair name (case-insensitive)"
        )
        
        # Ranking type
        rating_type = st.radio(
            "Ranking Type",
            ["Players", "Pairs"],
            index=0,
            key="elo_rating_type",
            horizontal=True,
            help="Switch between individual and partnership rankings"
        )
        
        # Choose which score type to use for Elo calculations
        score_type = st.radio(
            "Elo Based On",
            ["Scratch", "Handicap"],
            index=0,
            key="elo_score_type",
            horizontal=True,
            help="Choose which percentage to use for Elo calculations (rankings always sorted by Elo)"
        )
        use_handicap = (score_type == "Handicap")
        
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
        
        # PDF Export
        generate_pdf = st.button("Export Report to PDF File", width='stretch')
        st.sidebar.markdown('<p style="color: #ffc107; font-weight: 600;">Morty\'s Automated Postmortem Apps</p>', unsafe_allow_html=True)
        st.sidebar.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
        st.sidebar.markdown("🔗 [French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
        #st.sidebar.markdown("🔗 [BridgeWebs Postmortem](https://bridgewebs.postmortem.chat)")

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
    
    # Placeholder for processing stats (created once, updated later)
    stats_placeholder = st.empty()
    
    # -------------------------------
    # Main Content
    # -------------------------------
    
    # Cache key includes API name to separate caches
    api_key = selected_api_name.replace(" ", "_")
    
    # Fetch all tournaments
    with st.spinner("Loading tournament data..."):
        all_tournaments = api_module.fetch_tournament_list(series_id="all", limit=None)
        
        if not all_tournaments:
            st.error("Failed to retrieve tournament data. Please check your connection or authentication.")
            return
    
    # Cache key for full dataset - only depends on API and fetch_iv
    # Both scratch and handicap Elo are computed in one pass and stored in the DataFrame
    # Tournaments are always sorted ascending (oldest first) for Elo calculations
    cache_key = f"elo_full_v3_{api_key}_{len(all_tournaments)}_iv_{int(fetch_iv)}"
    
    if 'elo_full_cache' not in st.session_state:
        st.session_state.elo_full_cache = {}
    
    full_cache = st.session_state.elo_full_cache
    
    if cache_key in full_cache:
        # Use cached data - both scratch and handicap Elo are already computed
        cached = full_cache[cache_key]
        full_results_df = cached['results_df']
        full_players_df = cached['players_df']
        current_ratings = cached['scratch_ratings'] if not use_handicap else cached['handicap_ratings']
        processing_stats = cached.get('processing_stats', {"cached": 0, "fetched": 0, "missing_ids": []})
    else:
        # Clear old cache entries for different API
        keys_to_remove = [k for k in full_cache.keys() if not k.startswith(f"elo_full_v3_{api_key}_")]
        for k in keys_to_remove:
            del full_cache[k]
        
        # Process all tournaments - computes BOTH scratch and handicap Elo in one pass
        # Always sort ascending (oldest first) for proper Elo calculation chronology
        full_results_df, full_players_df, current_ratings, processing_stats = process_tournaments_to_elo(
            all_tournaments, api_module, initial_players=None, use_handicap=use_handicap, fetch_iv=fetch_iv, sort_ascending=True
        )
        
        # Apply club name mapping for APIs that don't provide club names directly (e.g., Lancelot)
        if not full_results_df.is_empty() and 'club_code' in full_results_df.columns:
            # Check if club_name column is mostly empty
            non_empty_names = full_results_df.filter(pl.col('club_name') != '').height
            if non_empty_names < full_results_df.height * 0.1:  # Less than 10% have names
                # Get unique club codes
                unique_codes = full_results_df.select('club_code').unique().to_series().to_list()
                unique_codes = [str(c) for c in unique_codes if c]
                
                # Build mapping using API's club name mapping function
                if hasattr(api_module, 'build_club_name_mapping'):
                    club_mapping = api_module.build_club_name_mapping(unique_codes, all_tournaments, full_results_df)
                    
                    # Apply mapping to DataFrame
                    if club_mapping:
                        full_results_df = full_results_df.with_columns(
                            pl.col('club_code').map_elements(
                                lambda c: club_mapping.get(str(c).lstrip('0') if c else '', str(c)),
                                return_dtype=pl.Utf8
                            ).alias('club_name')
                        )
        
        # Extract both scratch and handicap ratings from players_df for caching
        scratch_ratings_dict = {}
        handicap_ratings_dict = {}
        if not full_players_df.is_empty():
            for row in full_players_df.iter_rows(named=True):
                pid = row['player_id']
                scratch_ratings_dict[pid] = row['scratch_elo']
                handicap_ratings_dict[pid] = row['handicap_elo']
        
        full_cache[cache_key] = {
            'results_df': full_results_df,
            'players_df': full_players_df,
            'scratch_ratings': scratch_ratings_dict,
            'handicap_ratings': handicap_ratings_dict,
            'processing_stats': processing_stats,
        }
    
    # Display processing stats using the placeholder (avoids duplicates)
    missing_ids = processing_stats.get('missing_ids', [])
    missing_str = str(missing_ids) if missing_ids else "none"
    stats_msg = f"Cached: {processing_stats['cached']}, Fetched: {processing_stats['fetched']}, Missing results: {missing_str}"
    stats_placeholder.caption(stats_msg)
    
    # Apply filters
    results_df = full_results_df
    
    if simultaneous_type != "all":
        if 'series_id' in results_df.columns:
            results_df = results_df.filter(pl.col('series_id') == simultaneous_type)
    
    if selected_club != "All Clubs" and not results_df.is_empty():
        if 'club_name' in results_df.columns:
            results_df = results_df.filter(pl.col('club_name') == selected_club)
    
    # Recalculate players_df from filtered results with both scratch and handicap stats
    if not results_df.is_empty():
        # Dynamic column selection based on use_handicap
        elo_col_p1 = "player1_handicap_elo_after" if use_handicap else "player1_scratch_elo_after"
        elo_col_p2 = "player2_handicap_elo_after" if use_handicap else "player2_scratch_elo_after"
        # Use COALESCE for handicap to fall back to scratch when handicap is null
        pct_expr = "COALESCE(handicap_percentage, scratch_percentage)" if use_handicap else "scratch_percentage"
        
        players_df = duckdb.sql(f"""
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
    else:
        players_df = pl.DataFrame()
        st.info("No results found for the selected filters.")
    
    # Populate club options
    if not full_results_df.is_empty() and 'club_name' in full_results_df.columns:
        unique_clubs = sorted(set(full_results_df.select('club_name').to_series().to_list()))
        unique_clubs = [c for c in unique_clubs if c and c.strip()]
        
        old_clubs = st.session_state.get('elo_available_clubs', ["All Clubs"])
        merged_names = {c.strip() for c in old_clubs if c != "All Clubs"}
        merged_names.update(c.strip() for c in unique_clubs if c.strip())
        
        current_selected = st.session_state.get("elo_club_selectbox", "All Clubs")
        if current_selected and current_selected != "All Clubs":
            merged_names.add(current_selected.strip())
        
        new_clubs = ["All Clubs"] + sorted(merged_names)
        
        if len(new_clubs) > len(old_clubs):
            st.session_state.elo_available_clubs = new_clubs
            st.rerun()
        else:
            st.session_state.elo_available_clubs = new_clubs
    
    # Display metrics
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f'<div class="metric-card"><small>Tournaments</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{len(all_tournaments)}</span></div>', unsafe_allow_html=True)
    
    # Calculate metrics based on Ranking Type (Players vs Pairs)
    # Initialize defaults to avoid any potential value retention issues
    active_count = 0
    avg_games = 0.0
    highest_elo = 0.0
    
    if rating_type == "Players":
        active_label = "Active Players"
        avg_label = "Avg Games"
        highest_label = "Highest Elo"
        
        # Filter by min_games to match the ranking table requirements
        metric_players = players_df.filter(pl.col('games_played') >= min_games) if not players_df.is_empty() else players_df
        
        if not metric_players.is_empty():
            active_count = len(metric_players)
            avg_games = metric_players.select(pl.col('games_played').mean()).item()
            highest_elo = metric_players.select(pl.col('elo_rating').max()).item()
    else:
        active_label = "Active Pairs"
        avg_label = "Avg Games/Pair"
        highest_label = "Highest Pair Elo"
        
        if not results_df.is_empty():
            # Calculate pair stats for metrics with min_games filter to match ranking table logic (Average Elo)
            elo_col = "handicap_pair_elo" if use_handicap else "scratch_pair_elo"
            pair_stats = duckdb.sql(f"""
                SELECT pair_id, COUNT(*) as games, AVG({elo_col}) as elo
                FROM results_df
                GROUP BY pair_id
                HAVING games >= {min_games}
            """).pl()
            
            if not pair_stats.is_empty():
                active_count = len(pair_stats)
                avg_games = pair_stats.select(pl.col('games').mean()).item()
                highest_elo = pair_stats.select(pl.col('elo').max()).item()

    with m2:
        st.markdown(f'<div class="metric-card"><small>{active_label}</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{active_count}</span></div>', unsafe_allow_html=True)
    
    with m3:
        st.markdown(f'<div class="metric-card"><small>{avg_label}</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{avg_games:.1f}</span></div>', unsafe_allow_html=True)
    
    with m4:
        # Handle infinity/NaN values safely
        if highest_elo is None or not math.isfinite(highest_elo):
            highest_elo_display = 0
        else:
            highest_elo_display = round(highest_elo)
        st.markdown(f'<div class="metric-card"><small>{highest_label}</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{highest_elo_display}</span></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Store for detail views
    st.session_state.full_results_df = results_df
    
    # Display tables
    if rating_type == "Players":
        st.markdown(f"### 🏆 Top {top_n} Players (Min. {min_games} games)")
        if not players_df.is_empty():
            top_players, sql_query = show_top_players(players_df, top_n, min_games, use_handicap)
            
            if sql_query:
                with st.expander("📝 SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
            
            if not top_players.is_empty():
                # Apply name filter if provided
                if name_filter and name_filter.strip():
                    name_filter_lower = name_filter.strip().lower()
                    top_players = top_players.filter(
                        pl.col('Player_Name').str.to_lowercase().str.contains(name_filter_lower, literal=False)
                    )
                
                if not top_players.is_empty():
                    st.caption("💡 Click a row to view player's tournament history")
                    # Create dynamic key based on filter parameters to reset selection when data changes
                    dynamic_key = f'players_table_selectable_{rating_type}_{simultaneous_type}_{selected_club}_{top_n}_{min_games}_{use_handicap}_{name_filter}'
                    grid_response = build_selectable_aggrid(top_players, dynamic_key)
                    
                    selected_rows = grid_response.get('selected_rows', None)
                    if selected_rows is not None and len(selected_rows) > 0:
                        selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                        player_id = selected_row.get('Player_ID')
                        player_name = selected_row.get('Player_Name', 'Unknown')
                        
                        if player_id and not results_df.is_empty():
                            # Show tournament history
                            st.markdown(f"#### 📊 Tournament History Details: **{player_name}**")
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
                                
                                # Also keep pair_id for tournament opponent lookup
                                detail_df = player_results.select(cols_to_select + [pl.col('pair_id').alias('_pair_id')])
                                # Optional reconciliation: for Octopus, prefer BridgeInterNet scratch/handicap when matchable.
                                detail_df = _maybe_override_octopus_pct_rows(detail_df, pair_name=player_name, use_handicap=use_handicap)
                                
                                # Display as selectable AgGrid (click row to see tournament opponents)
                                st.caption("Click a row to see tournament opponents")
                                # Hide _pair_id from display
                                display_detail = detail_df.drop('_pair_id')
                                detail_key = f"detail_player_{player_id}_{rating_type}_{use_handicap}"
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
                                _show_partner_aggregation(display_detail, key_suffix=f"player_{player_id}")
                                
                                # Opponent History Details + Opponent Summary
                                _show_opponent_history(results_df, player_results, exclude_id=str(player_id), exclude_mode='player', key_suffix=f"player_{player_id}")
                                _show_opponent_summary(results_df, player_results, exclude_id=str(player_id), exclude_mode='player', key_suffix=f"player_{player_id}")
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
        st.markdown(f"### 🏆 Top {top_n} Pairs (Min. {min_games} games)")
        if not results_df.is_empty():
            top_pairs, sql_query = show_top_pairs(results_df, top_n, min_games, use_handicap, players_df)
            
            if sql_query:
                with st.expander("📝 SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
            
            if not top_pairs.is_empty():
                # Apply name filter if provided
                if name_filter and name_filter.strip():
                    name_filter_lower = name_filter.strip().lower()
                    top_pairs = top_pairs.filter(
                        pl.col('Pair_Name').str.to_lowercase().str.contains(name_filter_lower, literal=False)
                    )
                
                if not top_pairs.is_empty():
                    st.caption("💡 Click a row to view pair's tournament history")
                    # Create dynamic key based on filter parameters to reset selection when data changes
                    dynamic_key = f'pairs_table_selectable_{rating_type}_{simultaneous_type}_{selected_club}_{top_n}_{min_games}_{use_handicap}_{name_filter}'
                    grid_response = build_selectable_aggrid(top_pairs, dynamic_key)
                    
                    selected_rows = grid_response.get('selected_rows', None)
                    if selected_rows is not None and len(selected_rows) > 0:
                        selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                        pair_id = selected_row.get('Pair_ID')
                        pair_name = selected_row.get('Pair_Name', 'Unknown')
                        
                        if pair_id and not results_df.is_empty():
                            st.markdown(f"### 📋 Tournament History Details for **{pair_name}**")
                            
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
                                
                                detail_df = pair_results.select(cols_to_select)
                                # Optional reconciliation: for Octopus, prefer BridgeInterNet scratch/handicap when matchable.
                                detail_df = _maybe_override_octopus_pct_rows(detail_df, pair_name=pair_name, use_handicap=use_handicap)
                                
                                # Display as selectable AgGrid (click row to see tournament opponents)
                                st.caption("Click a row to see tournament opponents")
                                detail_key = f"detail_pair_{pair_id}_{rating_type}_{use_handicap}"
                                detail_resp = _render_detail_aggrid_ff(detail_df, key=detail_key, selectable=True)
                                
                                # 3rd df: tournament opponents for clicked row
                                if detail_resp is not None:
                                    d_sel = detail_resp.get('selected_rows', None)
                                    if d_sel is not None and len(d_sel) > 0:
                                        d_row = d_sel.iloc[0] if hasattr(d_sel, 'iloc') else d_sel[0]
                                        event_id = str(d_row.get('Event_ID', ''))
                                        if event_id:
                                            _show_tournament_opponents(results_df, event_id, exclude_pair_id=str(pair_id))
                                
                                # 4th df: club aggregation across all tournaments
                                _show_club_aggregation(detail_df, results_df, str(pair_id), key_suffix=f"pair_{pair_id}")
                                
                                # Opponent History Details + Opponent Summary
                                _show_opponent_history(results_df, pair_results, exclude_id=str(pair_id), exclude_mode='pair', key_suffix=f"pair_{pair_id}")
                                _show_opponent_summary(results_df, pair_results, exclude_id=str(pair_id), exclude_mode='pair', key_suffix=f"pair_{pair_id}")
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
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"FFBridge_Elo_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
    
    # Footer
    st.markdown(f"""
        <div style="text-align: center; color: #80cbc4; font-size: 0.8rem; opacity: 0.7;">
            Project lead is Robert Salita research@AiPolice.org. Code written in Python by Cursor AI. UI written in streamlit. Data engine is polars. Repo: <a href="https://github.com/BSalita/Elo_Ratings" target="_blank" style="color: #80cbc4;">github.com/BSalita/Elo_Ratings</a><br>
            Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}<br>
            Streamlit:{st.__version__} Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} duckdb:{duckdb.__version__} endplay:{ENDPLAY_VERSION}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0; color: #80cbc4; font-size: 0.9rem; opacity: 0.8;">
            Data sourced using {selected_api_name} • {selected_tournament_label}<br>
            System Current Date: {datetime.now().strftime('%Y-%m-%d')}
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

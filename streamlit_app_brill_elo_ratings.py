# streamlit_app_brill_elo_ratings.py
"""
Brill Elo Ratings - Streamlit application (https://brill-elo.7nt.info, port 8532)

Elo ratings and skill rankings for the daily events at Brill Bridge Challenge
(https://brill.aalborgdata.dk), where humans and robots each play the same
boards at their own table beside three Brill robots.

The dataset is built by elo_brill.py (see build_brill_elo_parquets.py). This
app loads it, asks the site for newly closed events when the data is older
than BRILL_ELO_MAX_AGE_HOURS, and shows the rankings.
"""

import os

# Prevent Intel Fortran runtime (libifcoremd.dll / MKL) from installing its own
# Ctrl+C handler that crashes with "forrtl: error (200)".
# Must be set before any numpy/scipy/MKL imports.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import pathlib
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl
import streamlit as st
from st_aggrid import AgGrid, AgGridTheme, ColumnsAutoSizeMode, DataReturnMode, GridOptionsBuilder, JsCode
from streamlit_scroll import scroll_to_here

from streamlitlib.streamlitlib import (
    _ranking_row_click_jscode,
    clear_row_click_hourglass,
    create_pdf,
    widen_scrollbars,
)

import elo_brill
from elo_common import (
    ASSISTANT_LOGO_URL,
    LEADERBOARD_PAGE_SIZE,
    LEADERBOARD_ROW_HEIGHT,
    LEADERBOARD_SORT_MODEL_KEY,
    NUMERIC_NULLS_LAST_COMPARATOR_JS,
    apply_aggrid_sort_model,
    apply_app_theme,
    apply_sort_model_to_grid_options,
    coerce_int,
    coerce_numeric_columns,
    current_leaderboard_sort,
    footer_streamlit_app_diagnostics_line,
    init_url_params_to_state,
    leaderboard_aggrid_viewport_height,
    remember_leaderboard_sort,
    render_app_footer,
    sync_state_to_url_params,
)

SITE_URL = "https://brill.aalborgdata.dk"
MAX_AGE_HOURS = float(os.environ.get("BRILL_ELO_MAX_AGE_HOURS", "1"))
# After a failed refresh, serve the data on disk this long before asking the site again.
REFRESH_RETRY_SECONDS = 15 * 60
DEFAULT_MIN_GAMES = 10
SCORING_OPTIONS = ("All", "MP", "IMP")

RANKING_TITLES = {
    "Elo": "Elo Rankings",
    "Declarer": "Best Declarers",
    "Defender": "Best Defenders",
    "Opening Lead": "Best Opening Leaders",
    "Par": "Par Rankings",
}
RANKING_NOTES = {
    "Elo": ("Elo", "Avg_Pct", "Avg_IMP", "Wins"),
    "Declarer": ("DD_Tricks_Diff_Avg", "Declarer_Pct"),
    "Defender": ("Defense_Tricks_Diff_Avg", "Defender_Pct"),
    "Opening Lead": ("Lead_Cost_Avg", "Lead_Best_Pct"),
    "Par": ("Par_Contract_Rate_Pct", "Par_Suit_Rate_Pct", "Sacrifice_Rate_Pct", "Par_Score_Rate_Pct", "Par_Diff_Avg"),
}

# URL query param -> sidebar widget session state.
BRILL_URL_PARAMS = {
    "rank": {"session_key": "brill_ranking", "parser": str, "valid_values": tuple(elo_brill.RANKINGS), "default": "Elo"},
    "players": {"session_key": "brill_player_type", "parser": str, "valid_values": elo_brill.PLAYER_TYPE_OPTIONS, "default": "All"},
    "event": {"session_key": "brill_event_type", "parser": str, "default": elo_brill.ALL_EVENTS},
    "scoring": {"session_key": "brill_scoring", "parser": str, "valid_values": SCORING_OPTIONS, "default": "All"},
    "name": {"session_key": "brill_name_filter", "parser": str, "default": ""},
    "date_range": {"session_key": "brill_date_range", "parser": str, "valid_values": elo_brill.DATE_RANGE_OPTIONS, "default": "All time"},
    "min_games": {"session_key": "brill_min_games", "parser": coerce_int(1, 100), "default": DEFAULT_MIN_GAMES},
}

_URL_CELL_RENDERER = JsCode("""
class UrlCellRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        var v = (params.value === null || params.value === undefined) ? '' : String(params.value).trim();
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
        a.textContent = 'Open postmortem';
        this.eGui.appendChild(a);
    }
    getGui() { return this.eGui; }
    refresh(params) { return false; }
}
""")

# Process-wide: when the last refresh failed, and why.
_refresh_failure: Dict[str, Any] = {}


# -------------------------------
# Dataset
# -------------------------------
@st.cache_resource(show_spinner=False, max_entries=1)
def _load_dataset(built_at: str) -> Dict[str, Any]:
    """The dataset on disk. built_at is the cache key: a rebuild loads anew."""
    return elo_brill.load_dataset()


def _refresh_dataset() -> Optional[str]:
    """Ask Brill for newly closed events when the data is stale.

    Returns a warning to show when the refresh failed and older data is served.
    With no data on disk at all, a failure is raised.
    """
    have_data = elo_brill.dataset_exists()
    age = elo_brill.dataset_age_hours() if have_data else None
    if have_data and age is not None and age < MAX_AGE_HOURS:
        return None
    failed_at = _refresh_failure.get("at")
    if have_data and failed_at and time.time() - failed_at < REFRESH_RETRY_SECONDS:
        return _refresh_failure["message"]
    label = "Checking Brill for new events…" if have_data else "Building the Brill Elo dataset (a few minutes the first time)…"
    try:
        with st.spinner(label):
            elo_brill.update_dataset(max_age_hours=MAX_AGE_HOURS)
    except Exception as exc:  # the site, the sign-in or the solver; older data still serves
        print(f"[brill-elo] refresh failed: {type(exc).__name__}: {exc}", flush=True)
        if not have_data:
            raise
        _refresh_failure["at"] = time.time()
        _refresh_failure["message"] = (
            f"Could not check Brill for new events ({type(exc).__name__}: {exc}). "
            "Showing the data from the last successful update."
        )
        return _refresh_failure["message"]
    _refresh_failure.clear()
    return None


# -------------------------------
# Grids
# -------------------------------
def _aggrid_key_part(value: str) -> str:
    return re.sub(r"[^\w\-]", "_", (value or "").strip())[:48]


def _configure_columns(gb: GridOptionsBuilder, display_df: pd.DataFrame, hidden: tuple[str, ...]) -> None:
    gb.configure_default_column(cellStyle={"color": "black", "font-size": "12px"}, suppressMenu=True)
    numeric_comparator = JsCode(NUMERIC_NULLS_LAST_COMPARATOR_JS)
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], comparator=numeric_comparator)
        if col.endswith("_URL"):
            gb.configure_column(col, cellRenderer=_URL_CELL_RENDERER, width=150)
    for col in hidden:
        if col in display_df.columns:
            gb.configure_column(col, hide=True)


def _selected_row(response: Any) -> Optional[Any]:
    selected = response.get("selected_rows", None) if response is not None else None
    if selected is None or len(selected) == 0:
        return None
    return selected.iloc[0] if hasattr(selected, "iloc") else selected[0]


def _leaderboard_grid(df: pl.DataFrame, key: str, rank_column: str) -> Any:
    """The leaderboard: paged, one row selectable, sort remembered for the PDF."""
    display_df = df.to_pandas(use_pyarrow_extension_array=False)
    coerce_numeric_columns(display_df)
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False, suppressRowClickSelection=False)
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=LEADERBOARD_PAGE_SIZE)
    _configure_columns(gb, display_df, hidden=("Player_ID",))
    gb.configure_column(rank_column, pinned="left", width=120)
    gb.configure_column("Player_Name", pinned="left", width=180)
    gb.configure_column("Type", width=90)
    grid_options = gb.build()
    grid_options["onRowClicked"] = _ranking_row_click_jscode()
    grid_options["rowHeight"] = LEADERBOARD_ROW_HEIGHT
    grid_options["headerHeight"] = 50
    grid_options["domLayout"] = "normal"
    default_sort = [{"colId": rank_column, "sort": "asc"}]
    apply_sort_model_to_grid_options(grid_options, current_leaderboard_sort(st.session_state, key, default_sort))
    response = AgGrid(
        display_df,
        gridOptions=grid_options,
        columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
        theme=AgGridTheme.BALHAM,
        height=leaderboard_aggrid_viewport_height(len(display_df), LEADERBOARD_PAGE_SIZE, pagination=True),
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        key=key,
        allow_unsafe_jscode=True,
        update_on=["selectionChanged", "sortChanged"],
    )
    remember_leaderboard_sort(st.session_state, key, response, default_sort)
    return response


def _detail_grid(df: pl.DataFrame, key: str, selectable: bool = False, hidden: tuple[str, ...] = ()) -> Any:
    """A detail table under the leaderboard, up to 25 rows tall."""
    display_df = df.to_pandas(use_pyarrow_extension_array=False)
    coerce_numeric_columns(display_df)
    gb = GridOptionsBuilder.from_dataframe(display_df)
    if selectable:
        gb.configure_selection(selection_mode="single", use_checkbox=False, suppressRowClickSelection=False)
    _configure_columns(gb, display_df, hidden=hidden)
    grid_options = gb.build()
    if selectable:
        grid_options["onRowClicked"] = _ranking_row_click_jscode()
    grid_options["rowHeight"] = 28
    grid_options["domLayout"] = "normal"
    visible = max(1, min(len(display_df), 25))
    grid_options["alwaysShowVerticalScroll"] = len(display_df) > 25
    return AgGrid(
        display_df,
        gridOptions=grid_options,
        height=50 + visible * 28 + 20,
        theme=AgGridTheme.BALHAM,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        key=key,
        allow_unsafe_jscode=True,
        update_on=["selectionChanged"] if selectable else [],
    )


def _metric_card(column, label: str, value: str) -> None:
    column.markdown(
        f'<div class="metric-card"><small>{label}</small><br>'
        f'<span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{value}</span></div>',
        unsafe_allow_html=True,
    )


# -------------------------------
# Player detail
# -------------------------------
def _show_player_history(sessions: pl.DataFrame, boards: pl.DataFrame, row: Any, key_suffix: str) -> None:
    player_id = str(row.get("Player_ID"))
    player_name = row.get("Player_Name", "Unknown")
    scroll_to_here(f"brill:player:{player_id}")
    st.markdown(f"#### Event History: **{player_name}** ({row.get('Type', '')})")
    history = elo_brill.player_sessions(sessions, player_id)
    if history.is_empty():
        st.info("No events in the current selection.")
        return
    st.caption(
        f"{history.height} events. Pct is the mean board matchpoint percentage against the other tables, "
        "Site_Rank is Brill's own final rank, Pct_Rank ranks the field by Pct. "
        "Click an event for its standings and this player's boards."
    )
    selected = _selected_row(_detail_grid(
        history, key=f"brill_sessions_{_aggrid_key_part(player_id)}_{key_suffix}",
        selectable=True, hidden=("_tournament_id",),
    ))
    if selected is None:
        return
    tournament_id = str(selected.get("_tournament_id"))
    scroll_to_here(f"brill:event:{tournament_id}:{player_id}")
    st.markdown(f"#### Standings: **{selected.get('Event', '')}**")
    _detail_grid(
        elo_brill.event_standings(sessions, tournament_id),
        key=f"brill_standings_{tournament_id[:8]}_{_aggrid_key_part(player_id)}",
    )
    st.markdown(f"#### Boards: **{player_name}**")
    st.caption(
        "DD_Tricks_Diff is tricks taken minus double-dummy tricks for the declarer. "
        "DD_Lead_Cost is the tricks the opening lead gave up against the best double-dummy lead."
    )
    _detail_grid(
        elo_brill.player_event_boards(boards, tournament_id, player_id),
        key=f"brill_boards_{tournament_id[:8]}_{_aggrid_key_part(player_id)}",
    )


# -------------------------------
# Main UI
# -------------------------------
def _initialize_session_state() -> None:
    if "app_datetime" not in st.session_state:
        st.session_state.app_datetime = datetime.fromtimestamp(
            pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S %Z")


def _sidebar(event_types: List[str]) -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            (
                '<div style="display:flex;align-items:center;gap:0.6rem;margin:0 0 0.75rem 0;">'
                f'<img src="{ASSISTANT_LOGO_URL}" alt="Morty" width="40" height="40" '
                'style="width:2.5rem;height:2.5rem;border-radius:50%;flex:0 0 auto;">'
                '<div style="font-size:1.05rem;font-weight:700;line-height:1.25;">Brill Elo Ratings</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"Build:{st.session_state.app_datetime}")
        st.markdown("[What is Elo Rating?](https://en.wikipedia.org/wiki/Elo_rating_system)")
        st.markdown(f"[Brill Bridge Challenge]({SITE_URL})")

        ranking = st.radio(
            "Rank By", list(elo_brill.RANKINGS), key="brill_ranking",
            help="Elo, or one skill: declarer play, defence, opening leads, bidding to par.",
        )
        player_type = st.radio(
            "Players", list(elo_brill.PLAYER_TYPE_OPTIONS), key="brill_player_type", horizontal=True,
            help="Show humans, robots or both. Ranks and Elo are always among all listed players.",
        )
        event_options = [elo_brill.ALL_EVENTS, *event_types]
        if st.session_state.get("brill_event_type") not in event_options:
            st.session_state.brill_event_type = elo_brill.ALL_EVENTS
        event_choice = st.selectbox(
            "Filter by Event", event_options, key="brill_event_type",
            help="Use only the events of one daily series.",
        )
        scoring = st.radio(
            "Scoring", list(SCORING_OPTIONS), key="brill_scoring", horizontal=True,
            help="Use only matchpoint events or only IMP events.",
        )
        name_filter = st.text_input("Player name", key="brill_name_filter", placeholder="Part of a name")
        date_range = st.selectbox(
            "Date range", list(elo_brill.DATE_RANGE_OPTIONS), key="brill_date_range",
            help="Use only the events played in this window.",
        )
        if "brill_min_games" not in st.session_state:
            st.session_state.brill_min_games = DEFAULT_MIN_GAMES
        min_games = st.slider(
            "Minimum Games", min_value=1, max_value=100, key="brill_min_games",
            help="Events played, in the current selection, to appear in the rankings.",
        )
        generate_pdf = st.button("Export Report to PDF File", width="stretch")
        st.caption("PDF uses the leaderboard's last column sort.")
        st.markdown('<p style="color: #ffc107; font-weight: 600;">Other Morty websites</p>', unsafe_allow_html=True)
        st.markdown("[Bridge Chatbot](https://chat.7nt.info)")
        st.markdown("[ACBL Elo Ratings](https://acbl-elo.7nt.info)")
        st.markdown("[ACBL Postmortem](https://acbl.postmortem.chat)")
        st.markdown("[ACBL Statistics](https://acbl-stats.7nt.info)")
        st.markdown("[Brill Elo Ratings](https://brill-elo.7nt.info)")
        st.markdown("[Brill Postmortem](https://brill.postmortem.chat)")
        st.markdown("[FFBridge Elo Ratings](https://ffbridge-elo.7nt.info)")
        st.markdown("[FFBridge Postmortem](https://ffbridge.postmortem.chat)")
        st.markdown("[FFBridge Statistics](https://ffbridge-stats.7nt.info)")
        st.markdown("[PBN/LIN Postmortem](https://pbn.postmortem.chat)")
    return {
        "ranking": ranking, "player_type": player_type, "event_choice": event_choice, "scoring": scoring,
        "name_filter": name_filter, "date_range": date_range, "min_games": int(min_games),
        "generate_pdf": generate_pdf,
    }


def main() -> None:
    st.set_page_config(
        page_title="Unofficial Brill Elo Ratings Playground",
        page_icon=ASSISTANT_LOGO_URL,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_session_state()
    init_url_params_to_state(st, BRILL_URL_PARAMS)
    apply_app_theme(st)
    widen_scrollbars()

    refresh_warning = _refresh_dataset()
    metadata = elo_brill.read_metadata()
    dataset = _load_dataset(metadata["built_at"])
    all_boards: pl.DataFrame = dataset["boards"]
    all_sessions: pl.DataFrame = dataset["sessions"]

    event_types = (
        all_sessions.group_by("Event_Type").agg(pl.col("tournament_id").n_unique().alias("n"))
        .sort("n", "Event_Type", descending=[True, False])["Event_Type"].to_list()
    )
    choices = _sidebar(event_types)
    sync_state_to_url_params(st, BRILL_URL_PARAMS)
    ranking = choices["ranking"]

    st.markdown(f"""
        <div style="text-align: center; padding: 0 0 1rem 0; margin-top: -2rem;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">
                <img src="{ASSISTANT_LOGO_URL}" style="height: 2.5rem; vertical-align: middle; margin-right: 0.5rem;">
                Morty's Unofficial Brill Elo Ratings Playground
            </h1>
            <p style="color: #ffc107; font-size: 1.2rem; font-weight: 500; opacity: 0.9;">
                 {choices["event_choice"]} at Brill Bridge Challenge
            </p>
        </div>
    """, unsafe_allow_html=True)

    players = all_sessions.select("Player_ID", "Player_Type").unique()
    st.caption(
        f"Elo dataset — {metadata['tournaments']} events • {all_boards.height:,} board results • "
        f"{players.height} players ({players.filter(pl.col('Player_Type') == elo_brill.HUMAN).height} humans, "
        f"{players.filter(pl.col('Player_Type') == elo_brill.ROBOT).height} robots) • "
        f"{metadata['first_date']} to {metadata['last_date']} • updated {metadata['built_at'][:16].replace('T', ' ')} UTC"
    )
    if refresh_warning:
        st.warning(refresh_warning)

    date_from = elo_brill.date_range_start(choices["date_range"])
    boards = elo_brill.filter_frame(all_boards, choices["event_choice"], choices["scoring"], date_from)
    sessions = elo_brill.filter_frame(all_sessions, choices["event_choice"], choices["scoring"], date_from)
    table = elo_brill.leaderboard(boards, sessions, min_games=choices["min_games"])

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, "Tournaments", str(sessions["tournament_id"].n_unique()))
    _metric_card(m2, "Active Players", str(table.height))
    _metric_card(m3, "Avg Games", f"{table['Games'].cast(pl.Float64).mean():.1f}" if table.height else "0")
    _metric_card(m4, "Highest Elo", str(table["Elo"].max()) if table.height else "0")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"### {RANKING_TITLES[ranking]} (Min. {choices['min_games']} games)")
    if choices["date_range"] != "All time":
        st.caption(f"Date range: {choices['date_range']} (from {date_from})")
    st.caption(
        "Every entrant sits South at their own table beside three Brill robots and plays the same boards "
        "as the rest of the field. The **Type** column says whether the entrant is a human or a robot."
    )
    if ranking == "Elo":
        st.caption(
            "Elo is replayed over every event in time order from each entrant's mean board matchpoint "
            "percentage against the other tables (IMP events included), then standardized to a chess-style "
            "scale (mean **1500**, sd **400**) so it lines up with the ACBL and FFBridge apps; the **Title** "
            "column uses standard bands (≥2400 IM, ≥2500 GM, ≥2600 SGM). **Elo_Raw** is the rating before "
            "standardizing."
        )
    for column in RANKING_NOTES[ranking]:
        if not (ranking == "Elo" and column == "Elo"):
            st.caption(f"**{column}**: {elo_brill.METRIC_DEFINITIONS[column]}")

    if table.is_empty():
        st.info(f"No players match the minimum requirement of {choices['min_games']} games.")
    else:
        shown = elo_brill.filter_players(
            elo_brill.order_for_ranking(table, ranking), choices["player_type"], choices["name_filter"])
        if shown.is_empty():
            st.info("No players match the Players and name filters.")
        else:
            st.caption("Click a row to view the player's event history")
            rank_column = elo_brill.RANKINGS[ranking][0]
            grid_key = "brill_lb_" + "_".join(_aggrid_key_part(str(part)) for part in (
                ranking, choices["player_type"], choices["event_choice"], choices["scoring"],
                choices["date_range"], choices["min_games"], choices["name_filter"], metadata["built_at"],
            ))
            row = _selected_row(_leaderboard_grid(shown, grid_key, rank_column))
            if row is not None:
                _show_player_history(sessions, boards, row, key_suffix=grid_key[-40:])
            st.session_state.display_df = shown.drop("Player_ID")
            st.session_state.report_title = f"Brill {RANKING_TITLES[ranking]} - {datetime.now().strftime('%Y-%m-%d')}"
            st.session_state.report_rank_column = rank_column

    if choices["generate_pdf"] and "display_df" in st.session_state and not st.session_state.display_df.is_empty():
        with st.spinner("Preparing PDF export..."):
            title = st.session_state.get("report_title", "Unofficial Brill Elo Ratings")
            pdf_df = apply_aggrid_sort_model(
                st.session_state.display_df,
                st.session_state.get(LEADERBOARD_SORT_MODEL_KEY)
                or [{"colId": st.session_state.report_rank_column, "sort": "asc"}],
            )
            pdf_bytes = create_pdf(
                [
                    f"# {title}",
                    f"### Processed on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "### Created by https://brill-elo.7nt.info",
                    pdf_df,
                ],
                title=title,
                shrink_to_fit=True,
            )
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"Brill_Elo_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )

    clear_row_click_hourglass()
    render_app_footer(
        st,
        source_line=f"Data sourced from Brill Bridge Challenge ({SITE_URL})",
        diagnostics_lines=[
            footer_streamlit_app_diagnostics_line(st),
            f"Cache: dir={elo_brill.CACHE_DIR} • events={metadata['tournaments']} • "
            f"checked {metadata.get('checked_at', metadata['built_at'])[:16].replace('T', ' ')} UTC",
        ],
    )


if __name__ == "__main__":
    main()

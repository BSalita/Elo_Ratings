# Elo_Ratings
Unofficial bridge game elo ratings


## Brill Elo Ratings (https://brill-elo.7nt.info, port 8532)

Elo ratings and skill rankings for the daily events at [Brill Bridge Challenge](https://brill.aalborgdata.dk). Every entrant, human or robot, sits South at their own table beside three Brill robots and plays the same boards as the rest of the field, so the app rates entrants, not pairs. The **Type** column says Human or Robot.

```
streamlit_app_brill_elo_ratings.py   UI: Elo, Declarer, Defender, Opening Lead and Par rankings; event history per player
elo_brill.py                         fetch, double dummy / par / lead cost, matchpoints, Elo replay, leaderboard
build_brill_elo_parquets.py          first build or full rebuild (--rebuild, --limit N)
test_elo_brill.py                    unit tests
```

- Data comes through `postmortem-brill/brill_client.py` (a sibling checkout, or `./postmortem-brill` in the container), which signs in with `BRILL_EMAIL` / `BRILL_PASSWORD` from its `.env`. Only closed events are read.
- The dataset lives in `data/brill` (`BRILL_ELO_CACHE_DIR`): one fragment per event in `boards/`, plus `brill_boards.parquet`, `brill_sessions.parquet` and `brill_elo_metadata.json`. The app asks the site for newly closed events when its last check is older than `BRILL_ELO_MAX_AGE_HOURS` (default 1).
- Elo: events are replayed in time order. An entrant's score is the mean of their board matchpoint percentages against the other tables (IMP events too); the expectation is the Elo curve of their rating against the mean rating of the rest of the field; K is 32 times the share of the boards played. Choosing one event series or one scoring (MP, IMP) replays Elo over those events alone, so an MP rating never sees IMP events. Every view is shown on one fixed scale: 1500 and 400 are the mean and standard deviation of the all-event ratings of the players with 10 or more events, so a number does not move with the filters or with Minimum Games. A date range keeps the rating as it stood after the last event in the window.
- Skill rankings: declarer play is tricks minus double-dummy tricks on boards North or South declared (the entrant plays both); defence is the same difference with the sign turned on boards East or West declared; opening leads are scored against the best double-dummy lead on boards East declared; par rankings compare the final contract and the score with par.
- Double dummy needs mlBridge's ddss library (`mlBridge/dds_ddss.py`).

```powershell
python build_brill_elo_parquets.py
streamlit run streamlit_app_brill_elo_ratings.py --server.port 8532
python -m unittest test_elo_brill
```

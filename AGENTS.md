# AGENTS.md

## Cursor Cloud specific instructions

This repo is a Python (Streamlit + FastAPI) monorepo of three independently
deployable services for "unofficial bridge Elo ratings". There is **no**
`package.json`, no test suite, and no lint config; the only dependency manifest
is `requirements.txt` (see the pinned `streamlit==1.53.1` note there — do not
upgrade Streamlit past 1.53.1 or `streamlit-aggrid` breaks).

Dependencies are installed to the user site via the update script. The console
scripts (`streamlit`, `uvicorn`) land in `~/.local/bin`, which is **not on
PATH**, so run them as modules: `python3 -m streamlit ...` and
`python3 -m uvicorn ...`.

### Services and how to run them (dev)

Start commands mirror the `railway-*.toml` files (substitute a local port for
`$PORT`). All three can run at once on different ports.

| Service | File | Dev run command | Notes |
|---|---|---|---|
| ACBL Elo API (FastAPI) | `acbl_api_server.py` | `python3 -m uvicorn acbl_api_server:app --host 0.0.0.0 --port 8000` | Health: `GET /health`. Data endpoints `/acbl/report` & `/acbl/detail`. |
| ACBL Elo UI (Streamlit) | `streamlit_app_acbl_elo_ratings.py` | `ACBL_API_BASE_URL=http://localhost:8000 python3 -m streamlit run streamlit_app_acbl_elo_ratings.py --server.port 8502 --server.address 0.0.0.0 --server.headless true` | **Requires** `ACBL_API_BASE_URL`; it is API-only and shows no data without the API above. |
| FFBridge Elo UI (Streamlit) | `streamlit_app_ffbridge_elo_ratings.py` | `python3 -m streamlit run streamlit_app_ffbridge_elo_ratings.py --server.port 8501 --server.address 0.0.0.0 --server.headless true` | Standalone; pulls live data from the public FFBridge Lancelot API (needs internet). |

### Non-obvious gotchas

- **ACBL API needs parquet data.** With no `R2_BUCKET` set, the API reads local
  files `./data/acbl_club_elo_ratings.parquet` and
  `./data/acbl_tournament_elo_ratings.parquet` (the `data/` dir is absent from
  the repo by default and is **not** currently in `.gitignore`, so avoid
  committing it — it holds large generated parquet + API-cache artifacts).
  Without them, `/health` still returns ok but
  `/acbl/report` and `/acbl/detail` raise `FileNotFoundError`. To exercise the
  ACBL data path locally without R2 credentials, place parquet files at those
  paths. Expected columns (per `_required_columns_for_mode` /
  `generate_top_*_sql`): `Date, session_id, is_virtual_game, Pct_NS, Round,
  Board, DD_Tricks_Diff, Is_Par_Suit, Is_Par_Contract, Is_Sacrifice,
  Pair_Number_NS/EW, Player_{ID,Name}_{N,E,S,W}, MasterPoints_{N,E,S,W},
  Elo_R_{N,E,S,W}` for Players, plus `Elo_R_NS`/`Elo_R_EW` for Pairs.
- The ACBL API **caches the full parquet frame in-process** on first query
  (keyed by source path). After changing the local parquet files you must
  **restart** the uvicorn process to pick them up.
- ACBL UI defaults filter aggressively: `Minimum sessions played = 30`,
  `Game type = Local Only` (keeps only rows where `is_virtual_game == False`),
  and the API applies an elite skill gate (`min_skill_z`), so a small dataset
  can legitimately show few or zero rows.
- **FFBridge cold start is slow.** On first load the app fetches per-tournament
  results for ~1200+ past tournaments from the Lancelot API (there is no UI
  control to limit this; tournament/club filters are applied *after* the full
  fetch). Results are cached under `FFBRIDGE_CACHE_DIR` (default
  `data/ffbridge/…`), so subsequent loads are fast; the first load can take
  many minutes. This cache lives under `data/` (see note above) and should not
  be committed — it can grow to ~1 GB.
- Streamlit apps only execute their script (and thus fetch/compute) once a
  browser/websocket connects — a plain `curl /` returns 200 without triggering
  data processing.

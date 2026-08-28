@echo off
set "acbl_source=e:\bridge\data\acbl"
set "ffbridge_quality_source=e:\bridge\data\ffbridge\quality_cache"
set "ffbridge_quality_destination=data\ffbridge\quality_cache"

rem Update the canonical Elo data directory. deploy_elo_ratings.ps1 mounts this
rem directory directly; postmortem_start.ps1 owns _wslc_host\SavedModels.
for %%F in (
    acbl_club_elo_ratings.parquet
    acbl_tournament_elo_ratings.parquet
    acbl_club_player_elo_ratings.parquet
    acbl_tournament_player_elo_ratings.parquet
    acbl_club_pair_elo_ratings.parquet
    acbl_tournament_pair_elo_ratings.parquet
    acbl_club_elo_shrinkage.json
    acbl_tournament_elo_shrinkage.json
) do (
    xcopy "%acbl_source%\%%F" "data\" /D /Y
    if errorlevel 1 exit /b 1
)

if not exist "%ffbridge_quality_destination%\" (
    mkdir "%ffbridge_quality_destination%"
    if errorlevel 1 exit /b 1
)

rem Synchronize the completed FFBridge quality artifacts used by the app,
rem API, and MCP reports.
for %%F in (
    ffbridge_quality_boards.parquet
    ffbridge_quality_players.parquet
    ffbridge_quality_pairs.parquet
    ffbridge_quality_metadata.json
) do (
    if not exist "%ffbridge_quality_source%\%%F" exit /b 1
    xcopy "%ffbridge_quality_source%\%%F" "%ffbridge_quality_destination%\" /D /Y
    if errorlevel 1 exit /b 1
)

exit /b 0

@echo off
set "acbl_source=e:\bridge\data\acbl"
set "shared=%~dp0..\data"
set "ffbridge_quality_source=e:\bridge\data\ffbridge\quality_cache"
set "ffbridge_quality_destination=%shared%\ffbridge\quality_cache"
set "ffbridge_hier_source=e:\bridge\data\ffbridge\postmortem_archive_hierarchical"
set "ffbridge_hier_destination=%shared%\ffbridge\postmortem_archive_hierarchical"

rem Publish into the sibling src\data hub. Start scripts mount that tree.
rem postmortem_start.ps1 owns _wslc_host\SavedModels.
if not exist "%shared%\" (
    mkdir "%shared%"
    if errorlevel 1 exit /b 1
)
for %%F in (
    acbl_club_elo_ratings.parquet
    acbl_tournament_elo_ratings.parquet
    acbl_club_player_elo_ratings.parquet
    acbl_tournament_player_elo_ratings.parquet
    acbl_club_pair_elo_ratings.parquet
    acbl_tournament_pair_elo_ratings.parquet
    acbl_club_elo_shrinkage.json
    acbl_tournament_elo_shrinkage.json
    acbl_club_player_awards.parquet
    acbl_tournament_player_awards.parquet
) do (
    xcopy "%acbl_source%\%%F" "%shared%\" /D /Y
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

rem Production archive only: latest-revision v3 fragments, a matching manifest,
rem and metadata.json (written last). Domain shards, dataset/, sqlite and logs
rem are builder artifacts and are not published. Fails if metadata is not v3.
set "pub_py=%USERPROFILE%\.venvs\bridge-postmortem\Scripts\python.exe"
if not exist "%pub_py%" exit /b 1
if not exist "%ffbridge_hier_source%\metadata.json" exit /b 1
if not exist "%ffbridge_hier_source%\manifest.parquet" exit /b 1
"%pub_py%" "%~dp0publish_ffbridge_hierarchical.py" --source "%ffbridge_hier_source%" --destination "%ffbridge_hier_destination%"
if errorlevel 1 exit /b 1

rem Publish the same artifacts to prod src\data. Do not /MIR — that would
rem replace _wslc_host, which postmortem_start.ps1 stages for the mount.
set "prod_elo=\\X1-pro-470-1tb\c\sw\bridge\ML-Contract-Bridge\src\data"
if not exist "%prod_elo%\" (
    mkdir "%prod_elo%"
    if errorlevel 1 exit /b 1
)
for %%F in (
    acbl_club_elo_ratings.parquet
    acbl_tournament_elo_ratings.parquet
    acbl_club_player_elo_ratings.parquet
    acbl_tournament_player_elo_ratings.parquet
    acbl_club_pair_elo_ratings.parquet
    acbl_tournament_pair_elo_ratings.parquet
    acbl_club_elo_shrinkage.json
    acbl_tournament_elo_shrinkage.json
    acbl_club_player_awards.parquet
    acbl_tournament_player_awards.parquet
) do (
    xcopy "%shared%\%%F" "%prod_elo%\" /D /Y
    if errorlevel 1 exit /b 1
)
if not exist "%prod_elo%\ffbridge\quality_cache\" (
    mkdir "%prod_elo%\ffbridge\quality_cache"
    if errorlevel 1 exit /b 1
)
for %%F in (
    ffbridge_quality_boards.parquet
    ffbridge_quality_players.parquet
    ffbridge_quality_pairs.parquet
    ffbridge_quality_metadata.json
) do (
    xcopy "%ffbridge_quality_destination%\%%F" "%prod_elo%\ffbridge\quality_cache\" /D /Y
    if errorlevel 1 exit /b 1
)
"%pub_py%" "%~dp0publish_ffbridge_hierarchical.py" --source "%ffbridge_hier_source%" --destination "%prod_elo%\ffbridge\postmortem_archive_hierarchical"
if errorlevel 1 exit /b 1

exit /b 0

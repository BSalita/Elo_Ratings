@echo off
set "acbl_source=e:\bridge\data\acbl"
set "ffbridge_quality_source=e:\bridge\data\ffbridge\quality_cache"
set "ffbridge_quality_destination=data\ffbridge\quality_cache"
set "ffbridge_hier_source=e:\bridge\data\ffbridge\postmortem_archive_hierarchical"
set "ffbridge_hier_destination=data\ffbridge\postmortem_archive_hierarchical"

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

rem Compacted hierarchical archive only (metadata, manifest, dataset/).
rem Skip fragments/, sqlite, and domain shards — those are builder artifacts.
if not exist "%ffbridge_hier_source%\metadata.json" exit /b 1
if not exist "%ffbridge_hier_source%\manifest.parquet" exit /b 1
if not exist "%ffbridge_hier_source%\dataset\" exit /b 1
if not exist "%ffbridge_hier_destination%\" (
    mkdir "%ffbridge_hier_destination%"
    if errorlevel 1 exit /b 1
)
xcopy "%ffbridge_hier_source%\metadata.json" "%ffbridge_hier_destination%\" /D /Y
if errorlevel 1 exit /b 1
xcopy "%ffbridge_hier_source%\manifest.parquet" "%ffbridge_hier_destination%\" /D /Y
if errorlevel 1 exit /b 1
robocopy "%ffbridge_hier_source%\dataset" "%ffbridge_hier_destination%\dataset" /E /XO /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 exit /b 1

exit /b 0

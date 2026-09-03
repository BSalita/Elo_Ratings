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

rem Publish the same artifacts to prod. Do not /MIR data\ — that would replace
rem _wslc_host, which postmortem_start.ps1 stages for the container mount.
set "prod_elo=\\X1-pro-470-1tb\c\sw\bridge\ML-Contract-Bridge\src\elo\data"
if not exist "%prod_elo%\" exit /b 1
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
    xcopy "data\%%F" "%prod_elo%\" /D /Y
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
if not exist "%prod_elo%\ffbridge\postmortem_archive_hierarchical\" (
    mkdir "%prod_elo%\ffbridge\postmortem_archive_hierarchical"
    if errorlevel 1 exit /b 1
)
xcopy "%ffbridge_hier_destination%\metadata.json" "%prod_elo%\ffbridge\postmortem_archive_hierarchical\" /D /Y
if errorlevel 1 exit /b 1
xcopy "%ffbridge_hier_destination%\manifest.parquet" "%prod_elo%\ffbridge\postmortem_archive_hierarchical\" /D /Y
if errorlevel 1 exit /b 1
robocopy "%ffbridge_hier_destination%\dataset" "%prod_elo%\ffbridge\postmortem_archive_hierarchical\dataset" /E /XO /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 exit /b 1

exit /b 0

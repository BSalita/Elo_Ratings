@echo off
set "source=e:\bridge\data\acbl"

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
    xcopy "%source%\%%F" "data\" /D /Y
    if errorlevel 1 exit /b 1
)

exit /b 0

@echo off
REM ==========================================================================
REM  Rebuild the offline datasets so the SAC replay-buffer preload carries the
REM  CURRENT reward weights from config\general_config.py.
REM
REM  Regenerates (from the demos in bc_config.DEMO_LIST, same as the existing
REM  buffer: imitation\data\demos\map1_30car, no DAgger extra dirs):
REM    imitation\data\processed\dataset_rl_buffer.npz      (+ .meta.json)
REM    imitation\data\processed\dataset_bc_continuous.npz  (+ .meta.json)
REM  The previous files are copied to imitation\data\processed\backup_<timestamp>\
REM
REM  Does not need a CARLA server. Run from an activated training environment,
REM  or point PYTHON at its interpreter first, e.g.
REM    set PYTHON=C:\path\to\env\python.exe
REM ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
if not defined PYTHON set "PYTHON=python"

set "PROC=imitation\data\processed"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"

if exist "%PROC%\dataset_rl_buffer.npz" (
    echo [BACKUP] %PROC%\backup_%TS%\
    mkdir "%PROC%\backup_%TS%" || goto :fail
    copy /Y "%PROC%\dataset_*.*" "%PROC%\backup_%TS%\" >nul || goto :fail
)

echo [BUILD] %PYTHON% -m imitation.build_dataset --action-mode continuous
"%PYTHON%" -m imitation.build_dataset --action-mode continuous || goto :fail

echo.
echo [DONE] Rebuilt datasets:
dir /T:W "%PROC%\dataset_*.npz" | findstr /I "npz"
exit /b 0

:fail
echo.
echo [ERROR] Dataset rebuild failed. If a backup was taken, the previous files are in %PROC%\backup_%TS%\
exit /b 1

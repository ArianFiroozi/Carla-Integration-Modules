@echo off
REM ==========================================================================
REM  Iteration 1 (forensics-driven lane departure fix): FRESH SAC run, 0 -> 1M
REM  steps, actor initialized from sac_config.BC_CHECKPOINT_PATH, replay buffer
REM  preloaded from the rebuilt imitation\data\processed\dataset_rl_buffer.npz.
REM
REM  Usage:   start_sac_forensics_run.bat [SEED]      (default seed 42)
REM           e.g. run seeds 42, 43, 44 one after another for the 3-seed protocol.
REM
REM  Before running:
REM    1. rebuild_buffer.bat   (once, after any reward-weight change)
REM    2. A FRESHLY STARTED CARLA server on port 2000, e.g.
REM         CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000
REM       Never reuse a server that survived a force-killed run (it stays wedged
REM       in synchronous mode). Optional: set CARLA_EXE=C:\...\CarlaUE4.exe and this
REM       script starts one off-screen if nothing answers on port 2000.
REM
REM  Rendering: training is headless. CarlaEnv defaults to no_rendering=True and
REM  switches the server to synchronous + no-rendering mode itself; train_sac uses
REM  no cameras, so no extra rendering flags are needed (or wanted) here.
REM
REM  Optional overrides: set PYTHON=C:\path\to\env\python.exe
REM                      set SKIP_BUFFER_CHECK=1   (edit did not touch reward weights)
REM                      set ALLOW_CPU=1           (train without a CUDA GPU - very slow)
REM ==========================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
if not defined PYTHON set "PYTHON=python"
set "PYTHONUNBUFFERED=1"

set "SEED=%~1"
if "%SEED%"=="" set "SEED=42"

set "PREFLIGHT_ARGS="
if "%SKIP_BUFFER_CHECK%"=="1" set "PREFLIGHT_ARGS=--skip-buffer-check"
if "%ALLOW_CPU%"=="1" set "PREFLIGHT_ARGS=%PREFLIGHT_ARGS% --allow-cpu"

REM --- Optional: start CARLA off-screen if CARLA_EXE is set and port 2000 is closed ---
if defined CARLA_EXE (
    powershell -NoProfile -Command "$c=New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1',2000); exit 0 } catch { exit 1 } finally { $c.Close() }"
    if errorlevel 1 (
        echo [CARLA] Nothing on port 2000 - starting "!CARLA_EXE!" off-screen
        start "CARLA" "!CARLA_EXE!" -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
        set "PREFLIGHT_ARGS=!PREFLIGHT_ARGS! --wait-carla 180"
    )
)

REM --- Preflight: deps, fresh-run config, BC checkpoint, buffer freshness, server state ---
"%PYTHON%" -m rl.sac.preflight %PREFLIGHT_ARGS%
if errorlevel 1 (
    echo.
    echo [ABORT] Preflight failed - fix the [FAIL] items above. Training was NOT started.
    exit /b 1
)

echo.
echo [TRAIN] %PYTHON% -m rl.sac.train_sac --vehicles 30 --seed %SEED%
echo         Ctrl+C stops it cleanly; if you force-kill it, restart CarlaUE4.exe before
echo         the next run and continue with: python -m rl.sac.train_sac --resume-dir ^<run dir^>
"%PYTHON%" -m rl.sac.train_sac --vehicles 30 --seed %SEED%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [DONE] Training finished. Latest run: experiments\rl\sac\
) else (
    echo [ERROR] train_sac exited with code %RC%.
    echo         Restart CarlaUE4.exe, then resume: python -m rl.sac.train_sac --resume-dir ^<run dir^>
)
exit /b %RC%

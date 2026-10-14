"""
Preflight checks for a fresh SAC run. Fails fast (exit code 1) on anything that would
otherwise crash or silently corrupt a long training run.

    python -m rl.sac.preflight [--wait-carla SECONDS] [--skip-buffer-check] [--ignore-sync]

Checks:
  1. Python deps (carla, gymnasium, torch) import, and the configured DEVICE is usable.
  2. sac_config describes a FRESH, BC-initialized run (no resume / branch).
  3. BC checkpoint exists and its experiment folder carries normalization stats.
  4. Replay-buffer preload file exists and is newer than config/general_config.py
     (reward weights live there; a stale buffer still carries the old rewards).
  5. A CARLA server answers on localhost:2000 and is NOT already in synchronous mode
     (sync mode before we connect = wedged by a force-killed run, or another trainer
     is already driving it).
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

from config import general_config
from config import sac_config as cfg

CARLA_HOST, CARLA_PORT = "localhost", 2000
failures = []


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def mtime(path):
    return datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)


def check_deps():
    print("1. Python environment")
    for mod in ("carla", "gymnasium", "torch"):
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:
            fail(f"import {mod} failed ({e}) - activate the training environment first")
    try:
        import torch
        if str(cfg.DEVICE).startswith("cuda") and not torch.cuda.is_available():
            fail(f"DEVICE={cfg.DEVICE} but torch.cuda.is_available() is False")
        else:
            ok(f"DEVICE={cfg.DEVICE}")
    except Exception:
        pass


def check_config():
    print("2. sac_config (fresh BC-initialized run)")
    expect = {"RESUME_CHECKPOINT": False, "BRANCH_FROM": None,
              "LOAD_BC_WEIGHTS": True, "PRELOAD_EXPERT_DATA": True}
    for key, want in expect.items():
        got = getattr(cfg, key)
        (ok if got == want else fail)(f"{key} = {got!r} (expected {want!r})")
    ok(f"MAX_TRAIN_STEPS = {cfg.MAX_TRAIN_STEPS:,}")
    ok(f"reward: PENALTY_LANE_INVASION={general_config.PENALTY_LANE_INVASION}, "
       f"WEIGHT_CENTERING={general_config.WEIGHT_CENTERING}, "
       f"PENALTY_STALLING={general_config.PENALTY_STALLING}")


def check_bc():
    print("3. BC checkpoint")
    ckpt = Path(cfg.BC_CHECKPOINT_PATH)
    if not ckpt.exists():
        fail(f"BC checkpoint missing: {ckpt}")
        return
    ok(f"{ckpt}")
    cfg_json = ckpt.parents[1] / "config.json"
    try:
        stats = json.load(open(cfg_json)).get("dataset_meta", {}).get("normalization_stats")
        (ok if stats else fail)(f"normalization stats in {cfg_json.name}: "
                                f"{len(stats) if stats else 0} keys")
    except Exception as e:
        fail(f"cannot read {cfg_json}: {e}")


def check_buffer(skip):
    print("4. Replay-buffer preload")
    buf = Path(cfg.COMPILED_DATASET_PATH)
    if not buf.exists():
        fail(f"buffer missing: {buf} - run rebuild_buffer.bat")
        return
    reward_cfg = Path(general_config.__file__)
    b, r = mtime(buf), mtime(reward_cfg)
    if b >= r:
        ok(f"buffer built {b:%Y-%m-%d %H:%M} (after reward config change {r:%Y-%m-%d %H:%M})")
    elif skip:
        ok(f"buffer older than reward config, but --skip-buffer-check given ({b:%Y-%m-%d %H:%M})")
    else:
        fail(f"buffer built {b:%Y-%m-%d %H:%M} is OLDER than general_config.py "
             f"({r:%Y-%m-%d %H:%M}) - it still carries old rewards; run rebuild_buffer.bat "
             f"(or pass --skip-buffer-check if the edit did not touch reward weights)")


def check_carla(wait_s, ignore_sync):
    print(f"5. CARLA server {CARLA_HOST}:{CARLA_PORT}")
    try:
        import carla
    except Exception:
        fail("carla module not importable - skipped server check")
        return
    deadline = time.time() + wait_s
    while True:
        try:
            client = carla.Client(CARLA_HOST, CARLA_PORT)
            client.set_timeout(10.0)
            version = client.get_server_version()
            settings = client.get_world().get_settings()
            break
        except Exception as e:
            if time.time() >= deadline:
                fail(f"no CARLA server answering ({e}). Start: "
                     f"CarlaUE4.exe -quality-level=Low -carla-rpc-port={CARLA_PORT}")
                return
            print(f"         waiting for CARLA... ({int(deadline - time.time())}s left)")
            time.sleep(5)
    ok(f"server version {version}, client {client.get_client_version()}")
    if settings.synchronous_mode and not ignore_sync:
        fail("server is already in synchronous mode: a force-killed run left it wedged, or "
             "another trainer is using it. Restart CarlaUE4.exe (or pass --ignore-sync).")
    else:
        ok(f"synchronous_mode={settings.synchronous_mode} (env enables sync + no_rendering itself)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-carla", type=int, default=0, help="seconds to wait for the server")
    ap.add_argument("--skip-buffer-check", action="store_true")
    ap.add_argument("--ignore-sync", action="store_true")
    args = ap.parse_args()

    print("=" * 70 + "\nSAC PREFLIGHT\n" + "=" * 70)
    check_deps()
    check_config()
    check_bc()
    check_buffer(args.skip_buffer_check)
    check_carla(args.wait_carla, args.ignore_sync)
    print("=" * 70)
    if failures:
        print(f"PREFLIGHT FAILED ({len(failures)} problem(s)) - training NOT started.")
        sys.exit(1)
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()

import getpass
import os
import socket
import subprocess


def get_experiment_creator() -> str:
    """
    Determine whether the current experiment is being run by 'Mani' or 'Mehdi'.
    Can be overridden via the EXPERIMENT_CREATOR environment variable.
    """
    override = os.environ.get("EXPERIMENT_CREATOR")
    if override and override.strip():
        return override.strip()

    try:
        host = socket.gethostname().lower()
        if "mani" in host:
            return "Mani"
    except Exception:
        pass

    try:
        user = getpass.getuser().lower()
        if "mani" in user:
            return "Mani"
    except Exception:
        pass

    try:
        git_user = (
            subprocess.check_output(
                ["git", "config", "user.name"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8", errors="ignore")
            .strip()
            .lower()
        )
        if "mani" in git_user:
            return "Mani"
    except Exception:
        pass

    return "Mehdi"


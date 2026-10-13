import subprocess
import sys


def _warn_internal_users() -> None:
    try:
        result = subprocess.run(
            ["git", "config", "--global", "--get", "user.email"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if result.returncode == 0 and result.stdout.strip().casefold().endswith("@valtech.ai"):
        print(
            "WARNING: This is the open-source version of OOps. "
            "Valhalla employees should use the internal high-performance version.",
            file=sys.stderr,
        )

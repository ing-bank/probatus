"""Run the same validation stages locally and in GitHub Actions."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run selected checks in the current Python environment, stopping on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=["lint", "tests", "docs"], default=["lint", "tests", "docs"])
    parser.add_argument("--notebooks", action="store_true", help="Also execute notebook tests (slower).")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["MPLBACKEND"] = "Agg"
    env["TEST_NOTEBOOKS"] = "1" if args.notebooks else "0"
    commands = {
        "lint": ["pre_commit", "run", "--all-files"],
        "tests": ["pytest", "tests", "--cov=probatus", "--cov-report=xml", "--cov-report=term-missing"],
        "docs": ["mkdocs", "build"],
    }
    for check in args.checks:
        print(f"\nRunning {check}...", flush=True)
        result = subprocess.run([sys.executable, "-m", *commands[check]], cwd=root, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())

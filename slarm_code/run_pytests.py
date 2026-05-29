#!/usr/bin/env python3
"""
Simple wrapper to run pytest when executed via the Slurm generic runner.

Usage: sbatch slarm_code/run_generic.slurm slarm_code/run_pytests.py [pytest args]
The slurm runner sources the project's .venv before invoking this script.
"""
import sys
import pytest


def main():
    # If no args provided, run tests under the top-level `tests/` directory only.
    args = sys.argv[1:]
    if not args:
        args = ["tests", "-q"]
    # Run pytest and forward exit code
    rc = pytest.main(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

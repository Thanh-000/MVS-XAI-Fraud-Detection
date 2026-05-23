"""Compatibility wrapper for the canonical academic E2E runner.

The project now exposes one official execution path. Keep this file only so
older commands that call run_production.py do not diverge from the experiment
pipeline.
"""
from run_academic_e2e import main


if __name__ == "__main__":
    main()

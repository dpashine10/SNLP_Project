"""Run the frozen Week 3 vs Week 4 evaluation once and save a new evidence run.

Run from the repository root::

    python -m src.week5.scripts.freeze_evaluation
"""

import logging

import pandas as pd

from src.week4.config import CONFIG
from src.week5.evaluation.frozen_run import METRICS_CSV, run_frozen_evaluation
from src.week5.paths import relative_to_root


def main() -> None:
    """Run the evaluation, then print the run folder and the summary table."""
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Third-party libraries stay at WARNING; Week 4 and Week 5 log at INFO.
    for name in ("src.week4", "src.week5"):
        logging.getLogger(name).setLevel(logging.INFO)
    run_dir = run_frozen_evaluation(CONFIG)
    print(f"\nRun saved to {relative_to_root(run_dir)}\n")
    print(pd.read_csv(run_dir / METRICS_CSV).to_string(index=False))


if __name__ == "__main__":
    main()

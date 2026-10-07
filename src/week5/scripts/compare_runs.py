"""Compare two frozen runs and save a reproducibility report.

Run from the repository root::

    python -m src.week5.scripts.compare_runs              # the two most recent runs
    python -m src.week5.scripts.compare_runs RUN_A RUN_B  # run IDs or folders
"""

import sys
from pathlib import Path

from src.week5.evaluation.reproducibility import compare_runs, write_report
from src.week5.paths import REPRODUCIBILITY_DIR, RUNS_DIR, relative_to_root


def main(argv: list[str]) -> None:
    """Compare the given runs (or the two latest), print the verdict and save the report."""
    if argv and len(argv) != 2:
        raise SystemExit("Usage: python -m src.week5.scripts.compare_runs [RUN_A RUN_B]")
    run_a, run_b = [_resolve(arg) for arg in argv] if argv else _latest_two()
    report = compare_runs(run_a, run_b)
    json_path, md_path = write_report(report, REPRODUCIBILITY_DIR)
    print(md_path.read_text(encoding="utf-8"))
    print(f"Saved {relative_to_root(json_path)} and {relative_to_root(md_path)}")


def _resolve(arg: str) -> Path:
    path = Path(arg)
    return path if path.is_dir() else RUNS_DIR / arg


def _latest_two() -> tuple[Path, Path]:
    runs = sorted(path for path in RUNS_DIR.glob("*") if path.is_dir())
    if len(runs) < 2:
        raise SystemExit(
            f"Need at least two runs in {relative_to_root(RUNS_DIR)}; "
            "run `python -m src.week5.scripts.freeze_evaluation` again."
        )
    return runs[-2], runs[-1]


if __name__ == "__main__":
    main(sys.argv[1:])

"""Multi-seed benchmark entry point.

Thin alias over :mod:`civictwin.experiments.run_rq1` so the benchmark can be
invoked by an obvious name:

    python -m civictwin.experiments.run_benchmark --seeds 1 2 3 4 5 --device cuda

All flags are identical to ``run_rq1``; strict mode is off here because a
benchmark sweep should report the verdict rather than exit non-zero.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from civictwin.experiments.run_rq1 import (  # re-exported for convenience
    DEFAULT_SEEDS,
    build_policy_scorecard,
    main as _run_rq1_main,
    run_forecast_experiment,
    run_multi_seed_benchmark,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--no-strict" not in args:
        args.append("--no-strict")
    _run_rq1_main(args)


__all__ = [
    "main",
    "run_multi_seed_benchmark",
    "run_forecast_experiment",
    "build_policy_scorecard",
    "DEFAULT_SEEDS",
]


if __name__ == "__main__":
    main()

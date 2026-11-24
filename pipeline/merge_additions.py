"""Merge whatever is pending in career-ops/batch/tracker-additions/ — after
recovering the rows a previous merge refused, and sanitizing every row first.

The step `run.sh`/`run.ps1` run once `batch-runner.sh` returns. That runner
merges its own additions as its last act, inside career-ops with no Python in
the process tree, so a row the agent CLI wrote with a bare `4.2` is refused
before anything here can see it — archived into `merged/` before
career-ops#5862, where nothing retries it, and kept in the queue since, where a
re-run refuses it again. `run_merge_tracker` is the one place that knows how to
repair either kind (pulling an archived one back first) and merge it again; this
module exists so the wrappers can reach it after the runner, which nothing else
in the pipeline does.

Also the right entry point for a manual retry: `python -m pipeline.merge_additions`.

Exits 1 unless the merge was clean (`run_merge_tracker`'s True): a partial merge,
or one where merge-tracker refused an addition, is not a success, and the log
above the exit names each one. That is what ends `run.sh`/`run.ps1 --batch` in
failure; `--evaluate-batch` ends the same way on the same result, through
orchestrate's exit code (`batch_evaluate.EvalRun`).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline._batch_common import run_merge_tracker
from pipeline.stdio import line_buffer_stdout
from pipeline.tracker_layout import career_ops_dir

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    # The same .env orchestrate.py reads, so CAREER_OPS_PATH resolves the same way.
    load_dotenv(ROOT / ".env")
    career_ops = career_ops_dir()
    if not (career_ops / "merge-tracker.mjs").exists():
        print(f"[batch] no career-ops checkout at {career_ops} — nothing to merge into")
        return 0
    return 0 if run_merge_tracker(career_ops) else 1


if __name__ == "__main__":
    line_buffer_stdout()
    sys.exit(main())

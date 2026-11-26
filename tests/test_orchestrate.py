"""Wiring tests for orchestrate.py's optional stages.

These don't exercise the stages themselves (each has its own test module) — they
pin the argparse-to-call wiring so the opt-in liveness re-check actually runs in
the pipeline (and therefore in the cloud workflow that invokes orchestrate),
and stays OFF by default so existing runs are unchanged.
"""

import json
import sys

import pytest

import orchestrate
from pipeline import batch_evaluate, handoff, recheck
from pipeline._batch_common import MERGE_NOT_CLEAN_LINE
from tests.conftest import evaluate_without_a_provider


@pytest.fixture
def quiet_pipeline(monkeypatch, tmp_path):
    """Neutralise the real stages so main() reaches the optional blocks cheaply:
    a tmp config that exists, a tmp career-ops, no toasts, an empty bridge."""
    cfg = tmp_path / "search.yml"
    cfg.write_text("searches: []\n", encoding="utf-8")
    co = tmp_path / "career-ops"
    (co / "data").mkdir(parents=True)
    monkeypatch.setenv("CAREER_OPS_PATH", str(co))
    monkeypatch.setattr(orchestrate.notify, "notify", lambda *a, **k: None)
    monkeypatch.setattr(orchestrate.bridge, "run", lambda *a, **k: [])
    return cfg, co


def _argv(cfg, *extra):
    # Skip every always-on stage; we only care about the optional tail.
    return ["orchestrate.py", "--config", str(cfg), "--skip-scrape", "--skip-filter",
            "--skip-screen", "--skip-bridge", "--skip-batch-prep", *extra]


class TestRecheckWiring:
    def test_flag_runs_recheck(self, quiet_pipeline, monkeypatch):
        cfg, co = quiet_pipeline
        calls = []
        monkeypatch.setattr(recheck, "run", lambda career_ops, **kw: calls.append((career_ops, kw)))
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--recheck-liveness"))
        assert orchestrate.main() == 0
        assert len(calls) == 1
        career_ops, kw = calls[0]
        assert career_ops == co.resolve()
        assert kw.get("timeout") == 8        # default --recheck-timeout

    def test_recheck_timeout_forwarded(self, quiet_pipeline, monkeypatch):
        cfg, _ = quiet_pipeline
        calls = []
        monkeypatch.setattr(recheck, "run", lambda career_ops, **kw: calls.append(kw))
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--recheck-liveness", "--recheck-timeout", "20"))
        orchestrate.main()
        assert calls and calls[0].get("timeout") == 20

    def test_off_by_default(self, quiet_pipeline, monkeypatch):
        cfg, _ = quiet_pipeline
        called = []
        monkeypatch.setattr(recheck, "run", lambda *a, **k: called.append(True))
        monkeypatch.setattr(sys, "argv", _argv(cfg))   # no --recheck-liveness
        assert orchestrate.main() == 0
        assert called == []

    def test_recheck_drain_routes_to_drain(self, quiet_pipeline, monkeypatch):
        """--recheck-drain loops the budgeted sweep until the backlog is covered;
        without it the re-check is a single sweep."""
        cfg, co = quiet_pipeline
        ran, drained = [], []
        monkeypatch.setattr(recheck, "run", lambda career_ops, **kw: ran.append(kw))
        monkeypatch.setattr(recheck, "drain", lambda career_ops, **kw: drained.append(kw))
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--recheck-liveness", "--recheck-drain"))
        assert orchestrate.main() == 0
        assert len(drained) == 1 and not ran          # drained, not single-swept
        assert drained[0].get("timeout") == 8

    def test_plain_recheck_does_not_drain(self, quiet_pipeline, monkeypatch):
        cfg, _ = quiet_pipeline
        ran, drained = [], []
        monkeypatch.setattr(recheck, "run", lambda career_ops, **kw: ran.append(kw))
        monkeypatch.setattr(recheck, "drain", lambda career_ops, **kw: drained.append(kw))
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--recheck-liveness"))   # no --recheck-drain
        orchestrate.main()
        assert len(ran) == 1 and not drained


class TestUncleanMergeExitCode:
    """--evaluate-batch's tracker merge decides the exit code, at the END: a
    merge that was not clean (`run_merge_tracker` False) still lets every later
    stage run, then exits 1 with one line saying why — which is what fails the
    cloud daily's evaluate step, where a refused addition is never retried. Both
    merges batch_evaluate runs count: the heal sweep of an earlier run's queue,
    and the merge after evaluating."""

    @staticmethod
    def _queue(co, *, evaluated: bool):
        """One job in the batch queue. `evaluated`: already done, with its
        addition still queued — the heal sweep's case; otherwise pending, and
        evaluated by a stand-in that makes no provider call."""
        batch = co / "batch"
        (batch / "tracker-additions").mkdir(parents=True)
        (batch / "batch-input.tsv").write_text(
            "id\turl\tsource\tnotes\n1\thttps://a.com\tAcme\tEng\n", encoding="utf-8")
        (co / "cv.md").write_text("# CV", encoding="utf-8")
        if evaluated:
            (batch / "batch-api-state.json").write_text(
                json.dumps({"jobs": {"1": {"status": "completed"}}}), encoding="utf-8")
            (batch / "tracker-additions" / "1.tsv").write_text("1\t2026-06-01\tAcme\n",
                                                                encoding="utf-8")

    @pytest.fixture
    def stages(self, monkeypatch):
        """The order things happened in: the merge, then the later stages."""
        events = []
        monkeypatch.setattr(recheck, "run", lambda *a, **k: events.append("recheck"))
        monkeypatch.setattr(handoff, "run", lambda *a, **k: events.append("handoff") or 0)
        return events

    @pytest.mark.parametrize("site", ["heal sweep", "after evaluating"])
    @pytest.mark.parametrize("clean", [True, False])
    def test_the_merge_decides_the_exit_code_after_every_stage(
            self, quiet_pipeline, monkeypatch, capsys, stages, site, clean):
        cfg, co = quiet_pipeline
        self._queue(co, evaluated=site == "heal sweep")
        if site == "after evaluating":
            evaluate_without_a_provider(monkeypatch)
        monkeypatch.setattr(batch_evaluate, "run_merge_tracker",
                            lambda career_ops: stages.append("merge") or clean)
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--evaluate-batch", "--batch-provider",
                                               "anthropic", "--recheck-liveness", "--handoff"))
        assert orchestrate.main() == (0 if clean else 1)
        assert stages == ["merge", "recheck", "handoff"]   # nothing stopped early
        err = capsys.readouterr().err
        if clean:
            assert MERGE_NOT_CLEAN_LINE not in err
        else:
            # One line, and the last thing the run says.
            assert err.rstrip().splitlines()[-1].startswith(MERGE_NOT_CLEAN_LINE)
            assert err.count(MERGE_NOT_CLEAN_LINE) == 1

    def test_a_failed_handoff_is_not_reported_as_only_the_merge(
            self, quiet_pipeline, monkeypatch, capsys, stages):
        # The UI reads MERGE_NOT_CLEAN_LINE as "the run finished, only the merge
        # was not clean"; with a handoff failure as well, that would hide it.
        cfg, co = quiet_pipeline
        self._queue(co, evaluated=True)
        monkeypatch.setattr(batch_evaluate, "run_merge_tracker",
                            lambda career_ops: stages.append("merge") or False)
        monkeypatch.setattr(handoff, "run", lambda *a, **k: stages.append("handoff") or 2)
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--evaluate-batch", "--batch-provider",
                                               "anthropic", "--handoff"))
        assert orchestrate.main() == 2
        err = capsys.readouterr().err
        assert not any(line.startswith(MERGE_NOT_CLEAN_LINE) for line in err.splitlines())
        assert "also: the tracker merge was not clean" in err

    def test_nothing_to_merge_exits_zero(self, quiet_pipeline, monkeypatch, stages):
        # Every job evaluated and nothing queued: no merge runs, which is not a
        # failure.
        cfg, co = quiet_pipeline
        self._queue(co, evaluated=True)
        (co / "batch" / "tracker-additions" / "1.tsv").unlink()
        monkeypatch.setattr(batch_evaluate, "run_merge_tracker",
                            lambda career_ops: stages.append("merge") or False)
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--evaluate-batch", "--batch-provider",
                                               "anthropic"))
        assert orchestrate.main() == 0
        assert stages == []

    def test_no_queue_at_all_exits_zero(self, quiet_pipeline, monkeypatch, stages):
        cfg, _ = quiet_pipeline                 # no batch/ under career-ops
        monkeypatch.setattr(batch_evaluate, "run_merge_tracker",
                            lambda career_ops: stages.append("merge") or False)
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--evaluate-batch", "--batch-provider",
                                               "anthropic"))
        assert orchestrate.main() == 0
        assert stages == []


class TestEvaluateBatchWiring:
    """--retry-abandoned reaches the evaluator: the jobs abandoned after
    MAX_ATTEMPTS failed runs (#206) have no other way back into the queue."""

    def _run(self, quiet_pipeline, monkeypatch, *extra):
        cfg, co = quiet_pipeline
        calls = []
        monkeypatch.setattr(orchestrate.batch_evaluate, "run",
                            lambda career_ops, **kw: calls.append((career_ops, kw))
                            or orchestrate.batch_evaluate.EvalRun(0))
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--evaluate-batch", *extra))
        assert orchestrate.main() == 0
        assert len(calls) == 1 and calls[0][0] == co.resolve()
        return calls[0][1]

    def test_retry_abandoned_is_forwarded(self, quiet_pipeline, monkeypatch):
        assert self._run(quiet_pipeline, monkeypatch, "--retry-abandoned")["retry_abandoned"] is True

    def test_retry_abandoned_needs_evaluate_batch(self, quiet_pipeline, monkeypatch):
        """Without --evaluate-batch the flag would be ignored after a full run,
        and `--batch --retry-abandoned` is the easy slip."""
        cfg, _ = quiet_pipeline
        monkeypatch.setattr(sys, "argv", _argv(cfg, "--retry-abandoned"))
        with pytest.raises(SystemExit) as exc:
            orchestrate.main()
        assert exc.value.code == 2

    def test_retry_abandoned_is_off_by_default(self, quiet_pipeline, monkeypatch):
        assert self._run(quiet_pipeline, monkeypatch)["retry_abandoned"] is False


class TestScrubsTheProfileFirst:
    """main() scrubs profile.yml before any stage (#227; the scrub's own
    contract is tests/test_batch_common.py::TestScrubEveryCheckout): once, for
    the CAREER_OPS_PATH checkout, ahead of the first stage."""

    def test_scrubbed_once_before_the_first_stage(self, quiet_pipeline, monkeypatch):
        cfg, co = quiet_pipeline
        events = []
        monkeypatch.setattr(orchestrate, "scrub_profile_yml_files",
                            lambda career_ops: events.append(("scrub", career_ops)))
        monkeypatch.setattr(orchestrate.scrape, "run", lambda *a, **k: events.append("scrape"))
        monkeypatch.setattr(sys, "argv", ["orchestrate.py", "--config", str(cfg),
                                          "--skip-filter", "--skip-screen", "--skip-bridge",
                                          "--skip-batch-prep"])
        assert orchestrate.main() == 0
        assert events == [("scrub", co.resolve()), "scrape"]

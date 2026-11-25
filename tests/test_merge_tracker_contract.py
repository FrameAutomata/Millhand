"""Contract test against the installed career-ops `merge-tracker.mjs`.

`_report_key` in [pipeline/_batch_common.py] is a claim about ONE line of that
script: on its guessed match tiers — entry number, fuzzy company+role — a merge
updates the existing row but keeps THAT ROW'S role title, while writing the
addition's report link and score through.

    role: (reportNumMatched || dupReason === 'url') ? addition.role : duplicate.role

The loss guard is built on that reading. If upstream flips it, or adds a tier
that doesn't carry the report number through, the guard either goes back to
reporting intact evaluations as lost (#152, the bug this exists to remove) or —
worse — starts scoring a genuinely lost row as landed. Unit fixtures cannot
catch either: a fixture that simulates the merge by moving files is a third copy
of the behaviour, and it stays green while the real script changes underneath.
So this drives the real one, the way `tests/test_app_onboard.py` drives the real
`setup-profile.mjs` and `tests/test_jobspy_contract.py` reads the installed
library.

It needs a career-ops checkout with its npm deps, so it runs locally (setup
clones and installs one) and in the contract gate, `career-ops-contract.yml`,
which does both; `tests.yml` installs neither and skips it. Read-only with respect to the
checkout — the tracker, the additions dir and batch-state all live in tmp_path
via merge-tracker's own env overrides.

It has to hold on both sides of two upstream changes, because the cloud daily
runs the pinned commit while a person's checkout tracks upstream `main`, and the
pin cannot move until this module passes on the candidate:

- **career-ops#6062** (2026-11-09) refuses a headerless addition. So every
  addition here is written the way `write_job_result` writes it for a checkout
  that reads labels — under the label line, through `labelled_addition` —
  except where a refusal of the raw row is the point. The gate requires that
  of the checkout (`test_the_checkout_reads_labelled_additions`), and each
  staged tree carries the checkout's own merge-tracker.mjs, so the pipeline's
  probe of it answers there what it answers for the checkout.
- **career-ops#5862** (2026-10-26) KEEPS a refused addition in the queue, where
  it used to archive it into `merged/`. What a test expects of a refusal is
  read off the checkout's own script (`KEEPS_REFUSALS`), the way headed support
  always was, so each release is held to what it says it does rather than
  either being let off.
"""

import contextlib
import io
import os
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from pipeline._batch_common import (
    ADDITION_COLUMNS, ADDITION_LABEL_LINE, ADDITION_LABELS, _REFUSES_HEADERLESS_MARKER,
    _REQUIRED_IDXS, _liveness_closed_rows, _pending_additions,
    _reopen_reposted, _sanitize_pending_additions, _warn_on_lost_additions,
    DECISION_SKIP_MARK, closed_by_recheck, jd_cache_path, labelled_addition,
    liveness_closed_mark, queued_additions, reads_labelled_additions, run_merge_tracker,
    sanitize_addition,
)
from pipeline.tracker_layout import career_ops_dir, data_rows
from pipeline.work_location import work_location_from_notes
from tests.conftest import tracker_row

HEADER = ("# Applications Tracker\n\n"
          "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
          "|---|------|---------|------|-------|--------|-----|--------|-------|\n")

# One scenario, used with and without req ids: one title at one company, and a
# re-wording of it that still fuzzy-matches — `(Remote)` is a suffix a re-post
# really picks up. NOT two levels of one title, deliberately: that was the
# original premise, and career-ops#3651 (merged 2026-09-02) made a stated level a
# disagreement, so `SPECIALIST I` vs `… II` stopped fuzzy-matching and the three
# tests on this fixture failed by design. This pair matches under both matchers,
# so the suite is honest against a checkout on either side of that release.
# Single-sourced, so if upstream tightens the matcher again the premise fails in
# ONE place rather than rotting in a second copy while the first is repaired.
BASE, VARIANT = "INSURANCE SPECIALIST II", "Insurance Specialist II (Remote)"
def _notes(req, jk):
    return (f"req {req} — " if req else "") + f"https://indeed.com/viewjob?jk={jk} — APPLY"


def _tracker(role, report, req=None):
    return HEADER + (f"| 10 | 2026-08-01 | UT Southwestern Medical Center | {role} "
                     f"| 4.0/5 | Evaluated | ❌ | [{report}](../reports/{report}-ut.md) "
                     f"| {_notes(req, 'old')} |\n")


def _labelled(row):
    """`row` as `write_job_result` writes it: under career-ops' label line."""
    text = labelled_addition(row)
    assert text is not None, f"the writer would not label {row!r}"
    return text


def _addition(role, report, req=None):
    return _labelled("11\t2026-09-01\tUT Southwestern Medical Center\t" + role +
                     f"\tEvaluated\t4.7/5\tnull\t[{report}](reports/{report}-ut.md)\t"
                     f"{_notes(req, 'new')}")


def _merge_tracker_runnable() -> bool:
    """True only if node, the career-ops checkout, and the npm packages
    merge-tracker imports all resolve."""
    career_ops = career_ops_dir()
    if shutil.which("node") is None or not (career_ops / "merge-tracker.mjs").exists():
        return False
    try:
        r = subprocess.run(["node", "-e", "require.resolve('js-yaml')"],
                           cwd=str(career_ops), capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        # Broad on purpose: this runs at MODULE SCOPE to build `pytestmark`, so
        # anything it raises — a TimeoutExpired from a wedged node, not an
        # OSError — is a collection error that fails the whole suite instead of
        # skipping this one file. Same reason `test_app_onboard.py` is broad.
        return False


_RUNNABLE = _merge_tracker_runnable()

pytestmark = pytest.mark.skipif(
    not _RUNNABLE,
    reason=(f"needs node and a career-ops checkout with its npm deps at {career_ops_dir()} "
            "(CAREER_OPS_PATH; `npm install --ignore-scripts` there)"))


def _merge_tracker_source() -> str:
    """The checkout's merge-tracker.mjs, or "" — read at import to settle what
    this release does, so broad for `_merge_tracker_runnable`'s reason."""
    try:
        return (career_ops_dir() / "merge-tracker.mjs").read_text(encoding="utf-8")
    except Exception:
        return ""


_SOURCE = _merge_tracker_source()
# Each picks which of two documented behaviours a test expects, by a text probe
# that asserts nothing itself: the tests below hold the running script to it,
# so a probe that went stale fails a test rather than excusing one.
KEEPS_REFUSALS = "refusedAdditions" in _SOURCE              # career-ops#5862, 2026-10-26
REFUSES_HEADERLESS = _REFUSES_HEADERLESS_MARKER in _SOURCE    # career-ops#6062, 2026-11-09


# A refused addition is in exactly one place: kept in the queue since
# career-ops#5862, archived into merged/ before it.
REFUSED_WHERE = (True, False) if KEEPS_REFUSALS else (False, True)     # (kept, archived)


def _refused_where(additions_dir, name):
    """(kept, archived) for one addition, to compare with `REFUSED_WHERE`."""
    return (additions_dir / name).exists(), (additions_dir / "merged" / name).exists()


def _stage(tmp_path, tracker_text, additions, batch_state=""):
    """Lay out a tracker, an additions dir ({filename: row}) and a batch-state
    file inside tmp_path — the additions where `run_merge_tracker` looks for
    them, so a test can drive either the script or the whole merge. Returns
    (career_ops_root, additions_dir, batch_state).

    The tree carries a copy of the checkout's merge-tracker.mjs, which nothing
    runs — node runs the checkout's own, from its cwd — but which
    `run_merge_tracker` asks for, and which `reads_labelled_additions` reads,
    so the writer and the sanitizer label here exactly when they would for the
    checkout."""
    career_ops = tmp_path / "co"
    (career_ops / "data").mkdir(parents=True)
    (career_ops / "merge-tracker.mjs").write_text(_SOURCE, encoding="utf-8")
    (career_ops / "data" / "applications.md").write_text(tracker_text, encoding="utf-8")
    additions_dir = career_ops / "batch" / "tracker-additions"
    additions_dir.mkdir(parents=True)
    for name, row in additions.items():
        (additions_dir / name).write_text(row, encoding="utf-8")
    state = tmp_path / "batch-state.tsv"
    state.write_text(batch_state, encoding="utf-8")
    return career_ops, additions_dir, state


def _merge_env(career_ops, state):
    """The parent environment PLUS merge-tracker's overrides naming a `_stage`d
    tree, never a hand-built dict: node needs SystemRoot/COMSPEC/TEMP to start
    at all on Windows, which is this repo's primary platform, and a stripped env
    would turn a local-only test into a hard failure rather than the skip above."""
    return {**os.environ,
            "CAREER_OPS_TRACKER": str(career_ops / "data" / "applications.md"),
            "CAREER_OPS_ADDITIONS": str(career_ops / "batch" / "tracker-additions"),
            "CAREER_OPS_BATCH_STATE": str(state)}


def _node_merge(career_ops, state):
    """One run of the real merge-tracker over a `_stage`d layout, which it
    reads and writes through its own env overrides — the checkout is only its
    cwd. Returns the completed process. Exit 0 on both sides of #5862: a
    refusal is a skip there, never a failure."""
    r = subprocess.run(
        ["node", "merge-tracker.mjs"], cwd=str(career_ops_dir()),
        capture_output=True, text=True, timeout=120, env=_merge_env(career_ops, state))
    assert r.returncode == 0, r.stderr
    return r


def _run_merge_tracker_for_real(career_ops, state):
    """`run_merge_tracker` itself — recovery, the sanitizer, the guard, the
    return value — over a `_stage`d tree, with its one `node merge-tracker.mjs`
    call pointed at the checkout under `_node_merge`'s env overrides. The unit
    tests drive the same function against a stand-in; this is what holds its
    reading of exit codes and kept files to the real script. Returns
    (its result, what it printed)."""
    real_run = subprocess.run

    def run(cmd, *args, cwd=None, **kw):
        return real_run(cmd, *args, cwd=str(career_ops_dir()), timeout=120,
                        env=_merge_env(career_ops, state), **kw)

    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(printed):
        mp.setattr(subprocess, "run", run)
        result = run_merge_tracker(career_ops)
    return result, printed.getvalue()


def _tracker_after(career_ops):
    return (career_ops / "data" / "applications.md").read_text(encoding="utf-8")


def _merge(tmp_path, tracker_text, additions, batch_state=""):
    """Run the real merge-tracker over `additions` ({filename: row}), entirely
    inside tmp_path. The loss guard's inputs are taken where `run_merge_tracker`
    takes them: the parsed snapshot and a listing of the queue BEFORE the
    merge, since the guard's question is what became of the queue as it stood,
    and a second listing after. Returns tracker (the text after), career_ops,
    additions, state, before, queued and pending (the guard's inputs) and
    proc."""
    career_ops, additions_dir, state = _stage(tmp_path, tracker_text, additions, batch_state)
    before, queued = _pending_additions(additions_dir), queued_additions(additions_dir)
    proc = _node_merge(career_ops, state)
    return SimpleNamespace(tracker=_tracker_after(career_ops), career_ops=career_ops,
                           additions=additions_dir, state=state, before=before,
                           queued=queued, pending=queued_additions(additions_dir), proc=proc)


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """One fuzzy-matching addition, no req ids on either side.

    Module-scoped: merge-tracker shells out to sync-pdf-flags.mjs, so each run is
    two node startups plus a tracker-lock acquisition, and every assertion below
    is read-only over one immutable result."""
    return _merge(tmp_path_factory.mktemp("merge"),
                  _tracker(BASE, 200), {"123.tsv": _addition(VARIANT, 229)})


class TestFuzzyMergeKeepsTheRowsTitle:
    def test_role_title_is_the_existing_rows_not_the_additions(self, merged):
        """The reading `_report_key` exists for. If this fails, upstream has
        changed which tier may rewrite a title — re-read merge-tracker's
        `reportNumMatched || dupReason === 'url'` ternary before touching the
        guard, and do not relax the test."""
        assert f"| {BASE} |" in merged.tracker
        assert VARIANT not in merged.tracker

    def test_report_and_score_do_write_through(self, merged):
        """The other half: the addition's report number reaches the row, which
        is what makes company + report a usable identity for it."""
        assert "[229]" in merged.tracker and "4.7/5" in merged.tracker
        assert "[200]" not in merged.tracker

    def test_the_tsv_is_archived_so_nothing_retries_it(self, merged):
        """Why a wrong answer here is permanent rather than self-healing: an
        APPLIED addition is archived on every release, whatever #5862 changed
        about a refused one."""
        assert not list(merged.additions.glob("*.tsv"))
        assert (merged.additions / "merged" / "123.tsv").exists()

    def test_guard_reports_the_retitle_not_a_loss(self, merged, capsys):
        """End to end over the real merge: the #152 run reported three intact
        evaluations as lost. It must now say what actually happened."""
        _warn_on_lost_additions(merged.before, merged.queued, merged.pending, merged.career_ops)
        out = capsys.readouterr().out
        assert "WARNING" not in out
        assert VARIANT in out and BASE in out


# A second run: an addition merge-tracker REFUSES (its report number is marked
# `failed` in batch-state.tsv), with exit 0 — archived into merged/ before
# career-ops#5862, kept in the queue since. Either way it is the evaluation the
# guard exists to report, and the reason line is the operator's only clue.
REFUSED_TRACKER = (HEADER + "| 10 | 2026-08-01 | Initech | SRE | 4.0/5 | Evaluated | ❌ "
                   "| [200](../reports/200-initech-2026-08-01.md) | note |\n")
REFUSED_ADDITION = _labelled("11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.7/5\t"
                             "null\t[229](reports/229-acme-2026-09-01.md)\tAPPLY — note")
REFUSED_STATE = "id\tx\tstatus\ty\tz\treport\n7\t-\tfailed\t-\t-\t229\n"


@pytest.fixture(scope="module")
def refused(tmp_path_factory):
    return _merge(tmp_path_factory.mktemp("refused"), REFUSED_TRACKER,
                  {"7.tsv": REFUSED_ADDITION}, REFUSED_STATE)


class TestRefusalReasonsReachTheOperator:
    """`run_merge_tracker` passes stdout AND stderr to the guard. This is why."""

    def test_a_refusal_is_reported_on_stderr_not_stdout(self, refused):
        """merge-tracker refuses with `console.warn`. Passing only `r.stdout`
        (as this did until #152) left the reasons block empty on every genuine
        loss, while the one refusal it logs to stdout is the deliberately BENIGN
        unscoreable-re-eval case — so the block could only ever print a reason
        belonging to an addition that was not lost.

        If a future release moves this to stdout, this failing is the
        notification; `run_merge_tracker` already reads both, so relax the
        stdout half rather than the combined one."""
        r = refused.proc
        assert "Skipping" in r.stderr
        assert "Skipping" not in r.stdout

    def test_the_refused_file_is_kept_or_archived_as_this_release_says(self, refused):
        """#5862 moved a refusal from merged/ back into the queue, which is
        what the guard's "kept" category reads."""
        assert "Platform Engineer" not in refused.tracker
        assert _refused_where(refused.additions, "7.tsv") == REFUSED_WHERE

    def test_the_guard_prints_that_reason_end_to_end(self, refused, capsys):
        """The wiring, over the real streams: warn about the refusal, in the
        category this release's handling of it calls for, and say why — on the
        line under the file it is about."""
        r = refused.proc
        assert _warn_on_lost_additions(refused.before, refused.queued, refused.pending,
                                       refused.career_ops, f"{r.stdout}\n{r.stderr}") == 1
        out = capsys.readouterr().out
        category = "were refused by merge-tracker and KEPT" if KEEPS_REFUSALS else \
            "without reaching applications.md"
        assert "WARNING: 1 evaluation(s)" in out and category in out
        lines = out.splitlines()
        named = lines.index("[batch]   7.tsv (Acme Corp — Platform Engineer)")
        assert lines[named + 1].startswith("[batch]     merge-tracker: ")
        assert 'Skipping 7.tsv: report #229 is marked "failed"' in lines[named + 1]


@pytest.fixture(scope="module")
def refused_by_the_pipeline(tmp_path_factory):
    """The same refusal through `run_merge_tracker` against the real script."""
    career_ops, _, state = _stage(tmp_path_factory.mktemp("refused-run"), REFUSED_TRACKER,
                                  {"7.tsv": REFUSED_ADDITION}, REFUSED_STATE)
    return _run_merge_tracker_for_real(career_ops, state)


def test_a_merge_that_refused_an_evaluation_is_not_a_success(refused_by_the_pipeline):
    """merge-tracker exits 0 and the guard names the row — and the merge is
    not reported clean, because `merge_additions` exits on it and the
    evaluation is not in the tracker. On a release that keeps the file, this
    is also the end of the silence #229 found: the guard used to look only at
    files that LEFT the queue."""
    ok, printed = refused_by_the_pipeline
    assert ok is False
    assert "[batch] tracker merged" in printed
    assert "7.tsv (Acme Corp — Platform Engineer)" in printed
    assert ("KEPT" in printed) == KEEPS_REFUSALS


@pytest.fixture(scope="module")
def distinct_reqs(tmp_path_factory):
    """The same fuzzy-matching pair, carrying DIFFERENT req ids — two openings
    that happen to share a title's wording."""
    return _merge(tmp_path_factory.mktemp("apart"),
                  _tracker(BASE, 200, req="5001"),
                  {"123.tsv": _addition(VARIANT, 229, req="5002")})


@pytest.fixture(scope="module")
def same_req(tmp_path_factory):
    """The same pair, carrying the SAME id — one requisition re-posted under a
    re-worded title. The wording has to differ or the scenario tests nothing:
    `roleFuzzyMatch` short-circuits on identical strings, and bridge drops an
    identically-titled addition long before the merge."""
    return _merge(tmp_path_factory.mktemp("same"),
                  _tracker(BASE, 200, req="5001"),
                  {"123.tsv": _addition(VARIANT, 229, req="5001")})


class TestReqIdOverridesTheFuzzyTitleMatch:
    """What `extract_req_id` buys, and what it must not cost. Both directions
    are load-bearing: the id has to split what the title match wrongly folds,
    without splitting a re-post of one requisition into a second row."""

    def test_different_ids_keep_two_rows(self, distinct_reqs):
        """The #154 case. Without the ids these fold, and the second opening
        stops existing — invisible to dedup, the handoff and the UI."""
        assert f"| {VARIANT} |" in distinct_reqs.tracker and f"| {BASE} |" in distinct_reqs.tracker
        assert "➕ Add" in distinct_reqs.proc.stdout and "🔄 Update" not in distinct_reqs.proc.stdout

    def test_the_same_id_still_folds(self, same_req):
        """The safety direction, and the one that has to survive a re-wording.
        An employer's req id is stable across re-posts, which is why it — and not
        the board's per-posting `jk=` key — is what we extract: keying on the
        posting would add a row every time a listing is re-published."""
        assert same_req.tracker.count(f"| {BASE} |") == 1       # kept its title
        assert VARIANT not in same_req.tracker                   # no second row
        assert "🔄 Update" in same_req.proc.stdout and "➕ Add" not in same_req.proc.stdout

    def test_the_id_survives_into_the_new_row(self, distinct_reqs):
        """The split above only holds for the NEXT merge if the id merge-tracker
        wrote into the row is still readable there — the guard needs it on both
        sides, and the row it just added is one of them."""
        added = [l for l in distinct_reqs.tracker.splitlines() if f"| {VARIANT} |" in l]
        assert added and "req 5002" in added[0]


# A row shaped the way an agent CLI wrote one on the `--batch` path before
# career-ops#3706 — and the way this repo's own writer did until #6062 refused
# it: headerless, the score cell "4.2" rather than "4.2/5". Nothing in Python
# touched it, because career-ops' batch-runner.sh owns tracker-additions/ on
# that path. Raw on purpose: what the real script does with it is the point.
CLI_ROW = ("11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t{score}\tnull\t"
           "[229](reports/229-acme.md)\tAPPLY strong match\n")


def _queue_url(career_ops):
    """A batch-input.tsv queuing job 7 at https://x/j/7 — where the merges this
    repo runs find the URL a queued job's row did not carry."""
    (career_ops / "batch").mkdir(exist_ok=True)
    (career_ops / "batch" / "batch-input.tsv").write_text(
        "id\turl\tsource\tnotes\n7\thttps://x/j/7\tAcme Corp\tPlatform Engineer\n",
        encoding="utf-8")


@pytest.fixture(scope="module")
def unsanitized(tmp_path_factory):
    return _merge(tmp_path_factory.mktemp("raw"), HEADER,
                  {"7.tsv": CLI_ROW.format(score="4.2")})


@pytest.fixture(scope="module")
def sanitized(tmp_path_factory):
    """The same raw row through the merge-time sanitizer, as `run_merge_tracker`
    runs it: the URL the job was queued under, the JD cached under its id —
    and, since #6062, the label line the row arrived without."""
    career_ops, additions, state = _stage(tmp_path_factory.mktemp("fixed"), HEADER,
                                          {"7.tsv": CLI_ROW.format(score="4.2")})
    _queue_url(career_ops)
    jd = jd_cache_path(career_ops, "7")
    jd.parent.mkdir(parents=True)
    jd.write_text("Job ID: 88214", encoding="utf-8")
    _sanitize_pending_additions(career_ops, additions)
    written = (additions / "7.tsv").read_text(encoding="utf-8")
    r = _node_merge(career_ops, state)
    return SimpleNamespace(tracker=_tracker_after(career_ops), written=written, proc=r)


class TestUnsanitizedRowsAreRefused:
    """Why `_sanitize_pending_additions` exists, proven against the real script
    rather than asserted. If upstream ever starts accepting the raw row, the
    first test fails — and that is the notification, not a reason to relax it."""

    def test_the_raw_row_is_refused(self, unsanitized):
        """Refused on either side of career-ops#6062, for a different reason on
        each: before it, the bare score (no cell in columns 5-6 reads as one);
        since, the missing label line, before the score is ever read. Before
        #5862 it was archived too, so nothing ever retried it: the permanent
        loss the `--batch` path was exposed to."""
        assert "Platform Engineer" not in unsanitized.tracker     # never reached the tracker
        reason = ("headerless tracker-addition refused" if REFUSES_HEADERLESS
                  else "cannot tell score from status")
        assert f"Skipping 7.tsv: {reason}" in unsanitized.proc.stderr
        assert _refused_where(unsanitized.additions, "7.tsv") == REFUSED_WHERE

    def test_the_same_row_sanitized_merges(self, sanitized):
        assert "Platform Engineer" in sanitized.tracker and "4.2/5" in sanitized.tracker
        assert "➕ Add" in sanitized.proc.stdout

    def test_under_the_label_line_the_sanitizer_gave_it(self, sanitized):
        """What gets a headerless row past #6062 at all: the sanitizer labels
        a row it can vouch for the positions of — on a checkout that reads
        labels, which this one must (`test_the_checkout_reads_labelled_additions`)."""
        assert sanitized.written.splitlines()[0] == ADDITION_LABEL_LINE

    def test_and_carries_its_url_and_req_id(self, sanitized):
        """The other two things that path lost: the UI's "Open posting" target,
        and the id that keeps two levels of one title apart."""
        assert "https://x/j/7" in sanitized.tracker and "req 88214" in sanitized.tracker


def _refused_then_merged(tmp_path, addition, *, queued_url=False, tracker=HEADER):
    """The `--batch` case end to end, against the real script: the runner's own
    merge refuses `addition` as 7.tsv — archiving it before #5862, keeping it
    since — and `run_merge_tracker`, which the wrappers run after the runner,
    gets it back into shape (recovery from merged/, or the sanitizer in the
    queue) and lands it. `queued_url` adds the batch-input.tsv a queued job has,
    after the runner's merge; `tracker` is the applications.md it merges into.
    Each stage's evidence is kept rather than asserted here, so a failure names
    the stage."""
    career_ops, additions, state = _stage(tmp_path, tracker, {"7.tsv": addition})
    first = _node_merge(career_ops, state)                # what batch-runner.sh's last step does
    after_first = _tracker_after(career_ops)
    refused_where = _refused_where(additions, "7.tsv")
    if queued_url:
        _queue_url(career_ops)
    ok, printed = _run_merge_tracker_for_real(career_ops, state)
    applied, kept = additions / "merged" / "7.tsv", additions / "7.tsv"
    return SimpleNamespace(first=first, after_first=after_first, refused_where=refused_where,
                           ok=ok, printed=printed,
                           applied=applied.read_text(encoding="utf-8") if applied.exists() else None,
                           kept=kept.read_text(encoding="utf-8") if kept.exists() else None,
                           tracker=_tracker_after(career_ops))


@pytest.fixture(scope="module")
def recovered(tmp_path_factory):
    """A raw headerless row with a bare score, its URL in batch-input.tsv."""
    return _refused_then_merged(tmp_path_factory.mktemp("recover"),
                                CLI_ROW.format(score="4.2"), queued_url=True)


class TestRecoveryAfterTheRunnersOwnMerge:
    def test_the_evaluation_lands_on_the_next_merge(self, recovered):
        assert "Platform Engineer" in recovered.tracker and "4.2/5" in recovered.tracker
        assert recovered.ok is True, recovered.printed         # and it is a clean merge

    def test_with_the_url_it_was_missing(self, recovered):
        assert "https://x/j/7" in recovered.tracker

    def test_recovery_acts_only_where_the_refusal_was_archived(self, recovered):
        """On a release that keeps a refusal, merged/ holds only what was
        applied, and recovery must find nothing — the sanitizer repairs the
        kept file in place. On one that archives, recovery is what brings it
        back."""
        assert ("recovered 1 evaluation" in recovered.printed) == (not KEEPS_REFUSALS)


# ── Headed additions (career-ops#3706) ───────────────────────────────────────
#
# Since #3706 (merged 2026-09-04) career-ops' batch worker writes a row of column
# LABELS above its one data row, and merge-tracker resolves that file by name.
# Our readers took the two-line file for one malformed row: the loss guard read
# the label line as a job called "role" at a company called "company", found no
# such row in the tracker and warned about an evaluation that had landed intact —
# the #152 false alarm by a new route — and recovery could not put a refused one
# back. So the two scenarios above are re-proven here on the form the worker now
# writes, against the script that reads it.
#
# The label line is `ADDITION_LABEL_LINE` (pipeline/_batch_common.py, which
# writes it above every row since #6062), verbatim from Step 5 of career-ops'
# batch/batch-prompt.md, which tells the worker to write it "exactly as shown".
# Every row below is built from it by label, so a relabelling upstream is one
# edit there and the rows follow.
#
# No alias-table pin is needed, unlike the unit tests: merge-tracker and our
# readers both resolve these labels through the same checkout's
# tracker-aliases.json — the pairing a real merge has — and every label in the
# worker's line is in `_FALLBACK_ALIASES` too, so the Python reading is the same
# either way.
def _headed_cells(score):
    """`CLI_ROW`'s cells by label, plus the URL a headed row carries in a column
    of its own — the same job, so each headed scenario below differs from its
    headerless twin only in the form."""
    values = CLI_ROW.format(score=score).rstrip("\n").split("\t") + ["https://x/j/7"]
    assert len(values) == len(ADDITION_LABELS), "a label with no value, or a value with no label"
    return dict(zip(ADDITION_LABELS, values))


def _headed_addition(score):
    """A worker-written headed addition of `_headed_cells`."""
    return f"{ADDITION_LABEL_LINE}\n" + "\t".join(_headed_cells(score).values()) + "\n"


@pytest.fixture(scope="module")
def labelled_merged(tmp_path_factory):
    """A labelled addition — what `write_job_result` writes for a checkout that
    reads labels — through the real merge."""
    return _merge(tmp_path_factory.mktemp("labelled"), HEADER, {"7.tsv": _labelled(
        "11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.7/5\tnull\t"
        "[229](reports/229-acme-2026-09-01.md)\thttps://x/j/7 — APPLY")})


def test_the_checkout_reads_labelled_additions(labelled_merged):
    """What the gate requires of a checkout, and what keeps the pipeline's
    probe honest. `write_job_result` labels every row for a checkout whose
    merge-tracker reads labels (`reads_labelled_additions`), because #6062
    refuses one without. A merge-tracker from before #3706 (2026-09-04) takes
    the label line for the data row and refuses the whole file as a
    score/status swap (`"status" | "score"`), so the probe keeps that checkout
    on headerless rows — but the pin and the headed tests below need #3706, so
    such a checkout fails here, on what its script did with a labelled row.

    The second half is the probe itself: a text match on either of two names.
    If upstream renames both, this checkout merges a labelled row while the
    probe says it cannot, and the writer would send every current checkout the
    headerless rows #6062 refuses — so that fails here too, before the pin
    moves."""
    m = labelled_merged
    assert "| Platform Engineer |" in m.tracker, (
        "this career-ops refused a labelled tracker addition — it predates "
        f"career-ops#3706: git pull it\n{m.proc.stderr}")
    assert reads_labelled_additions(career_ops_dir()), (
        "this merge-tracker reads labelled additions, but `reads_labelled_additions` "
        "says it does not: upstream has renamed both names it looks for "
        "(`_HEADED_PARSER_MARKER` and `_REFUSES_HEADERLESS_MARKER` in "
        "pipeline/_batch_common.py)")


# A labelled row whose PDF the model left blank, as the labeller writes it, and
# the same row labelled with the blank left in — one merge, two files.
BLANK_PDF_ROW = ("11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.7/5\t\t"
                 "[229](reports/229-acme-2026-09-01.md)\thttps://x/j/7 — APPLY")


@pytest.fixture(scope="module")
def blank_pdf(tmp_path_factory):
    return _merge(tmp_path_factory.mktemp("blank-pdf"), HEADER, {
        "7.tsv": _labelled(BLANK_PDF_ROW),
        "8.tsv": f"{ADDITION_LABEL_LINE}\n"
                 + BLANK_PDF_ROW.replace("Platform Engineer", "SRE").replace("[229]", "[230]")
                 + "\t\n"})


class TestLabelsDoNotTurnABlankCellIntoARefusal:
    """The headerless parser never checked a required cell for a value; the
    headed one refuses a blank, so a model that left PDF empty used to merge
    and, labelled as written, is refused — archived at a checkout before
    career-ops#5862, where recovery will not re-queue a readable score."""

    def test_a_blank_pdf_labelled_as_written_is_refused(self, blank_pdf):
        """The premise, off the real script. If upstream stops refusing a blank
        required cell, this failing is the notification."""
        assert "| SRE |" not in blank_pdf.tracker
        assert "Skipping 8.tsv: required cell(s) present but empty: pdf" in blank_pdf.proc.stderr

    def test_as_the_labeller_writes_it_the_row_merges(self, blank_pdf):
        row = tracker_row(blank_pdf.tracker, "11")
        assert (row["role"], row["score"]) == ("Platform Engineer", "4.7/5")
        assert "Skipping 7.tsv" not in blank_pdf.proc.stderr

    def test_the_required_cells_are_the_scripts(self):
        """The cells `labelled_addition` will not label blank are a copy of
        tracker-parse.mjs's `TSV_REQUIRED_FIELDS`, held to it here."""
        source = (career_ops_dir() / "tracker-parse.mjs").read_text(encoding="utf-8")
        listed = re.search(r"TSV_REQUIRED_FIELDS\s*=\s*\[([^\]]*)\]", source)
        assert listed, "tracker-parse.mjs no longer declares TSV_REQUIRED_FIELDS"
        assert set(re.findall(r"'([^']+)'", listed.group(1))) == {
            ADDITION_COLUMNS[i] for i in _REQUIRED_IDXS}


@pytest.fixture(scope="module")
def headed_merged(tmp_path_factory):
    """A headed addition with a readable score."""
    return _merge(tmp_path_factory.mktemp("headed"), HEADER,
                  {"7.tsv": _headed_addition("4.7/5")})


class TestHeadedAdditionLands:
    def test_it_merges_and_is_archived(self, headed_merged):
        h = headed_merged
        row = tracker_row(h.tracker, "11")
        assert (row["company"], row["role"], row["score"]) == (
            "Acme Corp", "Platform Engineer", "4.7/5")
        assert "➕ Add" in h.proc.stdout
        assert not list(h.additions.glob("*.tsv"))
        assert (h.additions / "merged" / "7.tsv").exists()

    def test_the_guard_does_not_report_it_lost(self, headed_merged, capsys):
        """The #152 false alarm this fix removes, over the real merge. The first
        assertion is what keeps the second from being vacuous: a snapshot that
        dropped the headed file warns about nothing too, and one that read its
        label line as a job warns about a role called "role"."""
        h = headed_merged
        assert [(a["company"], a["role"], a["report"]) for a in h.before] == [
            ("Acme Corp", "Platform Engineer", "229")]
        _warn_on_lost_additions(h.before, h.queued, h.pending, h.career_ops,
                                f"{h.proc.stdout}\n{h.proc.stderr}")
        assert "WARNING" not in capsys.readouterr().out


@pytest.fixture(scope="module")
def headed_recovered(tmp_path_factory):
    """`recovered` on the headed form, with a bare score. No batch-input.tsv:
    the headed row brings its URL in its own column."""
    return _refused_then_merged(tmp_path_factory.mktemp("headed-recover"),
                                _headed_addition("4.2"))


class TestHeadedRecoveryAfterTheRunnersOwnMerge:
    def test_a_bare_score_is_refused_by_label(self, headed_recovered):
        """The refusal has to name the column LABELLED score. A pre-#3706 script
        refuses this file too, on the label line, so a bare "Skipping" would pass
        even if the header were never read as one."""
        h = headed_recovered
        assert "Platform Engineer" not in h.after_first
        assert 'the column labelled "score" reads "4.2"' in h.first.stderr
        assert h.refused_where == REFUSED_WHERE

    def test_the_next_merge_repairs_it_under_its_label_line_and_lands_it(
            self, headed_recovered):
        """Repaired as a headed file, not flattened into a headerless row —
        merge-tracker takes a labelled file by name, and that is the one form
        in which a never-scored `—`/`—` row has an order at all (#3517). Read
        off the file merge-tracker archived, which is the one it applied."""
        h = headed_recovered
        assert h.applied is not None, h.printed
        label, data = h.applied.rstrip("\n").split("\n")     # one data row
        assert label == ADDITION_LABEL_LINE
        assert dict(zip(ADDITION_LABELS, data.split("\t")))["score"] == "4.2/5"
        row = tracker_row(h.tracker, "11")
        assert (row["role"], row["score"]) == ("Platform Engineer", "4.2/5")
        assert h.ok is True, h.printed
        # Recovery's part where the refusal was archived; the sanitizer's alone
        # where it was kept.
        assert ("recovered 1 evaluation" in h.printed) == (not KEEPS_REFUSALS)


# #191: a headed addition labelled in another order — the `via` column an
# agent CLI adds after `company`, where the tracker's own Via layout has it —
# with a bare score. merge-tracker reads it by name and refuses the score, and
# the chain addresses cells by index, so until #191 it was left as written and
# nothing here could bring it back. Merged into both tracker layouts: the Via
# one an agent adds the column for, and the canonical one, where merge-tracker
# drops a `via` value (with a `--migrate-via` hint) whatever order it came in.
VIA_TRACKER = ("# Applications Tracker\n\n"
               "| # | Date | Company | Via | Role | Score | Status | PDF | Report | Notes |\n"
               "|---|------|---------|-----|------|-------|--------|-----|--------|-------|\n")
VIA_LABELS = ("num", "date", "company", "via", "role", "status", "score", "pdf", "report",
              "notes", "url")


def _via_addition(score):
    """`_headed_cells` with an agency, labelled in `VIA_LABELS` order."""
    cells = {**_headed_cells(score), "via": "Hays"}
    return "\t".join(VIA_LABELS) + "\n" + "\t".join(cells[label] for label in VIA_LABELS) + "\n"


@pytest.fixture(scope="module", params=[VIA_TRACKER, HEADER], ids=["via-layout", "canonical"])
def via_reordered(request, tmp_path_factory):
    h = _refused_then_merged(tmp_path_factory.mktemp("via-reordered"),
                             _via_addition("4.2"), tracker=request.param)
    h.via_column = request.param is VIA_TRACKER
    return h


class TestAHeadedRowInAnotherOrderIsRepaired:
    def test_as_written_its_bare_score_is_refused_by_label(self, via_reordered):
        """The premise, off the real script: the order is legal — the refusal
        names the column LABELLED score, wherever it sits — and the score is
        not. If upstream starts accepting a bare score, this failing is the
        notification."""
        h = via_reordered
        assert "Platform Engineer" not in h.after_first
        assert 'Skipping 7.tsv: the column labelled "score" reads "4.2"' in h.first.stderr
        assert h.refused_where == REFUSED_WHERE

    def test_put_in_order_by_its_labels_it_merges(self, via_reordered):
        """Read off the file merge-tracker applied: still labelled — the nine
        in the batch prompt's order, then `url`, then `via` — with the score
        repaired. The agency lands in the Via column where the
        tracker has one, merge-tracker reading the `via` label by name, and is
        dropped where it has none, exactly as it would be in order."""
        h = via_reordered
        assert h.applied is not None, h.printed
        label, data = h.applied.rstrip("\n").split("\n")     # one data row
        assert label.split("\t") == [*ADDITION_COLUMNS, "url", "via"]
        assert dict(zip(label.split("\t"), data.split("\t")))["score"] == "4.2/5"
        rows = {cells[0]: dict(zip(columns, cells)) for columns, cells in data_rows(h.tracker)}
        row = rows["11"]
        assert (row["company"], row["role"], row["score"]) == (
            "Acme Corp", "Platform Engineer", "4.2/5")
        assert "https://x/j/7" in row["notes"]
        if h.via_column:
            assert row["via"] == "Hays"
        else:
            assert "via" not in row and "Hays" not in h.tracker
        assert h.ok is True, h.printed


# What putting a file in order must NOT repair: headed rows the real script
# refuses as written, each with a bare score, where the chain's score repair is
# the one thing that would let it through — with cells under labels they were
# not written under. merge-tracker lets a row stop before its optional cells
# and reads every present cell by its label, so a row that left out an
# INTERIOR cell is read one label off from there on, and it says so only when
# the shift reaches the score; and it refuses a bare URL under a label that is
# not `url` while the `url` cell is absent, which padding would make present.
# A refusal the loss guard names is recoverable; a merged garbage row is not.
# {id: (the file, the reason merge-tracker gives for refusing it as written)}
_LEFT_OUT = "[229](reports/229-acme.md)\tAPPLY\n"
NOT_REPAIRED = {
    # The PDF value left out, labels in another order: read by name, the
    # report link is the PDF and the note is the Report.
    "interior-cell-another-order": (
        "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
        "Acme Corp\tPlatform Engineer\t11\t2026-09-01\t4.2\tEvaluated\t" + _LEFT_OUT,
        'the column labelled "score" reads "4.2"'),
    # The same in the batch prompt's order, where the chain used to pad the
    # 8-cell row to nine and take it as it stood.
    "interior-cell-prompt-order": (
        f"{ADDITION_LABEL_LINE}\n"
        "11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.2\t" + _LEFT_OUT,
        'the column labelled "score" reads "4.2"'),
    # A `via` label after Report, its empty cell left out: nine cells, the
    # Report link in place, no required label after it and no URL misplaced
    # — read by name, the note is the agency, and the shift never reaches the
    # score. A short row may have left out only the prompt's own tail there.
    "optional-cell-after-report": (
        "num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\tvia\tnotes\turl\n"
        "11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.2\tnull\t" + _LEFT_OUT,
        'the column labelled "score" reads "4.2"'),
    # A tab inside Notes: one cell more than the labels. By name the second
    # half of the note is the URL and the URL is past every label; the score
    # repair would merge that shift.
    "a-cell-past-the-labels": (
        f"{ADDITION_LABEL_LINE}\n"
        "11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.2\tnull\t"
        "[229](reports/229-acme.md)\tnote a\tnote b\thttps://www.linkedin.com/jobs/view/1\n",
        'the column labelled "score" reads "4.2"'),
    # The posting URL under `notes` and no `url` cell, after a `location`.
    "bare-url-no-url-cell": (
        "num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\tlocation\tnotes\turl\n"
        "11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.2\tnull\t"
        "[229](reports/229-acme.md)\tDallas\thttps://x/j/7\n",
        'a URL sits under "notes" while the "url" cell is absent'),
}


@pytest.fixture(scope="module", params=list(NOT_REPAIRED))
def not_repaired(request, tmp_path_factory):
    text, reason = NOT_REPAIRED[request.param]
    h = _refused_then_merged(tmp_path_factory.mktemp(request.param), text)
    h.text, h.reason = text, reason
    return h


class TestWhatItsLabelsCannotPlaceIsNotRepaired:
    def test_as_written_it_is_refused(self, not_repaired):
        """The premise, off the real script — and the reason is the one that
        makes the shape what it is. If upstream starts merging one of these
        as written, this failing is the notification."""
        h = not_repaired
        assert "Platform Engineer" not in h.after_first
        assert f"Skipping 7.tsv: {h.reason}" in h.first.stderr
        assert h.refused_where == REFUSED_WHERE

    def test_it_is_still_refused_left_exactly_as_written(self, not_repaired):
        """Never merged on the sanitizer's guess. Where the release keeps a
        refusal, the next merge leaves the file as the writer wrote it, is not
        a success, and names it with merge-tracker's own reason; where it
        archives one, recovery leaves it in the archive untouched."""
        h = not_repaired
        assert "Platform Engineer" not in h.tracker, h.printed
        assert "sanitized" not in h.printed and "recovered" not in h.printed
        if KEEPS_REFUSALS:
            assert h.kept == h.text
            assert h.ok is False
            assert "refused by merge-tracker and KEPT" in h.printed
            assert h.reason in h.printed
        else:
            assert h.kept is None and h.applied == h.text


@pytest.fixture(scope="module")
def headed_short(tmp_path_factory):
    """A headed row that stops before its `url` cell with its Notes empty — both
    of which merge-tracker allows — and a queued URL for the sanitizer to write
    into Notes. A bare URL in Notes with no `url` cell is the shape merge-tracker
    refuses as every optional cell shifted one left."""
    tmp_path = tmp_path_factory.mktemp("headed-short")
    cells = _headed_cells("4.7/5")
    row = "\t".join(cells[label] for label in ADDITION_LABELS[:-2]) + "\t"   # empty notes, no url
    career_ops, additions, state = _stage(tmp_path, HEADER,
                                          {"7.tsv": f"{ADDITION_LABEL_LINE}\n{row}\n"})
    _queue_url(career_ops)
    _sanitize_pending_additions(career_ops, additions)
    return SimpleNamespace(proc=_node_merge(career_ops, state),
                           tracker=_tracker_after(career_ops))


class TestHeadedRowShortOfItsUrlCell:
    def test_the_repaired_row_still_merges_with_its_url(self, headed_short):
        """Unpadded, the sanitizer turned a row merge-tracker accepts into one it
        refuses — `a URL sits under "notes" while the "url" cell is absent` —
        and, its score being readable, recovery never retried it."""
        h = headed_short
        assert "Skipping" not in h.proc.stderr, h.proc.stderr
        row = tracker_row(h.tracker, "11")
        assert row["role"] == "Platform Engineer" and "https://x/j/7" in row["notes"]


def test_the_label_line_is_the_one_the_prompt_prescribes():
    """`ADDITION_LABEL_LINE` is a copy — CI has no checkout to read it from — so
    here, where the checkout is, it is held to the line career-ops' batch prompt
    tells the worker to write "exactly as shown". A relabelling upstream fails
    this, rather than leaving `write_job_result` labelling every row with a line
    no worker writes."""
    prompt = (career_ops_dir() / "batch" / "batch-prompt.md").read_text(encoding="utf-8")
    shown = [line.strip() for line in prompt.splitlines()
             if line.strip().startswith("num\\tdate\\t")]
    assert shown, "batch-prompt.md no longer shows a `num\\tdate\\t…` label line"
    assert shown[0].replace("\\t", "\t") == ADDITION_LABEL_LINE


class TestRecheckDiscardIsReopenedOnRepost:
    """The premise of #163 against the real script, then our half of the fix.

    merge-tracker's update tier writes a re-eval's score, report, date and
    notes through and KEEPS the row's status — so a re-post of a role the
    liveness re-check had Discarded was folded onto the Discarded row and never
    surfaced again. `_reopen_reposted` resets such a row (its report number
    changed in the merge, and the snapshot taken before the merge saw the
    re-check's mark) to Evaluated, and `extract_url` then reads the new
    posting's URL — the one the next re-check must verify. A higher score is
    used so the case holds on a checkout on either side of upstream's #2411
    (the older script skips a lower one)."""

    @pytest.fixture
    def reopened(self, tmp_path):
        text = (_tracker(BASE, 200)
                .replace("| Evaluated |", "| Discarded |")
                .replace("— APPLY |", f"— APPLY — {liveness_closed_mark('2026-08-20', 'HTTP 404')} |"))
        pre = tmp_path / "pre.md"
        pre.write_text(text, encoding="utf-8")
        closed = _liveness_closed_rows(pre)
        m = _merge(tmp_path, text, {"123.tsv": _addition(VARIANT, 229)})
        _reopen_reposted(closed, m.career_ops)
        return closed, m.tracker, _tracker_after(m.career_ops)

    @staticmethod
    def _row(text):
        return tracker_row(text, "10")

    def test_upstream_writes_through_and_keeps_the_status(self, reopened):
        closed, merged_text, _ = reopened
        assert set(closed) == {"10"}
        row = self._row(merged_text)
        assert row["report_num"] == "229" and row["status_canonical"] == "Discarded"

    def test_reopen_resets_to_evaluated_and_points_at_the_new_posting(self, reopened):
        from pipeline.app import data
        _, _, after = reopened
        row = self._row(after)
        assert row["status_canonical"] == "Evaluated"
        assert data.extract_url(row["notes"]) == "https://indeed.com/viewjob?jk=new"
        assert not closed_by_recheck(row["notes"])          # a later Discard is a person's



# ── #183: the Skip decision mark through the update tier ─────────────────────

def _reeval(score, status, notes, decision="", url="https://indeed.com/viewjob?jk=new", **kw):
    """A re-evaluation of `_tracker`'s row (the same company and title, so the
    exact tier pairs them) as `write_job_result` writes it: through the
    sanitizer, which puts the posting URL, any `Decision: Skip` mark and any
    other mark `kw` names in Notes, and under the label line."""
    row = ("11\t2026-09-01\tUT Southwestern Medical Center\t" + BASE +
           f"\t{status}\t{score}\tnull\t[229](reports/229-ut.md)\t{notes}")
    return _labelled(sanitize_addition(row, url, decision=decision, **kw))


@pytest.fixture(scope="module")
def dropped_to_skip(tmp_path_factory):
    """A queued Apply re-evaluated as a Skip. The score goes DOWN, which the
    pinned checkout writes through (career-ops#2411 — the older script skipped a
    lower re-eval, and this case would then be the evaluation lost outright)."""
    return _merge(tmp_path_factory.mktemp("skip"), _tracker(BASE, 200),
                  {"123.tsv": _reeval("2.4/5", "SKIP", "SKIP weak claims match",
                                      decision=DECISION_SKIP_MARK)})


@pytest.fixture(scope="module")
def skip_then_apply(tmp_path_factory):
    """A row that already carries the mark, re-evaluated as an Apply (higher, so
    it holds on either side of #2411). An Apply writes no mark of its own."""
    text = (_tracker(BASE, 200).replace("| 4.0/5 |", "| 2.4/5 |")
            .replace("— APPLY |", f"— {DECISION_SKIP_MARK} — SKIP weak |"))
    return _merge(tmp_path_factory.mktemp("apply"), text,
                  {"123.tsv": _reeval("4.3/5", "Evaluated", "APPLY strong match")})


# The cell `_reeval` writes for a Skip, which `skip_reevaluated_identically`
# also puts on the tracker row, so the re-evaluation repeats it exactly.
SKIP_CELL = f"https://indeed.com/viewjob?jk=new — {DECISION_SKIP_MARK} — SKIP weak claims match"


@pytest.fixture(scope="module")
def skip_reevaluated_identically(tmp_path_factory):
    """A Skip re-evaluated to the cell it already has. mergeNotes finds the
    incoming notes among the row's clauses and writes the `Re-eval` marker
    BARE — no `: <notes>` — which is the shape `skip_decided` must not read as
    a newer verdict."""
    text = (_tracker(BASE, 200).replace("| 4.0/5 |", "| 2.4/5 |")
            .replace(f"| {_notes(None, 'old')} |", f"| {SKIP_CELL} |"))
    addition = _reeval("2.4/5", "SKIP", "SKIP weak claims match", decision=DECISION_SKIP_MARK)
    assert f"\t{SKIP_CELL}\t" in addition           # the premise: an exact repeat
    return _merge(tmp_path_factory.mktemp("same"), text, {"123.tsv": addition})


@pytest.fixture(scope="module")
def skip_opens_the_re_eval(tmp_path_factory):
    """A Skip whose addition has no req id, URL or Work location mark, so the
    Decision mark opens its cell — which mergeNotes writes after the marker's
    `: `, where a ` — `-only clause anchor cannot see it. A downgrade, so the
    marker also carries the superseded report between the two (#2411)."""
    return _merge(tmp_path_factory.mktemp("opens"), _tracker(BASE, 200),
                  {"123.tsv": _reeval("2.4/5", "SKIP", "SKIP now", url="",
                                      decision=DECISION_SKIP_MARK)})


@pytest.fixture(scope="module")
def location_opens_the_re_eval(tmp_path_factory):
    """The Work location mark in the same position, against a row whose cell
    already carries one — so mergeNotes joins with `; `, its rule for a cell
    that ends in a location label."""
    text = _tracker(BASE, 200).replace(
        "— APPLY |", "— Work location: On-site Chicago, IL — APPLY |")
    return _merge(tmp_path_factory.mktemp("where"), text,
                  {"123.tsv": _reeval("4.3/5", "Evaluated", "APPLY now", url="",
                                      work_location="Work location: Remote")})


@pytest.fixture(scope="module")
def location_ends_the_cell(tmp_path_factory):
    """A cell that ENDS with its Work location mark, re-evaluated with no mark
    of its own: the label trips mergeNotes' location rule, so the re-eval is
    joined with `; `, and the old mark is still the newest one (#180)."""
    text = _tracker(BASE, 200).replace("— APPLY |", "— Work location: On-site Chicago, IL |")
    return _merge(tmp_path_factory.mktemp("ends"), text,
                  {"123.tsv": _reeval("4.3/5", "Evaluated", "APPLY now", url="")})


class TestDecisionSkipThroughTheUpdateTier:
    """Why the Skip has to be on the row, and that it survives being put there.

    merge-tracker's update tier keeps the EXISTING row's status, so a re-eval
    that drops a queued role to Skip leaves it Evaluated whatever Status the
    addition carried — the Status cell cannot say Skip on that path, only the
    Notes can. And its `mergeNotes` keeps the old cell first and appends the new
    one after a `Re-eval` marker, which is the shape `skip_decided`'s
    newest-mark rule is written against."""

    def test_the_update_keeps_the_rows_status(self, dropped_to_skip):
        row = tracker_row(dropped_to_skip.tracker, "10")
        assert row["report_num"] == "229" and row["status_canonical"] == "Evaluated"

    def test_the_mark_lands_after_the_re_eval_marker_and_reads_as_a_skip(self, dropped_to_skip):
        row = tracker_row(dropped_to_skip.tracker, "10")
        assert row["notes"].index("Re-eval") < row["notes"].index(DECISION_SKIP_MARK)
        assert row["skip_decided"] is True

    def test_an_existing_mark_is_kept_and_a_newer_apply_retires_it(self, skip_then_apply):
        row = tracker_row(skip_then_apply.tracker, "10")
        assert row["report_num"] == "229"
        assert row["notes"].index(DECISION_SKIP_MARK) < row["notes"].index("Re-eval")
        assert row["skip_decided"] is False

    def test_an_identical_skip_re_evaluation_stays_a_skip(self, skip_reevaluated_identically):
        """The bare marker is real: the pinned script writes it for an exact
        repeat, and the row must still read as the Skip it is."""
        notes = tracker_row(skip_reevaluated_identically.tracker, "10")["notes"]
        assert notes.startswith(SKIP_CELL)
        assert re.fullmatch(r"Re-eval 2026-09-01 \([^)]*\)", notes[notes.index("Re-eval"):])
        assert tracker_row(skip_reevaluated_identically.tracker, "10")["skip_decided"] is True

    def test_a_mark_that_opens_the_re_evaluations_cell_is_read(self, skip_opens_the_re_eval):
        row = tracker_row(skip_opens_the_re_eval.tracker, "10")
        assert re.search(r"Re-eval 2026-09-01 \([^)]*\) — Superseded report \[200\] "
                         r"\(was [^)]*\): " + DECISION_SKIP_MARK, row["notes"])
        assert row["skip_decided"] is True

    def test_a_location_mark_that_ends_the_cell_survives_a_re_eval(self, location_ends_the_cell):
        """Read past the `; ` join, the mark's value swallowed the re-eval clause
        and lost its state, so a re-evaluated out-of-area role stopped being
        demoted."""
        notes = tracker_row(location_ends_the_cell.tracker, "10")["notes"]
        assert re.search(r"Work location: On-site Chicago, IL; Re-eval 2026-09-01 ", notes)
        where = work_location_from_notes(notes)
        assert (where.mode, where.metro, where.state) == ("onsite", "Chicago, IL", "IL")

    def test_so_is_a_work_location_mark_there(self, location_opens_the_re_eval):
        notes = tracker_row(location_opens_the_re_eval.tracker, "10")["notes"]
        assert re.search(r"; Re-eval 2026-09-01 \([^)]*\): Work location: Remote", notes)
        assert work_location_from_notes(notes).mode == "remote"

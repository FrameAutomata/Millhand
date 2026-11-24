"""Tests for pipeline/_batch_common.py"""

import csv
import re
import json
import os
import threading
import time
from pathlib import Path

import pytest

from pipeline._batch_common import (
    ADDITION_COLUMNS,
    IncompleteResponse,
    ADDITION_LABEL_LINE,
    ADDITION_LABELS,
    _HEADED_PARSER_MARKER,
    _dead_link_repair,
    labelled_addition,
    _recover_refused_additions,
    _sanitize_pending_additions,
    _set_report_link,
    find_report_file,
    headed_addition_row,
    read_addition,
    read_report,
    resolve_report,
    sanitize_addition,
    _inject_req_id_into_notes,
    _normalize_score_cell,
    _pending_additions,
    _warn_on_lost_additions,
    assign_job_numbers,
    build_system_prompt,
    build_user_message,
    eval_system_prompt,
    extract_req_id,
    extract_tag,
    load_pending,
    load_state,
    max_report_num,
    max_tracker_num,
    parse_json_loose,
    queued_additions,
    read_text,
    reads_labelled_additions,
    run_merge_tracker,
    tail_text,
    write_job_result,
    write_process_lock,
    _report_location_mark,
    _pending_report_marks,
    _row_report,
)
from pipeline._batch_common import (   # the #163 marks, a separate block on purpose
    _liveness_closed_rows, _reopen_reposted, by_hand_mark, closed_by_recheck,
    liveness_closed_mark, reopened_mark,
)
from pipeline._batch_common import (   # the #183 mark
    DECISION_SKIP_MARK, _report_marks, is_skip_decision, skip_decided,
)
from pipeline import _batch_common
from tests.conftest import tracker_row


# A stand-in merge-tracker.mjs for a CURRENT checkout — one that reads labelled
# additions (career-ops#3706), as the pin and upstream `main` do, so the writer
# and the sanitizer label for it (`reads_labelled_additions`). Nothing runs it:
# every test that reaches `node` patches `subprocess.run`.
_CURRENT_MERGE_TRACKER = f"// stands in for a current checkout's {_HEADED_PARSER_MARKER}"


def _data_cells(path):
    """The cells of an addition file's DATA row — its last line, which is the
    whole file for a headerless row and the line under the labels for one
    `write_job_result` wrote (career-ops#6062 refuses a headerless addition).
    Split on `\\r?\\n` and stripped of line endings only: a trailing tab is
    the empty last cell."""
    return re.split(r"\r?\n", read_addition(path).rstrip("\r\n"))[-1].split("\t")


def _label_line(path):
    """An addition file's first line, the label line when it has one."""
    return re.split(r"\r?\n", read_addition(path))[0]


# What the evaluation prompt tells the model to write in the PDF column, read
# off the prompt itself, so a test of what the labeller fills in is held to the
# prompt rather than to a second copy of the value.
_PROMPT_PDF_CELL = re.search(r"^- Col 7 PDF: (\S+)", build_system_prompt("", ""), re.M).group(1)


_APPS_HEADER = ("# Applications Tracker\n\n"
                "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
                "|---|------|---------|------|-------|--------|-----|--------|-------|\n")


def _career_ops(tmp_path, rows=(), additions=()):
    """A career-ops with a stand-in merge-tracker.mjs, a tracker holding `rows`
    and one labelled addition per name in `additions` — the form
    `write_job_result` writes. Returns (career_ops, its queue)."""
    co = tmp_path / "career-ops"
    (co / "data").mkdir(parents=True)
    (co / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
    (co / "data" / "applications.md").write_text(_APPS_HEADER + "".join(rows), encoding="utf-8")
    queue = co / "batch" / "tracker-additions"
    queue.mkdir(parents=True)
    for i, name in enumerate(additions):
        (queue / name).write_text(labelled_addition("\t".join([
            str(11 + i), "2026-09-01", f"Co{i}", f"Role{i}", "Evaluated", "4.2/5", "null",
            f"[{229 + i}](reports/{229 + i}-co{i}.md)", "APPLY"])), encoding="utf-8")
    return co, queue


def _fake_merge(mocker, co, *, apply=(), archive=(), rewrite=None, returncode=0, stderr=""):
    """Stand in for merge-tracker over `co`: append a tracker row for each
    addition in `apply`, rewrite the tracker as its update tier would
    (`rewrite`), archive `apply` and `archive` as it archives each addition it
    applies, leave the rest queued — the shape of career-ops#5862, which keeps
    a refusal — and exit `returncode` (1 is its `failedAdditions`, after the
    others were written and archived)."""
    apps = co / "data" / "applications.md"
    queue = co / "batch" / "tracker-additions"

    def run(*a, **kw):
        text = apps.read_text(encoding="utf-8")
        text = rewrite(text) if rewrite else text
        for name in apply:
            row = headed_addition_row(read_addition(queue / name))
            text += (f"| {row['num']} | {row['date']} | {row['company']} | {row['role']} "
                     f"| {row['score']} | Evaluated | ❌ | {row['report']} | {row['notes']} |\n")
        apps.write_text(text, encoding="utf-8")
        (queue / "merged").mkdir(exist_ok=True)
        for name in (*apply, *archive):
            (queue / name).rename(queue / "merged" / name)
        return mocker.MagicMock(returncode=returncode, stdout="", stderr=stderr)
    mocker.patch("pipeline._batch_common.subprocess.run", side_effect=run)


class TestTailText:
    def test_returns_last_n_lines(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("\n".join(f"line {i}" for i in range(10)), encoding="utf-8")
        assert tail_text(f, 3) == "line 7\nline 8\nline 9"

    def test_missing_file_returns_empty(self, tmp_path):
        assert tail_text(tmp_path / "nope.txt", 5) == ""

    def test_zero_lines_returns_empty_not_whole_window(self, tmp_path):
        # [-0:] would slice the WHOLE file; the guard must return "" for 0.
        f = tmp_path / "log.txt"
        f.write_text("a\nb\nc", encoding="utf-8")
        assert tail_text(f, 0) == ""

    def test_reads_only_tail_bytes(self, tmp_path):
        # A line beyond the byte window is excluded even if the line count allows
        # it — the read is bounded to the last max_bytes of the file.
        f = tmp_path / "log.txt"
        f.write_text("OLD-LINE\n" + "x" * 100 + "\nNEW-LINE", encoding="utf-8")
        out = tail_text(f, 100, max_bytes=16)
        assert "OLD-LINE" not in out
        assert "NEW-LINE" in out


class TestReadText:
    def test_reads_file_content(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_text(f) == "hello world"

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("  trimmed  \n", encoding="utf-8")
        assert read_text(f) == "trimmed"

    def test_returns_default_on_missing(self, tmp_path):
        assert read_text(tmp_path / "missing.txt") == ""
        assert read_text(tmp_path / "missing.txt", default="fallback") == "fallback"


class TestLoadState:
    def test_returns_empty_jobs_on_missing_file(self, tmp_path):
        result = load_state(tmp_path / "state.json")
        assert result == {"jobs": {}}

    def test_loads_valid_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"batch_id": "abc", "jobs": {"1": {"status": "completed"}}}), encoding="utf-8")
        result = load_state(state_file)
        assert result["batch_id"] == "abc"
        assert result["jobs"]["1"]["status"] == "completed"

    def test_returns_empty_jobs_on_invalid_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json {{{", encoding="utf-8")
        result = load_state(state_file)
        assert result == {"jobs": {}}


class TestMaxReportNum:
    def test_returns_zero_for_missing_dir(self, tmp_path):
        state = {"jobs": {}}
        assert max_report_num(tmp_path / "reports", state, tmp_path / "none.md") == 0

    def test_reads_highest_from_filenames(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "001-acme-2026-01-01.md").write_text("x")
        (reports / "005-globex-2026-01-02.md").write_text("x")
        (reports / "003-initech-2026-01-03.md").write_text("x")
        assert max_report_num(reports, {"jobs": {}}, tmp_path / "none.md") == 5

    def test_reads_from_state_when_higher(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "003-acme-2026-01-01.md").write_text("x")
        state = {"jobs": {"j1": {"report_num": "010"}}}
        assert max_report_num(reports, state, tmp_path / "none.md") == 10

    def test_ignores_non_numeric_prefixes(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "no-number.md").write_text("x")
        (reports / "abc-company-2026.md").write_text("x")
        assert max_report_num(reports, {"jobs": {}}, tmp_path / "none.md") == 0


    # The tracker is the one record that survives a cache eviction (#203).
    def _tracker(self, tmp_path, rows, header=None):
        apps = tmp_path / "applications.md"
        apps.write_text((header or _APPS_HEADER) + "".join(rows), encoding="utf-8")
        return apps

    def test_tracker_links_count_when_files_and_state_are_gone(self, tmp_path):
        """A tracker-only restore: reports/ and the batch state are empty, the
        tracker links up to 3200. The next number must be 3201, not 001."""
        apps = self._tracker(tmp_path, [_apps_row(1, "Evaluated", "0001", "-"),
                                        _apps_row(2, "Applied", "3200", "-")])
        assert max_report_num(tmp_path / "reports", {"jobs": {}}, apps) == 3200

    def test_the_highest_source_wins(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "040-acme-2026-01-01.md").write_text("x")
        apps = self._tracker(tmp_path, [_apps_row(1, "Evaluated", "12", "-")])
        assert max_report_num(reports, {"jobs": {"j": {"report_num": "007"}}}, apps) == 40

    def test_a_report_link_in_notes_counts(self, tmp_path):
        """merge-tracker puts the link in Notes for a tracker with no Report
        column, and row_report_num reads it there; numbering must agree."""
        apps = self._tracker(tmp_path, [
            "| 1 | 2026-01-01 | Acme | Eng | 4.0 | Evaluated | see [57](reports/057-acme.md) |\n"],
            header="| # | Date | Company | Role | Score | Status | Notes |\n|---|---|---|---|---|---|---|\n")
        assert max_report_num(tmp_path / "reports", {}, apps) == 57

    def test_the_via_layout_is_read_by_name(self, tmp_path):
        apps = self._tracker(tmp_path, [
            "| 1 | 2026-01-01 | Acme | Hays | Eng | 4.0/5 | Evaluated | | [88](reports/088-acme.md) | |\n"],
            header="| # | Date | Company | Via | Role | Score | Status | PDF | Report | Notes |\n"
                   "|---|---|---|---|---|---|---|---|---|---|\n")
        assert max_report_num(tmp_path / "reports", {}, apps) == 88

    def test_a_000_link_claims_nothing(self, tmp_path):
        """`[000]` is what write_job_result writes with no number at all."""
        apps = self._tracker(tmp_path, [_apps_row(1, "Evaluated", "000", "-")])
        assert max_report_num(tmp_path / "reports", {}, apps) == 0

    def test_every_link_in_notes_counts_not_just_the_first(self, tmp_path):
        """A re-evaluated row in a tracker with no Report column carries its old
        and new links in Notes, oldest first; numbering must see the newest."""
        apps = self._tracker(tmp_path, [
            "| 1 | 2026-01-01 | Acme | Eng | 4.0 | Evaluated | see [57](reports/057-acme.md)"
            " — Re-eval 2026-02-01 (4.0→4.2): [3201](reports/3201-acme.md) |\n"],
            header="| # | Date | Company | Role | Score | Status | Notes |\n|---|---|---|---|---|---|---|\n")
        assert max_report_num(tmp_path / "reports", {}, apps) == 3201

    def test_a_posting_url_in_notes_is_not_a_report_number(self, tmp_path):
        apps = self._tracker(tmp_path, [
            "| 1 | 2026-01-01 | Acme | Eng | 4.0/5 | Evaluated | | | [9999](https://example.com/jobs/9999) |\n"])
        assert max_report_num(tmp_path / "reports", {}, apps) == 0


class TestMaxTrackerNum:
    def test_returns_zero_for_missing_file(self, tmp_path):
        state = {"jobs": {}}
        assert max_tracker_num(tmp_path / "missing.md", state) == 0

    def test_reads_highest_from_table(self, tmp_path):
        apps = tmp_path / "applications.md"
        apps.write_text("| 1 | stuff |\n| 5 | stuff |\n| 3 | stuff |\n", encoding="utf-8")
        assert max_tracker_num(apps, {"jobs": {}}) == 5

    def test_reads_from_state_when_higher(self, tmp_path):
        apps = tmp_path / "applications.md"
        apps.write_text("| 1 | stuff |\n", encoding="utf-8")
        state = {"jobs": {"j1": {"tracker_num": 20}}}
        assert max_tracker_num(apps, state) == 20

    def test_ignores_non_numeric_table_values(self, tmp_path):
        apps = tmp_path / "applications.md"
        apps.write_text("| # | Company | Role |\n| --- | --- | --- |\n| 7 | Acme | Eng |\n", encoding="utf-8")
        assert max_tracker_num(apps, {"jobs": {}}) == 7


class TestExtractTag:
    def test_extracts_simple_tag(self):
        assert extract_tag("<foo>bar</foo>", "foo") == "bar"

    def test_extracts_multiline_content(self):
        text = "<report>\nline1\nline2\n</report>"
        assert extract_tag(text, "report") == "line1\nline2"

    def test_returns_empty_when_tag_missing(self):
        assert extract_tag("no tags here", "missing") == ""

    def test_extracts_first_match(self):
        text = "<tag>first</tag> ... <tag>second</tag>"
        assert extract_tag(text, "tag") == "first"


class TestParseJsonLoose:
    def test_parses_clean_json(self):
        result = parse_json_loose('{"key": "value", "num": 3.5}')
        assert result == {"key": "value", "num": 3.5}

    def test_extracts_json_from_surrounding_text(self):
        text = 'some text before {"score": 4.2} and after'
        result = parse_json_loose(text)
        assert result["score"] == 4.2

    def test_returns_none_on_no_json(self):
        assert parse_json_loose("no json here at all") is None

    def test_returns_none_on_empty_string(self):
        assert parse_json_loose("") is None


class TestBuildUserMessage:
    def test_includes_all_fields(self):
        meta = {
            "id": "42",
            "report_num": "007",
            "tracker_num": 99,
            "url": "https://example.com/job",
            "company": "Acme Corp",
            "role": "Software Engineer",
            "jd_text": "Must know Python.",
        }
        msg = build_user_message(meta, "2026-05-24")
        assert "42" in msg
        assert "007" in msg
        assert "99" in msg
        assert "https://example.com/job" in msg
        assert "Acme Corp" in msg
        assert "Software Engineer" in msg
        assert "Must know Python." in msg
        assert "2026-05-24" in msg

    def test_uses_fallback_for_missing_jd_text(self):
        meta = {"id": "1", "report_num": "001", "tracker_num": 1, "url": "", "company": "", "role": "", "jd_text": ""}
        msg = build_user_message(meta, "2026-05-24")
        assert "no JD cached" in msg

    def test_uses_fallback_for_missing_company(self):
        meta = {"id": "1", "report_num": "001", "tracker_num": 1, "url": "", "company": None, "role": "", "jd_text": "some text"}
        msg = build_user_message(meta, "2026-05-24")
        assert "unknown" in msg


class TestLoadPending:
    def _write_tsv(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "url", "source", "notes"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_pending(tmp_path / "missing.tsv", {"jobs": {}})
        assert result == []

    def test_returns_all_when_none_done(self, tmp_path):
        tsv = tmp_path / "batch-input.tsv"
        self._write_tsv(tsv, [
            {"id": "1", "url": "https://a.com", "source": "Acme", "notes": "Eng"},
            {"id": "2", "url": "https://b.com", "source": "Globex", "notes": "Dev"},
        ])
        result = load_pending(tsv, {"jobs": {}})
        assert len(result) == 2

    def test_skips_completed_by_default(self, tmp_path):
        tsv = tmp_path / "batch-input.tsv"
        self._write_tsv(tsv, [
            {"id": "1", "url": "https://a.com", "source": "Acme", "notes": "Eng"},
            {"id": "2", "url": "https://b.com", "source": "Globex", "notes": "Dev"},
        ])
        state = {"jobs": {"1": {"status": "completed"}}}
        result = load_pending(tsv, state)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_custom_done_statuses_skips_pending(self, tmp_path):
        tsv = tmp_path / "batch-input.tsv"
        self._write_tsv(tsv, [
            {"id": "1", "url": "https://a.com", "source": "Acme", "notes": "Eng"},
            {"id": "2", "url": "https://b.com", "source": "Globex", "notes": "Dev"},
        ])
        state = {"jobs": {"1": {"status": "pending"}}}
        result = load_pending(tsv, state, done_statuses=frozenset({"pending", "completed"}))
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_includes_failed_by_default(self, tmp_path):
        tsv = tmp_path / "batch-input.tsv"
        self._write_tsv(tsv, [{"id": "1", "url": "https://a.com", "source": "Acme", "notes": "Eng"}])
        state = {"jobs": {"1": {"status": "failed"}}}
        result = load_pending(tsv, state)
        assert len(result) == 1

    def test_skips_abandoned_by_default(self, tmp_path):
        # A job that used up its allowance of failed runs (#206) stays out of
        # the queue; only --retry-abandoned brings it back.
        tsv = tmp_path / "batch-input.tsv"
        self._write_tsv(tsv, [
            {"id": "1", "url": "https://a.com", "source": "Acme", "notes": "Eng"},
            {"id": "2", "url": "https://b.com", "source": "Globex", "notes": "Dev"},
        ])
        state = {"jobs": {"1": {"status": "abandoned", "attempts": 3, "error": "x"}}}
        assert [r["id"] for r in load_pending(tsv, state)] == ["2"]

    def test_skips_blank_ids(self, tmp_path):
        tsv = tmp_path / "batch-input.tsv"
        with open(tsv, "w", newline="", encoding="utf-8") as f:
            f.write("id\turl\tsource\tnotes\n")
            f.write("\thttps://a.com\tAcme\tEng\n")
            f.write("2\thttps://b.com\tGlobex\tDev\n")
        result = load_pending(tsv, {"jobs": {}})
        assert len(result) == 1
        assert result[0]["id"] == "2"


class TestAssignJobNumbers:
    """The entry is rebuilt every run, so a job's failure history survives only
    if it is carried over by name — a counter written where the job fails would
    otherwise be wiped on the next run, and the #206 allowance never reached."""

    def _rows(self):
        return [{"id": "1", "url": "https://a.com", "source": "Acme", "notes": "Eng"},
                {"id": "2", "url": "https://b.com", "source": "Globex", "notes": "Dev"}]

    def test_carries_attempts_forward(self, tmp_path):
        state = {"jobs": {"1": {"status": "failed", "attempts": 2, "error": "boom",
                                "report_num": "007", "score": 1.0}}}
        assign_job_numbers(self._rows(), state, 10, 20, tmp_path, load_jd_text=False)
        job = state["jobs"]["1"]
        assert job["attempts"] == 2
        # The error is this run's to write: a failure sets it, a success has none.
        assert "error" not in job
        # Everything else is this run's: fresh numbers, pending, no stale score.
        assert job["status"] == "pending" and job["report_num"] == "011"
        assert "score" not in job

    def test_a_fresh_job_has_neither(self, tmp_path):
        state = {"jobs": {}}
        assign_job_numbers(self._rows(), state, 0, 0, tmp_path, load_jd_text=False)
        assert not {"attempts", "error"} & set(state["jobs"]["2"])


class TestWriteJobResult:
    def _make_response(self, report="report content", tracker="1\t2026-01-01\tAcme\tEng\tEvaluada\t4.0/5\tnull\t[001](reports/001-acme-2026-01-01.md)\tAPPLY", score=4.0,
                       company="Acme", report_num="001"):
        summary = json.dumps({"status": "completed", "id": "42", "report_num": report_num, "company": company, "role": "Eng", "score": score, "legitimacy": "High Confidence", "pdf": None, "report": "reports/001-acme-2026-01-01.md", "error": None})
        return f"<evaluation><report>{report}</report><tracker_tsv>{tracker}</tracker_tsv><summary>{summary}</summary></evaluation>"

    def test_the_addition_is_written_atomically(self, tmp_path, mocker):
        # A merge's sanitizer elsewhere must never read a half-written row.
        spy = mocker.spy(_batch_common, "atomic_write_text")
        reports, tracker = tmp_path / "reports", tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme"}
        write_job_result(self._make_response(), meta, reports, tracker, "2026-01-01")
        assert any(c.args[0] == tracker / "42.tsv" for c in spy.call_args_list)

    def test_writes_report_file(self, tmp_path):
        reports = tmp_path / "reports"
        tracker = tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme"}
        out = write_job_result(self._make_response(), meta, reports, tracker, "2026-01-01")
        assert out["report_file"] == "001-acme-2026-01-01.md"
        assert (reports / "001-acme-2026-01-01.md").exists()

    def test_writes_tracker_file(self, tmp_path):
        reports = tmp_path / "reports"
        tracker = tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme"}
        out = write_job_result(self._make_response(), meta, reports, tracker, "2026-01-01")
        assert out["tracker_file"] == "42.tsv"
        assert (tracker / "42.tsv").exists()

    def test_returns_score_from_summary(self, tmp_path):
        reports = tmp_path / "reports"
        tracker = tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme"}
        out = write_job_result(self._make_response(score=3.7), meta, reports, tracker, "2026-01-01")
        assert out["summary"]["score"] == 3.7

    # `{n}` is the response's length: the reason quotes it instead of the text.
    @pytest.mark.parametrize("response, reason", [
        pytest.param("<evaluation><summary>{}</summary></evaluation>",
                     "the response carried no <report> or <tracker_tsv> ({n} chars)", id="neither"),
        pytest.param("<evaluation><report># R</report><summary>{}</summary></evaluation>",
                     "the response carried no <tracker_tsv> ({n} chars)", id="no-row"),
        pytest.param("<evaluation><tracker_tsv>a row</tracker_tsv><summary>{}</summary></evaluation>",
                     "the response carried no <report> ({n} chars)", id="no-report"),
        pytest.param('<evaluation><summary>{"status": "failed", "error": "JD\\n  was empty"}'
                     "</summary></evaluation>",
                     "the model reported a failed evaluation (14-char reason withheld)",
                     id="model-failed"),
        pytest.param("I could not evaluate this posting.",
                     "the response carried no <report> or <tracker_tsv> ({n} chars)", id="prose"),
    ])
    def test_an_incomplete_evaluation_raises_and_writes_nothing(self, tmp_path, response, reason):
        """#204: a response with either tag missing used to be written as far as
        it went and returned normally, so the job was recorded completed — never
        re-queued — and the role never reached the tracker. The check comes
        before any write, and the reason names what was missing (or the model's
        own error), never the response, which can carry profile text."""
        reports, tracker = tmp_path / "reports", tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        meta = {"id": "99", "report_num": "002", "company": "Globex"}
        with pytest.raises(IncompleteResponse) as exc:
            write_job_result(response, meta, reports, tracker, "2026-01-01")
        assert str(exc.value) == reason.format(n=len(response))
        assert list(reports.iterdir()) == [] and list(tracker.iterdir()) == []

    def test_both_tags_win_over_a_failed_status(self, tmp_path):
        # The tags are the evaluation; a stray "failed" in a summary beside them
        # must not turn a whole one into a counted failure.
        reports, tracker = tmp_path / "reports", tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme"}
        response = self._make_response().replace('"status": "completed"', '"status": "failed"')
        assert '"status": "failed"' in response
        out = write_job_result(response, meta, reports, tracker, "2026-01-01")
        assert (reports / out["report_file"]).exists() and (tracker / "42.tsv").exists()

    @pytest.mark.parametrize("summary", ['["completed"]', '"done"', "7"])
    def test_a_summary_that_is_not_an_object_is_no_summary(self, tmp_path, summary):
        reports, tracker = tmp_path / "reports", tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme"}
        response = self._make_response().split("<summary>")[0] + f"<summary>{summary}</summary></evaluation>"
        out = write_job_result(response, meta, reports, tracker, "2026-01-01")
        assert out["summary"] == {} and (tracker / "42.tsv").exists()

    def test_a_row_that_cannot_be_built_leaves_no_report(self, tmp_path, monkeypatch):
        # The row is built before anything is written: a report left behind
        # would share its number with the re-queued job's next one.
        reports, tracker = tmp_path / "reports", tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()

        def broken(*a, **k):
            raise ValueError("a row the sanitizer cannot read")
        monkeypatch.setattr(_batch_common, "sanitize_addition", broken)
        with pytest.raises(ValueError):
            write_job_result(self._make_response(), {"id": "42", "report_num": "001"},
                             reports, tracker, "2026-01-01")
        assert list(reports.iterdir()) == [] and list(tracker.iterdir()) == []

    def test_slugifies_company_name(self, tmp_path):
        reports = tmp_path / "reports"
        tracker = tmp_path / "tracker"
        reports.mkdir()
        tracker.mkdir()
        summary = json.dumps({"company": "Acme Corp LLC", "score": 4.0})
        response = f"<evaluation><report>content</report><tracker_tsv>row</tracker_tsv><summary>{summary}</summary></evaluation>"
        meta = {"id": "1", "report_num": "001"}
        out = write_job_result(response, meta, reports, tracker, "2026-01-01")
        assert "acme-corp-llc" in out["report_file"]

    def test_report_link_names_the_file_the_writer_created(self, tmp_path):
        """The Report cell is model-authored from a shape alone, so it invents
        the slug — keeps the `&`, appends the role, uses an en-dash — while the
        file is written under the writer's own slug. Six of 230 rows in a real
        tracker linked a file that never existed; the tailor then silently built
        from the JD alone (#162). The writer knows the name: it wins."""
        reports, tracker = tmp_path / "reports", tmp_path / "tracker"
        reports.mkdir(); tracker.mkdir()
        company = "Baylor Scott & White Health"
        meta = {"id": "42", "report_num": "271", "company": company}
        row = (f"11\t2026-09-03\t{company}\tFront Desk Lead – Heart & Vascular"
               "\tEvaluated\t4.2/5\tnull"
               "\t[271](reports/271-baylor-scott-&-white-health-front-desk-lead-–-heart-2026-09-03.md)"
               "\tAPPLY")
        response = self._make_response(report="report", tracker=row, score=4.2,
                                       company=company, report_num="271")
        write_job_result(response, meta, reports, tracker, "2026-09-03")
        cells = _data_cells(tracker / "42.tsv")
        assert cells[7] == "[271](reports/271-baylor-scott-white-health-2026-09-03.md)"
        assert (reports / "271-baylor-scott-white-health-2026-09-03.md").exists()

    def test_url_spliced_into_notes_cell(self, tmp_path):
        # The UI's "Open posting" link reads the URL out of the notes cell.
        # Per our prompt, the LLM writes a freeform one-sentence summary there
        # — typically no URL — so without this splice the link points to `#`
        # for every row. Confirm we write the URL into the notes column.
        reports = tmp_path / "reports"; reports.mkdir()
        tracker = tmp_path / "tracker"; tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme",
                "url": "https://www.linkedin.com/jobs/view/12345"}
        write_job_result(self._make_response(), meta, reports, tracker, "2026-01-01")
        cells = _data_cells(tracker / "42.tsv")
        assert len(cells) == len(ADDITION_LABELS)
        assert "https://www.linkedin.com/jobs/view/12345" in cells[8]
        # Original LLM notes are preserved alongside the URL.
        assert "APPLY" in cells[8]
        assert cells[9] == ""          # the `url` column: the URL lives in Notes

    def test_url_injection_skipped_when_notes_already_have_a_url(self, tmp_path):
        # Don't double up if the LLM already included a URL.
        reports = tmp_path / "reports"; reports.mkdir()
        tracker = tmp_path / "tracker"; tracker.mkdir()
        existing = "https://other.example/posting"
        tracker_row = f"1\t2026-01-01\tAcme\tEng\tEvaluada\t4.0/5\tnull\t[001](reports/001-acme-2026-01-01.md)\t{existing}"
        meta = {"id": "42", "report_num": "001", "company": "Acme",
                "url": "https://canonical.example/job"}
        write_job_result(self._make_response(tracker=tracker_row), meta, reports, tracker, "2026-01-01")
        notes = _data_cells(tracker / "42.tsv")[8]
        # Existing URL is preserved; we don't append the canonical one.
        assert notes == existing

    def test_url_injection_no_op_when_meta_lacks_url(self, tmp_path):
        # Older callers / legacy state may pass meta without a url field.
        # Leave the row untouched rather than write the literal "—  ".
        reports = tmp_path / "reports"; reports.mkdir()
        tracker = tmp_path / "tracker"; tracker.mkdir()
        meta = {"id": "42", "report_num": "001", "company": "Acme"}  # no url
        write_job_result(self._make_response(), meta, reports, tracker, "2026-01-01")
        assert _data_cells(tracker / "42.tsv")[8] == "APPLY"

    def test_url_injection_no_op_on_malformed_row(self, tmp_path):
        # If the LLM returned a row with the wrong number of columns, leave it
        # alone rather than corrupt it by splitting/rejoining at the wrong
        # boundary — and without a label line, which would assert positions
        # nothing has checked (merge-tracker refuses it either way, and the
        # guard names it).
        reports = tmp_path / "reports"; reports.mkdir()
        tracker = tmp_path / "tracker"; tracker.mkdir()
        malformed = "1\t2026-01-01\tAcme"  # only 3 cols
        meta = {"id": "42", "report_num": "001", "company": "Acme",
                "url": "https://example.com/job"}
        write_job_result(self._make_response(tracker=malformed), meta, reports, tracker, "2026-01-01")
        written = (tracker / "42.tsv").read_text(encoding="utf-8").rstrip("\n")
        assert written == malformed
        assert "example.com" not in written

    def test_a_blank_pdf_and_score_still_go_out_labelled(self, tmp_path):
        """The headed parser refuses a blank required cell, which the headerless
        one never checked, so a model that left PDF or score empty used to
        merge and, labelled, would be refused. The chain turns a blank score
        into `N/A` before the labels go on, and the labeller fills the PDF."""
        reports, tracker = tmp_path / "reports", tmp_path / "tracker"
        reports.mkdir(); tracker.mkdir()
        row = "1\t2026-01-01\tAcme\tEng\tEvaluated\t\t\t[001](reports/001-acme-2026-01-01.md)\tAPPLY"
        write_job_result(self._make_response(tracker=row), {"id": "42", "report_num": "001"},
                         reports, tracker, "2026-01-01")
        assert _label_line(tracker / "42.tsv") == ADDITION_LABEL_LINE
        assert _data_cells(tracker / "42.tsv")[5:7] == ["N/A", _PROMPT_PDF_CELL]


class TestRunMergeTracker:
    def test_returns_false_when_script_missing(self, tmp_path, capsys):
        # Not clean — and said, since --evaluate-batch now fails the run on it
        # and points at the [batch] lines for why.
        career_ops = tmp_path / "career-ops"
        career_ops.mkdir()
        assert run_merge_tracker(career_ops) is False
        assert "no merge-tracker.mjs" in capsys.readouterr().out

    def test_no_node_is_a_merge_that_did_not_run(self, tmp_path, mocker, capsys):
        # Not a traceback: the run's later stages still have to run.
        career_ops = tmp_path / "career-ops"
        career_ops.mkdir()
        (career_ops / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        mocker.patch("pipeline._batch_common.subprocess.run",
                     side_effect=FileNotFoundError(2, "No such file or directory", "node"))
        assert run_merge_tracker(career_ops) is False
        assert "could not run merge-tracker.mjs" in capsys.readouterr().out

    def test_an_empty_queue_is_a_clean_merge(self, tmp_path, mocker):
        career_ops = tmp_path / "career-ops"
        career_ops.mkdir()
        (career_ops / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        mock_run = mocker.patch("pipeline._batch_common.subprocess.run")
        mock_run.return_value = mocker.MagicMock(returncode=0)
        assert run_merge_tracker(career_ops) is True

    def test_refusal_reasons_are_read_off_stderr(self, tmp_path, mocker, capsys):
        """merge-tracker refuses a row with console.warn — stderr. Reading only
        stdout left the reasons block empty on every genuine loss, and printed
        the one refusal it does log to stdout (the benign unscoreable re-eval)
        as the explanation for a different addition's disappearance.
        `tests/test_merge_tracker_contract.py` pins the stream itself."""
        career_ops = tmp_path / "career-ops"
        (career_ops / "data").mkdir(parents=True)
        (career_ops / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        additions = career_ops / "batch" / "tracker-additions"
        (additions / "merged").mkdir(parents=True)
        (additions / "7.tsv").write_text(
            "11\t2026-09-01\tAcme Corp\tPlatform Engineer\tEvaluated\t4.7/5\tnull\t"
            "[229](reports/229-x.md)\tnote", encoding="utf-8")

        def archive_like_merge_tracker(*a, **kw):
            """merge-tracker archives a row it refused, and still exits 0."""
            for f in additions.glob("*.tsv"):
                f.rename(additions / "merged" / f.name)
            return mocker.MagicMock(
                returncode=0, stdout="📊 Existing: 1 entries\n",
                stderr='⚠️  Skipping 7.tsv: report #229 is marked "failed"\n')

        mocker.patch("pipeline._batch_common.subprocess.run",
                     side_effect=archive_like_merge_tracker)
        run_merge_tracker(career_ops)
        out = capsys.readouterr().out
        assert "WARNING: 1 evaluation(s)" in out
        assert 'Skipping 7.tsv: report #229 is marked "failed"' in out

    # ── partial merges: what happened is read off the disk, not the exit code ──

    def test_a_clean_merge_is_true(self, tmp_path, mocker, capsys):
        co, _ = _career_ops(tmp_path, additions=["7.tsv", "8.tsv"])
        _fake_merge(mocker, co, apply=["7.tsv", "8.tsv"])
        assert run_merge_tracker(co) is True
        assert "WARNING" not in capsys.readouterr().out

    def test_a_partial_merge_runs_the_guard_and_is_not_a_success(self, tmp_path, mocker, capsys):
        """merge-tracker exits 1 for an addition it could not apply AFTER it has
        written the tracker and archived the rest. Read as total failure, the
        guard never ran — the one it kept went unnamed — and the run printed
        "failed" over rows that had just merged."""
        co, _ = _career_ops(tmp_path, additions=["7.tsv", "8.tsv"])
        _fake_merge(mocker, co, apply=["7.tsv"], returncode=1,
                    stderr="❌ 8.tsv: could not locate tracker row #4 (Co1 — Role1) to update; "
                           "this evaluation was NOT merged.\n")
        assert run_merge_tracker(co) is False
        out = capsys.readouterr().out
        assert "merge-tracker failed" not in out
        assert "exited 1 after a partial merge" in out
        assert "KEPT" in out and "8.tsv (Co1 — Role1)" in out
        assert "merge-tracker: ❌ 8.tsv: could not locate tracker row" in out
        assert "7.tsv" not in out.split("KEPT", 1)[1]           # the applied one is not named

    def test_a_tracker_rewritten_by_someone_else_is_not_a_partial_merge(
            self, tmp_path, mocker, capsys):
        """The tracker changed during a failed merge, but nothing left the
        queue — so the change was another writer's, not this merge's: the UI's
        Add-Job merge times out on merge-tracker's tracker lock while a pipeline
        merge rewrites the tracker, and exits 1 having touched nothing. Read as
        a partial merge, every valid addition in the queue was reported refused
        and KEPT, with advice to delete it. It is the failure path."""
        co, additions = _career_ops(tmp_path, additions=["7.tsv"])
        apps = co / "data" / "applications.md"

        def run(*a, **kw):
            apps.write_text(apps.read_text(encoding="utf-8") + "| 10 | someone else's |\n",
                            encoding="utf-8")
            return mocker.MagicMock(returncode=1, stdout="",
                                    stderr="Error: timed out waiting for the tracker lock\n")
        mocker.patch("pipeline._batch_common.subprocess.run", side_effect=run)
        assert run_merge_tracker(co) is False
        out = capsys.readouterr().out
        assert "merge-tracker failed:\nError: timed out waiting for the tracker lock" in out
        assert "partial merge" not in out and "KEPT" not in out and "WARNING" not in out
        assert "deleted" not in out
        assert (additions / "7.tsv").exists()                  # left for the next merge

    def test_nothing_happened_is_still_a_failure_with_the_hint(self, tmp_path, mocker, capsys):
        """A non-zero exit that wrote nothing and moved nothing is the failure
        it always was — and where a missing `npm install` shows up."""
        co, _ = _career_ops(tmp_path, additions=["7.tsv"])
        mocker.patch("pipeline._batch_common.subprocess.run", return_value=mocker.MagicMock(
            returncode=1, stdout="",
            stderr="Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'js-yaml'"))
        assert run_merge_tracker(co) is False
        out = capsys.readouterr().out
        assert "merge-tracker failed" in out and "npm install --ignore-scripts" in out
        assert "WARNING" not in out and "partial merge" not in out

    def test_an_exit_0_merge_that_kept_a_refusal_is_not_a_success(self, tmp_path, mocker, capsys):
        """career-ops#5862: a refused addition is kept and the run still exits
        0. True would tell `merge_additions` the evaluation is in the tracker."""
        co, _ = _career_ops(tmp_path, additions=["7.tsv", "8.tsv"])
        _fake_merge(mocker, co, apply=["7.tsv"],
                    stderr="⚠️  Skipping 8.tsv: the column labelled \"score\" reads \"x\"\n")
        assert run_merge_tracker(co) is False
        out = capsys.readouterr().out
        assert "tracker merged" in out and "KEPT" in out and "8.tsv (Co1 — Role1)" in out

    @pytest.mark.parametrize("text, who", [
        ("| 11 | 2026-09-01 | Acme Corp | Platform Engineer | Evaluated | 4.7/5 | null "
         "| [229](reports/229-x.md) | note |\n", "Acme Corp — Platform Engineer"),
        ("| 11 | 2026-09-01 | Acme Corp |\n", "no readable row"),
        ("11\tAcme Corp\n", "no readable row"),
    ], ids=["pipe-row", "short-pipe-row", "two-cell-row"])
    def test_a_kept_file_is_named_whether_or_not_it_parses(self, tmp_path, mocker, capsys,
                                                           text, who):
        """#241. Kept is decided by the FILE: `_pending_additions` reads nothing
        from a row short of its role cell, and #6062 refuses every headerless
        row and #5862 keeps it all the same. Judged on what parsed, the short
        ones went unnamed, merge-tracker's reason was dropped and the merge
        read as clean — #229's silent refusal on another row shape."""
        co, additions = _career_ops(tmp_path)
        (additions / "7.tsv").write_text(text, encoding="utf-8")
        mocker.patch("pipeline._batch_common.subprocess.run", return_value=mocker.MagicMock(
            returncode=0, stdout="",
            stderr="⚠️  Skipping 7.tsv: headerless tracker-addition refused\n"))
        assert run_merge_tracker(co) is False
        out = capsys.readouterr().out
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text   # left as written
        kept = out.split("KEPT", 1)[1]
        assert f"[batch]   7.tsv ({who})" in kept
        assert "merge-tracker: ⚠️  Skipping 7.tsv: headerless tracker-addition refused" in kept

    # ── one merge at a time: the listings around merge-tracker are ours ──

    @staticmethod
    def _in_threads(*targets):
        """Run each target in its own thread, as Add-Job's background merges
        run, and return their results in order once every one has finished."""
        results = [None] * len(targets)

        def call(i, target):
            results[i] = target()
        threads = [threading.Thread(target=call, args=(i, t)) for i, t in enumerate(targets)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not any(t.is_alive() for t in threads)
        return results

    def test_two_merges_at_once_take_turns(self, tmp_path, mocker):
        """Two quick Add-Jobs merge from two background threads. merge-tracker
        serializes its own writes, but what a merge applied and what it kept
        is read off the queue before and after it, so the two subprocess
        windows must not overlap."""
        co, queue = _career_ops(tmp_path, additions=["7.tsv"])
        windows = []

        def run(*a, **kw):
            start = time.monotonic()
            time.sleep(0.2)
            (queue / "merged").mkdir(exist_ok=True)
            for f in queue.glob("*.tsv"):
                f.rename(queue / "merged" / f.name)
            windows.append((start, time.monotonic()))
            return mocker.MagicMock(returncode=0, stdout="", stderr="")
        mocker.patch("pipeline._batch_common.subprocess.run", side_effect=run)
        self._in_threads(lambda: run_merge_tracker(co), lambda: run_merge_tracker(co))
        first, second = sorted(windows)
        assert first[1] <= second[0]

    def test_a_merge_that_timed_out_while_another_applied_is_not_partial(
            self, tmp_path, mocker, capsys):
        """The misread the merge lock exists for. A pipeline merge applies
        7.tsv; an Add-Job merge started meanwhile, its own 8.tsv queued, times
        out on merge-tracker's tracker lock and exits 1 having applied nothing.
        Overlapping, it listed 7.tsv before and found it gone after — a
        "partial merge" — and reported its valid 8.tsv refused and KEPT, with
        advice to delete it. Taking turns, it lists the queue once the other
        is done, sees nothing leave, and fails as it should."""
        co, queue = _career_ops(tmp_path, additions=["7.tsv"])
        apps = co / "data" / "applications.md"
        inside, second_listed, applied = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def run(*a, **kw):
            calls.append(None)
            if len(calls) == 1:            # the pipeline's merge
                inside.set()
                second_listed.wait(0.5)     # overlapping, the other has listed by now
                row = headed_addition_row(read_addition(queue / "7.tsv"))
                apps.write_text(apps.read_text(encoding="utf-8") + (
                    f"| 11 | 2026-09-01 | {row['company']} | {row['role']} | 4.2/5 "
                    f"| Evaluated | ❌ | {row['report']} | APPLY |\n"), encoding="utf-8")
                (queue / "merged").mkdir(exist_ok=True)
                (queue / "7.tsv").rename(queue / "merged" / "7.tsv")
                applied.set()
                return mocker.MagicMock(returncode=0, stdout="", stderr="")
            second_listed.set()             # the Add-Job's, past its listing
            applied.wait(5)
            return mocker.MagicMock(returncode=1, stdout="",
                                    stderr="Error: timed out waiting for the tracker lock\n")
        mocker.patch("pipeline._batch_common.subprocess.run", side_effect=run)

        def add_job():
            inside.wait(5)
            (queue / "8.tsv").write_text(labelled_addition(
                "12\t2026-09-01\tGlobex\tSRE\tEvaluated\t4.1/5\tnull"
                "\t[230](reports/230-globex.md)\tAPPLY"), encoding="utf-8")
            return run_merge_tracker(co)
        pipeline_merge, add_job_merge = self._in_threads(lambda: run_merge_tracker(co), add_job)
        out = capsys.readouterr().out
        assert (pipeline_merge, add_job_merge) == (True, False)
        assert "merge-tracker failed:\nError: timed out waiting for the tracker lock" in out
        assert "partial merge" not in out and "KEPT" not in out and "deleted" not in out
        assert (queue / "8.tsv").exists()                     # left for the next merge

    @pytest.mark.parametrize("holder", ["another-process", "unwritable"])
    def test_a_lock_held_past_the_bound_is_merged_without(
            self, tmp_path, mocker, monkeypatch, capsys, holder):
        """Every caller is best-effort, so a merge never waits for the lock
        for good: past the bound — a holder that never lets go, or a lock
        file that cannot be written — it merges without it, says so once, and
        leaves another holder's lock where it was."""
        co, queue = _career_ops(tmp_path, additions=["7.tsv"])
        lock = co / "batch" / ".merge-lock"
        monkeypatch.setattr(_batch_common, "_MERGE_LOCK_WAIT", 0.3)
        if holder == "another-process":
            write_process_lock(lock, os.getppid())          # live, fresh and not us
        else:
            mocker.patch("pipeline._batch_common.acquire_process_lock",
                         side_effect=PermissionError("lock file not writable"))
        _fake_merge(mocker, co, apply=["7.tsv"])
        assert run_merge_tracker(co) is True
        out = capsys.readouterr().out
        assert out.count("merging without it") == 1
        assert not (queue / "7.tsv").exists()                 # the merge ran
        if holder == "another-process":
            assert lock.read_text(encoding="utf-8").split()[0] == str(os.getppid())

    def test_a_lock_file_that_cannot_be_written_is_not_waited_out(
            self, tmp_path, mocker, monkeypatch, capsys):
        # A run of OSErrors means the file cannot be created: merging without it
        # then beats making every merge (and Add-Job's request) poll the bound.
        co, queue = _career_ops(tmp_path, additions=["7.tsv"])
        monkeypatch.setattr(_batch_common, "_MERGE_LOCK_WAIT", 30.0)
        mocker.patch("pipeline._batch_common.acquire_process_lock",
                     side_effect=PermissionError("lock file not writable"))
        _fake_merge(mocker, co, apply=["7.tsv"])
        started = time.monotonic()
        assert run_merge_tracker(co) is True
        assert time.monotonic() - started < 10
        assert capsys.readouterr().out.count("merging without it") == 1

    def test_seeds_applications_md_before_merge(self, tmp_path, mocker):
        # Regression: merge-tracker no-ops if applications.md doesn't exist.
        # run_merge_tracker must seed the header first so the merge lands.
        career_ops = tmp_path / "career-ops"
        career_ops.mkdir()
        (career_ops / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        mocker.patch("pipeline._batch_common.subprocess.run",
                     return_value=mocker.MagicMock(returncode=0))
        apps_md = career_ops / "data" / "applications.md"
        assert not apps_md.exists()
        run_merge_tracker(career_ops)
        assert apps_md.exists()
        assert "Applications Tracker" in apps_md.read_text(encoding="utf-8")

    def test_does_not_clobber_existing_applications_md(self, tmp_path, mocker):
        # If applications.md already exists (e.g. restored from cache with the
        # user's status edits), seeding must NOT overwrite it.
        career_ops = tmp_path / "career-ops"
        (career_ops / "data").mkdir(parents=True)
        (career_ops / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        apps_md = career_ops / "data" / "applications.md"
        apps_md.write_text("# Applications Tracker\n\nEXISTING USER DATA\n", encoding="utf-8")
        mocker.patch("pipeline._batch_common.subprocess.run",
                     return_value=mocker.MagicMock(returncode=0))
        run_merge_tracker(career_ops)
        assert "EXISTING USER DATA" in apps_md.read_text(encoding="utf-8")


class TestLabelledAddition:
    """`labelled_addition` decides whether a sanitized headerless row may go
    under career-ops' label line, and in what shape. Labels settle by position
    what the headerless parser settled by content, so the row must not change
    from one merge-tracker accepted into one the headed parser refuses."""

    @staticmethod
    def _cells(text):
        label, data = text.rstrip("\n").split("\n")
        assert label == ADDITION_LABEL_LINE
        return dict(zip(ADDITION_LABELS, data.split("\t")))

    def test_a_swapped_status_and_score_go_out_in_label_order(self):
        """The headerless parser told the pair apart by content, and the chain
        leaves a swapped pair for it to (`_normalize_score_cell`). Labelled as
        written, the `score` label would sit over `Evaluated` and merge-tracker
        would refuse a row it used to merge — so the order is settled first,
        by the same rule."""
        cells = self._cells(labelled_addition(_tracker_row(score="Evaluated", status="4.2/5")))
        assert (cells["status"], cells["score"]) == ("Evaluated", "4.2/5")

    def test_a_blank_pdf_gets_what_the_prompt_asks_for(self):
        """The headed parser refuses a blank required cell ("present but empty"),
        which the headerless one never checked: a model that left PDF empty
        used to merge, and labelled as written would be refused."""
        row = _tracker_row().replace("\tnull\t", "\t\t")
        assert self._cells(labelled_addition(row))["pdf"] == _PROMPT_PDF_CELL

    @pytest.mark.parametrize("label", ["num", "date", "company", "role", "status", "score"])
    def test_any_other_blank_required_cell_declines_the_row(self, label):
        """No value to put there that the model meant, so the row is written
        as it stands — a checkout before career-ops#6062 merges it headerless
        as it always did."""
        cells = _tracker_row().split("\t")
        cells[ADDITION_COLUMNS.index(label)] = " "
        assert labelled_addition("\t".join(cells)) is None

    def test_a_tenth_cell_is_a_url_in_any_case(self):
        """As merge-tracker's own URL test is: `HTTPS://` is a URL there, so it
        is one here, and labelled `url`."""
        cells = self._cells(labelled_addition(_tracker_row() + "\tHTTPS://X.example/j/7"))
        assert cells["url"] == "HTTPS://X.example/j/7"


class TestLabelsOnlyForACheckoutThatReadsThem:
    """A merge-tracker from before career-ops#3706 reads a label line as a data
    row and refuses the file — archiving it, exit 0 — and recovery will not
    re-queue it, since its score is readable. That checkout merges the same
    row headerless, so the writer and the sanitizer label only for a checkout
    whose merge-tracker reads labels (`reads_labelled_additions`)."""

    # The row `TestWriteJobResult` writes, as the chain leaves it for a posting
    # at https://x/j/42 — exactly what this wrote before it learned to label.
    ROW = "1\t2026-01-01\tAcme\tEng\tEvaluada\t4.0/5\tnull\t[001](reports/001-acme-2026-01-01.md)\tAPPLY"
    SANITIZED = ("1\t2026-01-01\tAcme\tEng\tEvaluada\t4/5\tnull"
                 "\t[001](reports/001-acme-2026-01-01.md)\thttps://x/j/42 — APPLY")
    # The file either writes, by whether the checkout reads labels: under the
    # label line padded to its `url` cell, or as it stands.
    EXPECTED = {True: f"{ADDITION_LABEL_LINE}\n{SANITIZED}\t\n", False: SANITIZED + "\n"}

    @staticmethod
    def _text(path):
        # Additions are written in text mode, so Windows ends their lines with
        # \r\n; merge-tracker splits on \r?\n, as every reader here does.
        return read_addition(path).replace("\r\n", "\n")

    CHECKOUTS = pytest.mark.parametrize("script, labels", [
        ("// a merge-tracker from before career-ops#3706", False),
        (_CURRENT_MERGE_TRACKER, True),
        ("// usage: node merge-tracker.mjs [--allow-headerless]", True),   # #6062's alone
        (None, True),                   # no merge-tracker.mjs at all
    ], ids=["before-3706", "current", "refuses-headerless-only", "no-script"])

    @staticmethod
    def _co(tmp_path, script):
        co = tmp_path / "career-ops"
        additions = co / "batch" / "tracker-additions"
        additions.mkdir(parents=True)
        (co / "reports").mkdir()
        if script is not None:
            (co / "merge-tracker.mjs").write_text(script, encoding="utf-8")
        return co, additions

    @CHECKOUTS
    def test_the_probe(self, tmp_path, script, labels):
        co, _ = self._co(tmp_path, script)
        assert reads_labelled_additions(co) is labels

    @CHECKOUTS
    def test_the_writer_labels_only_where_it_is_read(self, tmp_path, script, labels):
        co, additions = self._co(tmp_path, script)
        response = TestWriteJobResult()._make_response(tracker=self.ROW)
        write_job_result(response, {"id": "42", "report_num": "001", "company": "Acme",
                                    "url": "https://x/j/42"},
                         co / "reports", additions, "2026-01-01")
        assert self._text(additions / "42.tsv") == self.EXPECTED[labels]

    @CHECKOUTS
    def test_the_sanitizer_labels_only_where_it_is_read(self, tmp_path, script, labels, capsys):
        """A headerless pending row is still sanitized either way — the score
        and URL repairs are not about labels — but goes under the label line
        only for a checkout that reads it."""
        co, additions = self._co(tmp_path, script)
        (co / "batch" / "batch-input.tsv").write_text(
            "id\turl\tsource\tnotes\n42\thttps://x/j/42\tAcme\tEng\n", encoding="utf-8")
        (additions / "42.tsv").write_text(self.ROW + "\n", encoding="utf-8")
        _sanitize_pending_additions(co, additions)
        assert self._text(additions / "42.tsv") == self.EXPECTED[labels]
        assert "sanitized 1 addition" in capsys.readouterr().out

    def test_an_already_sanitized_headerless_row_is_left_alone_before_3706(
            self, tmp_path, capsys):
        """Nothing to repair and no labels to add: not rewritten, not counted."""
        co, additions = self._co(tmp_path, "// a merge-tracker from before career-ops#3706")
        (additions / "42.tsv").write_text(self.SANITIZED + "\n", encoding="utf-8")
        _sanitize_pending_additions(co, additions)
        assert self._text(additions / "42.tsv") == self.EXPECTED[False]
        assert capsys.readouterr().out == ""

    def test_the_answer_follows_the_script_when_it_changes(self, tmp_path):
        """Cached on the script's mtime, so a `git pull` that brings #3706 is
        seen by the next write rather than the next process."""
        co, _ = self._co(tmp_path, "// a merge-tracker from before career-ops#3706")
        script = co / "merge-tracker.mjs"
        assert reads_labelled_additions(co) is False
        mtime = script.stat().st_mtime_ns
        script.write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        os.utime(script, ns=(mtime + 1_000_000_000, mtime + 1_000_000_000))
        assert reads_labelled_additions(co) is True
        script.unlink()
        assert reads_labelled_additions(co) is True        # gone: labelled, the default


class TestEnsureApplicationsMd:
    def test_creates_header_when_missing(self, tmp_path):
        from pipeline._batch_common import ensure_applications_md
        career_ops = tmp_path / "career-ops"
        p = ensure_applications_md(career_ops)
        assert p.exists()
        text = p.read_text(encoding="utf-8")
        assert text.startswith("# Applications Tracker")
        assert "| # | Date | Company | Role |" in text

    def test_idempotent_preserves_content(self, tmp_path):
        from pipeline._batch_common import ensure_applications_md
        career_ops = tmp_path / "career-ops"
        (career_ops / "data").mkdir(parents=True)
        (career_ops / "data" / "applications.md").write_text("custom", encoding="utf-8")
        ensure_applications_md(career_ops)
        assert (career_ops / "data" / "applications.md").read_text(encoding="utf-8") == "custom"


class TestBuildSystemPrompt:
    def test_includes_cv(self):
        result = build_system_prompt("MY CV CONTENT", "profile: yaml")
        assert "MY CV CONTENT" in result

    def test_includes_profile_yml(self):
        result = build_system_prompt("cv", "profile: yaml content")
        assert "profile: yaml content" in result

    def test_includes_profile_md_when_provided(self):
        result = build_system_prompt("cv", "profile", profile_md="custom instructions")
        assert "custom instructions" in result

    def test_profile_md_omitted_when_empty(self):
        result = build_system_prompt("cv", "profile", profile_md="")
        assert "User Customizations" not in result

    def test_includes_article_digest_when_provided(self):
        result = build_system_prompt("cv", "profile", article_digest="proof points here")
        assert "proof points here" in result

    def test_article_digest_omitted_when_empty(self):
        result = build_system_prompt("cv", "profile", article_digest="")
        assert "Proof Points" not in result

    # --- Commit 4: PROFILE.md as the authoritative candidate profile ----------
    # When a living PROFILE.md is resolved, it supersedes the 4 seed fragments
    # (cv.md / profile.yml / _profile.md / article-digest.md) as the candidate
    # profile — one master drives evaluation. Only the profile *content* swaps;
    # the evaluation framework is unchanged.

    def test_profile_master_supersedes_seed_fragments(self):
        result = build_system_prompt(
            "MYCV_XYZ", "PROFILEYAML_XYZ",
            profile_md="CUSTOM_XYZ", article_digest="PROOF_XYZ",
            profile_master="LIVING MASTER PROFILE",
        )
        assert "LIVING MASTER PROFILE" in result
        # the raw seed fragments are NOT separately dumped when a master is present
        for frag in ("MYCV_XYZ", "PROFILEYAML_XYZ", "CUSTOM_XYZ", "PROOF_XYZ"):
            assert frag not in result

    def test_profile_master_keeps_evaluation_framework(self):
        result = build_system_prompt("cv", "profile", profile_master="MASTER")
        assert "EVALUATION FRAMEWORK" in result
        assert "Machine Summary" in result

    def test_empty_profile_master_is_current_behavior(self):
        # regression: no master → the 4 seed fragments render exactly as before
        result = build_system_prompt(
            "MYCV_XYZ", "PROFILEYAML_XYZ",
            profile_md="CUSTOM_XYZ", article_digest="PROOF_XYZ",
        )
        for frag in ("MYCV_XYZ", "PROFILEYAML_XYZ", "CUSTOM_XYZ", "PROOF_XYZ"):
            assert frag in result
        assert "EVALUATION FRAMEWORK" in result

    def test_whitespace_profile_master_falls_back_to_fragments(self):
        # a blank/whitespace master must not blank out the candidate profile
        result = build_system_prompt("MYCV_XYZ", "profile", profile_master="   \n  ")
        assert "MYCV_XYZ" in result

    def test_none_profile_master_falls_back_to_fragments(self):
        # defensive: a None master degrades like the sibling params, never crashes
        result = build_system_prompt("MYCV_XYZ", "profile", profile_master=None)
        assert "MYCV_XYZ" in result


class TestEvalSystemPrompt:
    """eval_system_prompt(career_ops) — the single candidate-profile resolution
    point shared by the batch (`--evaluate-batch`) and UI add-job eval paths, so
    they can never disagree: PROFILE.md wins when present, else the seed files."""

    def _career_ops(self, tmp_path, profile_yml="PROFILEYAML_XYZ"):
        co = tmp_path / "career-ops"
        (co / "config").mkdir(parents=True)
        (co / "cv.md").write_text("MYCV_XYZ", encoding="utf-8")
        (co / "config" / "profile.yml").write_text(profile_yml, encoding="utf-8")
        return co

    def _master(self, tmp_path, monkeypatch):
        handoff_dir = tmp_path / "handoff"
        handoff_dir.mkdir()
        (handoff_dir / "PROFILE.md").write_text("LIVING MASTER PROFILE", encoding="utf-8")
        monkeypatch.setenv("HANDOFF_OUT_DIR", str(handoff_dir))

    def test_uses_profile_master_when_present(self, tmp_path, monkeypatch):
        co = self._career_ops(tmp_path)
        self._master(tmp_path, monkeypatch)
        result = eval_system_prompt(co)
        assert "LIVING MASTER PROFILE" in result
        assert "MYCV_XYZ" not in result            # master supersedes the seeds
        assert "PROFILEYAML_XYZ" not in result

    def test_falls_back_to_seeds_when_absent(self, tmp_path, monkeypatch):
        co = self._career_ops(tmp_path)
        handoff_dir = tmp_path / "handoff"
        handoff_dir.mkdir()                          # no PROFILE.md anywhere
        monkeypatch.setenv("HANDOFF_OUT_DIR", str(handoff_dir))
        result = eval_system_prompt(co)
        assert "MYCV_XYZ" in result
        assert "PROFILEYAML_XYZ" in result
        assert "EVALUATION FRAMEWORK" in result

    def test_target_roles_come_from_profile_yml_under_a_master(self, tmp_path, monkeypatch):
        """PROFILE.md supersedes profile.yml as the candidate profile, but a master
        seeded before #209 names only two target roles, so the list still comes
        from profile.yml (#209). The one level setup guesses for every role is
        not stated."""
        co = self._career_ops(tmp_path, (
            "target_roles:\n  primary: [Outreach, Translation]\n  archetypes:\n"
            "    - {name: Outreach, level: staff/principal, fit: primary}\n"
            "    - {name: Translation, level: staff/principal, fit: secondary}\n"
            "    - {name: Youth programs, level: staff/principal, fit: secondary}\n"))
        self._master(tmp_path, monkeypatch)
        prompt = eval_system_prompt(co)
        for line in ("- Outreach (primary)\n", "- Translation (secondary)\n",
                     "- Youth programs (secondary)\n"):
            assert line in prompt
        assert "staff/principal" not in prompt

    @pytest.mark.parametrize("profile_yml", [
        "target_roles: [unclosed\n",
        # PyYAML raises a plain ValueError, not a YAMLError, for an impossible date
        "available_from: 2026-02-30\ntarget_roles:\n  primary: [Outreach]\n",
    ])
    def test_a_malformed_profile_yml_falls_back_to_the_profile(self, tmp_path, monkeypatch, profile_yml):
        co = self._career_ops(tmp_path, profile_yml)
        monkeypatch.setenv("HANDOFF_OUT_DIR", str(tmp_path / "nowhere"))
        assert "The candidate profile above states their target roles" in eval_system_prompt(co)


class TestScoreBands:
    """The Global score is defined, not left to each model's reading (#208).

    The number is what the digest, the role selector and the tracker's SKIP status
    all act on, so its bands are stated in the prompt, and the two lines the rest
    of the pipeline uses are floors of those bands."""

    def test_bands_cover_the_scale_without_gaps(self):
        floors = [b[0] for b in _batch_common.SCORE_BANDS]
        assert floors == sorted(floors, reverse=True)
        assert floors[0] < 5.0 and floors[-1] == 1.0
        # one decimal each, or the table's "floor-(next-0.1)" ranges overlap
        assert all(round(f, 1) == f for f in floors)
        # every band has a meaning and evidence, not a bare number
        assert all(meaning and evidence for _, meaning, evidence in _batch_common.SCORE_BANDS)

    def test_every_band_is_in_the_prompt(self):
        prompt = build_system_prompt("cv", "profile")
        assert _batch_common._score_band_table() in prompt
        # the ends of the scale, as the model reads them
        assert "| 4.5-5.0 | Strong match: apply now |" in prompt
        assert "| 1.0-1.9 | No fit |" in prompt

    def test_old_location_score_lines_are_dropped_from_profile_md(self):
        """Setups before #208 wrote these into every _profile.md, which is never
        regenerated; they assigned a Global score by location."""
        old = ("## Your Location Policy\n\n**In evaluations (scoring):**\n"
               "- **Remote roles:** Score 5.0 (ideal)\n"
               "- **On-site 4-5 days/week, no exceptions:** Score 1.0 (not viable)\n"
               "- **My own note:** I could relocate for the right role\n")
        prompt = build_system_prompt("cv", "profile", old)
        assert "Score 1.0 (not viable)" not in prompt and "Score 5.0 (ideal)" not in prompt
        assert "I could relocate for the right role" in prompt

    def test_profile_md_cannot_override_the_bands(self):
        """_profile.md weighs dimensions; it must not re-score a hard stop. Old
        setups wrote "On-site ...: Score 1.0" into it, which sank a location-
        blocked role to Skip and out of every queue, undoing #180."""
        prompt = build_system_prompt("cv", "profile", "MODES_PROFILE")
        assert "but not the score bands, the decision rule or the hard-stop rule" in prompt

    def test_decision_and_status_rules_use_the_same_lines(self):
        prompt = build_system_prompt("cv", "profile")
        apply_, skip = _batch_common.APPLY_MIN_SCORE, _batch_common.SKIP_BELOW
        assert f'"Apply" at {apply_:.1f} and above' in prompt
        assert f'"Skip" below {skip:.1f}' in prompt
        assert f'"NO APLICAR" if score < {skip:.1f}' in prompt

    def test_hard_stops_cap_the_decision_not_the_score(self):
        """#180 demotes an out-of-area role and keeps its score, and the digest
        marks it rather than hiding it; a prompt that sank the score for location
        would undo both."""
        prompt = build_system_prompt("cv", "profile")
        assert "A location hard stop caps the decision, not the score" in prompt
        # ...while a credential they cannot hold still lowers the score
        assert "(a licence, credential or clearance) lowers the score as well" in prompt


class TestTargetRolesInPrompt:
    """The archetype list is the candidate's own target roles (#209), not
    career-ops' six AI-engineering defaults."""

    AI_ARCHETYPES = ("AI Platform", "LLMOps", "Forward Deployed", "AI Transformation")

    def test_candidate_roles_replace_the_ai_archetypes(self):
        prompt = build_system_prompt("cv", "profile", target_roles=[
            {"name": "Bilingual community outreach", "fit": "primary"},
            {"name": "Policy analyst", "fit": ""}])
        assert not [ai for ai in self.AI_ARCHETYPES if ai in prompt]
        assert "fit: primary = the role they most want" in prompt
        assert "- Bilingual community outreach (primary)\n" in prompt
        assert "- Policy analyst\n" in prompt          # no empty "()" without a fit
        assert '"Outside target roles (<' in prompt     # an answer, not a failure
        assert "it is newer: use those" in prompt     # a master's own list wins

    def test_the_outside_marker_parses_unquoted(self):
        """The Machine Summary is YAML. A marker carrying ": " fails to parse
        when a model leaves the value unquoted, which loses work_location and
        with it the #180 out-of-area gate."""
        import yaml
        prompt = build_system_prompt("cv", "profile")
        hint = re.search(r'archetype: "([^"]*)"', prompt).group(1)
        marker = hint.split(", or ", 1)[1].replace("<function>", "community outreach")
        assert yaml.safe_load(f"archetype: {marker}") == {"archetype": marker}

    def test_no_fit_legend_when_no_role_has_a_fit(self):
        prompt = build_system_prompt("cv", "profile", target_roles=[{"name": "Policy analyst", "fit": ""}])
        assert "The candidate's target roles, as they named them at setup:\n- Policy analyst\n" in prompt

    def test_without_target_roles_points_at_the_profile(self):
        """A hand-written profile.yml with no target_roles still gets a usable
        instruction: a PROFILE.md master states them in prose."""
        prompt = build_system_prompt("cv", "profile")
        assert "The candidate profile above states their target roles" in prompt
        assert not [ai for ai in self.AI_ARCHETYPES if ai in prompt]


def _tracker_row(score="4.2/5", status="Evaluated", notes="note", role="SRE"):
    """One nine-field tracker-additions row, the shape write_job_result emits."""
    return "\t".join(["3", "2026-08-25", "Initech", role, status,
                      score, "null", "[003](reports/003-x.md)", notes])


# A ten-cell row whose columns are NOT in place (an early extra cell), which
# every positional step of the chain must decline.
_SHIFTED_ROW = "12\t2026-09-01\tAcme\tInc\tSRE\tEvaluated\t4.2/5\tnull\t[12](reports/12-a.md)\tAPPLY"


def _reports(tmp_path, *names, **files):
    """A career-ops dir at tmp_path/co holding these files under reports/:
    `names` get a placeholder body, `files` map name → body."""
    co = tmp_path / "co"; (co / "reports").mkdir(parents=True)
    for n in names:
        (co / "reports" / n).write_text("#", encoding="utf-8")
    for n, body in files.items():
        (co / "reports" / n).write_text(body, encoding="utf-8")
    return co


class TestNormalizeScoreCell:
    """merge-tracker.mjs decides which of columns 5-6 is the score by asking
    whether exactly one of them matches `N/5` (or an N/A / DUP / dash sentinel).
    When neither does it skips the addition — counted as `skipped`, so it still
    exits 0, and before career-ops#5862 it archived the TSV to merged/ too. The
    evaluation was then lost with no error anywhere. The older merge-tracker's fallback assumed this row
    order, so any score string merged and the prompt's "format X.X/5" rule was
    advisory; it is load-bearing now."""

    def _score_of(self, raw):
        return _normalize_score_cell(_tracker_row(raw)).split("\t")[5]

    @pytest.mark.parametrize("raw,expected", [
        ("4.2/5", "4.2/5"),      # already correct — untouched
        ("4.2", "4.2/5"),        # the common model drift
        ("4.2/5.0", "4.2/5"),
        ("4.2 / 5", "4.2/5"),
        ("4,2/5", "4.2/5"),      # comma decimal
        ("**4.5/5**", "4.5/5"),  # bold
        ("5/5", "5/5"),
        ("0/5", "0/5"),
    ])
    def test_coerced_to_mergeable_shape(self, raw, expected):
        assert self._score_of(raw) == expected

    @pytest.mark.parametrize("raw", ["", "unknown", "TBD"])
    def test_unscorable_becomes_the_na_sentinel(self, raw):
        """N/A is a shape merge-tracker recognises, so the row still merges and
        the evaluation stays visible. A scoreless row beats a discarded one."""
        assert self._score_of(raw) == "N/A"

    @pytest.mark.parametrize("raw", ["84%", "8/10", "9.5"])
    def test_other_scales_are_not_clamped_to_a_top_score(self, raw):
        """Out of range means some other scale, and we cannot know which.
        Clamping to 5 would invent a perfect score — and the handoff work-order
        ranks by score descending, so a fabricated 5 puts the role first."""
        assert self._score_of(raw) == "N/A"

    @pytest.mark.parametrize("raw", ["N/A", "DUP"])
    def test_sentinels_pass_through(self, raw):
        assert self._score_of(raw) == raw

    def test_row_with_unexpected_column_count_is_left_alone(self):
        assert _normalize_score_cell("a\tb\tc") == "a\tb\tc"

    def test_applied_by_write_job_result(self, tmp_path):
        """The normalization has to be on the write path, not just available."""
        reports, tracker = tmp_path / "reports", tmp_path / "tsv"
        reports.mkdir(); tracker.mkdir()
        response = TestWriteJobResult()._make_response(tracker=_tracker_row("4.2"))
        write_job_result(response, {"id": 3, "url": "https://x/j/3"},
                         reports, tracker, "2026-08-25")
        assert _data_cells(tracker / "3.tsv")[5] == "4.2/5"


class TestScoreNormalizationDoesNotInvent:
    """The normalizer must never manufacture a score, and never turn a row
    merge-tracker would accept into one it refuses."""

    @pytest.mark.parametrize("raw", [
        "Top 5%",                 # -> 5/5: a fabricated TOP score, ranked first
        "not scored (4 blockers)",
        "see report 003",
        "N/A - 3 red flags",
    ])
    def test_prose_does_not_become_a_score(self, raw):
        """Searching anywhere in the cell turned prose into a plausible score.
        The handoff work-order ranks by score descending, so an invented 5 puts
        that role at the top of the queue."""
        assert _normalize_score_cell(_tracker_row(raw)).split("\t")[5] == "N/A"

    def test_swapped_status_and_score_are_left_alone(self):
        """When the model writes the pair the other way round, merge-tracker
        resolves it — it asks which ONE of the two looks like a score. Writing
        N/A into col 6 makes BOTH look like one, so it refuses a row it would
        have merged: the loss this function exists to prevent, caused by it."""
        swapped = _tracker_row("Evaluated", status="4.2/5")
        assert _normalize_score_cell(swapped) == swapped


class TestEmptyTrailingCell:
    """A row whose Notes cell is empty arrives one field short — `extract_tag`
    strips the tag body, taking the trailing tab with it. Every sanitizer then
    no-ops on its 9-column guard, so the score is never normalized and
    merge-tracker refuses the row: the row that arrives short is exactly the row
    the chain exists to save."""

    def test_short_row_is_restored_and_normalized(self, tmp_path):
        reports, tracker = tmp_path / "reports", tmp_path / "tsv"
        reports.mkdir(); tracker.mkdir()
        response = (
            "<report>body</report><tracker_tsv>"
            "3\t2026-08-25\tInitech\tSRE | Remote\tEvaluada\t4.2\tnull\t[003](reports/003-x.md)\t"
            "</tracker_tsv><summary>{\"company\": \"Initech\"}</summary>"
        )
        write_job_result(response, {"id": 3, "url": "https://x/j/3"},
                         reports, tracker, "2026-08-25")
        cells = _data_cells(tracker / "3.tsv")
        assert len(cells) == len(ADDITION_LABELS)   # restored, then padded to the labels
        assert cells[3] == "SRE"          # role pipe stripped
        assert cells[5] == "4.2/5"        # score normalized to the mergeable shape
        assert cells[8].startswith("https://x/j/3")   # url spliced into notes
        assert cells[9] == ""


class TestMaxReportNumCountsLocks:
    """`max_report_num` must COUNT a `NNN-RESERVED.md` lock, which is the exact
    opposite of `data.find_report_file`'s rule — see the docstrings on both.
    Guarded because the tempting refactor is to make them agree: a reader who
    finds the skip in one and propagates it here would break nothing visible in
    the suite, while `server.py`'s Add-Job and `batch_evaluate` would start
    handing out a number career-ops has already reserved, so two writers race
    for the same `NNN-` and one evaluation overwrites the other."""

    def test_reserved_lock_claims_its_number(self, tmp_path):
        from pipeline._batch_common import max_report_num
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "004-acme-2026-01-01.md").write_text("# real", encoding="utf-8")
        (reports / "011-RESERVED.md").write_text('{"pid":1,"token":"x"}', encoding="utf-8")
        assert max_report_num(reports, {}, tmp_path / "none.md") == 11, "a lock must reserve its number"


class TestHeadedAdditionDetection:
    """career-ops' batch worker writes a label row above its one data row since
    career-ops#3706, and merge-tracker still accepts the legacy headerless row,
    so both arrive — and every reader here tells them apart by merge-tracker's
    own rule (`looksLikeTsvHeaderRow`): the first cell is not a tracker number
    AND at least three cells are recognised labels. Misjudged one way, the
    label line is read as a job called "role" at a company called "company";
    the other way, a real row is taken for table furniture.

    Every test here reads the baked fallback table unless it pins another, so
    none depends on whether ./career-ops is checked out."""

    @pytest.fixture(autouse=True)
    def _fallback_table(self, alias_table):
        alias_table()

    def test_the_workers_label_row_is_a_header(self):
        row = headed_addition_row(
            ADDITION_LABEL_LINE + "\n" + _tracker_row(score="4.2") + "\thttps://x/j/7\n")
        assert (row["company"], row["role"]) == ("Initech", "SRE")
        assert (row["status"], row["score"]) == ("Evaluated", "4.2")
        assert row["report"] == "[003](reports/003-x.md)"

    def test_a_legacy_row_is_not_a_header(self):
        # None, not {}: the caller's cue to read it positionally, as before.
        assert headed_addition_row(_tracker_row() + "\n") is None

    def test_two_recognised_labels_are_not_enough(self):
        """Three is merge-tracker's threshold. One short of it, merge-tracker
        reads the file positionally, and so must every reader here — or the
        two would disagree about which cell is the role."""
        assert headed_addition_row("Company\tRole\tRemote\n" + _tracker_row() + "\n") is None

    def test_a_row_leading_with_a_number_is_data_whatever_its_cells_say(self):
        """A company called Company hiring for a Role, status Status: three
        header words in three cells, enough for the label count. The leading
        tracker number is what keeps it a row — a data row always has one."""
        row = "12\t2026-09-01\tCompany\tRole\tStatus\t4.2/5\tnull\t[12](reports/12-x.md)\tnote"
        assert headed_addition_row(row + "\n") is None

    def test_labels_come_from_career_ops_own_table(self, alias_table):
        """career-ops ships localised labels (`fecha`, `empresa`, `puesto`) in
        tracker-aliases.json and detects a header against that table. A
        hand-mirrored label list would read a file merge-tracker resolves by
        name as a positional row."""
        headed = "fecha\tempresa\tpuesto\n2026-09-01\tAcme Corp\tPlatform Engineer\n"
        assert headed_addition_row(headed) is None        # the fallback knows none of them
        alias_table({"fecha": "date", "empresa": "company", "puesto": "role"})
        assert headed_addition_row(headed) == {
            "date": "2026-09-01", "company": "Acme Corp", "role": "Platform Engineer"}

    def test_the_first_column_carrying_a_label_wins(self):
        """merge-tracker's `resolveTsvColumns` maps a repeated label to its
        FIRST column (and then refuses the file). The loss guard is what
        reports that refusal, so it must name the company that map reads, not
        the repeat's."""
        text = (ADDITION_LABEL_LINE + "\tcompany\n"
                + _tracker_row() + "\thttps://x/j/7\tOther Co\n")
        assert headed_addition_row(text)["company"] == "Initech"

    def test_headed_without_one_data_row_is_empty_not_headerless(self):
        """`{}` is "headed, but not the one data row a reader can take" — no
        row, or more than the one merge-tracker accepts. It is falsy, so a
        caller that tests truthiness instead of `is None` falls through to the
        positional read, which takes a lone label line for a job called "role"
        at a company called "company"."""
        assert headed_addition_row(ADDITION_LABEL_LINE + "\n") == {}
        assert headed_addition_row(
            ADDITION_LABEL_LINE + "\n" + _tracker_row() + "\n" + _tracker_row() + "\n") == {}

    def test_a_lone_cr_in_a_cell_does_not_split_the_row(self, tmp_path):
        """merge-tracker splits lines on `\\r?\\n`, so a stray `\\r` inside a
        cell stays in that cell. `read_text` turns it into a line break, and a
        headed file read that way has two data rows — `{}`, which the loss
        guard, the sanitizer and the UI all pass over, while merge-tracker
        merges or refuses the one row it sees."""
        path = tmp_path / "7.tsv"
        path.write_bytes((ADDITION_LABEL_LINE + "\n" + _tracker_row(notes="one\rtwo")
                          + "\thttps://x/j/7\n").encode("utf-8"))
        row = headed_addition_row(read_addition(path))
        assert (row["role"], row["notes"]) == ("SRE", "one\rtwo")

    def test_a_bom_does_not_hide_the_first_label(self, tmp_path):
        """merge-tracker's `trim()` drops a leading BOM from the first label;
        Python's `strip()` keeps it, and `\\ufeffnum` is no label — so the
        header map lost `num`, the file no longer read as in place, and a bare
        score was neither repaired nor recovered."""
        path = tmp_path / "7.tsv"
        path.write_bytes(b"\xef\xbb\xbf" + (ADDITION_LABEL_LINE + "\n" + _tracker_row()
                                            + "\n").encode("utf-8"))
        row = headed_addition_row(read_addition(path))
        assert row["num"] == _tracker_row().split("\t")[0]


class TestLostAdditionGuard:
    """`run_merge_tracker`'s loss guard asks the filesystem "did this evaluation
    reach applications.md", because merge-tracker exits 0 on a row it refused —
    archiving it before career-ops#5862, keeping it in the queue since (the
    guard's "kept" category). The question has TWO readings, and asking only for
    company::role reported three intact evaluations as lost on a real run (#152):
    merge-tracker's guessed match tiers (entry number, fuzzy title) update the row
    but keep THAT row's title, so the key the addition carried is not the key it
    landed under. Verified against the real merge-tracker.mjs — the score and
    report link write through while the title does not."""

    HEADER = ("# Applications Tracker\n\n"
              "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
              "|---|------|---------|------|-------|--------|-----|--------|-------|\n")

    @staticmethod
    def _addition(tracker_dir, name, company, role, report="229"):
        (tracker_dir / name).write_text(
            "\t".join(["11", "2026-09-01", company, role, "Evaluated", "4.7/5", "null",
                       f"[{report}](reports/{report}-x-2026-09-01.md)", "APPLY — note"]),
            encoding="utf-8")

    def _setup(self, tmp_path, rows="", header=None):
        career_ops = tmp_path / "career-ops"
        (career_ops / "data").mkdir(parents=True)
        (career_ops / "data" / "applications.md").write_text(
            (self.HEADER if header is None else header) + rows, encoding="utf-8")
        tracker_dir = career_ops / "batch" / "tracker-additions"
        (tracker_dir / "merged").mkdir(parents=True)
        return career_ops, tracker_dir

    @staticmethod
    def _row(company, role, report="229", score="4.7/5"):
        return (f"| 10 | 2026-09-01 | {company} | {role} | {score} | Evaluated | ❌ "
                f"| [{report}](../reports/{report}-x-2026-09-01.md) | note |\n")

    def _run(self, career_ops, tracker_dir, capsys, merge_output="", archive=True):
        """Snapshot the pending additions and list the queue, archive them the
        way merge-tracker archives what it applies (`archive=False` keeps them,
        the way it keeps a refusal), list the queue again, then report — the
        real sequence around the subprocess call. Returns (printed, the guard's
        unapplied count)."""
        before, queued = _pending_additions(tracker_dir), queued_additions(tracker_dir)
        if archive:
            for f in tracker_dir.glob("*.tsv"):
                f.rename(tracker_dir / "merged" / f.name)
        unapplied = _warn_on_lost_additions(before, queued, queued_additions(tracker_dir),
                                            career_ops, merge_output)
        return capsys.readouterr().out, unapplied

    def test_retitled_row_is_not_reported_lost(self, tmp_path, capsys):
        """The #152 case: merged into an existing row on a fuzzy tier, which
        keeps the row's own title. Report and company survive; the key doesn't."""
        career_ops, tracker_dir = self._setup(
            tmp_path, self._row("UT Southwestern Medical Center", "INSURANCE SPECIALIST II"))
        self._addition(tracker_dir, "123.tsv", "UT Southwestern Medical Center",
                       "INSURANCE SPECIALIST I")
        out, _ = self._run(career_ops, tracker_dir, capsys)
        assert "WARNING" not in out
        assert "123.tsv" in out
        assert "INSURANCE SPECIALIST I" in out and "INSURANCE SPECIALIST II" in out

    def test_landed_under_its_own_identity_is_silent(self, tmp_path, capsys):
        career_ops, tracker_dir = self._setup(tmp_path, self._row("Acme Corp", "Platform Engineer"))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer")
        assert self._run(career_ops, tracker_dir, capsys) == ("", 0)

    def test_unscoreable_reeval_keeps_the_rows_old_report_and_stays_silent(self, tmp_path, capsys):
        """merge-tracker deliberately keeps the row's score, report and PDF when a
        re-eval produces no score. The new report number never lands, so only the
        company::role reading can see this one — which is why both are kept."""
        career_ops, tracker_dir = self._setup(
            tmp_path, self._row("Acme Corp", "Platform Engineer", report="200", score="4.0/5"))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer", report="229")
        assert self._run(career_ops, tracker_dir, capsys) == ("", 0)

    def test_genuinely_lost_addition_still_warns(self, tmp_path, capsys):
        """Neither identity in the tracker, and the TSV is gone from the queue —
        merge-tracker refused it (a `failed` report number, an unreadable score)
        and archived it anyway, so nothing retries it."""
        career_ops, tracker_dir = self._setup(tmp_path, self._row("Initech", "SRE", report="200"))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer")
        out, unapplied = self._run(career_ops, tracker_dir, capsys,
                                   merge_output='⚠️  Skipping 7.tsv: report #229 is marked "failed"')
        assert unapplied == 1
        assert "WARNING: 1 evaluation(s)" in out
        assert "7.tsv (Acme Corp — Platform Engineer)" in out
        assert 'Skipping 7.tsv' in out          # merge-tracker's own reason, captured

    def test_report_number_alone_does_not_count_as_landed(self, tmp_path, capsys):
        """Report-file and tracker-row sequences drift, so a bare number can name
        an unrelated row — merge-tracker's own report tier requires the company
        too (#912 upstream), and so must this."""
        career_ops, tracker_dir = self._setup(tmp_path, self._row("Initech", "SRE", report="229"))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer", report="229")
        assert "WARNING: 1 evaluation(s)" in self._run(career_ops, tracker_dir, capsys)[0]

    def test_a_still_pending_addition_is_reported_as_kept_not_lost(self, tmp_path, capsys):
        """merge-tracker leaves a TSV it could not apply in place — every refusal
        since career-ops#5862, and its exit-1 `failedAdditions` before that — so
        a later run retries it. Not a loss, and not silence either: the retry is
        refused again until the file changes, and on a fresh install that was
        every evaluation, with this guard saying nothing about any of them. Its
        own category, with merge-tracker's reason and what to do."""
        career_ops, tracker_dir = self._setup(tmp_path)
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer")
        out, unapplied = self._run(
            career_ops, tracker_dir, capsys,
            "⚠️  Skipping 7.tsv: headerless tracker-addition refused — a positional file\n"
            "⚠️  1 skipped addition(s) left in /x for a re-run: 7.tsv — fix each",
            archive=False)
        assert unapplied == 1
        assert "WARNING: 1 evaluation(s) were refused by merge-tracker and KEPT" in out
        assert "without reaching applications.md" not in out       # not the lost line
        assert "7.tsv (Acme Corp — Platform Engineer)" in out
        assert "merge-tracker: ⚠️  Skipping 7.tsv: headerless tracker-addition refused" in out
        assert "left in /x" not in out          # the run summary is not a reason

    def test_a_kept_file_says_what_to_do_wherever_the_merge_ran(self, tmp_path, capsys):
        """One message, true in both places a merge runs. Locally the file stays
        queued, so every merge retries it and reports failure until it is
        corrected or deleted — which is why `merge_additions`,
        `run.sh`/`run.ps1 --batch` and `--evaluate-batch` keep exiting 1. The
        cloud daily does not keep batch/tracker-additions/, so there nothing
        retries it and the run's artifact is where the file is. No "re-evaluate": the job is done
        in the batch state, so nothing would queue it again."""
        career_ops, tracker_dir = self._setup(tmp_path)
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer")
        out, _ = self._run(career_ops, tracker_dir, capsys, archive=False)
        assert "Locally every merge retries each one, and reports failure" in out
        assert "or deleted to drop the evaluation" in out
        assert "the cloud daily does not keep that directory, so there nothing retries it" in out
        assert "pipeline-output artifact for 7 days" in out
        assert "re-evaluate" not in out

    def test_a_kept_re_evaluation_is_not_hidden_by_the_row_it_was_for(self, tmp_path, capsys):
        """A refused re-eval of a role already in the tracker matches that row by
        company::role, so the "did it land" test would call it landed — while
        its new report never arrived. A kept file is reported on being kept."""
        career_ops, tracker_dir = self._setup(
            tmp_path, self._row("Acme Corp", "Platform Engineer", report="200"))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer", report="229")
        out, unapplied = self._run(career_ops, tracker_dir, capsys, archive=False)
        assert unapplied == 1 and "KEPT" in out

    def test_each_reason_is_printed_under_its_own_file(self, tmp_path, capsys):
        """`7.tsv` is a substring of `17.tsv`: matched as a bare substring, each
        file would be printed with the other's refusal."""
        career_ops, tracker_dir = self._setup(tmp_path)
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer", report="229")
        self._addition(tracker_dir, "17.tsv", "Globex", "SRE", report="230")
        out, _ = self._run(career_ops, tracker_dir, capsys,
                           "⚠️  Skipping 17.tsv: reason seventeen\n"
                           "⚠️  Skipping 7.tsv: reason seven", archive=False)
        out = out.splitlines()
        seven = out.index("[batch]   7.tsv (Acme Corp — Platform Engineer)")
        seventeen = out.index("[batch]   17.tsv (Globex — SRE)")
        assert out[seventeen + 1].endswith("Skipping 17.tsv: reason seventeen")
        assert out[seven + 1].endswith("Skipping 7.tsv: reason seven")
        assert sum("Skipping" in line for line in out) == 2

    def test_two_additions_sharing_one_key_are_reported_separately(self, tmp_path, capsys):
        """A re-evaluation of a queued role puts two TSVs under one company::role.
        Collapsing them sends the operator after one report while the other stays
        buried."""
        career_ops, tracker_dir = self._setup(tmp_path)
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer", report="229")
        self._addition(tracker_dir, "8.tsv", "Acme Corp", "Platform Engineer", report="230")
        out, _ = self._run(career_ops, tracker_dir, capsys)
        assert "WARNING: 2 evaluation(s)" in out
        assert "7.tsv" in out and "8.tsv" in out

    def test_blank_role_in_the_matched_row_is_not_a_loss(self, tmp_path, capsys):
        """A tracker row with an empty Role cell (hand-added, half-migrated) is
        still a row the addition merged into. Testing the matched title for
        truthiness rather than the key for membership put this back in the loud
        WARNING — the #152 cry-wolf by another route."""
        career_ops, tracker_dir = self._setup(tmp_path, self._row("Acme Corp", ""))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer")
        out, _ = self._run(career_ops, tracker_dir, capsys)
        assert "WARNING" not in out
        assert "7.tsv" in out

    def test_pipe_delimited_addition_is_still_seen(self, tmp_path, capsys):
        """A model that ignores the prompt's tab rule and emits a markdown row
        produces a file merge-tracker reads — and can therefore refuse and
        archive. Splitting on tabs alone made that addition invisible here, so
        its loss would have been silent, uncounted and unnamed."""
        career_ops, tracker_dir = self._setup(tmp_path, self._row("Initech", "SRE", report="200"))
        (tracker_dir / "7.tsv").write_text(
            "| 11 | 2026-09-01 | Acme Corp | Platform Engineer | Evaluated | 4.7/5 "
            "| null | [229](reports/229-x.md) | note |", encoding="utf-8")
        out, _ = self._run(career_ops, tracker_dir, capsys)
        assert "WARNING: 1 evaluation(s)" in out
        assert "7.tsv (Acme Corp — Platform Engineer)" in out

    def test_report_zero_is_not_an_identity(self, tmp_path, capsys):
        """`000` is what write_job_result writes when it has NO number, and
        merge-tracker's own `if (reportNum && …)` treats 0 as absent. Reading it
        as an identity would let two numberless additions at one company be
        taken for the same evaluation, and report a real loss as a retitle."""
        career_ops, tracker_dir = self._setup(
            tmp_path, self._row("Acme Corp", "Other Role", report="000"))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer", report="000")
        assert "WARNING: 1 evaluation(s)" in self._run(career_ops, tracker_dir, capsys)[0]

    def test_via_layout_is_read_by_header_name(self, tmp_path, capsys):
        """The tracker's optional Via column shifts Role right by one. Read
        positionally the agency lands where the role belongs, and every addition
        looks lost."""
        career_ops, tracker_dir = self._setup(tmp_path, header=(
            "# Applications Tracker\n\n"
            "| # | Date | Company | Via | Role | Score | Status | PDF | Report | Notes |\n"
            "|---|------|---------|-----|------|-------|--------|-----|--------|-------|\n"
            "| 10 | 2026-09-01 | Acme Corp | Hays | Platform Engineer | 4.7/5 | Evaluated | ❌ "
            "| [229](../reports/229-x-2026-09-01.md) | note |\n"))
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "Platform Engineer")
        assert self._run(career_ops, tracker_dir, capsys) == ("", 0)

    def test_a_headed_addition_is_read_by_its_labels(self, tmp_path):
        """career-ops' own worker writes a label row above its data row
        (career-ops#3706). Read positionally the two lines are one row at a
        company called "company" for a role called "role" — neither reading of
        "did it land" can find that, so every such evaluation, landed or not,
        was reported lost under a name that is not a job. By NAME, so a header
        in any order reads the same (the sanitizer declines that one; this
        guard is what still sees it)."""
        _, tracker_dir = self._setup(tmp_path)
        (tracker_dir / "7.tsv").write_text(
            ADDITION_LABEL_LINE + "\n" + "\t".join([
                "11", "2026-09-01", "Acme Corp", "Platform Engineer", "Evaluated", "4.7/5",
                "null", "[229](reports/229-x-2026-09-01.md)", "APPLY — note", "https://x/j/7"]) + "\n",
            encoding="utf-8")
        (tracker_dir / "8.tsv").write_text(
            "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
            "Globex\tSRE\t12\t2026-09-01\t4.2/5\tEvaluated\tnull\t[230](reports/230-x.md)\tnote\n",
            encoding="utf-8")
        assert _pending_additions(tracker_dir) == [
            {"name": "7.tsv", "company": "Acme Corp", "role": "Platform Engineer", "report": "229"},
            {"name": "8.tsv", "company": "Globex", "role": "SRE", "report": "230"},
        ]

    def test_a_headed_file_with_no_single_row_is_still_reported(self, tmp_path, capsys):
        """merge-tracker refuses a headed file with two data rows ("one addition
        per file") and archives it with exit 0, so both evaluations are gone.
        Passed over as unreadable, the guard said nothing about it at all —
        where the positional misreading had at least warned, under a job
        called "role"."""
        career_ops, tracker_dir = self._setup(tmp_path)
        (tracker_dir / "7.tsv").write_text(
            ADDITION_LABEL_LINE + "\n" + _tracker_row() + "\n" + _tracker_row(role="PM") + "\n",
            encoding="utf-8")
        out, _ = self._run(career_ops, tracker_dir, capsys)
        assert "WARNING: 1 evaluation(s)" in out
        assert "7.tsv (no readable row)" in out

    def test_a_row_with_a_company_but_no_role_still_names_the_company(self, tmp_path, capsys):
        """"no readable row" is for a file with no identity at all. A row that
        lost only its role still says whose evaluation went missing."""
        career_ops, tracker_dir = self._setup(tmp_path)
        self._addition(tracker_dir, "7.tsv", "Acme Corp", "")
        assert "7.tsv (Acme Corp — )" in self._run(career_ops, tracker_dir, capsys)[0]

    def test_an_undecodable_addition_does_not_stop_the_guard(self, tmp_path, capsys):
        """An agent CLI on Windows can write cp1252. merge-tracker reads it with
        replacement characters and archives it; a UnicodeDecodeError here then
        stopped every later merge at recovery, which reads merged/ first."""
        career_ops, tracker_dir = self._setup(tmp_path)
        (tracker_dir / "7.tsv").write_bytes("\t".join([
            "11", "2026-09-01", "Société Générale", "Analyst", "Evaluated", "4.7/5", "null",
            "[229](reports/229-x-2026-09-01.md)", "note"]).encode("cp1252"))
        assert "WARNING: 1 evaluation(s)" in self._run(career_ops, tracker_dir, capsys)[0]


class TestExtractReqId:
    """A req id in the Notes column is what stops merge-tracker's fuzzy title
    match folding two distinct requisitions into one row (#154) — its matcher
    drops tokens of ≤3 characters, so `SPECIALIST I` and `SPECIALIST II` compare
    equal. The extractor is deliberately strict: no id is exactly today's
    behaviour, while a WRONG id splits two rows that should fold."""

    @pytest.mark.parametrize("jd,expected", [
        ("Job ID: 88214", "88214"),
        ("Requisition Number R-2291", "R-2291"),
        ("Posting ID 5340", "5340"),
        ("Job Reference Number JR00124259 — apply today", "JR00124259"),
        ("job id 65136", "65136"),                     # case-insensitive
        ("Position Number: 00012345", "00012345"),
        ("Job ID: 88214 … and in the footer, Job ID 88214", "88214"),
        ("Req #1311", "1311"),
        ("Req # 1311", "1311"),            # `#` and the space are both non-word
        ("Position #: 12345", "12345"),
    ])
    def test_reads_a_labelled_id(self, jd, expected):
        assert extract_req_id(jd) == expected

    @pytest.mark.parametrize("jd,expected", [
        ("**Job ID:** 88214", "88214"),
        ("- **Job ID**: 65136", "65136"),
        ("Requisition ID: R\\_1488728", "R_1488728"),
        ("Job ID: JR\\-00124259", "JR-00124259"),
        # Bold closed with no whitespace before the next label. Deleting the
        # asterisks glued `Remote` to `Requisition` and killed the label's
        # leading word boundary — found by measuring a real corpus, where two
        # postings carried their req number three newlines below exactly this.
        ("Location: Remote**Requisition Number\n\n\nR2857 Insurance Service Associate", "R2857"),
    ])
    def test_reads_the_markdown_its_own_cache_is_written_in(self, jd, expected):
        """JobSpy's description_format defaults to MARKDOWN and scrape.py
        forwards it, so an Indeed JD is cached as `**Job ID:** 88214` with `_`
        and `-` backslash-escaped mid-word. LinkedIn's is backfilled as plain
        text by screen.py — so a pattern written for prose worked on one board
        and silently never fired on the other."""
        assert extract_req_id(jd) == expected

    @pytest.mark.parametrize("jd", [
        "Job ID: 5340-Nurse-Practitioner-Days-FT",          # id + slugged title
        "Requisition ID REQ-2026-320610000000000000000000",  # not an id at all
    ])
    def test_an_overlong_token_is_declined_not_truncated(self, jd):
        """A bounded capture truncated instead of declining: this yielded
        `5340-NURSE-PRACTITIONER-` and `REQ-2026-`, and merge-tracker reads the
        latter back as `REQ-2026` — a different string from the one we wrote, and
        one that two different reqs sharing a prefix would both produce. Over-long
        means we cannot tell where the id ends."""
        assert extract_req_id(jd) == ""

    def test_a_pay_grade_is_not_a_requisition(self):
        """In public-sector and healthcare postings — the shape this exists for —
        "Job Code" is a classification shared by every posting of that title. One
        board's copy stating only that, against another stating a real req id,
        would split a cross-board pair."""
        assert extract_req_id("Job Code: 4021 — Insurance Specialist I") == ""

    @pytest.mark.parametrize("jd", [
        "This job 2 of 5 in the series",       # a bare label + a number is prose
        "benefits include a 401k plan",
        "Job Type: Full-time. Remote.",        # `Type` is not a qualifier
        "reference checks required",
        "",
    ])
    def test_prose_does_not_become_an_id(self, jd):
        """merge-tracker's own regex accepts a BARE `job`/`ref` label because a
        human wrote that cell deliberately. A JD is prose we did not write."""
        assert extract_req_id(jd) == ""

    @pytest.mark.parametrize("jd", ["Job ID: TBD", "Requisition Number: Pending"])
    def test_a_labelled_value_with_no_digit_is_not_an_id(self, jd):
        """The digit filter, which every other negative case reaches the regex
        too early to exercise. Without it, "reference number for our EEO policy
        is 11246" yields `FOR`."""
        assert extract_req_id(jd) == ""

    def test_two_different_ids_are_ambiguous_and_yield_nothing(self):
        """Guessing which one is the requisition splits a row that should fold;
        declining leaves today's behaviour untouched."""
        assert extract_req_id("Job ID 123 … Requisition Number ABC9") == ""

    def test_none_of_this_reads_the_boards_own_job_key(self):
        """A `jk=`/`currentJobId` key identifies the POSTING, so a re-post of one
        requisition carries a new one — keying on it would add a tracker row
        every time a listing is re-published."""
        assert extract_req_id("https://www.indeed.com/viewjob?jk=1a2b3c4d5e") == ""
        assert extract_req_id("https://www.linkedin.com/jobs/view/4123456789/") == ""


class TestInjectReqIdIntoNotes:
    def test_leads_the_cell(self):
        """Ahead of the URL: merge-tracker's extractReqNumber takes the FIRST
        match in the cell, so leading with the resolved id makes the reading
        deterministic rather than dependent on the free-text sentence."""
        out = _inject_req_id_into_notes(_tracker_row(notes="https://x/j — APPLY"), "R-2291")
        assert out.split("\t")[-1] == "req R-2291 — https://x/j — APPLY"

    def test_empty_notes(self):
        assert _inject_req_id_into_notes(_tracker_row(notes=""), "5340").split("\t")[-1] == "req 5340"

    def test_no_id_is_a_no_op(self):
        row = _tracker_row(notes="APPLY")
        assert _inject_req_id_into_notes(row, "") == row

    def test_does_not_restate_an_id_the_model_already_wrote(self):
        row = _tracker_row(notes="req 5340 — APPLY")
        assert _inject_req_id_into_notes(row, "5340") == row

    def test_malformed_row_is_left_alone(self):
        assert _inject_req_id_into_notes("a\tb\tc", "5340") == "a\tb\tc"

    def test_a_whitespace_only_row_is_left_alone(self):
        """`_restore_trailing_cells` pads a short row to nine fields, so an empty
        row can reach here WITH the right column count. Writing into it produces
        a half-populated garbage row instead of one visibly left untouched."""
        blank = "\t" * (len(ADDITION_COLUMNS) - 1)
        assert _inject_req_id_into_notes(blank, "5340") == blank

    def test_applied_by_write_job_result(self, tmp_path):
        """On the write path, not merely available — and after the URL, so the
        id ends up ahead of it."""
        reports, tracker = tmp_path / "reports", tmp_path / "tsv"
        reports.mkdir(); tracker.mkdir()
        response = TestWriteJobResult()._make_response(tracker=_tracker_row(notes="APPLY strong match"))
        write_job_result(response,
                         {"id": 3, "url": "https://x/j/3", "jd_text": "Job ID: 88214"},
                         reports, tracker, "2026-08-25")
        notes = _data_cells(tracker / "3.tsv")[8]
        assert notes == "req 88214 — https://x/j/3 — APPLY strong match"



# #191's own case: an agent CLI told to "add a `via` column" puts it where the
# tracker's Via layout has it, after `company`, and writes a bare score. Read by
# the sanitizer's tests and by recovery's.
_VIA_LABELS = "num\tdate\tcompany\tvia\trole\tstatus\tscore\tpdf\treport\tnotes\turl"
_VIA_ROW = ("11\t2026-09-01\tAcme Corp\tHays\tPlatform Engineer | Remote\tEvaluated\t4.2\t"
            "null\t[229](reports/229-acme.md)\tAPPLY\thttps://own/j/9")


class TestSanitizePendingAdditions:
    """career-ops' own evaluators write straight into tracker-additions/, never
    entering Python — so those rows reached a merge with none of the chain
    applied, and an unreadable score cell got the row REFUSED and (before
    career-ops#5862) archived, exit 0, never retried: permanent loss. Since
    career-ops#6062 a row with no label line is refused too, so the pass labels
    one it can vouch for."""

    CLI_ROW = _tracker_row(role="Platform Engineer | Remote", score="4.2",
                           notes="APPLY strong match")

    def _tree(self, tmp_path, rows=None, queued=True, reports=None):
        co = tmp_path / "career-ops"
        additions = co / "batch" / "tracker-additions"
        additions.mkdir(parents=True)
        (co / "batch" / "jds").mkdir()
        for name, body in (reports or {}).items():
            (co / "reports").mkdir(exist_ok=True)
            (co / "reports" / name).write_text(body, encoding="utf-8")
        for name, row in (rows or {"7.tsv": self.CLI_ROW}).items():
            (additions / name).write_text(row + "\n", encoding="utf-8")
            # A JD is cached under a QUEUE id only — that is what the
            # "unknown filename" tests below are about, so it must be absent there.
            if Path(name).stem.isdigit():
                (co / "batch" / "jds" / f"{Path(name).stem}.txt").write_text(
                    "Job ID: 88214", encoding="utf-8")
        if queued:
            (co / "batch" / "batch-input.tsv").write_text(
                "id\turl\tsource\tnotes\n7\thttps://x/j/7\tAcme Corp\tPlatform Engineer\n",
                encoding="utf-8")
        return co, additions

    @staticmethod
    def _cells(additions, name="7.tsv"):
        return _data_cells(additions / name)

    def test_repairs_a_row_no_python_writer_touched(self, tmp_path):
        co, additions = self._tree(tmp_path)
        _sanitize_pending_additions(co, additions)
        cells = self._cells(additions)
        assert cells[3] == "Platform Engineer"          # pipe suffix stripped
        assert cells[5] == "4.2/5"                      # merge-tracker can read it
        assert cells[8] == "req 88214 — https://x/j/7 — APPLY strong match"

    def test_says_so(self, tmp_path, capsys):
        """In a cloud run the log is all the operator has."""
        co, additions = self._tree(tmp_path)
        _sanitize_pending_additions(co, additions)
        assert "sanitized 1 addition" in capsys.readouterr().out

    def test_a_row_already_sanitized_is_left_exactly_alone(self, tmp_path, capsys):
        """The chain runs at two points and must not be two mechanisms — on
        the file as `write_job_result` writes it, label line and padding."""
        done = labelled_addition(_tracker_row(
            score="4.2/5", notes="req 88214 — https://x/j/7 — APPLY strong match"))
        co, additions = self._tree(tmp_path, rows={"7.tsv": done.rstrip("\n")})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == done
        assert "sanitized" not in capsys.readouterr().out

    def test_a_headerless_row_goes_back_under_the_label_line(self, tmp_path, capsys):
        """What an older version of this repo wrote, or Add-Job before the
        writer labelled its rows: on a checkout with career-ops#6062 it sits
        refused in the queue for good. Already sanitized, it is labelled anyway
        — the label line is a repair in its own right — padded, and counted."""
        done = _tracker_row(score="4.2/5", notes="req 88214 — https://x/j/7 — APPLY strong match")
        co, additions = self._tree(tmp_path, rows={"7.tsv": done})
        _sanitize_pending_additions(co, additions)
        assert _label_line(additions / "7.tsv") == ADDITION_LABEL_LINE
        assert self._cells(additions) == done.split("\t") + [""]
        assert "sanitized 1 addition" in capsys.readouterr().out
        once = (additions / "7.tsv").read_bytes()
        _sanitize_pending_additions(co, additions)              # and only once
        assert (additions / "7.tsv").read_bytes() == once
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("row", [
        # A tenth cell that is not a URL — a location, a `via=` tag. Labelled
        # `url`, its value would be dropped (upstream's own refusal says so).
        _tracker_row(score="4.2") + "\tDallas, TX",
        _tracker_row(score="4.2") + "\tvia=Hays",
        # No `[N](path)` Report cell to vouch for the positions: a row one cell
        # short in the middle would be read by name as the link under `pdf`.
        "\t".join(["3", "2026-08-25", "Initech", "SRE", "Evaluated", "4.2", "null", "—", "note"]),
        # Two cells after Notes: only `url` has a label there.
        _tracker_row(score="4.2") + "\thttps://x/j/7\tDallas, TX",
    ])
    def test_a_row_the_labels_cannot_vouch_for_stays_headerless(self, tmp_path, row):
        """Anything but a positional row the chain already addresses is left
        without labels — still repaired, for a merge-tracker that reads it."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": row})
        _sanitize_pending_additions(co, additions)
        assert len(read_addition(additions / "7.tsv").splitlines()) == 1
        assert self._cells(additions)[5] == "4.2/5"

    def test_the_rows_own_trailing_url_is_the_fallback(self, tmp_path):
        """career-ops' batch worker is told to append the URL as a 10th field, so
        a row nothing queued can still carry its own. The QUEUED URL wins when
        there is one — handoff routes by the notes URL, and an agent may have
        written a resolved careers-page URL as the 10th field."""
        co, additions = self._tree(
            tmp_path, rows={"229-acme.tsv": self.CLI_ROW + "\thttps://own/j/9"},
            queued=False)
        _sanitize_pending_additions(co, additions)
        assert "https://own/j/9" in self._cells(additions, "229-acme.tsv")[8]

    def test_a_ten_column_row_gets_its_score_fixed(self, tmp_path):
        """Nine columns is OUR prompt's shape; career-ops tells its own worker to
        write nine PLUS a trailing url, and merge-tracker parses that natively. A
        `== 9` guard skipped the score normalization while the pipe strip still
        rewrote the row — so the repair was COUNTED and the row refused anyway."""
        co, additions = self._tree(
            tmp_path, rows={"7.tsv": self.CLI_ROW + "\thttps://own/j/9"})
        _sanitize_pending_additions(co, additions)
        cells = self._cells(additions)
        assert cells[5] == "4.2/5"
        assert cells[9] == "https://own/j/9"            # extras carried through

    def test_an_unknown_filename_still_gets_the_score_repair(self, tmp_path):
        """career-ops' other writers name additions `{num}-{slug}.tsv`, so the
        job-id lookups find nothing. The repair that does not depend on them must
        still happen."""
        co, additions = self._tree(tmp_path, rows={"229-acme.tsv": self.CLI_ROW},
                                   queued=False)
        _sanitize_pending_additions(co, additions)
        cells = self._cells(additions, "229-acme.tsv")
        assert cells[5] == "4.2/5"
        assert "req " not in cells[8]                  # no JD to read one from

    def test_the_queued_url_wins_over_the_rows_own(self, tmp_path):
        co, additions = self._tree(tmp_path, rows={"7.tsv": self.CLI_ROW + "\thttps://own/j/9"})
        _sanitize_pending_additions(co, additions)
        assert "https://x/j/7" in self._cells(additions)[8]

    def test_a_shifted_wide_row_is_left_alone_and_not_counted(self, tmp_path, capsys):
        """Ten cells with the extra one EARLY — a tab inside the company name. A
        width guard cannot tell it from career-ops' "9 plus a trailing url"; the
        Report cell can. Sanitizing it wrote N/A into the STATUS and a URL into
        the REPORT LINK, and merge-tracker — now seeing one score-shaped cell —
        accepted the garbage it had been refusing, so the loss guard saw it land."""
        shifted = "12\t2026-09-01\tAcme\tInc\tPlatform Engineer\tEvaluated\t4.2/5\tnull\t[12](reports/12-acme.md)\tAPPLY"
        co, additions = self._tree(tmp_path, rows={"12.tsv": shifted})
        _sanitize_pending_additions(co, additions)
        assert (additions / "12.tsv").read_text(encoding="utf-8") == shifted + "\n"
        assert "sanitized" not in capsys.readouterr().out

    @pytest.mark.parametrize("text", [
        labelled_addition(_tracker_row(score="4.2/5", notes="")),       # the row ends in a tab
        ADDITION_LABEL_LINE + "\n" + _tracker_row(score="4.2/5", notes="") + "\n",
    ], ids=["padded", "worker-short"])
    def test_an_empty_notes_cell_is_not_repaired_on_every_merge(self, tmp_path, capsys, text):
        """read_text's strip() eats the trailing tab of an empty Notes cell, so an
        already-correct row read as 8 cells, was padded back to 9, differed from
        what was read, and was rewritten and counted — on every merge, forever.
        Labelled the way `write_job_result` writes it, and as career-ops'
        worker does, one cell short of its labels."""
        co, additions = self._tree(tmp_path, rows={"229-acme.tsv": text.rstrip("\n")},
                                   queued=False)
        _sanitize_pending_additions(co, additions)
        assert (additions / "229-acme.tsv").read_text(encoding="utf-8") == text
        assert "sanitized" not in capsys.readouterr().out

    def test_one_unreadable_row_does_not_hold_up_the_rest(self, tmp_path):
        co, additions = self._tree(tmp_path)
        (additions / "8.tsv").write_bytes(b"11\t2026\tAcme \x96 Inc\tSRE\t...")   # cp1252
        _sanitize_pending_additions(co, additions)
        assert self._cells(additions)[5] == "4.2/5"         # 7.tsv still repaired

    def test_a_pipe_delimited_row_is_left_alone(self, tmp_path):
        """merge-tracker parses that shape natively; every step here splits on
        tabs. Rewriting it would change which of upstream's parsers reads it."""
        piped = "| 11 | 2026-09-01 | Acme Corp | Platform Engineer | Evaluated | 4.2 | null | [229](reports/229-acme.md) | note |"
        co, additions = self._tree(tmp_path, rows={"7.tsv": piped})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == piped + "\n"

    def test_a_dead_report_link_is_pointed_at_the_file_with_its_number(self, tmp_path):
        """A row from a writer that trusted the model's slug (#162): the link
        names no file, but a report with that number exists — the same lookup
        the UI resolves by points the cell at it."""
        co, additions = self._tree(tmp_path, reports={"003-initech-2026-08-25.md": "# report"})
        _sanitize_pending_additions(co, additions)
        assert self._cells(additions)[7] == "[003](reports/003-initech-2026-08-25.md)"

    def test_a_live_report_link_is_left_as_written(self, tmp_path, capsys):
        """Whichever relative shape it has: merge-tracker relativises the cell to
        data/applications.md, so `../reports/` is a valid link too. And the link
        is the one cell that knows which of two same-numbered files was meant."""
        row = labelled_addition(_tracker_row(
            score="4.2/5", notes="req 88214 — https://x/j/7 — APPLY strong match"
        ).replace("[003](reports/003-x.md)", "[003](../reports/003-x.md)"))
        co, additions = self._tree(tmp_path, rows={"7.tsv": row.rstrip("\n")}, reports={
            "003-x.md": "# report", "003-other.md": "# a second file with the number"})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == row
        assert "sanitized" not in capsys.readouterr().out

    def test_a_row_that_lost_its_trailing_notes_tab_gets_the_link_repair_too(self, tmp_path):
        """The 8-cell foreign row is what _restore_trailing_cells exists to
        save; the link repair saw the un-restored row and was the one step
        that skipped it, so the row merged with its dead link for good."""
        eight = "\t".join(_tracker_row(score="4.2", notes="").split("\t")[:-1])
        co, additions = self._tree(tmp_path, rows={"7.tsv": eight},
                                   reports={"003-initech-2026-08-25.md": "# report"})
        _sanitize_pending_additions(co, additions)
        cells = self._cells(additions)
        # Restored to nine, then padded to the label line's width.
        assert len(cells) == len(ADDITION_LABELS)
        assert cells[7] == "[003](reports/003-initech-2026-08-25.md)"

    def test_a_reserved_lock_is_not_a_repair_target(self, tmp_path):
        # `003-RESERVED.md` is career-ops' number lock, not a report.
        co, additions = self._tree(tmp_path, reports={"003-RESERVED.md": '{"pid": 1}'})
        _sanitize_pending_additions(co, additions)
        assert self._cells(additions)[7] == "[003](reports/003-x.md)"

    def test_no_report_with_the_number_leaves_the_link(self, tmp_path):
        co, additions = self._tree(tmp_path)
        _sanitize_pending_additions(co, additions)                  # no reports/ at all
        assert self._cells(additions)[7] == "[003](reports/003-x.md)"

    def test_a_headed_row_is_repaired_under_its_label_line(self, tmp_path):
        """career-ops' own worker is the writer this pass exists for, and since
        career-ops#3706 it writes a label row above its data row. The chain
        repairs the data row; the label line goes back byte-for-byte, because
        it is what merge-tracker resolves the row's fields by."""
        co, additions = self._tree(tmp_path, rows={
            "7.tsv": ADDITION_LABEL_LINE + "\n" + self.CLI_ROW + "\thttps://x/j/7"})
        _sanitize_pending_additions(co, additions)
        text = (additions / "7.tsv").read_text(encoding="utf-8")
        assert text.startswith(ADDITION_LABEL_LINE + "\n")
        assert text.count("\n") == 2                    # still two lines, not one
        cells = text.split("\n")[1].split("\t")
        assert cells[3] == "Platform Engineer"
        assert cells[5] == "4.2/5"
        assert cells[8] == "req 88214 — https://x/j/7 — APPLY strong match"

    def test_a_repaired_headed_file_is_left_alone_on_the_next_pass(self, tmp_path, capsys):
        """The chain runs at two points and must not be two mechanisms — with a
        label line in the file too. A second pass that re-joined the lines
        differently, or counted the file, would rewrite it on every merge."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": ADDITION_LABEL_LINE + "\n" + self.CLI_ROW})
        _sanitize_pending_additions(co, additions)
        once = (additions / "7.tsv").read_bytes()
        # The first pass DID repair it, under its label line — or two identical
        # passes over a file nothing ever touched would pass this too.
        #
        # `splitlines()`, not a `\n` split, and the bytes kept for the pass-two
        # comparison only: `atomic_write_text` translates newlines, so on Windows
        # this file lands CRLF, which every reader here takes in its stride (they
        # split on `\r?\n`, as merge-tracker does) and a byte-exact expectation
        # here does not.
        lines = once.decode("utf-8").splitlines()
        assert lines[0] == ADDITION_LABEL_LINE
        assert lines[1].split("\t")[5] == "4.2/5"
        capsys.readouterr()
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_bytes() == once
        assert capsys.readouterr().out == ""

    def test_a_headed_row_short_of_its_url_cell_gets_one(self, tmp_path):
        """merge-tracker lets a headed row stop before its optional cells, but
        refuses one where a URL sits under `notes` while the `url` cell is
        absent — it reads that as every optional value shifted left. Injecting
        the queued URL into an empty Notes cell makes exactly that shape, so the
        repaired row is padded to the label line's width (the contract test
        drives the refusal it prevents)."""
        co, additions = self._tree(tmp_path, rows={
            "7.tsv": ADDITION_LABEL_LINE + "\n" + _tracker_row(score="4.2/5", notes="")})
        _sanitize_pending_additions(co, additions)
        cells = (additions / "7.tsv").read_text(encoding="utf-8").split("\n")[1].split("\t")
        assert len(cells) == len(ADDITION_LABEL_LINE.split("\t"))
        assert "https://x/j/7" in cells[8] and cells[9] == ""

    def test_a_headed_row_with_a_sentinel_report_is_still_repaired(self, tmp_path):
        """`—` is a Report value merge-tracker documents. `_row_parts` takes a
        nine-cell row as it stands but anchors a longer one on a `[N](path)`
        Report cell — so padding the row to its label line's width BEFORE the
        chain declined every repair on it, and its bare score was refused."""
        sentinel = "\t".join(["3", "2026-08-25", "Initech", "SRE", "Evaluated", "4.2",
                              "null", "—", "note"])
        co, additions = self._tree(tmp_path, rows={"7.tsv": ADDITION_LABEL_LINE + "\n" + sentinel})
        _sanitize_pending_additions(co, additions)
        cells = (additions / "7.tsv").read_text(encoding="utf-8").split("\n")[1].split("\t")
        assert cells[5] == "4.2/5" and len(cells) == len(ADDITION_LABEL_LINE.split("\t"))

    def test_a_headed_row_in_another_order_is_repaired_still_labelled(self, tmp_path, capsys):
        """merge-tracker reads a headed file by NAME, in any order; the chain
        rewrites cells by INDEX. In this order index 5 is the status, so the
        chain run on the row as written would write N/A over `Evaluated` and
        leave the bare `4.2` it was for — and the Report cell, still at index 7,
        gets that row past every shape guard. So this file used to be left as
        written, and its bare score refused for good (#191). It is now put into
        the batch prompt's order BY ITS LABELS first, repaired there, and
        written back labelled — the label line proving the order, the by-name
        read proving each value went with its label."""
        reordered = ("company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
                     "Initech\tPlatform Engineer\t3\t2026-08-25\t4.2\tEvaluated\tnull\t"
                     "[003](reports/003-x.md)\tAPPLY")
        co, additions = self._tree(tmp_path, rows={"7.tsv": reordered})
        _sanitize_pending_additions(co, additions)
        assert _label_line(additions / "7.tsv") == "\t".join(ADDITION_COLUMNS)
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["num"], row["company"], row["role"], row["status"], row["score"]) == (
            "3", "Initech", "Platform Engineer", "Evaluated", "4.2/5")
        assert row["notes"] == "req 88214 — https://x/j/7 — APPLY"
        assert "sanitized 1 addition" in capsys.readouterr().out

    def test_a_via_column_after_company_goes_after_the_nine(self, tmp_path):
        """Every cell moves with its label, and the labels that are not among
        the nine follow them — `url` first, where the chain looks for the
        row's own URL, then the rest (`via`) in the order they were written."""
        co, additions = self._tree(
            tmp_path, rows={"229-acme.tsv": _VIA_LABELS + "\n" + _VIA_ROW}, queued=False)
        _sanitize_pending_additions(co, additions)
        assert _label_line(additions / "229-acme.tsv") == "\t".join(
            [*ADDITION_COLUMNS, "url", "via"])
        assert self._cells(additions, "229-acme.tsv") == [
            "11", "2026-09-01", "Acme Corp", "Platform Engineer", "Evaluated", "4.2/5", "null",
            "[229](reports/229-acme.md)", "https://own/j/9 — APPLY", "https://own/j/9", "Hays"]

    def test_a_reordered_file_is_in_place_on_the_next_pass(self, tmp_path, capsys):
        """Idempotent after the first rewrite: the file is then in the order
        the chain addresses, so a second pass reads it as written and neither
        rewrites nor counts it."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": _VIA_LABELS + "\n" + _VIA_ROW})
        _sanitize_pending_additions(co, additions)
        once = (additions / "7.tsv").read_bytes()
        # The first pass did reorder it, or two passes over an untouched file
        # would pass this too.
        assert once.decode("utf-8").splitlines()[0].startswith("\t".join(ADDITION_COLUMNS))
        assert "sanitized 1 addition" in capsys.readouterr().out
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_bytes() == once
        assert capsys.readouterr().out == ""

    def test_another_order_with_nothing_to_repair_is_not_rewritten(self, tmp_path, capsys):
        """The order is not itself a defect — merge-tracker reads it by name —
        so a file the chain has nothing to do to stays exactly as written."""
        done = _VIA_ROW.replace("\t4.2\t", "\t4.2/5\t").replace(" | Remote", "").replace(
            "\tAPPLY\t", "\treq 88214 — https://x/j/7 — APPLY\t")
        text = _VIA_LABELS + "\n" + done
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    def test_a_missing_notes_label_is_added(self, tmp_path):
        """`notes` is optional to merge-tracker, but it is where the chain
        writes the URL and the req id. Added with its cell, so the `via` and
        `url` cells after it keep their labels."""
        text = ("num\tdate\tcompany\tvia\trole\tstatus\tscore\tpdf\treport\turl\n"
                "3\t2026-08-25\tInitech\tHays\tSRE\tEvaluated\t4.2\tnull\t"
                "[003](reports/003-x.md)\thttps://own/j/9")
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert _label_line(additions / "7.tsv") == "\t".join([*ADDITION_COLUMNS, "url", "via"])
        assert (row["score"], row["via"], row["url"]) == ("4.2/5", "Hays", "https://own/j/9")
        assert row["notes"] == "req 88214 — https://x/j/7"

    def test_a_never_scored_row_keeps_its_order_by_keeping_its_labels(self, tmp_path):
        """A discarded, never-scored row carries `—` as both its score and its
        status, which no content rule can order — the headerless parser refuses
        it, and only a label line says which cell is which (career-ops#3517). So
        the file is reordered, never flattened: still two lines, the pair under
        the labels it was written under."""
        text = ("company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
                "Initech\tSRE\t3\t2026-08-25\t—\t—\tnull\t—\tclosed before scoring")
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        lines = read_addition(additions / "7.tsv").splitlines()
        assert len(lines) == 2 and lines[0] == "\t".join(ADDITION_COLUMNS)
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["company"], row["role"], row["status"], row["score"], row["report"]) == (
            "Initech", "SRE", "—", "—", "—")
        assert row["notes"] == "req 88214 — https://x/j/7 — closed before scoring"

    @pytest.mark.parametrize("text", [
        # No `pdf` label: merge-tracker refuses the header outright.
        "num\tdate\tcompany\tvia\trole\tstatus\tscore\treport\tnotes\n"
        "3\t2026-08-25\tInitech\tHays\tSRE\tEvaluated\t4.2\t[003](reports/003-x.md)\tAPPLY",
        # `company` labelled twice: merge-tracker will not pick which one is it.
        "num\tdate\tcompany\tcompany\trole\tstatus\tscore\tpdf\treport\n"
        "3\t2026-08-25\tInitech\tGlobex\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)",
        # The nine in place and a label repeated after them: refused the same
        # way, so the batch prompt's order is no shortcut past it.
        ADDITION_LABEL_LINE + "\turl\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\tAPPLY\t"
        "https://x/j/7\thttps://x/j/7",
        # The row stops before its required `report` cell.
        "num\tdate\tcompany\tvia\trole\tstatus\tscore\tpdf\treport\tnotes\n"
        "3\t2026-08-25\tInitech\tHays\tSRE\tEvaluated\t4.2\tnull",
        # A pipe row under labels in another order: a guess at its cells.
        "| company | role | num | date | score | status | pdf | report | notes |\n"
        "| Initech | SRE | 3 | 2026-08-25 | 4.2 | Evaluated | null | [003](reports/003-x.md) | APPLY |",
    ], ids=["no-pdf-label", "company-twice", "url-twice-in-place", "short-of-report", "pipe"])
    def test_a_file_merge_tracker_refuses_whatever_is_done_is_left_as_written(
            self, tmp_path, capsys, text):
        """Putting a file in order is only for a file whose labels say where
        every required cell is. One they do not is refused by merge-tracker
        whatever is done to it, and repairing it by a guess would be the
        silent column swap headers exist to prevent."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    @pytest.mark.parametrize("text", [
        # The reviewer's case: labelled in another order, the PDF value left
        # out. merge-tracker lets a row stop early and reads every present
        # cell by its label, so it reads the report link as `pdf` and the note
        # as `report` — and refuses the row only for its bare score. Repaired
        # here, it merged exactly that way.
        "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
        "Initech\tSRE\t3\t2026-08-25\t4.2\tEvaluated\t[003](reports/003-x.md)\tAPPLY",
        # The same hole in the batch prompt's order: the 8-cell row was padded
        # to nine by the chain and taken as it stood.
        ADDITION_LABEL_LINE + "\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\t[003](reports/003-x.md)\tAPPLY",
        # The Report cell is in place, but a required label comes after it and
        # the row is short: the score was left out, `null` reads as the score,
        # and the anchor cannot say so.
        "num\tdate\tcompany\trole\treport\tstatus\tscore\tpdf\tnotes\n"
        "3\t2026-08-25\tInitech\tSRE\t[003](reports/003-x.md)\tEvaluated\tnull\tAPPLY",
    ], ids=["another-order", "prompt-order", "required-after-report"])
    def test_a_row_that_left_out_an_interior_cell_is_left_as_written(
            self, tmp_path, capsys, text):
        """merge-tracker refuses each of these as written, and the chain's
        score repair is the one thing that would let it through with every
        later cell under the wrong label. A Report cell that is not a link (or
        the `—` sentinel) is how a short row says it is short in the middle."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    def test_a_reordered_row_that_stops_before_notes_is_still_repaired(self, tmp_path):
        """The short row merge-tracker means to allow: only the last, optional
        cell left out, so the Report cell is still a link and every required
        cell is ahead of it."""
        text = ("company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
                "Initech\tSRE\t3\t2026-08-25\t4.2\tEvaluated\tnull\t[003](reports/003-x.md)")
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["pdf"], row["report"], row["score"]) == (
            "null", "[003](reports/003-x.md)", "4.2/5")
        assert row["notes"] == "req 88214 — https://x/j/7"

    @pytest.mark.parametrize("text", [
        # The reviewer's case: a `location` label before `notes`, and the URL
        # under `notes` with no `url` cell.
        "num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\tlocation\tnotes\turl\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\t"
        "Dallas\thttps://x/j/7",
        # The batch prompt's order: the same refusal, the same answer.
        ADDITION_LABEL_LINE + "\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\t"
        "HTTPS://x/j/7",
    ], ids=["another-order", "prompt-order"])
    def test_a_bare_url_under_another_label_with_no_url_cell_is_left_as_written(
            self, tmp_path, capsys, text):
        """merge-tracker refuses this on purpose: a URL under a label that is
        not `url`, while the `url` cell is absent, means the optional cells may
        have shifted one left. Repaired, the row would go out padded — its
        `url` cell present and empty — and merge on our guess at which cell
        was left out."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    def test_the_same_url_with_its_url_cell_written_empty_is_repaired(self, tmp_path):
        """What merge-tracker's refusal asks a writer for — "write an empty
        cell for each optional value you omit" — so the row it then reads by
        name is repaired here too."""
        text = ("num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\tlocation\tnotes\turl\n"
                "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\t"
                "Dallas\thttps://x/j/7\t")
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        # By the labels as written, not `headed_addition_row`: `location` is in
        # career-ops' alias table and not in the fallback, and either way it is
        # a label after the nine.
        labels = _label_line(additions / "7.tsv").split("\t")
        row = dict(zip(labels, self._cells(additions)))
        assert labels == [*ADDITION_COLUMNS, "url", "location"]
        assert (row["score"], row["location"], row["url"]) == ("4.2/5", "Dallas", "")
        assert row["notes"] == "req 88214 — https://x/j/7"

    @pytest.mark.parametrize("label", ["num", "date", "company", "role", "status", "report"])
    @pytest.mark.parametrize("labels", [
        ("company", "role", "num", "date", "score", "status", "pdf", "report", "notes"),
        ADDITION_COLUMNS,
    ], ids=["another-order", "prompt-order"])
    def test_a_blank_required_cell_with_no_value_to_give_it_is_left_as_written(
            self, tmp_path, capsys, label, labels):
        """merge-tracker refuses a required cell present but blank, whatever
        else is repaired, so rewriting the file only to have it refused would
        count a repair that is not one — and recovery would retry it. The rule
        is `labelled_addition`'s, for the rows this repo labels."""
        values = dict(zip(ADDITION_COLUMNS, _tracker_row(score="4.2").split("\t")))
        values[label] = ""
        text = "\t".join(labels) + "\n" + "\t".join(values[name] for name in labels)
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    @pytest.mark.parametrize("labels", [
        ("company", "role", "num", "date", "score", "status", "pdf", "report", "notes"),
        ADDITION_COLUMNS,
    ], ids=["another-order", "prompt-order"])
    def test_a_blank_pdf_gets_what_the_prompt_asks_for(self, tmp_path, capsys, labels):
        """The one blank required cell that has a value to give it, as
        `labelled_addition` gives it — in either order — and a repair in its
        own right, so it is written and counted even with nothing else to do."""
        values = dict(zip(ADDITION_COLUMNS, _tracker_row(notes="https://x/j/7").split("\t")))
        values["pdf"] = " "
        text = "\t".join(labels) + "\n" + "\t".join(values[name] for name in labels)
        co, additions = self._tree(tmp_path, rows={"229-acme.tsv": text}, queued=False)
        _sanitize_pending_additions(co, additions)
        row = headed_addition_row(read_addition(additions / "229-acme.tsv"))
        assert (row["pdf"], row["score"], row["notes"]) == (
            _PROMPT_PDF_CELL, "4.2/5", "https://x/j/7")
        assert "sanitized 1 addition" in capsys.readouterr().out

    def test_a_blank_score_in_another_order_is_the_chains_n_a(self, tmp_path):
        """As in a headerless row: the chain writes the `N/A` sentinel into a
        score cell with no number, so a blank one is not a reason to decline."""
        text = ("company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
                "Initech\tSRE\t3\t2026-08-25\t\tEvaluated\tnull\t[003](reports/003-x.md)\tAPPLY")
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert headed_addition_row(read_addition(additions / "7.tsv"))["score"] == "N/A"

    @pytest.mark.parametrize("text", [
        # A `via` label after Report, its empty cell left out: nine cells, and
        # merge-tracker reads the note as the agency. The Report link is in
        # place, no required label follows it and no URL is misplaced, so
        # nothing else here sees it.
        "num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\tvia\tnotes\turl\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\tAPPLY",
        # The same with `location`: career-ops' alias table knows it and the
        # fallback does not, and it is declined either way.
        "num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\tlocation\tnotes\turl\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\tAPPLY",
        # `url` before `notes`, the url left out: the note is read as the URL,
        # and it is under `url`, so the misplaced-URL rule cannot see it.
        "num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\turl\tnotes\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\tAPPLY",
        # An unrecognised label there is no better: merge-tracker drops its
        # cell, and which cell that was is the same guess.
        "num\tdate\tcompany\trole\tstatus\tscore\tpdf\treport\tsource\tnotes\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\tAPPLY",
        # The `via` case in another order.
        "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tvia\tnotes\n"
        "Initech\tSRE\t3\t2026-08-25\t4.2\tEvaluated\tnull\t[003](reports/003-x.md)\tAPPLY",
    ], ids=["via", "location", "url-before-notes", "unrecognised", "via-another-order"])
    def test_a_short_row_with_another_label_after_report_is_left_as_written(
            self, tmp_path, capsys, text):
        """A row may stop before its optional cells, and merge-tracker reads
        every present cell by its label — so a short row whose labels after
        the Report cell are anything but the batch prompt's own tail (`notes`,
        then `url`) may have left out an INTERIOR optional cell, and is then
        read one label off from it. The shift never reaches the score, so
        merge-tracker refuses the row only for the bare score the chain would
        repair: repaired, it merged with the note under `via`."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    @pytest.mark.parametrize("text", [
        # The batch prompt's labels, the row stopping before `url`.
        ADDITION_LABEL_LINE + "\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)\tAPPLY",
        # ...and before `notes`: both trailing cells left out, in order.
        ADDITION_LABEL_LINE + "\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)",
        # No `notes` label, the row stopping before `url`: the only cell it
        # can have left out.
        "\t".join([*ADDITION_COLUMNS[:-1], "url"]) + "\n"
        "3\t2026-08-25\tInitech\tSRE\tEvaluated\t4.2\tnull\t[003](reports/003-x.md)",
        # Another order before the Report cell, the prompt's tail after it.
        "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\turl\n"
        "Initech\tSRE\t3\t2026-08-25\t4.2\tEvaluated\tnull\t[003](reports/003-x.md)\tAPPLY",
    ], ids=["no-url", "no-notes-no-url", "no-notes-label", "another-order"])
    def test_a_short_row_with_the_prompts_tail_after_report_is_repaired(self, tmp_path, text):
        """The short row merge-tracker means to allow: its labels after the
        Report cell are `notes` then `url`, or one of them, so the only cells
        it can have left out are its last ones, and the misplaced-URL rule
        settles the one ambiguity left (a URL where the note should be)."""
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["report"], row["score"]) == ("[003](reports/003-x.md)", "4.2/5")
        assert row["notes"].startswith("req 88214 — https://x/j/7")

    @pytest.mark.parametrize("labels", [
        ADDITION_LABELS,
        ("company", "role", "num", "date", "score", "status", "pdf", "report", "notes", "url"),
    ], ids=["prompt-order", "another-order"])
    def test_a_blank_last_label_does_not_make_a_whole_row_short(self, tmp_path, labels):
        """A label line ending in a tab has a blank last label, which
        merge-tracker skips. Short is counted as it counts it — against the
        labels it recognises — so a row with a cell for every one of them is
        whole, and repaired, however its labels after the Report cell run."""
        values = dict(zip(ADDITION_LABELS, [*_tracker_row(score="4.2").split("\t"),
                                            "https://own/j/9"]))
        text = "\t".join(labels) + "\t\n" + "\t".join(values[name] for name in labels)
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["score"], row["url"]) == ("4.2/5", "https://own/j/9")
        assert row["notes"] == "req 88214 — https://x/j/7 — note"

    @pytest.mark.parametrize("text", [
        # Stops before its optional `notes` cell.
        "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
        "Initech\tSRE\t3\t2026-08-25\t4.2/5\tEvaluated\tnull\t[003](reports/003-x.md)",
        # Carries no `notes` label at all.
        "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\n"
        "Initech\tSRE\t3\t2026-08-25\t4.2/5\tEvaluated\tnull\t[003](reports/003-x.md)",
    ], ids=["notes-left-out", "no-notes-label"])
    def test_another_order_with_no_notes_cell_and_nothing_to_repair_is_not_rewritten(
            self, tmp_path, capsys, text):
        """Put in order, such a row has no Notes cell, and the chain's first
        step gives it an empty one — which is the chain normalising the row,
        not repairing it. Counted as a repair, the file was rewritten in the
        batch prompt's order with nothing done to it."""
        co, additions = self._tree(tmp_path, rows={"229-acme.tsv": text}, queued=False)
        _sanitize_pending_additions(co, additions)
        assert (additions / "229-acme.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    @pytest.mark.parametrize("labels", [
        ADDITION_LABELS,
        ("num", "date", "company", "via", "role", "status", "score", "pdf", "report", "notes",
         "url"),
    ], ids=["prompt-order", "via-after-company"])
    def test_a_row_wider_than_nine_with_the_report_sentinel_gets_the_whole_chain(
            self, tmp_path, labels):
        """The chain's own guard takes a row wider than nine only when its
        Report cell is a `[N](path)` link — for a headerless row, the anchor
        that says its nine are in place. A headed row's labels have already
        said so, so its nine are what the chain is given: the batch prompt's
        own ten-cell row with the `—` sentinel used to get only its PDF
        filled, with no score fix, no URL and no req id."""
        values = {"num": "3", "date": "2026-08-25", "company": "Initech", "via": "Hays",
                  "role": "SRE | Remote", "status": "Discarded", "score": "4.2", "pdf": "null",
                  "report": "—", "notes": "closed", "url": "https://own/j/9"}
        text = "\t".join(labels) + "\n" + "\t".join(values[name] for name in labels)
        co, additions = self._tree(tmp_path, rows={"7.tsv": text}, queued=False)
        _sanitize_pending_additions(co, additions)
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["role"], row["score"], row["report"], row["url"]) == (
            "SRE", "4.2/5", "—", "https://own/j/9")
        assert row["notes"] == "req 88214 — https://own/j/9 — closed"
        if "via" in labels:
            assert row["via"] == "Hays"

    @pytest.mark.parametrize("labels", [
        (*ADDITION_COLUMNS, "apply link", "url"),
        ("company", "role", "num", "date", "score", "status", "pdf", "report", "notes",
         "apply link", "url"),
    ], ids=["prompt-order", "another-order"])
    def test_the_url_cell_is_the_url_the_row_carries_into_notes(self, tmp_path, labels):
        """With no queued URL, a row's own URL goes into Notes, where handoff
        routes the role by it — and of a row's cells after the nine the first
        URL is taken. An `apply link` written before `url` is a careers page,
        which would send the role to the 'other' site session, so `url` is put
        first among them."""
        values = dict(zip(ADDITION_COLUMNS, _tracker_row(score="4.2").split("\t")))
        values.update({"apply link": "https://careers.initech.example/apply/9",
                       "url": "https://www.linkedin.com/jobs/view/9"})
        text = "\t".join(labels) + "\n" + "\t".join(values[name] for name in labels)
        co, additions = self._tree(tmp_path, rows={"229-acme.tsv": text}, queued=False)
        _sanitize_pending_additions(co, additions)
        assert _label_line(additions / "229-acme.tsv") == "\t".join(
            [*ADDITION_COLUMNS, "url", "apply link"])
        assert self._cells(additions, "229-acme.tsv")[ADDITION_COLUMNS.index("notes")] == (
            "https://www.linkedin.com/jobs/view/9 — note")

    @pytest.mark.parametrize("labels", [
        (*ADDITION_COLUMNS, "apply link", "url"),
        ("company", "role", "num", "date", "score", "status", "pdf", "report", "notes",
         "url", "apply link"),
    ], ids=["prompt-order-apply-link-first", "another-order"])
    def test_an_empty_url_cell_is_no_url_not_the_apply_link(self, tmp_path, labels):
        """When the header labels `url`, that cell alone is the row's own URL:
        empty means none, not the careers page another column carries."""
        values = dict(zip(ADDITION_COLUMNS, _tracker_row(score="4.2").split("\t")))
        values.update({"apply link": "https://careers.initech.example/apply/9", "url": ""})
        text = "\t".join(labels) + "\n" + "\t".join(values[name] for name in labels)
        co, additions = self._tree(tmp_path, rows={"229-acme.tsv": text}, queued=False)
        _sanitize_pending_additions(co, additions)
        notes = headed_addition_row(read_addition(additions / "229-acme.tsv"))["notes"]
        assert "careers.initech" not in notes and notes == "note"

    @pytest.mark.parametrize("labels", [
        ADDITION_LABELS,
        ("company", "role", "num", "date", "score", "status", "pdf", "report", "notes", "url"),
    ], ids=["prompt-order", "another-order"])
    def test_a_cell_past_the_last_label_is_left_as_written(self, tmp_path, labels):
        """A tab inside Notes gives one cell more than the labels — the mirror
        of a short row: which label each cell after the Report belongs to is a
        guess, so the file is left for merge-tracker to refuse, not repaired."""
        values = dict(zip(ADDITION_COLUMNS, _tracker_row(score="4.2").split("\t")))
        values.update({"notes": "note a\tnote b", "url": "https://www.linkedin.com/jobs/view/1"})
        text = "\t".join(labels) + "\n" + "\t".join(values[name] for name in labels)
        co, additions = self._tree(tmp_path, rows={"229-acme.tsv": text})
        before = (additions / "229-acme.tsv").read_bytes()
        _sanitize_pending_additions(co, additions)
        assert (additions / "229-acme.tsv").read_bytes() == before

    def test_a_blank_line_under_the_labels_is_not_a_repair(self, tmp_path, capsys):
        """merge-tracker drops blank lines before it looks, so a blank line
        between the label row and the data row is nothing. Writing the file
        back drops it here too — so comparing the re-joined FILE rather than
        the row would count a row that needed nothing as repaired, and rewrite
        it on every merge."""
        done = _tracker_row(score="4.2/5", notes="req 88214 — https://x/j/7 — APPLY strong match")
        text = ADDITION_LABEL_LINE + "\n\n" + done + "\thttps://x/j/7"
        co, additions = self._tree(tmp_path, rows={"7.tsv": text})
        _sanitize_pending_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == text + "\n"
        assert "sanitized" not in capsys.readouterr().out

    def test_run_merge_tracker_wires_it_in(self, tmp_path, mocker):
        """The one test that goes through the merge: sanitizing must happen, and
        must happen BEFORE the loss guard's snapshot, since stripping a role's
        pipe suffix changes half the identity that guard matches on."""
        co, additions = self._tree(tmp_path)
        (co / "data").mkdir(parents=True)
        (co / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        mocker.patch("pipeline._batch_common.subprocess.run",
                     return_value=mocker.MagicMock(returncode=0, stdout="", stderr=""))
        run_merge_tracker(co)
        assert self._cells(additions)[5] == "4.2/5"


class TestSanitizeAddition:
    @pytest.mark.parametrize("raw", [
        _tracker_row(score="4.2", role="Platform Engineer | Remote", notes="APPLY"),
        _tracker_row(score="4.0/5"),                         # first pass rewrites to 4/5
        _tracker_row(score="4.2", notes=""),                 # empty notes
        _tracker_row(score="4.2", notes="APPLY") + "\tHTTPS://X/J/7",   # upper-case scheme
        _tracker_row(score="4.2", notes="APPLY") + "\tvia=Hays\thttps://x/j",
    ])
    def test_is_idempotent(self, raw):
        """What lets the chain run on the write path AND at the merge without
        being two mechanisms: sanitize(sanitize(x)) == sanitize(x), over the
        shapes the chain actually meets."""
        once = sanitize_addition(raw, "https://x/j", "Job ID: 88214")
        assert sanitize_addition(once, "https://x/j", "Job ID: 88214") == once

    def test_declines_a_row_whose_columns_are_not_in_place(self):
        assert sanitize_addition(_SHIFTED_ROW, "https://x/j", "Job ID: 5") == _SHIFTED_ROW
        assert sanitize_addition(_SHIFTED_ROW, "https://x/j", "Job ID: 5",
                                 report_file="12-acme-2026-09-01.md") == _SHIFTED_ROW

    def test_is_idempotent_with_a_report_file(self):
        raw = _tracker_row(score="4.2", role="SRE | Remote") + "\thttps://own/j/9"
        once = sanitize_addition(raw, "https://x/j", "Job ID: 88214", report_file="003-initech-2026.md")
        assert "[003](reports/003-initech-2026.md)" in once
        assert sanitize_addition(once, "https://x/j", "Job ID: 88214",
                                 report_file="003-initech-2026.md") == once


class TestSetReportLink:
    """The Report cell is `[N](reports/N-slug-DATE.md)` with N the FILE's own
    leading number, so the two cannot disagree. Empty means nothing known."""

    def test_points_the_cell_at_the_file(self):
        out = _set_report_link(_tracker_row(), "003-initech-2026-08-25.md")
        assert out.split("\t")[7] == "[003](reports/003-initech-2026-08-25.md)"

    def test_keeps_trailing_extras(self):
        out = _set_report_link(_tracker_row() + "\thttps://own/j/9", "003-i.md")
        assert out.split("\t")[9] == "https://own/j/9"

    def test_empty_or_unnumbered_file_leaves_the_row(self):
        assert _set_report_link(_tracker_row(), "") == _tracker_row()
        assert _set_report_link(_tracker_row(), "notes.md") == _tracker_row()

    def test_declines_a_shifted_row(self):
        assert _set_report_link(_SHIFTED_ROW, "12-acme.md") == _SHIFTED_ROW


class TestDeadLinkRepair:
    """`_dead_link_repair` takes the row's already-resolved report, because
    `_sanitize_pending_additions` resolves it once and shares it with
    `_pending_report_marks` — a dead link sends `resolve_report` through a full
    `reports/` glob, and two independent callers meant two scans per addition.
    These drive the real composition rather than a two-argument shorthand."""

    def test_dead_link_resolves_by_the_filename_number(self, tmp_path):
        co = _reports(tmp_path, "003-initech-2026.md")
        assert _dead_link_repair(co, _tracker_row(), _row_report(co, _tracker_row())) == "003-initech-2026.md"

    def test_padding_of_the_number_is_tolerated(self, tmp_path):
        co = _reports(tmp_path, "3-initech-2026.md")
        assert _dead_link_repair(co, _tracker_row(), _row_report(co, _tracker_row())) == "3-initech-2026.md"

    def test_live_link_is_not_repaired(self, tmp_path):
        co = _reports(tmp_path, "003-x.md", "003-y.md")
        assert _dead_link_repair(co, _tracker_row(), _row_report(co, _tracker_row())) == ""

    def test_link_text_is_the_fallback_number(self, tmp_path):
        # A slug so mangled it lost its numeric prefix: the `[N]` text still says.
        co = _reports(tmp_path, "003-initech-2026.md")
        row = _tracker_row().replace("[003](reports/003-x.md)", "[003](reports/initech.md)")
        assert _dead_link_repair(co, row, _row_report(co, row)) == "003-initech-2026.md"

    def test_shifted_row_declined(self, tmp_path):
        co = _reports(tmp_path, "12-a.md")
        shifted = _SHIFTED_ROW.replace("12-a.md", "12-b.md")      # a dead link, so only the shape declines it
        assert _dead_link_repair(co, shifted, _row_report(co, shifted)) == ""

    def test_two_reports_with_the_number_resolve_by_the_rows_company(self, tmp_path):
        """A synced cloud report and a local-only one can share a number; the
        first by filename order was another company's evaluation, and this
        pass would have written that binding into the tracker for good."""
        co = _reports(tmp_path, "042-globex-2026-05-27.md", "042-zeta-corp-2026-08-25.md")
        row = (_tracker_row().replace("Initech", "Zeta Corp")
               .replace("[003](reports/003-x.md)", "[042](reports/042-zeta-&-corp-2026-08-25.md)"))
        assert _dead_link_repair(co, row, _row_report(co, row)) == "042-zeta-corp-2026-08-25.md"

    def test_two_reports_with_the_number_and_no_company_match_are_declined(self, tmp_path):
        co = _reports(tmp_path, "042-globex-2026-05-27.md", "042-acme-2026-08-25.md")
        row = (_tracker_row().replace("Initech", "Zeta Corp")
               .replace("[003](reports/003-x.md)", "[042](reports/042-zeta-corp.md)"))
        assert _dead_link_repair(co, row, _row_report(co, row)) == ""                   # a guess is worse than a dead link

    def test_link_text_wins_over_a_disagreeing_filename_prefix(self, tmp_path):
        # `[N]` is the row's identity to the UI, row_report_num and the loss
        # guard; a dead path whose prefix says otherwise does not repoint the row.
        co = _reports(tmp_path, "271-acme-2026.md", "270-globex-2026.md")
        row = _tracker_row().replace("[003](reports/003-x.md)", "[271](reports/270-acme-&-co.md)")
        assert _dead_link_repair(co, row, _row_report(co, row)) == "271-acme-2026.md"

    def test_a_row_that_lost_its_trailing_notes_tab_is_still_repaired(self, tmp_path):
        co = _reports(tmp_path, "003-initech-2026.md")
        eight_cells = "\t".join(_tracker_row(notes="").split("\t")[:-1])
        assert _dead_link_repair(co, eight_cells, _row_report(co, eight_cells)) == "003-initech-2026.md"


class TestReadReport:
    """The readers — cover letters and both tailors — resolved `base / link`
    and got "" from read_text on a dead link, building from the JD alone with
    no log line (#162). Now they resolve like the UI, and say so."""

    def test_live_link_reads_directly_and_quietly(self, tmp_path, capsys):
        co = _reports(tmp_path, **{"271-baylor-2026.md": "PROOF"})
        assert read_report(co, "reports/271-baylor-2026.md") == "PROOF"
        assert capsys.readouterr().out == ""

    def test_ascent_form_of_a_live_link_reads_directly(self, tmp_path, capsys):
        co = _reports(tmp_path, **{"271-baylor-2026.md": "PROOF"})
        assert read_report(co, "../reports/271-baylor-2026.md") == "PROOF"
        assert capsys.readouterr().out == ""

    def test_dead_link_resolves_by_number_and_says_so(self, tmp_path, capsys):
        co = _reports(tmp_path, **{"271-baylor-scott-white-health-2026-09-03.md": "PROOF"})
        text = read_report(co, "reports/271-baylor-scott-&-white-health-2026-09-03.md", label="build")
        assert text == "PROOF"
        out = capsys.readouterr().out
        assert out.startswith("[build]") and "271-baylor-scott-white-health-2026-09-03.md" in out

    def test_missing_report_warns_and_returns_empty(self, tmp_path, capsys):
        co = _reports(tmp_path)
        assert read_report(co, "reports/999-gone.md", label="cover") == ""
        assert "[cover] report reports/999-gone.md not found" in capsys.readouterr().out

    def test_the_rows_number_resolves_a_dead_link_that_lost_its_own(self, tmp_path, capsys):
        # A slug so mangled it lost its numeric prefix: the UI and the merge-time
        # repair resolve it from the row's `[N]`; the readers can too, given it.
        co = _reports(tmp_path, **{"271-acme-2026.md": "PROOF"})
        assert read_report(co, "reports/acme.md", num_text="271") == "PROOF"
        assert "is dead — using 271-acme-2026.md" in capsys.readouterr().out
        assert read_report(co, "reports/acme.md") == ""

    def test_two_reports_with_the_number_read_the_companys_or_nothing(self, tmp_path, capsys):
        co = _reports(tmp_path, **{"042-globex-2026-05-27.md": "GLOBEX",
                                   "042-zeta-corp-2026-08-25.md": "ZETA"})
        dead = "reports/042-zeta-&-corp-2026-08-25.md"
        assert read_report(co, dead, company="Zeta Corp") == "ZETA"
        assert read_report(co, dead) == ""                        # no company to ask: not Globex's
        assert "not found" in capsys.readouterr().out

    def test_lock_is_not_a_report(self, tmp_path):
        co = _reports(tmp_path, **{"271-RESERVED.md": '{"pid": 1}'})
        assert resolve_report(co, "reports/271-baylor.md") is None

    def test_empty_path_is_silent(self, tmp_path, capsys):
        assert read_report(tmp_path, "") == ""
        assert capsys.readouterr().out == ""

    def test_find_report_file_is_the_shared_lookup(self, tmp_path):
        # app/data re-exports it; one implementation for the UI and the readers.
        from pipeline.app import data
        assert data.find_report_file is find_report_file



class TestRecoverRefusedAdditions:
    """The half of #156 the merge-time sanitizer cannot reach: `--batch` runs
    career-ops' own runner, whose last step is its OWN merge-tracker call, so a
    row with a bare `4.2` is refused and archived into merged/ before Python
    runs. From there nothing retries it. This pulls it back — but ONLY a row
    that did not land AND whose score fix would change it."""

    HEADER = ("# Applications Tracker\n\n"
              "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
              "|---|------|---------|------|-------|--------|-----|--------|-------|\n")

    def _tree(self, tmp_path, merged_rows, tracker_rows=""):
        co = tmp_path / "career-ops"
        (co / "data").mkdir(parents=True)
        (co / "data" / "applications.md").write_text(self.HEADER + tracker_rows, encoding="utf-8")
        additions = co / "batch" / "tracker-additions"
        (additions / "merged").mkdir(parents=True)
        (co / "batch" / "jds").mkdir()
        for name, row in merged_rows.items():
            (additions / "merged" / name).write_text(row + "\n", encoding="utf-8")
        return co, additions

    def test_a_refused_row_with_a_bad_score_comes_back_repaired(self, tmp_path, capsys):
        co, additions = self._tree(tmp_path, {"7.tsv": _tracker_row(score="4.2")})
        _recover_refused_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8").split("\t")[5] == "4.2/5"
        assert "recovered 1 evaluation" in capsys.readouterr().out

    def test_a_row_that_landed_is_left_in_the_archive(self, tmp_path):
        """Landed by either reading — including a #152 retitle, found under
        company + report number."""
        landed = ("| 10 | 2026-08-25 | Initech | SRE (Remote) | 4.2/5 | Evaluated | ❌ "
                  "| [003](../reports/003-x.md) | note |\n")
        co, additions = self._tree(tmp_path, {"7.tsv": _tracker_row(score="4.2")}, landed)
        _recover_refused_additions(co, additions)
        assert not (additions / "7.tsv").exists()

    def test_a_row_refused_for_a_reason_we_cannot_fix_is_not_retried(self, tmp_path, capsys):
        """A readable score means the refusal was something else — a report
        number marked `failed`, which upstream refuses on purpose. Re-queuing it
        would have it refused again on every run. This rule is also what keeps a
        row the user DELETED from the tracker from coming back."""
        co, additions = self._tree(tmp_path, {"7.tsv": _tracker_row(score="4.2/5")})
        _recover_refused_additions(co, additions)
        assert not (additions / "7.tsv").exists()
        assert "recovered" not in capsys.readouterr().out

    def test_a_row_already_back_in_the_queue_is_not_clobbered(self, tmp_path):
        co, additions = self._tree(tmp_path, {"7.tsv": _tracker_row(score="4.2")})
        (additions / "7.tsv").write_text("in progress", encoding="utf-8")
        _recover_refused_additions(co, additions)
        assert (additions / "7.tsv").read_text(encoding="utf-8") == "in progress"

    def test_a_refused_headed_row_comes_back_under_its_label_line(self, tmp_path, capsys):
        """career-ops' worker writes a label row above its data row since
        career-ops#3706, and merge-tracker refuses a headed row whose `score`
        cell reads `4.2` exactly as it refuses a headerless one — then
        archives it. The score rule has to read the DATA row: asked of the
        whole two-line file it sees no score to fix, and nothing headed would
        ever come back. The label line is kept, since merge-tracker resolves
        the row's fields by it."""
        co, additions = self._tree(
            tmp_path, {"7.tsv": ADDITION_LABEL_LINE + "\n" + _tracker_row(score="4.2")})
        _recover_refused_additions(co, additions)
        lines = (additions / "7.tsv").read_text(encoding="utf-8").split("\n")
        assert lines[0] == ADDITION_LABEL_LINE
        assert lines[1].split("\t")[5] == "4.2/5"
        assert "recovered 1 evaluation" in capsys.readouterr().out

    def test_a_refused_headed_row_in_another_order_comes_back_in_order(self, tmp_path, capsys):
        """#191: merge-tracker refuses the bare score of a file labelled in any
        order, by name. Recovery read only the batch prompt's order, so a
        `via` column after `company` kept the refusal in the archive for good;
        it now comes back put in order, still labelled, the score repaired."""
        co, additions = self._tree(tmp_path, {"7.tsv": _VIA_LABELS + "\n" + _VIA_ROW})
        _recover_refused_additions(co, additions)
        assert _label_line(additions / "7.tsv") == "\t".join([*ADDITION_COLUMNS, "url", "via"])
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["company"], row["role"], row["score"], row["via"]) == (
            "Acme Corp", "Platform Engineer", "4.2/5", "Hays")
        assert "recovered 1 evaluation" in capsys.readouterr().out

    @pytest.mark.parametrize("text", [
        _VIA_LABELS + "\n" + _VIA_ROW.replace("\tEvaluated\t", "\t\t"),     # a blank status
        # The PDF value left out: the 8-cell row, put in order, is padded to
        # nine by the chain and taken as it stands.
        "company\trole\tnum\tdate\tscore\tstatus\tpdf\treport\tnotes\n"
        "Acme Corp\tPlatform Engineer\t11\t2026-09-01\t4.2\tEvaluated\t"
        "[229](reports/229-acme.md)\tAPPLY",
    ], ids=["blank-status", "pdf-left-out"])
    def test_a_headed_row_refused_whatever_is_repaired_is_not_retried(
            self, tmp_path, capsys, text):
        """Its bare score would change — the rule that lets a row back — but
        merge-tracker refuses a blank status whatever is repaired, and a row
        that left out its PDF would merge with its report link under `pdf`.
        Neither is the sanitizer's to touch, so neither comes back."""
        co, additions = self._tree(tmp_path, {"7.tsv": text})
        _recover_refused_additions(co, additions)
        assert not (additions / "7.tsv").exists()
        assert "recovered" not in capsys.readouterr().out

    def test_a_refused_headed_row_with_the_report_sentinel_comes_back(self, tmp_path, capsys):
        """The batch prompt's ten-cell row with `—` for its Report. The chain's
        own guard declines a row that wide without a `[N](path)` link, so its
        bare score read as one the fix would not change and it stayed in the
        archive; its labels have placed its nine, which is what the score rule
        is now asked of."""
        text = (ADDITION_LABEL_LINE + "\n"
                "3\t2026-08-25\tInitech\tSRE\tDiscarded\t4.2\tnull\t—\tclosed\thttps://own/j/9")
        co, additions = self._tree(tmp_path, {"7.tsv": text})
        _recover_refused_additions(co, additions)
        assert _label_line(additions / "7.tsv") == ADDITION_LABEL_LINE
        row = headed_addition_row(read_addition(additions / "7.tsv"))
        assert (row["score"], row["report"], row["url"]) == ("4.2/5", "—", "https://own/j/9")
        assert row["notes"] == "https://own/j/9 — closed"
        assert "recovered 1 evaluation" in capsys.readouterr().out

    def test_a_headed_row_with_a_readable_score_is_not_retried(self, tmp_path, capsys):
        # The readable-score rule holds through a label line too: refused with
        # a score merge-tracker can read is refused for a reason we cannot fix.
        co, additions = self._tree(
            tmp_path, {"7.tsv": ADDITION_LABEL_LINE + "\n" + _tracker_row(score="4.2/5")})
        _recover_refused_additions(co, additions)
        assert not (additions / "7.tsv").exists()
        assert "recovered" not in capsys.readouterr().out

    def test_no_archive_is_fine(self, tmp_path):
        co = tmp_path / "career-ops"
        (co / "data").mkdir(parents=True)
        _recover_refused_additions(co, co / "batch" / "tracker-additions")

    def test_a_refusal_kept_in_the_queue_leaves_nothing_to_recover(self, tmp_path, capsys):
        """Since career-ops#5862 a refused row stays in the queue and only an
        APPLIED one is archived — so merged/ holds a row that landed, and the
        refused one is the sanitizer's on the next merge, not recovery's.
        Recovery must neither re-queue the applied row nor touch the kept one."""
        landed = ("| 10 | 2026-08-25 | Initech | SRE | 4.2/5 | Evaluated | ❌ "
                  "| [003](../reports/003-x.md) | note |\n")
        co, additions = self._tree(tmp_path, {"6.tsv": _tracker_row(score="4.2/5")}, landed)
        kept = ADDITION_LABEL_LINE + "\n" + _tracker_row(score="4.2", role="PM") + "\n"
        (additions / "7.tsv").write_text(kept, encoding="utf-8")
        _recover_refused_additions(co, additions)
        assert sorted(f.name for f in additions.glob("*.tsv")) == ["7.tsv"]
        assert (additions / "7.tsv").read_text(encoding="utf-8") == kept
        assert capsys.readouterr().out == ""

    def test_run_merge_tracker_recovers_before_it_merges(self, tmp_path, mocker):
        co, additions = self._tree(tmp_path, {"7.tsv": _tracker_row(score="4.2")})
        (co / "merge-tracker.mjs").write_text(_CURRENT_MERGE_TRACKER, encoding="utf-8")
        seen = {}
        def snapshot(*a, **kw):
            seen["pending"] = sorted(f.name for f in additions.glob("*.tsv"))
            return mocker.MagicMock(returncode=0, stdout="", stderr="")
        mocker.patch("pipeline._batch_common.subprocess.run", side_effect=snapshot)
        run_merge_tracker(co)
        assert seen["pending"] == ["7.tsv"]         # back in the queue when node ran


class TestLivenessMarks:
    """'Discarded' is two things (#163): a person's decision, or the liveness
    re-check's verdict that a POSTING died. The re-check marks its own in Notes;
    the newest mark decides."""

    def test_closed_mark_is_recognised(self):
        assert closed_by_recheck("https://x — APPLY — " + liveness_closed_mark("2026-09-06", "HTTP 404"))

    def test_no_mark_is_a_persons_discard(self):
        assert not closed_by_recheck("https://x — APPLY — not for me")
        assert not closed_by_recheck("")

    def test_reopened_after_closed_is_a_persons_discard_again(self):
        notes = liveness_closed_mark("2026-09-01", "HTTP 404") + " — " + reopened_mark("2026-09-08")
        assert not closed_by_recheck(notes)

    def test_closed_again_after_reopened_is_the_rechecks(self):
        notes = (liveness_closed_mark("2026-09-01", "HTTP 404") + " — " + reopened_mark("2026-09-08")
                 + ". Re-eval 2026-09-09 (4→4.5): https://y — " + liveness_closed_mark("2026-09-20", "HTTP 410"))
        assert closed_by_recheck(notes)

    def test_mark_survives_upstreams_notes_merge(self):
        # merge-tracker keeps existing Notes first and appends the re-eval clause.
        notes = ("https://old — APPLY — " + liveness_closed_mark("2026-09-06", "HTTP 404")
                 + ". Re-eval 2026-09-08 (4.1→4.7): https://new — APPLY")
        assert closed_by_recheck(notes)

    def test_reason_cannot_break_the_cell_or_hijack_the_url(self):
        # A `|` is a cell boundary; a URL would be read back as the posting.
        from pipeline.app.data import extract_url
        mark = liveness_closed_mark("2026-09-06", "error redirect: https://x.com/jobs?expired=1|foo")
        assert "|" not in mark and "http" not in mark
        assert extract_url("https://posting — APPLY — " + mark) == "https://posting"

    def test_reason_is_capped_and_one_line(self):
        mark = liveness_closed_mark("2026-09-06", "body: (?:closed|filled)" + " x\n" * 200)
        assert "\n" not in mark and len(mark) < 140

    def test_empty_reason_still_marks(self):
        assert closed_by_recheck(liveness_closed_mark("2026-09-06", ""))

    def test_a_by_hand_mark_after_closed_makes_the_discard_a_persons(self):
        # A person's write leaves this mark on a re-check-Closed row (#163 follow-up);
        # newest still wins, so a later re-check Closed is the re-check's again.
        closed = liveness_closed_mark("2026-09-06", "HTTP 404")
        notes = f"https://x — APPLY — {closed} — {by_hand_mark('2026-09-08', 'Discarded')}"
        assert not closed_by_recheck(notes)
        assert closed_by_recheck(notes + f" — {liveness_closed_mark('2026-09-10', 'HTTP 410')}")
        assert by_hand_mark("2026-09-08", "Discarded | x") == "Set Discarded x 2026-09-08 (by hand)"


def _apps_row(num, status, report, notes, score="4.0/5", role="Nurse"):
    return (f"| {num} | 2026-09-01 | Acme | {role} | {score} | {status} | ❌ "
            f"| [{report}](../reports/{report}-acme.md) | {notes} |\n")


class TestReopenRepostedDiscards:
    """A re-post of a role the liveness re-check Discarded: its re-eval lands on
    the Discarded row (merge-tracker's update tier writes score, report, date
    and notes through but keeps the status), and nothing ever surfaced it
    again (#163). After a merge, a re-check Discard whose report number changed
    goes back to Evaluated. A person's Discard is never bounced."""

    CLOSED = liveness_closed_mark("2026-09-06", "HTTP 404")

    @staticmethod
    def _row(co, num):
        return tracker_row(co / "data" / "applications.md", num)

    def _merge_update(self, text):
        # Score/report/date through, status kept, existing notes first + re-eval clause.
        return (text.replace("[050](../reports/050-acme.md)", "[070](../reports/070-acme.md)")
                    .replace("4.0/5", "4.7/5")
                    .replace(f"{self.CLOSED} |", f"{self.CLOSED}. Re-eval 2026-09-08 (4→4.7): https://new — APPLY |"))

    def test_reopens_a_recheck_discard_whose_report_changed(self, tmp_path, mocker, capsys):
        from pipeline.app import data
        co, _ = _career_ops(tmp_path, [_apps_row(5, "Discarded", "050", f"https://old — APPLY — {self.CLOSED}")])
        _fake_merge(mocker, co, rewrite=self._merge_update)
        assert run_merge_tracker(co)
        row = self._row(co, "5")
        assert row["status_canonical"] == "Evaluated"
        assert "Reopened" in row["notes"] and not closed_by_recheck(row["notes"])
        assert data.extract_url(row["notes"]) == "https://new"          # the next re-check verifies the LIVE posting
        override = json.loads(data.STATUS_OVERRIDES_FILE.read_text(encoding="utf-8"))["5"]
        assert override.pop("note").startswith("Reopened ")   # the mark rides to the cloud with the status
        assert override == {"status": "Evaluated", "company": "Acme", "role": "Nurse"}
        assert "reopened 1 role" in capsys.readouterr().out

    def test_reopens_under_the_older_notes_replacing_merge(self, tmp_path, mocker):
        """The older merge-tracker REPLACES Notes on an update, so the mark
        is gone afterwards — which is why the snapshot is taken before."""
        co, _ = _career_ops(tmp_path, [_apps_row(5, "Discarded", "050", f"https://old — APPLY — {self.CLOSED}")])
        _fake_merge(mocker, co, rewrite=lambda t: (
            t.replace("[050](../reports/050-acme.md)", "[070](../reports/070-acme.md)")
             .replace(f"https://old — APPLY — {self.CLOSED}", "Re-eval 2026-09-08 (4→4.7). https://new — APPLY")))
        assert run_merge_tracker(co)
        assert self._row(co, "5")["status_canonical"] == "Evaluated"

    def test_reopens_after_a_partial_merge_too(self, tmp_path, mocker):
        """A non-zero exit after an addition was applied — archived, the
        evidence of a merge — is a partial merge, and the re-eval that wrote
        through is as re-posted as on a clean one."""
        co, additions = _career_ops(
            tmp_path, [_apps_row(5, "Discarded", "050", f"https://old — APPLY — {self.CLOSED}")])
        for name, cells in {
                "8.tsv": ["5", "2026-09-08", "Acme", "Nurse", "Evaluated", "4.7/5", "null",
                          "[070](reports/070-acme.md)", "https://new — APPLY"],
                "9.tsv": ["6", "2026-09-08", "Globex", "SRE", "Evaluated", "4.1/5", "null",
                          "[071](reports/071-globex.md)", "APPLY"]}.items():
            (additions / name).write_text(labelled_addition("\t".join(cells)), encoding="utf-8")
        _fake_merge(mocker, co, rewrite=self._merge_update, archive=["8.tsv"],
                    returncode=1, stderr="❌ 9.tsv: NOT merged.")
        assert run_merge_tracker(co) is False
        assert self._row(co, "5")["status_canonical"] == "Evaluated"

    def test_a_persons_discard_is_not_bounced(self, tmp_path, mocker):
        co, _ = _career_ops(tmp_path, [_apps_row(5, "Discarded", "050", "https://old — APPLY — not for me")])
        _fake_merge(mocker, co, rewrite=lambda t: t.replace(
            "[050](../reports/050-acme.md)", "[070](../reports/070-acme.md)"))
        assert run_merge_tracker(co)
        assert self._row(co, "5")["status_canonical"] == "Discarded"

    def test_a_discard_after_a_reopen_is_a_persons(self, tmp_path, mocker):
        notes = f"https://old — APPLY — {self.CLOSED} — {reopened_mark('2026-09-08')} — no thanks"
        co, _ = _career_ops(tmp_path, [_apps_row(5, "Discarded", "050", notes)])
        _fake_merge(mocker, co, rewrite=lambda t: t.replace(
            "[050](../reports/050-acme.md)", "[070](../reports/070-acme.md)"))
        assert run_merge_tracker(co)
        assert self._row(co, "5")["status_canonical"] == "Discarded"

    def test_no_re_eval_means_no_reopen(self, tmp_path, mocker, capsys):
        co, _ = _career_ops(tmp_path, [_apps_row(5, "Discarded", "050", f"https://old — APPLY — {self.CLOSED}")])
        _fake_merge(mocker, co, rewrite=lambda t: t)
        assert run_merge_tracker(co)
        assert self._row(co, "5")["status_canonical"] == "Discarded"
        assert "reopened" not in capsys.readouterr().out

    def test_snapshot_keys_on_status_and_mark(self, tmp_path):
        co, _ = _career_ops(tmp_path, [
            _apps_row(1, "Discarded", "010", f"x — {self.CLOSED}"),                # the re-check's
            _apps_row(2, "Descartada", "020", f"x — {self.CLOSED}"),               # alias spelling counts
            _apps_row(3, "Discarded", "030", "x — not for me"),                    # a person's
            _apps_row(4, "Evaluated", "040", f"x — {self.CLOSED}"),                # already reopened by hand
            _apps_row(5, "Discarded", "050", f"x — {self.CLOSED} — {reopened_mark('2026-09-08')}"),
        ])
        closed = _liveness_closed_rows(co / "data" / "applications.md")
        assert set(closed) == {"1", "2"}
        assert closed["1"] == {"company": "Acme", "role": "Nurse", "report": "10"}   # row_report_num int-normalizes

    def test_reopen_is_a_no_op_without_a_snapshot(self, tmp_path):
        co, _ = _career_ops(tmp_path, [_apps_row(5, "Discarded", "050", "x")])
        _reopen_reposted({}, co)
        assert self._row(co, "5")["status_canonical"] == "Discarded"


class TestProtectedCharacteristicRedaction:
    """Protected characteristics must never reach an evaluation prompt (#165).

    Two routes in and they need separate passes: profile.yml's
    `voluntary_disclosures:` section on the seed path, and PROFILE.md's standing
    answers on the master path — which supersedes the seeds AND is non-clobber,
    so an install seeded by an older version keeps those bullets forever.
    """

    YML = (
        "candidate:\n"
        "  full_name: Jane Roe\n"
        "voluntary_disclosures:\n"
        "  gender: Female\n"
        "  race_ethnicity: Hispanic or Latino\n"
        "# a genuinely column-0 comment inside the block\n"
        "  veteran_status: I am a protected veteran\n"
        "  disability_status: Yes, I have a disability\n"
        "  data_processing_consent: true\n"
        "\n"
        "compensation:\n"
        "  target_range: 60-70k\n"
    )
    SECRETS = ("Female", "Hispanic", "protected veteran", "have a disability")

    def test_seed_path_drops_the_section_and_keeps_the_rest(self):
        out = _batch_common.build_system_prompt("CV text", self.YML)
        for leak in self.SECRETS:
            assert leak not in out, leak
        assert "data_processing_consent" not in out   # apply-path form data too
        assert "Jane Roe" in out and "60-70k" in out  # everything else survives

    def test_master_path_drops_the_standing_answer_bullets(self):
        master = (
            "## Standing answers\n"
            "- **Work authorization:** US citizen — no sponsorship required\n"
            "- **Gender:** Female\n"
            "- **Race / ethnicity:** Hispanic or Latino\n"
            "- **Veteran status:** I am a protected veteran\n"
            "- **Disability status:** Yes, I have a disability\n"
        )
        out = _batch_common.build_system_prompt("CV", self.YML, profile_master=master)
        for leak in self.SECRETS:
            assert leak not in out, leak
        assert "US citizen" in out

    def test_a_column_zero_comment_does_not_end_the_block_early(self):
        """The conservative read: a comment at column 0 inside the section is
        dropped with it rather than reopening the file and leaking the KEY NAMES
        after it — which the prompt-level test above cannot see, since it only
        checks the values."""
        red = _batch_common.redact_profile_yml(self.YML)
        assert "veteran_status" not in red and "disability_status" not in red
        assert "compensation:" in red

    def test_stems_catch_labels_no_generator_ever_wrote(self):
        """PROFILE.md is agent-grown and append-only, so an exact-label
        allowlist is defeated by the next wording the agent invents."""
        grown = ("- **Gender identity:** Non-binary\n"
                 "- **Protected veteran status:** Yes\n"
                 "- **Disability (self-ID):** Prefer not to say\n"
                 "- **Compensation target:** 60-70k\n")
        out = _batch_common.redact_profile_master(grown)
        assert "Non-binary" not in out and "Prefer not to say" not in out
        assert "**Protected veteran status:**" not in out
        assert "60-70k" in out

    def test_stems_are_word_anchored_and_keep_the_fact_bank(self):
        """PROFILE.md is the résumé build's ONLY source of truth, so an
        unanchored "race" would silently delete real experience."""
        bank = ("- **Grace Health:** ran intake for 12 clinics\n"
                "- **Terrace Senior Living:** 3 years front desk\n"
                "- **Traceability:** built the audit trail\n")
        assert _batch_common.redact_profile_master(bank) == bank

    def test_redaction_is_idempotent(self):
        once = _batch_common.redact_profile_yml(self.YML)
        assert _batch_common.redact_profile_yml(once) == once

    # ── #216: the label alone was the whole test ──────────────────────────────

    def test_disclosure_labels_with_no_protected_stem(self):
        """The five stems never fire on these labels, and they are the labels
        an agent writes. The disclosure stems name nothing but a disclosure, so
        they match a label anywhere in the file."""
        grown = ("- **Self-ID:** Female, Hispanic or Latino\n"
                 "- **Self identification:** declined\n"
                 "- **EEO:** Prefer not to say\n"
                 "- **Pronouns:** they/them\n"
                 "- **Voluntary disclosures:** see the form\n"
                 "- **Compensation target:** 60-70k\n")
        assert _batch_common.redact_profile_master(grown) == "- **Compensation target:** 60-70k\n"

    def test_disclosure_stems_stop_at_a_word_boundary(self):
        """`eeo` and `pronoun` are bounded on the right too: the EEOC is an
        employer, and "pronounced" is an ordinary word in a fact-bank label."""
        bank = ("- **EEOC:** investigator, 2019-2022\n"
                "- **Pronounced gains:** cut intake time 40%\n"
                "- **Selfridges:** 2 years retail\n")
        assert _batch_common.redact_profile_master(bank) == bank

    def test_a_dropped_bullet_takes_its_continuation(self):
        master = ("- **Disability status:**\n"
                  "  Yes, I have a disability\n"
                  "    (or previously had one)\n"
                  "- **Location:** Dallas, TX\n"
                  "  open to hybrid\n")
        assert _batch_common.redact_profile_master(master) == (
            "- **Location:** Dallas, TX\n"
            "  open to hybrid\n")

    def test_a_dropped_bullet_takes_its_nested_items(self):
        master = ("- **Voluntary disclosures:**\n"
                  "  - Gender: Female\n"
                  "  - Hispanic or Latino\n"
                  "\t- Not a protected veteran\n"
                  "- **Compensation target:** 60-70k\n")
        assert _batch_common.redact_profile_master(master) == "- **Compensation target:** 60-70k\n"

    def test_a_continuation_follows_markdown_and_ends_at_a_sibling_or_heading(self):
        """What Markdown reads as part of the dropped item goes with it: deeper
        lines, across a blank line (a loose list nests that way), and an
        unindented lazy line straight after it. A sibling at its own indent, a
        new paragraph after a blank, and a heading end it."""
        master = ("- **Gender:** Female\n"
                  "Hispanic or Latino\n"                     # lazy continuation
                  "\n"
                  "  - nested after a blank line\n"           # still the item's
                  "\n"
                  "a new paragraph\n"
                  "## Next\n"
                  "  - **Veteran status:** no\n"
                  "  - a sibling at its own indent\n"
                  "- **Race:** x\n"
                  "## After\n"
                  "  indented under a heading\n")
        assert _batch_common.redact_profile_master(master) == (
            "\n"
            "\n"
            "a new paragraph\n"
            "## Next\n"
            "  - a sibling at its own indent\n"
            "## After\n"
            "  indented under a heading\n")

    @pytest.mark.parametrize("heading", [
        "## Voluntary disclosures", "## Self-ID", "## EEO self-identification",
        "### Pronouns", "## voluntary   disclosure answers ##",
    ])
    def test_a_disclosure_heading_drops_its_whole_section(self, heading):
        """Mirrors `redact_profile_yml`'s whole-section rule: an unlabelled
        bullet under such a heading has nothing to match, so the heading is the
        only thing that can tell. Sub-sections go with it; a heading at the same
        level or above ends it."""
        master = ("## Identity & contact\n"
                  "- **Name:** Jane Roe\n"
                  "\n"
                  f"{heading}\n"
                  "- Female\n"
                  "- Hispanic or Latino\n"
                  "\n"
                  "#### Details\n"
                  "Not a protected veteran.\n"
                  "\n"
                  "## Tailoring rules\n"
                  "- Keep every metric.\n")
        assert _batch_common.redact_profile_master(master) == (
            "## Identity & contact\n"
            "- **Name:** Jane Roe\n"
            "\n"
            "## Tailoring rules\n"
            "- Keep every metric.\n")

    def test_standing_answers_test_the_whole_line(self):
        """Inside the generator's own section a stem in the value, in an
        unlabelled bullet or in a sub-heading is a disclosure, not a fact."""
        master = ("## Standing answers\n"
                  "What applications keep asking:\n"
                  "\n"
                  "- **Work authorization:** US citizen\n"
                  "- **Demographics:** Female; not a protected veteran\n"
                  "- Ethnicity - Hispanic or Latino\n"
                  "- **Location:** Dallas, TX\n"
                  "### Gender\n"
                  "Female\n"
                  "### Commute\n"
                  "- Up to 45 minutes\n"
                  "- I have a disability parking permit\n")
        assert _batch_common.redact_profile_master(master) == (
            "## Standing answers\n"
            "What applications keep asking:\n"
            "\n"
            "- **Work authorization:** US citizen\n"
            "- **Location:** Dallas, TX\n"
            "### Commute\n"
            "- Up to 45 minutes\n")

    # Values carrying every stem, in the fact-bank shapes: a bold label, a plain
    # bullet, a nested one, an employer heading. None of them is a disclosure.
    VALUE_DECOYS = ("- **VA North Texas:** scheduled care for 2,000 veterans\n"
                    "- **Accessibility:** audited intake forms for users with disabilities\n"
                    "- Led a gender pay-equity review covering race and ethnicity data\n"
                    "  - Filed the EEO-1 report; ran a self-identification drive\n"
                    "  - Rewrote pronouns in the style guide\n"
                    "### Disability Rights Texas — Veterans Advocate\n"
                    "- Represented 40 claimants at hearings\n")

    def test_fact_bank_values_are_never_scanned(self):
        """`\\bveteran` and `disab\\w*` match "veterans" and "disabilities", so a
        global value scan would be the Grace Health deletion moved from labels
        to values. Text before the first heading is not scanned either."""
        doc = ("# Jane Roe — candidate profile\n"
               "A nurse who served veterans and patients with disabilities.\n"
               "\n"
               "## Role fact bank\n" + self.VALUE_DECOYS)
        assert _batch_common.redact_profile_master(doc) == doc

    def test_the_scan_ends_with_the_standing_answers_section(self):
        doc = ("## Standing answers\n"
               "- **Gender:** Female\n"
               "\n"
               "## Role fact bank\n" + self.VALUE_DECOYS)
        assert _batch_common.redact_profile_master(doc) == (
            "## Standing answers\n"
            "\n"
            "## Role fact bank\n" + self.VALUE_DECOYS)

    def test_master_redaction_is_idempotent(self):
        master = ("# Jane Roe\n"
                  "- **Self-ID:** Female\n"
                  "  - Hispanic or Latino\n"
                  "- **Name:** Jane Roe\n"
                  "\n"
                  "## Role fact bank\n" + self.VALUE_DECOYS +
                  "\n"
                  "## Standing answers\n"
                  "- **Location:** Dallas, TX\n"
                  "- not a protected veteran\n"
                  "    deeper\n"
                  "  shallower but still deeper than nothing\n"
                  "### Pronouns\n"
                  "they/them\n"
                  "### Commute\n"
                  "- 45 minutes\n"
                  "## Voluntary disclosures\n"
                  "- Female\n"
                  "## Tailoring rules\n"
                  "- Keep every metric.\n")
        once = _batch_common.redact_profile_master(master)
        assert _batch_common.redact_profile_master(once) == once
        for leak in ("Female", "Hispanic", "protected veteran", "they/them", "shallower"):
            assert leak not in once, leak
        assert self.VALUE_DECOYS in once and "45 minutes" in once

    def test_the_generators_own_answers_survive_and_its_section_is_scanned(self):
        """`render_profile_md` writes the section this scan keys on, so a
        disclosure the agent appends there is caught — and everything the
        generator itself writes survives except its self-identification
        instruction, which names the stems and has no use in a prompt."""
        from pipeline import handoff
        md = handoff.render_profile_md(
            cv_md="## Experience\n### VA North Texas — Scheduler\n- Booked care for 2,000 veterans\n",
            profile={"compensation": {"target_range": "60-70k", "location_flexibility": "Hybrid"},
                     "location": {"city": "Dallas", "state": "TX", "timezone": "CST"},
                     "work_authorization": {"citizenship": "US", "requires_sponsorship": False},
                     "voluntary_disclosures": {"gender": "Female", "veteran_status": "No"}})
        out = _batch_common.redact_profile_master(md)
        assert out.splitlines() == [
            ln for ln in md.splitlines() if not ln.startswith("- **Self-identification")]
        location = next(ln for ln in md.splitlines() if ln.startswith("- **Location:**"))
        grown = md.replace(location, f"{location}\n- Hispanic or Latino, not a protected veteran")
        assert _batch_common.redact_profile_master(grown) == out

    def test_evidence_sections_keep_the_pre_216_rule(self):
        """The disclosure stems are words a job carries too. In the generator's
        fact bank and skills inventory, an EEO role or skill survives whole,
        while a bullet LABELLED with a protected stem still goes, as before."""
        bank = (
            f"## {_batch_common.EVIDENCE_HEADINGS[0]}\n"
            "### EEO Specialist — U.S. Department of Labor\n"
            "- Investigated 40 complaints a quarter\n"
            "- **Pronoun-aware intake scripting:** rewrote the intake form\n"
            f"## {_batch_common.EVIDENCE_HEADINGS[1]}\n"
            "- **EEO-1 reporting:** strong\n"
            "- **Gender:** Female\n")
        out = _batch_common.redact_profile_master(bank)
        assert out == bank.replace("- **Gender:** Female\n", "")

    def test_the_evidence_rule_ends_at_the_next_generator_section(self):
        """The fact bank holds a CV verbatim, its own `## Experience` heading
        included; ending the section there put an "EEO Specialist" role under
        the stricter rule, which deleted it. It ends at a section the generator
        writes instead."""
        bank = _batch_common.EVIDENCE_HEADINGS[0]
        md = (f"## {bank}\n## Experience\n### EEO Specialist — U.S. Department of Labor\n"
              "- Investigated 40 complaints a quarter\n"
              "## How this profile grows\n### Pronouns\nthey/them\n")
        assert _batch_common.redact_profile_master(md) == (
            f"## {bank}\n## Experience\n### EEO Specialist — U.S. Department of Labor\n"
            "- Investigated 40 complaints a quarter\n"
            "## How this profile grows\n")

    def test_generator_headings_are_recognised_before_any_is_tested(self):
        """An agent widening a generator heading must not delete the section."""
        md = ("## Standing answers (incl. EEO)\n- **Location:** Dallas, TX\n- **Gender:** x\n"
              "## Skills inventory (incl. EEO-1 reporting)\n- **EEO-1 reporting:** strong\n")
        assert _batch_common.redact_profile_master(md) == (
            "## Standing answers (incl. EEO)\n- **Location:** Dallas, TX\n"
            "## Skills inventory (incl. EEO-1 reporting)\n- **EEO-1 reporting:** strong\n")

    def test_a_protected_heading_in_an_agent_section_drops_it(self):
        md = "## How this profile grows\nnotes\n## Application answers\n### Gender\nFemale\n### Notes\nok\n"
        assert _batch_common.redact_profile_master(md) == (
            "## How this profile grows\nnotes\n## Application answers\n### Notes\nok\n")

    def test_the_title_and_fenced_lines_are_never_headings_to_drop(self):
        """A surname carrying a stem must not take the whole profile, and a `#`
        inside a code fence must not end the Standing answers scan."""
        md = ("# Eli Disabato — candidate profile\n## Standing answers\n- **Location:** Dallas\n"
              "```\n# not a heading\n```\n- Gender: Female\n")
        assert _batch_common.redact_profile_master(md) == (
            "# Eli Disabato — candidate profile\n## Standing answers\n- **Location:** Dallas\n"
            "```\n# not a heading\n```\n")

    def test_numbered_and_underscore_bold_bullets_are_read(self):
        md = "## Notes the agent added\n1. **Gender:** x\n* __Veteran status:__ no\n- **Gender identity** kept? no\n- ok\n"
        assert _batch_common.redact_profile_master(md) == "## Notes the agent added\n- ok\n"

    def test_race_needs_a_word_boundary(self):
        """`race` unbounded on the right redacted a town ("Raceland, LA") out of
        the candidate's location."""
        md = "## Standing answers\n- **Location:** Raceland, LA (CST)\n- **Race:** x\n- Racial identity: y\n"
        assert _batch_common.redact_profile_master(md) == "## Standing answers\n- **Location:** Raceland, LA (CST)\n"

    def test_every_generator_section_is_listed(self):
        """An evidence section ends only at a heading in PROFILE_SECTION_HEADINGS,
        so a section render_profile_md adds must be listed there."""
        from pipeline import handoff
        cv = "## Experience\n- Did things\n"
        md = handoff.render_profile_md(cv_md=cv, profile={})
        written = [ln[3:].strip() for ln in md.splitlines()
                   if ln.startswith("## ") and ln not in cv.splitlines()]   # the CV's own are pasted in
        assert written and set(written) <= set(_batch_common.PROFILE_SECTION_HEADINGS), written

    def test_the_generator_writes_the_evidence_headings(self):
        """Renaming a heading in render_profile_md must not quietly move the fact
        bank onto the stricter rule and delete real experience."""
        from pipeline import handoff
        md = handoff.render_profile_md(cv_md="## Experience\n- Did things\n", profile={})
        for h in _batch_common.EVIDENCE_HEADINGS:
            assert f"## {h}" in md

    def test_empty_and_none_degrade_quietly(self):
        for f in (_batch_common.redact_profile_yml, _batch_common.redact_profile_master):
            assert f("") == "" and f(None) == ""


class TestNoUnredactedProfileReads:
    """No prompt may read profile.yml or PROFILE.md raw (#165).

    Redacting inside ONE builder is opt-in, and this repo has five that eat the
    candidate files — so the sixth would be safe only if its author remembered.
    Every read is therefore either redacted at the call site or listed here with
    a reason it never reaches a model. A new raw read fails this test, and the
    fix is to route it through `profile_yml_for_prompt` /
    `profile_master_for_prompt` — or to add it below and say why.
    """

    READS = re.compile(r'(_read|read_text|_read_or_empty|_load_yaml_or_empty|safe_load)\s*\(')
    PATHS = ('profile.yml"', "HANDOFF_PROFILE")
    SAFE = ("redact_", "_for_prompt")

    # The exact source line -> why that read never reaches a model. Keyed on the
    # LINE, not the module: a per-module exemption would wave through a NEW
    # prompt builder added to skills.py or handoff.py, which are precisely the
    # two files that hold both a safe read and a prompt read. Line text rather
    # than line number so an edit above it doesn't spuriously fail.
    ALLOWED = {
        'path = Path(career_ops) / "config" / "profile.yml"':
            "candidate_profile.ApplyProfile projects four contact fields",
        'derived = derive_from_profile(_load_yaml_or_empty(career_ops / "config" / "profile.yml"))':
            "wizard prefill, not a prompt",
        'text = _read(local / "config" / "profile.yml")':
            "_candidate_slug regexes out a name for a filename",
        'if not read_text(co / "cv.md") and not read_text(co / "config" / "profile.yml"):':
            "existence check for the onboarding gate",
        '"profile": _load_yaml_or_empty(co / "config" / "profile.yml"),':
            "render_profile_md's own sources; it no longer writes the EEO fields",
        "master = _read_or_empty(Path(out_dir or default_out_dir()) / HANDOFF_PROFILE)":
            "resolve_profile_md; both prompt consumers redact on the way out",
        "return _read_or_empty(_career_ops_dir(career_ops) / HANDOFF_PROFILE)":
            "resolve_profile_md's fallback arm, same reason",
        'profile_yml = read_text(career_ops / "config" / "profile.yml")':
            "eval_system_prompt; build_system_prompt redacts both inputs, and "
            "only target_roles' names and fit are parsed out of it",
    }

    def test_every_raw_read_is_redacted_or_declared(self):
        root = Path(__file__).resolve().parent.parent / "pipeline"
        offenders = []
        for py in sorted(root.rglob("*.py")):
            for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if not any(t in line for t in self.PATHS) or not self.READS.search(line):
                    continue
                if any(t in line for t in self.SAFE) or line.strip() in self.ALLOWED:
                    continue
                offenders.append(f"{py.relative_to(root.parent)}:{n}: {line.strip()}")
        assert not offenders, (
            "raw read of a candidate profile file — route it through "
            "profile_yml_for_prompt / profile_master_for_prompt, or add the "
            "module to ALLOWED with a reason:\n" + "\n".join(offenders))

    def test_the_two_builders_this_commit_fixed_stay_fixed(self):
        """Named explicitly because the allowlist is per-MODULE: skills.py and
        resume_content.py each hold both a safe read and a prompt read, so the
        scan above cannot tell them apart."""
        root = Path(__file__).resolve().parent.parent / "pipeline"
        skills = (root / "app" / "skills.py").read_text(encoding="utf-8")
        assert 'redact_profile_yml(_read(local / "config" / "profile.yml"))' in skills
        content = (root / "resume_content.py").read_text(encoding="utf-8")
        assert "profile_master_for_prompt(profile_path)" in content


class TestGeographyRuleInThePrompt:
    """The belt half of #180's belt-and-braces. `work_location` is what the code
    gates on, and the code degrades to "leave it alone" when the field is absent
    — so a prompt that stops asking for the field disables the whole feature
    silently, on every copy, with every test below this one still green.

    The rule's own wording is guarded too, not just the schema key. Every clause
    here is one the evaluator got wrong on a real report: it filed "requires
    local presence in Chicago" as a soft gap because nothing defined a hard
    stop, and it padded that gap with flexibility the posting never offered
    ("may offer remote", "or candidate is open to relocation").
    """

    def _prompt(self):
        return build_system_prompt("cv", "profile", profile_master="MASTER")

    def test_the_machine_summary_asks_for_the_structured_field(self):
        p = self._prompt()
        i = p.index("**Machine Summary**")
        block = p[i:i + 900]
        assert "work_location:" in block
        for key in ("mode:", "metro:", "state:"):
            assert key in block, key
        assert "onsite | hybrid | remote" in block

    def test_hard_stops_are_defined_at_all(self):
        """`hard_stops: []` shipped in the schema for months with no definition
        anywhere of what earns one, which is how "cannot physically be there"
        ended up beside "unfamiliar with the neighbourhood"."""
        p = self._prompt()
        assert "**Hard stops**" in p
        assert "hard_stops" in p

    def test_the_geography_rule_is_stated(self):
        p = self._prompt().lower()
        assert "physical presence" in p
        # Hybrid must be named: the failure mode is reading one office day a
        # week as near enough to remote.
        assert "hybrid" in p
        assert "commutable" in p or "has not said they can work in" in p

    def test_invented_flexibility_is_forbidden(self):
        p = self._prompt().lower()
        assert "do not invent flexibility" in p

    def test_inherently_local_work_is_called_out(self):
        """The report that prompted this said "remote coordination", so the
        keyword guard in remote_signal could never have caught it."""
        p = self._prompt().lower()
        assert "community-based" in p and "field" in p

    def test_the_mode_is_what_the_posting_requires_not_prefers(self):
        p = self._prompt()
        assert "REQUIRES" in p
        assert "headquarters" in p.lower()

    def _sentences_with(self, p, phrase):
        return [s for s in p.split(". ") if phrase in s]

    def test_a_board_location_never_fills_metro_or_state(self):
        """#205. A board location is often the employer's headquarters, and a
        metro recorded from it is what #180's out-of-area gate demotes on — so a
        place only the board names never fills the two fields the gate reads,
        whether or not the candidate can work there. Said in the work_location
        rules AND beside the geography rule, which is where a model deciding
        that a board location is "confirmed" reads it."""
        p = self._prompt()
        i = p.index("work_location rules:")
        [rule] = self._sentences_with(p[i:], "whatever place the job board names")
        assert 'both are "" for a fully remote role and "" when the posting does not say, ' \
            "whatever place the job board names" in rule
        [geo] = self._sentences_with(p, "location naming a metro is the evidence")
        assert 'for the decision only (`metro` and `state` stay "")' in geo
        # The reason sits in the geography rule, once.
        assert "a board often lists the headquarters" in geo

    def test_a_board_only_place_outside_the_area_is_a_soft_gap_capped_at_consider(self):
        """#205. A board location inside the candidate's area confirms the
        location the description left out. One outside it is NOT a hard stop —
        it may be the headquarters — but a soft gap naming the board's place,
        and it caps the decision at Consider (what a description-silent role
        got before #205), not the score — and not at "Research first", which
        the decision rule otherwise offers for a fact the posting leaves out.
        The decision rule, the hard-stop paragraph, the geography rule and the
        work_location rules must all say one thing, or the model reads one of
        them as overriding the others: a decision rule capping on "a location
        only the job board names" read as capping the candidate's OWN metro too,
        which is #205's harm."""
        p = self._prompt()
        [rule] = self._sentences_with(p, "location naming a metro is the evidence")
        assert "says neither where the job is nor that it is remote" in rule
        assert "inside the candidate's area it confirms the location" in rule
        assert "outside, it is not a hard stop" in rule
        assert "but a soft gap naming it" in rule
        assert ('caps `final_decision` at "Consider", not "Research first", '
                "and leaves the score alone") in rule
        assert "confirm\")" not in rule
        hard_stops = p[p.index("**Hard stops**"):p.index("Geography is the one most often missed")]
        assert "A place only the job board names is never one" in hard_stops
        assert ("unless a hard stop or a job-board location outside the candidate's area "
                "caps it") in p
        assert "a location only the job board names caps it" not in p

    def test_a_board_location_broader_than_a_metro_is_no_location(self):
        """#205. JobSpy's location is often not a metro — LinkedIn's usual one
        for a remote role is "United States", Indeed's can be "Texas" or
        "Remote" — and read by the two-case rule as "not in the candidate's
        area" each would cap a role on nothing. It is no location at all, and
        the word remote in it does not make a role remote either: the board's
        remote flag was dropped on purpose, and this must not re-admit it."""
        p = self._prompt()
        sentences = p.split(". ")
        [i] = [n for n, s in enumerate(sentences)
               if "location naming a metro is the evidence" in s]
        broader = sentences[i + 1]  # "one" is the board location just named
        assert broader.startswith("A broader one - a country, a state, or one saying remote - ")
        assert "is no location at all: only the description decides remote" in broader
        assert "it wins any disagreement with the board" in broader

    def test_a_board_salary_is_not_the_posting_stating_pay(self):
        """A board salary may be Indeed's own estimate — it arrives labelled
        direct data like an employer's figure — so it may inform the comp score
        but must not move the band the way stated pay does."""
        p = self._prompt()
        assert '"(job board)" salary may inform the comp score' in p
        assert "it is not the posting stating pay" in p
        assert "a range the description states wins" in p

    def test_the_board_remote_flag_is_not_mentioned(self):
        """#205 dropped it: JobSpy's is_remote is a keyword guess, wrong in both
        directions, so the description alone decides remote."""
        assert "Remote (job board)" not in self._prompt()


class TestWorkLocationMark:
    """#180's storage half: the evaluation records WHERE the role is on the row,
    once, so the three surfaces that rank roles (the digest, the work-order, the
    UI) read one fact instead of each opening the report. Notes rather than a new
    column or a state file, for `liveness_closed_mark`'s own reasons — it rides
    the cached cloud tracker and every Refresh."""

    ROW = ("12\t2026-09-14\tUCAN\tOutreach Specialist\tEvaluada\t4.8/5\tnull\t"
           "[047](reports/047-ucan-2026-09-14.md)\tAPPLY strong outreach match")

    def _report(self, mode="onsite", metro="Chicago, IL", state="IL"):
        return ("# Evaluacion\n\n## Machine Summary\n\n```yaml\n"
                f'final_decision: "Apply"\nwork_location:\n  mode: "{mode}"\n'
                f'  metro: "{metro}"\n  state: "{state}"\nhard_stops: []\n```\n')

    def test_write_job_result_records_the_location_it_evaluated(self, tmp_path):
        reports, additions = tmp_path / "reports", tmp_path / "additions"
        reports.mkdir(); additions.mkdir()
        response = (f"<report>{self._report()}</report>"
                    f"<tracker_tsv>{self.ROW}</tracker_tsv>"
                    '<summary>{"status": "completed", "company": "UCAN"}</summary>')
        write_job_result(response, {"id": "j1", "report_num": "047",
                                    "url": "https://x.example/47"},
                         reports, additions, "2026-09-14")
        notes = _data_cells(additions / "j1.tsv")[8]
        assert "Work location: On-site Chicago, IL" in notes

    def test_a_report_without_the_field_gets_no_mark(self):
        """Every row written before #180. An unmarked row has never been judged,
        which must stay distinguishable from a row judged remote."""
        plain = '# Evaluacion\n\n## Machine Summary\n\n```yaml\nfinal_decision: "Apply"\n```\n'
        assert _report_location_mark(plain) == ""
        assert "Work location" not in sanitize_addition(self.ROW, work_location="")

    def test_the_mark_lands_after_the_url_and_req_id(self):
        """Those two are read by machines that take the FIRST match in the cell
        (merge-tracker's extractReqNumber, the UI's Open posting); the mark is
        read by a regex that does not care where it sits."""
        out = sanitize_addition(self.ROW, "https://x.example/47", "Job ID: 88214",
                                work_location="Work location: On-site Chicago, IL")
        notes = out.split("\t")[-1]
        assert notes.index("req ") < notes.index("https://") < notes.index("Work location:")

    def test_it_is_idempotent(self):
        """sanitize_addition runs at two points; a second pass must not grow a
        second mark."""
        once = sanitize_addition(self.ROW, "https://x.example/47",
                                 work_location="Work location: On-site Chicago, IL")
        twice = sanitize_addition(once, "https://x.example/47",
                                  work_location="Work location: On-site Chicago, IL")
        assert once == twice and twice.count("Work location:") == 1

    def test_a_batch_row_gets_the_mark_from_the_report_it_links(self, tmp_path):
        """The `--batch` path: career-ops' batch-runner lets the agent CLI write
        the TSV directly, so the row reaches the merge with none of the chain
        applied — but the report it names is on disk from the same run."""
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "047-ucan-2026-09-14.md").write_text(
            self._report(), encoding="utf-8")
        assert _pending_report_marks(self.ROW, _row_report(tmp_path, self.ROW)) == ("Work location: On-site Chicago, IL", "")

    def test_a_dead_report_link_still_finds_its_report(self, tmp_path):
        """The Report cell is model-authored (#162), so the slug may name no file;
        resolution is by number, exactly as every other reader does it."""
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "047-ucan-2026-09-14.md").write_text(
            self._report(), encoding="utf-8")
        dead = self.ROW.replace("047-ucan-2026-09-14.md", "047-ucan-&-co-2026-09-14.md")
        assert _pending_report_marks(dead, _row_report(tmp_path, dead)) == ("Work location: On-site Chicago, IL", "")

    def test_no_report_means_no_mark(self, tmp_path):
        (tmp_path / "reports").mkdir()
        assert _pending_report_marks(self.ROW, _row_report(tmp_path, self.ROW)) == ("", "")


class TestDecisionSkipMark:
    """#183: the evaluator's `final_decision: Skip` is recorded on the row, the
    way #180 records where the job is, so the work-order and the role selector —
    which never open reports — can leave the role out. Only a Skip is marked:
    nothing any reader does changes for Apply, Consider or Research first."""

    ROW = ("12\t2026-11-20\tInitech\tClaims Analyst\tEvaluada\t2.6/5\tnull\t"
           "[047](reports/047-initech-2026-11-20.md)\tSKIP weak claims match")

    @staticmethod
    def _report(decision="Skip", work_location=True):
        return ("# Evaluacion\n\n## Machine Summary\n\n```yaml\n"
                f'final_decision: "{decision}"\n'
                + ('work_location:\n  mode: "onsite"\n  metro: "Dallas, TX"\n  state: "TX"\n'
                   if work_location else "")
                + "hard_stops: []\n```\n")

    def _write(self, tmp_path, decision):
        reports, additions = tmp_path / "reports", tmp_path / "additions"
        reports.mkdir(); additions.mkdir()
        response = (f"<report>{self._report(decision)}</report>"
                    f"<tracker_tsv>{self.ROW}</tracker_tsv>"
                    '<summary>{"status": "completed", "company": "Initech"}</summary>')
        write_job_result(response, {"id": "j1", "report_num": "047",
                                    "url": "https://x.example/47"},
                         reports, additions, "2026-11-20")
        return _data_cells(additions / "j1.tsv")[8]

    def test_write_job_result_marks_a_skip(self, tmp_path):
        notes = self._write(tmp_path, "Skip")
        assert DECISION_SKIP_MARK in notes and skip_decided(notes)

    @pytest.mark.parametrize("decision", ["Apply", "Consider", "Research first"])
    def test_and_nothing_else(self, tmp_path, decision):
        notes = self._write(tmp_path, decision)
        assert "Decision:" not in notes and not skip_decided(notes)

    @pytest.mark.parametrize("value, want", [
        ("Skip", True), ("skip", True), ("SKIP - weak match", True), (["Skip"], True),
        ("Consider", False), ("", False), (None, False), ([], False),
    ])
    def test_the_one_reading_of_final_decision(self, value, want):
        assert is_skip_decision(value) is want

    def test_a_report_with_no_summary_earns_no_mark(self):
        assert _report_marks("# Evaluacion\n\nNo summary here.") == ("", "")

    def test_one_parse_yields_both_marks(self, mocker):
        """The writer and the merge-time pass each want both marks of one report,
        and the YAML parse is the cost the merge-time pass's guard exists to
        avoid, so one call parses the Machine Summary once."""
        spy = mocker.spy(_batch_common, "parse_machine_summary")
        assert _report_marks(self._report()) == (
            "Work location: On-site Dallas, TX", DECISION_SKIP_MARK)
        assert spy.call_count == 1

    def test_the_mark_lands_after_the_url_the_req_id_and_the_location(self):
        out = sanitize_addition(self.ROW, "https://x.example/47", "Job ID: 88214",
                                work_location="Work location: On-site Dallas, TX",
                                decision=DECISION_SKIP_MARK)
        notes = out.split("\t")[-1]
        assert (notes.index("req ") < notes.index("https://") < notes.index("Work location:")
                < notes.index(DECISION_SKIP_MARK) < notes.index("SKIP weak"))

    def test_it_is_idempotent(self):
        """sanitize_addition runs at two points; a second pass must not grow a
        second mark — TestSanitizeAddition's rule, with the new step in it."""
        kw = dict(work_location="Work location: On-site Dallas, TX", decision=DECISION_SKIP_MARK)
        once = sanitize_addition(self.ROW, "https://x.example/47", "Job ID: 88214", **kw)
        twice = sanitize_addition(once, "https://x.example/47", "Job ID: 88214", **kw)
        assert once == twice and twice.count(DECISION_SKIP_MARK) == 1

    def test_a_cell_the_mark_opens_is_idempotent_too(self):
        """No URL, req id or location to land after: the mark opens the cell, and
        the guard must still find it there."""
        once = sanitize_addition(self.ROW, decision=DECISION_SKIP_MARK)
        assert once.split("\t")[-1].startswith(DECISION_SKIP_MARK)
        assert sanitize_addition(once, decision=DECISION_SKIP_MARK) == once

    def test_a_batch_row_gets_both_marks_from_the_report_it_links(self, tmp_path):
        """The `--batch` path, as for the Work location mark: one report read
        yields both, since they come from the same Machine Summary."""
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "047-initech-2026-11-20.md").write_text(
            self._report(), encoding="utf-8")
        assert _pending_report_marks(self.ROW, _row_report(tmp_path, self.ROW)) == (
            "Work location: On-site Dallas, TX", DECISION_SKIP_MARK)

    def test_a_marked_skip_is_not_read_again(self, tmp_path):
        """A Decision mark is the evidence the row's writer read its report: that
        writer wrote the Work location mark from the same Machine Summary."""
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "047-initech-2026-11-20.md").write_text(
            self._report(), encoding="utf-8")
        row = self.ROW.replace(
            "\tSKIP", f"\tWork location: On-site Dallas, TX — {DECISION_SKIP_MARK} — SKIP")
        assert _pending_report_marks(row, _row_report(tmp_path, row)) == ("", "")

    def test_a_pre_183_addition_gets_its_decision_at_merge_time(self, tmp_path):
        """A Work location mark is NOT that evidence: `write_job_result` before
        #183 wrote it and no Decision mark, and its additions can still be
        pending after the upgrade (a merge that failed leaves them for the next).
        The report is read, the Skip is marked, and the Work location mark the
        row already carries is not written a second time."""
        co = tmp_path / "co"
        (co / "reports").mkdir(parents=True)
        (co / "reports" / "047-initech-2026-11-20.md").write_text(self._report(), encoding="utf-8")
        additions = co / "batch" / "tracker-additions"
        additions.mkdir(parents=True)
        row = self.ROW.replace("\tSKIP", "\thttps://x.example/47 — Work location: On-site Dallas, TX — SKIP")
        (additions / "5.tsv").write_text(row + "\n", encoding="utf-8")
        _sanitize_pending_additions(co, additions)
        notes = _data_cells(additions / "5.tsv")[8]
        assert skip_decided(notes)
        assert notes.count("Work location:") == 1 and notes.count(DECISION_SKIP_MARK) == 1

    def test_the_merge_time_pass_writes_it_onto_a_batch_row(self, tmp_path):
        co = tmp_path / "co"
        (co / "reports").mkdir(parents=True)
        (co / "reports" / "047-initech-2026-11-20.md").write_text(self._report(), encoding="utf-8")
        additions = co / "batch" / "tracker-additions"
        additions.mkdir(parents=True)
        (additions / "5.tsv").write_text(self.ROW + "\n", encoding="utf-8")
        _sanitize_pending_additions(co, additions)
        assert skip_decided(_data_cells(additions / "5.tsv")[8])


class TestSkipDecidedReading:
    """`skip_decided` is the verdict every reader acts on, derived per read from
    the marks in Notes — the newest one that bears on it wins, for the reason
    `work_location_from_notes` gives: merge-tracker keeps old Notes first."""

    BASE = "req 88214 — https://x.example/47 — Work location: On-site Dallas, TX"

    def test_the_mark_alone(self):
        assert skip_decided(f"{self.BASE} — {DECISION_SKIP_MARK} — SKIP weak match")
        assert skip_decided(DECISION_SKIP_MARK)                     # a cell with nothing else
        assert not skip_decided(f"{self.BASE} — APPLY strong match")
        assert not skip_decided("")

    @pytest.mark.parametrize("notes", [
        # Mid-clause: an ordinary sentence the model writes in that cell.
        "https://x.example/47 — SKIP: the final decision: Skip unless remote is offered",
        "https://x.example/47 — CONSIDER, decision: Skip if the pay is low",
        # Clause-leading, but not our spelling.
        "https://x.example/47 — decision: skip — weak match",
        "https://x.example/47 — Decision: Skipped for now",
    ])
    def test_a_model_sentence_is_not_the_mark(self, notes):
        assert not skip_decided(notes)

    def test_a_re_evaluation_retires_it(self):
        """The update tier appends the new evaluation after a `Re-eval` marker;
        an Apply writes no mark of its own, so the old Skip must not outlive it."""
        notes = (f"{self.BASE} — {DECISION_SKIP_MARK} — SKIP weak. "
                 f"Re-eval 2026-11-22 (2.6→4.3): {self.BASE} — APPLY better fit")
        assert not skip_decided(notes)

    def test_a_re_evaluation_that_skips_again_is_a_skip(self):
        notes = (f"{self.BASE} — APPLY. Re-eval 2026-11-22 (4.3→2.4): "
                 f"{self.BASE} — {DECISION_SKIP_MARK} — SKIP now")
        assert skip_decided(notes)

    @pytest.mark.parametrize("tail", [
        ". Re-eval 2026-11-22 (2.6→2.6)",
        # A downgrade's marker carries the superseded report, still bare.
        ". Re-eval 2026-11-22 (2.6→2.4) — Superseded report [31] (was 2.6/5)",
        # …and a sighting note after it has a colon of its own to reach for.
        ". Re-eval 2026-11-22 (2.6→2.4) — Superseded report [31] (was 2.6/5). "
        "Also listed at https://agg.example/9",
        "; Re-eval 2026-11-22 (2.6→2.6)",
        # One of OUR later " — " clauses after a bare marker: its colon is not
        # the marker's ": " (only mergeNotes' Superseded-report text sits there).
        "; Re-eval 2026-11-01 (3.0→3.0) — Closed 2026-11-05 (liveness re-check: posting 404)",
        ". Re-eval 2026-11-01 (3.0→3.0) — Work location: On-site Chicago, IL",
    ])
    def test_a_bare_re_eval_marker_does_not_retire_it(self, tail):
        """mergeNotes writes the marker BARE when the new notes repeat a clause
        already in the cell — a Skip re-evaluated to the same cell. That is no
        newer verdict, and reading it as one queued the Skip again."""
        assert skip_decided(f"{self.BASE} — {DECISION_SKIP_MARK} — SKIP weak{tail}")

    def test_a_re_eval_that_carries_notes_past_a_superseded_report_retires_it(self):
        notes = (f"{self.BASE} — {DECISION_SKIP_MARK} — SKIP weak. Re-eval 2026-11-22 "
                 f"(2.6→2.4) — Superseded report [31] (was 2.6/5): {self.BASE} — CONSIDER")
        assert not skip_decided(notes)

    @pytest.mark.parametrize("marker", [
        "Re-eval 2026-11-22 (4.3→2.4)",
        "Re-eval 2026-11-22 (4.3→2.4) — Superseded report [31] (was 4.3/5)",
    ])
    def test_a_mark_that_opens_the_re_evaluations_cell_is_read(self, marker):
        """A row with no req id, URL or Work location mark puts the Decision mark
        FIRST in its cell, and mergeNotes writes that cell after the marker's
        `: ` — a clause start, or the mark is not seen at all."""
        notes = f"https://x.example/47 — APPLY. {marker}: {DECISION_SKIP_MARK} — SKIP now"
        assert skip_decided(notes)

    @pytest.mark.parametrize("cell", [
        "the final decision: Skip unless remote",
        "Decision: Skipped for now",
    ])
    def test_after_a_marker_a_model_sentence_is_still_not_the_mark(self, cell):
        notes = f"https://x.example/47 — APPLY. Re-eval 2026-11-22 (4.3→2.4): {cell}"
        assert not skip_decided(notes)

    def test_the_converse_repeat_is_the_documented_misread(self):
        """What reading a bare marker as "no change" costs: Apply → Skip → an
        Apply that repeats the FIRST evaluation word for word stays a Skip. Rarer
        than the identical Skip re-evaluation it fixes, and the by-hand lift
        recovers it — pinned so a change to the trade is a decision."""
        notes = (f"{self.BASE} — APPLY. Re-eval 2026-11-21 (4.3→2.4): {self.BASE} — "
                 f"{DECISION_SKIP_MARK} — SKIP now. Re-eval 2026-11-22 (2.4→4.3)")
        assert skip_decided(notes)

    def test_a_status_set_by_hand_lifts_it(self):
        notes = f"{self.BASE} — {DECISION_SKIP_MARK} — SKIP weak"
        lifted = f"{notes} — {by_hand_mark('2026-11-23', 'Interview')}"
        assert skip_decided(notes) and not skip_decided(lifted)

    def test_the_rechecks_marks_do_not_bear_on_it(self):
        """Closed and Reopened are about the POSTING, not the evaluator's verdict
        on the role: a reopened re-post is re-evaluated, and that re-evaluation's
        own marks after its `Re-eval` marker decide."""
        notes = (f"{self.BASE} — {DECISION_SKIP_MARK} — SKIP weak — "
                 f"{liveness_closed_mark('2026-11-21', 'HTTP 404')}")
        assert skip_decided(notes)
        assert skip_decided(f"{notes} — {reopened_mark('2026-11-22')}")

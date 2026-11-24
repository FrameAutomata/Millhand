"""The job board's location and salary reach the evaluator through a sidecar
beside the cached JD, `batch/jds/<id>.board.json` (#205) — and a job with no
sidecar gets the user message byte for byte as before, because every queue
cached before the sidecar existed has none.

The board's remote flag is deliberately NOT carried: JobSpy's `is_remote` is a
keyword guess (on LinkedIn over the title, the location and a description still
empty at scrape time), wrong in both directions, so the description alone
decides remote.
"""

import csv
import json

import pytest

from pipeline import batch_prep, bridge
from pipeline._batch_common import (
    assign_job_numbers,
    board_meta_from_row,
    board_meta_json,
    board_meta_path,
    build_user_message,
    jd_cache_path,
    load_pending,
    normalize_board_meta,
    read_board_meta,
)
from pipeline.rowio import write_rows

TODAY = "2026-11-23"
URL = "https://example.com/jobs/1"


def _write_sidecar(career_ops, job_id, content):
    path = board_meta_path(career_ops, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif isinstance(content, dict):
        path.write_text(json.dumps(content), encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return path

# The message every job got before #205, spelled out byte for byte: a job with
# no board fields must still get exactly this.
META = {
    "id": "7", "report_num": "012", "tracker_num": 40,
    "url": "https://example.com/jobs/7", "company": "Initech",
    "role": "Outreach Coordinator", "jd_text": "Bilingual outreach to families.",
}
PRE_205_MESSAGE = (
    "Evaluate this job posting.\n\n"
    "**Job ID:** 7\n"
    "**Report Number:** 012\n"
    "**Tracker Number:** 40\n"
    f"**Evaluation Date:** {TODAY}\n"
    "**URL:** https://example.com/jobs/7\n"
    "**Company (source field):** Initech\n"
    "**Role (source field):** Outreach Coordinator\n\n"
    "**Job Description:**\nBilingual outreach to families."
)


class TestNoBoardIsTodaysMessage:
    def test_no_board_key(self):
        assert build_user_message(dict(META), TODAY) == PRE_205_MESSAGE

    @pytest.mark.parametrize("board", [
        None, {}, "Dallas, TX", ["Dallas, TX"],
        {"location": "", "min_amount": "", "max_amount": "nan", "interval": "nan"},
        {"location": float("nan"), "min_amount": True},
        # A currency or an interval with no figure beside it says nothing.
        {"currency": "USD", "interval": "yearly"},
        {"unknown": "key"},
        # The remote flag is not a field, whatever value it carries.
        {"board_remote": True}, {"is_remote": "True"},
    ])
    def test_an_empty_or_malformed_board_adds_nothing(self, board):
        assert build_user_message({**META, "board": board}, TODAY) == PRE_205_MESSAGE


class TestBoardLines:
    def test_every_field_present(self):
        board = {"location": "Dallas, TX", "min_amount": 60000, "max_amount": 75000,
                 "currency": "USD", "interval": "yearly"}
        msg = build_user_message({**META, "board": board}, TODAY)
        assert ("**Role (source field):** Outreach Coordinator\n"
                "**Location (job board):** Dallas, TX\n"
                "**Salary (job board):** 60,000–75,000 USD / yearly\n\n"
                "**Job Description:**\n") in msg

    def test_only_the_fields_present_get_a_line(self):
        msg = build_user_message({**META, "board": {"location": "Dallas, TX"}}, TODAY)
        assert "**Location (job board):** Dallas, TX\n" in msg
        assert "Salary (job board)" not in msg

    def test_no_remote_line_even_when_the_board_says_remote(self):
        msg = build_user_message({**META, "board": {"location": "Dallas, TX",
                                                    "board_remote": True}}, TODAY)
        assert "Remote (job board)" not in msg
        assert "**Location (job board):** Dallas, TX\n" in msg

    @pytest.mark.parametrize("board, line", [
        ({"min_amount": 18.5, "max_amount": 22, "interval": "hourly"},
         "18.50–22.00 / hourly"),
        ({"min_amount": 18.5, "interval": "hourly"}, "from 18.50 / hourly"),
        ({"min_amount": 60000}, "from 60,000"),
        ({"max_amount": 75000, "interval": "yearly"}, "up to 75,000 / yearly"),
        ({"min_amount": 70000, "max_amount": 70000, "interval": "yearly"}, "70,000 / yearly"),
        ({"min_amount": 60000, "max_amount": 75000, "currency": "CAD", "interval": "yearly"},
         "60,000–75,000 CAD / yearly"),
        ({"min_amount": 18, "currency": "USD", "interval": "hourly"}, "from 18 USD / hourly"),
        ({"max_amount": 75000, "currency": "EUR"}, "up to 75,000 EUR"),
    ])
    def test_salary_shapes(self, board, line):
        msg = build_user_message({**META, "board": board}, TODAY)
        assert f"**Salary (job board):** {line}\n" in msg

    @pytest.mark.parametrize("low, high, text", [
        (18.5, 25, "18.50–25.00"),
        (18, 25.75, "18.00–25.75"),
        ("18.5", "25.0", "18.50–25.00"),  # a CSV cell: "25.0" is a whole figure
        (60000, 75000, "60,000–75,000"),
    ])
    def test_both_ends_of_a_range_share_one_precision(self, low, high, text):
        """Formatted end by end, 18.5–25 read "18.50–25", as if the two ends were
        different kinds of figure."""
        board = board_meta_from_row({"min_amount": low, "max_amount": high})
        msg = build_user_message({**META, "board": board}, TODAY)
        assert f"**Salary (job board):** {text}\n" in msg

    @pytest.mark.parametrize("currency", ["", "nan", "NaN", "  ", None, float("nan")])
    def test_a_blank_currency_is_left_out(self, currency):
        board = {"min_amount": 60000, "max_amount": 75000, "currency": currency,
                 "interval": "yearly"}
        assert "currency" not in normalize_board_meta(board)
        msg = build_user_message({**META, "board": board}, TODAY)
        assert "**Salary (job board):** 60,000–75,000 / yearly\n" in msg

    @pytest.mark.parametrize("low, high", [("", ""), ("nan", "nan"), ("NaN", ""),
                                           ("0", ""), ("-5", "inf"), ("sixty", None)])
    def test_no_figure_no_salary_line(self, low, high):
        """An interval alone says nothing, and "nan" in a prompt is not a pay
        figure — pandas' NaN is the one a scraped CSV actually carries."""
        board = board_meta_from_row({"min_amount": low, "max_amount": high,
                                     "currency": "USD", "interval": "yearly"})
        assert board == {}
        msg = build_user_message({**META, "board": board}, TODAY)
        assert "Salary (job board)" not in msg


class TestReversedRange:
    """One valid shape: min never above max, wherever the dict came from."""

    def test_a_reversed_range_is_put_in_order(self):
        assert normalize_board_meta({"min_amount": 75000, "max_amount": 60000}) == {
            "min_amount": 60000, "max_amount": 75000}

    def test_from_the_csv_and_into_the_line(self):
        board = board_meta_from_row({"min_amount": "75000.0", "max_amount": "60000.0",
                                     "currency": "USD", "interval": "yearly"})
        assert (board["min_amount"], board["max_amount"]) == (60000, 75000)
        msg = build_user_message({**META, "board": board}, TODAY)
        assert "**Salary (job board):** 60,000–75,000 USD / yearly\n" in msg

    def test_an_ordered_range_is_unchanged(self):
        board = {"min_amount": 60000, "max_amount": 75000}
        assert normalize_board_meta(board) == board


class TestBoardMetaFromRow:
    """The CSV half: pandas writes the salary columns as floats and a missing
    one blank, and filter carries every column scrape got from JobSpy —
    `salary_source` and `currency` included."""

    def test_reads_filtered_jobs_cells(self):
        row = {"location": "Dallas, TX", "is_remote": "False", "remote_only": "False",
               "salary_source": "direct_data", "min_amount": "60000.0",
               "max_amount": "75000.0", "currency": "USD", "interval": "yearly"}
        assert board_meta_from_row(row) == {
            "location": "Dallas, TX", "min_amount": 60000, "max_amount": 75000,
            "currency": "USD", "interval": "yearly",
        }

    def test_a_row_with_no_board_fields_has_none(self):
        assert board_meta_from_row({"title": "Analyst", "company": "Globex"}) == {}

    def test_a_multi_line_location_stays_one_line(self):
        assert board_meta_from_row({"location": "Dallas,\n  TX"}) == {"location": "Dallas, TX"}

    @pytest.mark.parametrize("is_remote, remote_only", [
        ("True", "True"), ("True", "False"), ("False", "True"), ("False", "False"),
    ])
    def test_the_remote_flag_is_never_carried(self, is_remote, remote_only):
        """A keyword guess, wrong both ways: a LinkedIn row from a remote pass
        arrives False because its description is empty at scrape time, and a
        True only repeats a word the remote guard already required in the JD."""
        board = board_meta_from_row({"location": "Dallas, TX", "is_remote": is_remote,
                                     "remote_only": remote_only})
        assert board == {"location": "Dallas, TX"}
        assert "Remote (job board)" not in build_user_message({**META, "board": board}, TODAY)

    @pytest.mark.parametrize("source", ["description", "Description", " description "])
    def test_a_salary_parsed_from_the_description_gets_no_line(self, source):
        """The model reads that figure in the JD itself, where it IS the posting
        stating pay; labelling it the board's would demote it to an estimate."""
        row = {"location": "Dallas, TX", "salary_source": source, "min_amount": "48000.0",
               "max_amount": "56000.0", "currency": "USD", "interval": "yearly"}
        board = board_meta_from_row(row)
        assert board == {"location": "Dallas, TX"}
        assert "Salary (job board)" not in build_user_message({**META, "board": board}, TODAY)

    @pytest.mark.parametrize("source", ["direct_data", "", "nan", None])
    def test_any_other_salary_is_the_boards(self, source):
        """`direct_data` is the board's figure — the employer's or Indeed's own
        estimate, indistinguishable — and a row with no `salary_source` cell
        (a CSV older than the column) is read the same way."""
        row = {"min_amount": "48000.0", "max_amount": "56000.0", "interval": "yearly"}
        if source is not None:
            row["salary_source"] = source
        assert board_meta_from_row(row) == {"min_amount": 48000, "max_amount": 56000,
                                            "interval": "yearly"}


class TestReadBoardMeta:
    def test_missing_sidecar(self, tmp_path):
        assert read_board_meta(tmp_path, "1", URL) == {}

    @pytest.mark.parametrize("content", [
        b"{not json", b"", b"[1, 2]", b'"Dallas, TX"', b"null", b"\xff\xfe\x00garbage",
    ])
    def test_malformed_sidecar_degrades_without_raising(self, tmp_path, content):
        _write_sidecar(tmp_path, "1", content)
        assert read_board_meta(tmp_path, "1", URL) == {}

    def test_wrong_typed_values_are_dropped_one_by_one(self, tmp_path):
        _write_sidecar(tmp_path, "1", {"url": URL, "location": "Dallas, TX",
                                       "board_remote": "False", "min_amount": [1],
                                       "interval": 3})
        assert read_board_meta(tmp_path, "1", URL) == {"location": "Dallas, TX"}

    def test_the_url_it_was_written_for_matches(self, tmp_path):
        _write_sidecar(tmp_path, "1", {"url": URL, "location": "Dallas, TX"})
        assert read_board_meta(tmp_path, "1", URL) == {"location": "Dallas, TX"}

    @pytest.mark.parametrize("sidecar_url", [
        "https://example.com/jobs/2",   # another posting's, under a reused id
        None,                           # no URL recorded at all
        "", 1,
    ])
    def test_a_sidecar_not_written_for_this_posting_is_rejected(self, tmp_path, sidecar_url):
        content = {"location": "Chicago, IL"}
        if sidecar_url is not None:
            content["url"] = sidecar_url
        _write_sidecar(tmp_path, "1", content)
        assert read_board_meta(tmp_path, "1", URL) == {}

    def test_a_row_with_no_url_matches_nothing(self, tmp_path):
        _write_sidecar(tmp_path, "1", {"url": "", "location": "Chicago, IL"})
        assert read_board_meta(tmp_path, "1", "") == {}

    def test_sits_beside_the_cached_jd(self, tmp_path):
        assert board_meta_path(tmp_path, "9").parent == jd_cache_path(tmp_path, "9").parent


class TestBoardMetaJson:
    def test_records_the_url_beside_the_fields(self):
        text = board_meta_json(URL, {"location": "Dallas, TX", "min_amount": 9, "max_amount": 5})
        assert json.loads(text) == {"url": URL, "location": "Dallas, TX",
                                    "min_amount": 5, "max_amount": 9}
        assert text.endswith("\n")

    def test_no_fields_no_sidecar(self):
        # The URL alone is not worth a file: there is nothing to tell the model.
        assert board_meta_json(URL, {}) == ""
        assert board_meta_json(URL, {"currency": "USD", "board_remote": True}) == ""


class TestAssignJobNumbers:
    def _rows(self):
        return [{"id": "1", "url": URL, "source": "Initech", "notes": "Outreach Coordinator"}]

    def test_the_sidecar_reaches_the_message_but_not_the_state(self, tmp_path):
        _write_sidecar(tmp_path, "1", {"url": URL, "location": "Dallas, TX"})
        state = {"jobs": {}}
        [job] = assign_job_numbers(self._rows(), state, 0, 0, tmp_path)
        assert job["board"] == {"location": "Dallas, TX"}
        assert "board" not in state["jobs"]["1"] and "jd_text" not in state["jobs"]["1"]
        # The state is what is written to batch-api-state.json.
        assert "Dallas" not in json.dumps(state)
        assert "**Location (job board):** Dallas, TX" in build_user_message(job, TODAY)

    def test_another_postings_sidecar_is_not_attached(self, tmp_path):
        """A hand-added queue row reusing an id whose sidecar an earlier posting
        left: that posting's location must not reach this job."""
        _write_sidecar(tmp_path, "1", {"url": "https://example.com/jobs/old",
                                       "location": "Chicago, IL"})
        [job] = assign_job_numbers(self._rows(), {"jobs": {}}, 0, 0, tmp_path)
        assert "board" not in job
        assert "(job board)" not in build_user_message(job, TODAY)

    def test_no_sidecar_no_board(self, tmp_path):
        [job] = assign_job_numbers(self._rows(), {"jobs": {}}, 0, 0, tmp_path)
        assert "board" not in job

    def test_a_malformed_sidecar_does_not_raise(self, tmp_path):
        _write_sidecar(tmp_path, "1", "{truncated")
        [job] = assign_job_numbers(self._rows(), {"jobs": {}}, 0, 0, tmp_path)
        assert "board" not in job
        assert "(job board)" not in build_user_message(job, TODAY)


class TestEndToEnd:
    """filtered_jobs.csv → bridge → batch_prep → the queue → the user message,
    in tmp_path, with the columns filter actually writes: its summary columns
    first, then the rest of what scrape got from JobSpy."""

    COLUMNS = ["title", "company", "location", "is_remote", "remote_only",
               "min_amount", "max_amount", "interval", "job_url", "date_posted",
               "relevance_score", "matched_keywords", "salary_source", "currency",
               "description"]
    ROWS = [
        # The #205 case: the description never says where the job is.
        {"title": "Outreach Coordinator", "company": "Initech", "location": "Dallas, TX",
         "is_remote": "False", "remote_only": "False", "min_amount": "48000.0",
         "max_amount": "56000.0", "interval": "yearly",
         "job_url": "https://www.linkedin.com/jobs/view/101", "date_posted": "2026-11-20",
         "relevance_score": "12", "matched_keywords": "outreach",
         "salary_source": "direct_data", "currency": "USD",
         "description": "Bilingual outreach to families. Spanish required."},
        # Flagged remote by the board, with the pay parsed out of the JD itself.
        {"title": "Patient Access Representative", "company": "Globex",
         "location": "Dallas, TX", "is_remote": "True", "remote_only": "True",
         "min_amount": "18.0", "max_amount": "22.0", "interval": "hourly",
         "job_url": "https://www.indeed.com/viewjob?jk=202", "date_posted": "2026-11-19",
         "relevance_score": "10", "matched_keywords": "patient access",
         "salary_source": "description", "currency": "USD",
         "description": "Remote patient registration. Pay: $18-$22 an hour."},
        # No board fields at all: exactly today's message.
        {"title": "Program Analyst", "company": "Acme", "location": "",
         "is_remote": "", "remote_only": "", "min_amount": "nan", "max_amount": "nan",
         "interval": "", "job_url": "https://www.indeed.com/viewjob?jk=303",
         "date_posted": "2026-11-18", "relevance_score": "9",
         "matched_keywords": "analyst", "salary_source": "", "currency": "",
         "description": "Analyse programs."},
    ]

    @pytest.fixture
    def queued(self, career_ops_dir, tmp_path, monkeypatch):
        filtered = tmp_path / "filtered_jobs.csv"
        write_rows(filtered, self.ROWS, self.COLUMNS)  # filter's own writer
        monkeypatch.setattr(bridge, "FILTERED_PATH", filtered)
        offers = bridge.run(career_ops_dir)
        assert batch_prep.run(career_ops_dir, offers) == 3
        return career_ops_dir

    def _ids(self, career_ops):
        tsv = career_ops / batch_prep.BATCH_INPUT
        return {r["source"]: r["id"] for r in csv.DictReader(
            tsv.open(encoding="utf-8"), delimiter="\t")}

    def _messages(self, career_ops):
        pending = load_pending(career_ops / batch_prep.BATCH_INPUT, {"jobs": {}})
        state = {"jobs": {}}
        jobs = assign_job_numbers(pending, state, 0, 0, career_ops)
        return {j["company"]: build_user_message(j, TODAY) for j in jobs}, state

    def test_the_board_lines_reach_the_evaluator(self, queued):
        messages, _ = self._messages(queued)
        initech = messages["Initech"]
        assert "**Location (job board):** Dallas, TX\n" in initech
        assert "**Salary (job board):** 48,000–56,000 USD / yearly\n" in initech
        globex = messages["Globex"]
        assert "**Location (job board):** Dallas, TX\n" in globex
        # Description-sourced pay: the model reads it in the JD.
        assert "Salary (job board)" not in globex
        assert "$18-$22 an hour" in globex
        assert "(job board)" not in messages["Acme"]
        # No remote line for anyone, the board-flagged row included.
        assert not any("Remote (job board)" in m for m in messages.values())

    def test_each_sidecar_records_its_posting(self, queued):
        ids = self._ids(queued)
        for row in self.ROWS[:2]:
            sidecar = json.loads(board_meta_path(queued, ids[row["company"]])
                                 .read_text(encoding="utf-8"))
            assert sidecar["url"] == row["job_url"]
            assert "board_remote" not in sidecar and "is_remote" not in sidecar

    def test_the_queue_and_the_jd_cache_are_unchanged(self, queued):
        tsv = queued / batch_prep.BATCH_INPUT
        lines = tsv.read_text(encoding="utf-8").splitlines()
        assert lines[0].split("\t") == batch_prep.FIELDNAMES
        assert all(len(line.split("\t")) == 4 for line in lines)
        ids = self._ids(queued)
        for row in self.ROWS:
            jd = jd_cache_path(queued, ids[row["company"]])
            assert jd.read_text(encoding="utf-8") == row["description"]
        # A row with no board fields gets no sidecar at all.
        assert not board_meta_path(queued, ids["Acme"]).exists()
        assert board_meta_path(queued, ids["Initech"]).exists()

    def test_pipeline_md_and_scan_history_do_not_carry_it(self, queued):
        pipeline_md = (queued / bridge.PIPELINE_MD).read_text(encoding="utf-8")
        assert "- [ ] https://www.linkedin.com/jobs/view/101 | Initech | Outreach Coordinator\n" in pipeline_md
        assert "Dallas" not in pipeline_md and "48000" not in pipeline_md
        history = (queued / bridge.SCAN_HISTORY).read_text(encoding="utf-8")
        assert history.splitlines()[0] == "url\tfirst_seen\tportal\ttitle\tcompany\tstatus"
        assert "Dallas" not in history


class TestReusedId:
    """batch_prep reuses an id only when batch-input.tsv was cleared and
    batch/jds was not; a file left under it would hand the new job another
    posting's location or description."""

    def test_a_reused_id_with_no_board_fields_loses_the_old_sidecar(self, tmp_path):
        stale = _write_sidecar(tmp_path, "1", {"url": "https://example.com/1",
                                               "location": "Chicago, IL"})
        batch_prep.run(tmp_path, [{"url": "https://example.com/1", "company": "Acme",
                                   "title": "Analyst", "description": "JD"}])
        assert not stale.exists()

    def test_a_reused_id_with_no_description_loses_the_old_jd(self, tmp_path):
        stale = jd_cache_path(tmp_path, "1")
        stale.parent.mkdir(parents=True)
        stale.write_text("Another posting's description.", encoding="utf-8")
        batch_prep.run(tmp_path, [{"url": "https://example.com/1", "company": "Acme",
                                   "title": "Analyst", "description": ""}])
        assert not stale.exists()
        [job] = assign_job_numbers(load_pending(tmp_path / batch_prep.BATCH_INPUT, {"jobs": {}}),
                                   {"jobs": {}}, 0, 0, tmp_path)
        assert job["jd_text"] == ""

    def test_a_reused_id_with_board_fields_gets_its_own(self, tmp_path):
        _write_sidecar(tmp_path, "1", {"url": "https://example.com/old",
                                       "location": "Chicago, IL"})
        batch_prep.run(tmp_path, [{"url": "https://example.com/1", "company": "Acme",
                                   "title": "Analyst", "description": "JD",
                                   "board": {"location": "Dallas, TX"}}])
        [job] = assign_job_numbers(load_pending(tmp_path / batch_prep.BATCH_INPUT, {"jobs": {}}),
                                   {"jobs": {}}, 0, 0, tmp_path)
        assert job["board"] == {"location": "Dallas, TX"}

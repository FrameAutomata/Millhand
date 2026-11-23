"""Tests for pipeline/state_sync.py — the Push and Refresh cores, FastAPI-free.

The route tests in test_app_server.py drive these through HTTP, and they skip
wholesale where fastapi is not installed; these run everywhere, which is the
point of the module."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline import state_sync
from pipeline._batch_common import read_process_lock
from pipeline.app import data

ROOT = Path(__file__).resolve().parent.parent

TRACKER = (
    "# Applications Tracker\n"
    "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
    "|---|------|---------|------|-------|--------|-----|--------|-------|\n"
    "| 1 | 2026-05-27 | Acme | Eng | 4.2/5 | Evaluated | ❌ | [001](reports/001-acme.md) | APPLY |\n"
)


@pytest.fixture
def career_ops(tmp_path):
    co = tmp_path / "career-ops"
    (co / "data").mkdir(parents=True)
    (co / "data" / "applications.md").write_text(TRACKER, encoding="utf-8")
    return co


@pytest.fixture
def offline(monkeypatch):
    """No successful run, so Push uses the local base; returns the dispatches."""
    monkeypatch.setattr(state_sync.gh, "latest_successful_run", lambda *a, **k: None)
    sent = []
    monkeypatch.setattr(state_sync.gh, "trigger_workflow", lambda *a: sent.append(a))
    return sent


# ── Import shape ─────────────────────────────────────────────────────────────

def test_importable_without_ui_deps():
    """A non-UI caller runs where the UI's dependencies are not installed — the
    daily's venv is requirements.txt only. A fresh interpreter, because conftest
    has already imported the server into this one, so blocking the names here
    would prove nothing."""
    probe = ("import sys; sys.modules['fastapi'] = None; sys.modules['pydantic'] = None; "
             "sys.modules['markdown'] = None; "
             "import pipeline.state_sync as m; "
             "assert m.push_pending and m.refresh and m.SyncError")
    res = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ── Path isolation ───────────────────────────────────────────────────────────

class TestPathIsolation:
    """UI_CACHE is rmtree'd before every download. The isolation that keeps a
    test off the real one used to be three per-fixture assignments on the
    server module; it is one conftest autouse fixture now, and these hold it."""

    def test_both_paths_point_into_this_tests_tmp(self, tmp_path):
        assert state_sync.UI_CACHE.is_relative_to(tmp_path)
        assert state_sync.PUSHED_OVERRIDES_FILE.is_relative_to(tmp_path)

    def test_the_server_binds_no_copy_of_them(self):
        """A copy bound in server.py is one a test could redirect while the
        core went on using — and wiping — the real path."""
        tree = ast.parse((ROOT / "pipeline" / "app" / "server.py").read_text(encoding="utf-8"))
        bound = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound |= {a.asname or a.name for a in node.names}
        assert "state_sync" in bound, "the scan stopped seeing the server's imports"
        assert not bound & {"UI_CACHE", "PUSHED_OVERRIDES_FILE"}


# ── Push ─────────────────────────────────────────────────────────────────────

class TestPushPending:
    def test_nothing_pending_is_a_400(self, career_ops, offline):
        with pytest.raises(state_sync.SyncError) as e:
            state_sync.push_pending(career_ops)
        assert e.value.status == 400
        assert offline == []

    def test_no_tracker_is_a_409(self, tmp_path, offline):
        data.record_status_override("1", "Applied")
        with pytest.raises(state_sync.SyncError) as e:
            state_sync.push_pending(tmp_path / "no-career-ops")
        assert e.value.status == 409

    def test_dispatches_and_moves_the_edit_to_the_pushed_overlay(self, career_ops, offline):
        data.record_status_override("1", "Applied")
        assert state_sync.push_pending(career_ops) == {"pushed": 1, "unresolved": 0, "base": "local"}
        ((workflow, fields),) = offline
        assert workflow == state_sync.EDIT_WORKFLOW
        assert json.loads(fields["status_overrides_json"]) == {"1": "Applied"}
        assert data.load_status_overrides() == {}
        assert state_sync.load_pushed_overrides() == {"1": "Applied"}

    def test_an_edit_made_during_the_dispatch_stays_pending(self, career_ops, monkeypatch):
        # The card is dragged again while gh is dispatching: the cloud receives
        # the old status, so the new one must survive the clear for the next push.
        monkeypatch.setattr(state_sync.gh, "latest_successful_run", lambda *a, **k: None)
        monkeypatch.setattr(state_sync.gh, "trigger_workflow",
                            lambda *a: data.record_status_override("1", "Rejected"))
        data.record_status_override("1", "Applied")
        state_sync.push_pending(career_ops)
        assert data.load_status_overrides() == {"1": "Rejected"}

    def test_a_scratch_dir_it_cannot_empty_is_an_error_not_offline(self, career_ops, monkeypatch):
        # gh.download_artifact ignores rmtree errors; a surviving file would make
        # `gh run download` fail as a GhError, which Push reads as offline and
        # so skips its clobber guard without saying so.
        monkeypatch.setattr(state_sync.gh, "latest_successful_run",
                            lambda *a, **k: {"databaseId": 1})
        state_sync.UI_CACHE.mkdir(parents=True)

        def held(path):
            raise PermissionError(32, "in use")
        monkeypatch.setattr(state_sync.shutil, "rmtree", held)
        data.record_status_override("1", "Applied")
        with pytest.raises(PermissionError):
            state_sync.push_pending(career_ops)
        assert data.load_status_overrides() == {"1": "Applied"}

    def test_a_failed_dispatch_is_a_502_and_the_edit_stays_pending(self, career_ops, monkeypatch):
        monkeypatch.setattr(state_sync.gh, "latest_successful_run", lambda *a, **k: None)

        def fail(*a):
            raise state_sync.gh.GhError("edit-tracker.yml not found")
        monkeypatch.setattr(state_sync.gh, "trigger_workflow", fail)
        data.record_status_override("1", "Applied")
        with pytest.raises(state_sync.SyncError) as e:
            state_sync.push_pending(career_ops)
        assert e.value.status == 502 and "edit-tracker" in e.value.detail
        assert data.load_status_overrides() == {"1": "Applied"}

    def test_both_override_files_are_written_under_their_process_lock(self, career_ops, offline,
                                                                      monkeypatch):
        """The overrides are edited from more than one process (the UI, a local
        run's subprocess, a Push outside the UI), and the thread lock alone
        serializes only threads."""
        data.record_status_override("1", "Applied")
        holders = []

        save = data.save_status_overrides

        def watching(overrides, path=None):
            path = path or data.STATUS_OVERRIDES_FILE
            holders.append((path.name, read_process_lock(data.override_lock_path(path))[0]))
            save(overrides, path)
        monkeypatch.setattr(data, "save_status_overrides", watching)
        state_sync.push_pending(career_ops)
        assert holders == [(data.STATUS_OVERRIDES_FILE.name, os.getpid()),        # the clear
                           (state_sync.PUSHED_OVERRIDES_FILE.name, os.getpid())]  # the overlay
        for path in (data.STATUS_OVERRIDES_FILE, state_sync.PUSHED_OVERRIDES_FILE):
            assert not data.override_lock_path(path).exists()   # released


# ── The pushed overlay ───────────────────────────────────────────────────────

class TestPushedOverlay:
    @staticmethod
    def _row(num, status):
        return {"num": num, "status": status, "status_canonical": status}

    def test_shows_a_pushed_status_until_an_artifact_catches_up(self):
        data.save_status_overrides({"1": "Applied", "2": "Rejected"}, state_sync.PUSHED_OVERRIDES_FILE)
        rows = [self._row("1", "Evaluated"), self._row("2", "Rejected")]
        state_sync.apply_pushed_overlay(rows)
        assert rows[0]["status_canonical"] == "Applied"
        assert state_sync.load_pushed_overrides() == {"1": "Applied"}   # 2 caught up

    def test_keeps_an_entry_a_push_rewrote_since_the_load(self, monkeypatch):
        """The self-clean re-reads under the lock: a Push that lands between the
        load and the clean has written a NEW status for that row, not yet in
        any artifact, and dropping it would revert the card."""
        data.save_status_overrides({"2": "Rejected"}, state_sync.PUSHED_OVERRIDES_FILE)
        load = data.load_status_overrides
        first = []

        def push_lands_after_the_first_load(path=None):
            got = load(path)
            if not first:
                first.append(got)
                data.save_status_overrides({"2": "Interview"}, path)
            return got
        monkeypatch.setattr(data, "load_status_overrides", push_lands_after_the_first_load)
        state_sync.apply_pushed_overlay([self._row("2", "Rejected")])
        assert load(state_sync.PUSHED_OVERRIDES_FILE) == {"2": "Interview"}


# ── Refresh ──────────────────────────────────────────────────────────────────

class TestRefresh:
    def test_no_successful_run_is_a_404(self, career_ops, offline):
        with pytest.raises(state_sync.SyncError) as e:
            state_sync.refresh(career_ops)
        assert e.value.status == 404

    def test_offline_is_a_502_and_leaves_local_intact(self, career_ops, monkeypatch):
        def fail(*a, **k):
            raise state_sync.gh.GhError("billing limit reached")
        monkeypatch.setattr(state_sync.gh, "latest_successful_run", fail)
        with pytest.raises(state_sync.SyncError) as e:
            state_sync.refresh(career_ops)
        assert e.value.status == 502 and "last-synced" in e.value.detail
        assert (career_ops / "data" / "applications.md").read_text(encoding="utf-8") == TRACKER

    def test_merges_the_artifact(self, career_ops, tmp_path, monkeypatch):
        art = tmp_path / "dl" / "pipeline-output-7"
        (art / "data").mkdir(parents=True)
        (art / "data" / "applications.md").write_text(
            TRACKER.replace("| 1 | 2026-05-27 | Acme |", "| 9 | 2026-05-27 | Refreshed Co |"),
            encoding="utf-8")
        monkeypatch.setattr(state_sync.gh, "latest_successful_run",
                            lambda *a, **k: {"databaseId": 7, "createdAt": "t", "displayTitle": "Daily"})
        monkeypatch.setattr(state_sync.gh, "download_artifact", lambda run_id, dest: art)
        result = state_sync.refresh(career_ops)
        assert (result["run_id"], result["created_at"], result["title"]) == (7, "t", "Daily")
        assert "Refreshed Co" in (career_ops / "data" / "applications.md").read_text(encoding="utf-8")

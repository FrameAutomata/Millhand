"""Route smoke tests for pipeline/app/server.py.

Skips entirely if FastAPI isn't installed (it's an optional UI dependency in
requirements-ui.txt, not part of the core pipeline deps)."""

import importlib
import shutil
from pathlib import Path

import pytest

from pipeline import agent_cli, state_sync
from tests.test_agent_cli import _which

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_app_onboard import RETIRED_FORM_VALUES  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Point the server at a tmp career-ops dir with one tracker row + report,
    then return a TestClient bound to a freshly-imported app."""
    career_ops = tmp_path / "career-ops"
    (career_ops / "data").mkdir(parents=True)
    (career_ops / "reports").mkdir(parents=True)
    (career_ops / "data" / "applications.md").write_text(
        "# Applications Tracker\n"
        "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
        "|---|------|---------|------|-------|--------|-----|--------|-------|\n"
        "| 1 | 2026-05-27 | Acme | Eng | 4.2/5 | Evaluated | ❌ | [001](reports/001-acme.md) | APPLY |\n",
        encoding="utf-8",
    )
    (career_ops / "reports" / "001-acme.md").write_text(
        "# Acme — Eng\n\n**Score:** 4.2/5\n\nGreat match.", encoding="utf-8"
    )
    monkeypatch.setenv("CAREER_OPS_PATH", str(career_ops))

    # Re-import the server so its module-level paths pick up the env var. The
    # _career_ops() helper reads the env at request time, so a plain import is
    # fine, but reload keeps the test hermetic across runs.
    from pipeline.app import server
    importlib.reload(server)
    # The .ui-cache paths (status-overrides, pushed-overrides, the artifact
    # scratch dir) are isolated by autouse conftest fixtures.
    return TestClient(server.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["applications_md_exists"] is True


def test_list_jobs(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    payload = r.json()
    assert payload["source"] == "applications"
    jobs = payload["rows"]
    assert len(jobs) == 1
    assert jobs[0]["company"] == "Acme"
    assert jobs[0]["score_value"] == 4.2
    assert jobs[0]["report_num"] == "001"


def test_get_report_renders(client):
    r = client.get("/api/reports/001")
    assert r.status_code == 200
    assert "Acme" in r.text
    assert "Great match" in r.text


def test_get_report_404_for_unknown(client):
    r = client.get("/api/reports/999")
    assert r.status_code == 404


def test_index_served(client):
    # The SPA shell should be served at root.
    r = client.get("/")
    assert r.status_code == 200
    assert "Triage" in r.text


def test_run_triggers_workflow(client, mocker):
    from pipeline.app import server
    trigger = mocker.patch.object(server.gh, "trigger_workflow")
    r = client.post("/api/run")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    trigger.assert_called_once_with(server.DAILY_WORKFLOW)


def test_run_surfaces_gh_error(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.gh, "trigger_workflow",
                        side_effect=server.gh.GhError("gh not authenticated"))
    r = client.post("/api/run")
    assert r.status_code == 502
    assert "not authenticated" in r.json()["detail"]


def test_refresh_404_when_no_runs(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.gh, "latest_successful_run", return_value=None)
    r = client.post("/api/refresh")
    assert r.status_code == 404


def test_refresh_downloads_and_repoints(client, tmp_path, mocker):
    from pipeline.app import server
    art = tmp_path / "dl" / "pipeline-output-7"
    (art / "data").mkdir(parents=True)
    (art / "reports").mkdir(parents=True)
    (art / "data" / "applications.md").write_text(
        "# Applications Tracker\n"
        "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
        "|---|------|---------|------|-------|--------|-----|--------|-------|\n"
        "| 9 | 2026-05-27 | Refreshed Co | Eng | 4.9/5 | Evaluated | ❌ | [009](reports/009-x.md) | APPLY |\n",
        encoding="utf-8",
    )
    mocker.patch.object(server.gh, "latest_successful_run",
                        return_value={"databaseId": 7, "createdAt": "t", "displayTitle": "Daily"})
    mocker.patch.object(server.gh, "download_artifact", return_value=art)
    r = client.post("/api/refresh")
    assert r.status_code == 200
    assert r.json()["run_id"] == 7
    # Refresh merges the artifact INTO local; /api/jobs (always local) shows it.
    jobs = client.get("/api/jobs").json()
    assert any(row["company"] == "Refreshed Co" for row in jobs["rows"])


def test_set_status_records_override(client):
    r = client.post("/api/status", json={"num": "1", "status": "Applied"})
    assert r.status_code == 200
    assert r.json()["pending"] == 1
    # /api/jobs should now overlay the pending status on that row.
    jobs = client.get("/api/jobs").json()
    assert jobs["pending"] == 1
    row = jobs["rows"][0]
    assert row["status_canonical"] == "Applied"
    assert row["pending"] is True


def test_set_status_rejects_unknown(client):
    r = client.post("/api/status", json={"num": "1", "status": "Bogus"})
    assert r.status_code == 400
    assert "Unknown status" in r.json()["detail"]


def test_refresh_offline_keeps_local_intact(client, mocker):
    # No credits / gh error: Refresh must not wipe or alter the local tracker —
    # the user keeps working against the last-synced data.
    from pipeline.app import server
    local_apps = server._career_ops_local() / "data" / "applications.md"
    before = local_apps.read_text(encoding="utf-8")
    mocker.patch.object(server.gh, "latest_successful_run",
                        side_effect=server.gh.GhError("billing limit reached"))
    r = client.post("/api/refresh")
    assert r.status_code == 502
    assert "last-synced" in r.json()["detail"]
    assert local_apps.read_text(encoding="utf-8") == before


def test_load_eval_system_prompt_uses_profile_master(tmp_path, monkeypatch):
    # The UI add-job eval must resolve the living PROFILE.md exactly like
    # --evaluate-batch (add_job's parity contract), not the stale seeds.
    career_ops = tmp_path / "career-ops"
    (career_ops / "config").mkdir(parents=True)
    (career_ops / "cv.md").write_text("MYCV_XYZ", encoding="utf-8")
    (career_ops / "config" / "profile.yml").write_text("PROFILEYAML_XYZ", encoding="utf-8")
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir()
    (handoff_dir / "PROFILE.md").write_text("LIVING MASTER PROFILE", encoding="utf-8")
    monkeypatch.setenv("CAREER_OPS_PATH", str(career_ops))
    monkeypatch.setenv("HANDOFF_OUT_DIR", str(handoff_dir))

    from pipeline.app import server
    prompt = server._load_eval_system_prompt()
    assert "LIVING MASTER PROFILE" in prompt
    assert "MYCV_XYZ" not in prompt          # master supersedes the seeds on the UI path too


def test_template_status_reports_available(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.self_update, "update_available",
                        return_value={"available": True, "template_sha": "abc123"})
    r = client.get("/api/template/status")
    assert r.status_code == 200
    assert r.json() == {"available": True, "template_sha": "abc123"}


def test_template_update_success(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.self_update, "apply_update",
                        return_value={"ok": True, "updated": True})
    r = client.post("/api/template/update")
    assert r.status_code == 200
    assert r.json()["updated"] is True


def test_template_update_conflict_returns_409(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.self_update, "apply_update",
                        return_value={"ok": False, "conflict": True, "error": "merge conflict"})
    r = client.post("/api/template/update")
    assert r.status_code == 409
    assert "conflict" in r.json()["detail"].lower()


def test_reset_requires_confirm(client, mocker):
    from pipeline.app import server
    rj = mocker.patch.object(server.reset, "reset_job_search")
    r = client.post("/api/reset", json={"confirm": "nope"})
    assert r.status_code == 400
    rj.assert_not_called()   # nothing wiped without the exact confirmation


def test_reset_runs_and_clears_cloud(client, mocker):
    from pipeline.app import server
    rj = mocker.patch.object(server.reset, "reset_job_search", return_value={"removed": ["a"], "count": 1})
    cc = mocker.patch.object(server.reset, "clear_cloud_caches", return_value={"deleted": ["pipeline-state-v1-1"]})
    r = client.post("/api/reset", json={"confirm": "RESET", "clear_cloud": True})
    assert r.status_code == 200
    rj.assert_called_once()
    cc.assert_called_once()
    assert r.json()["cloud"]["deleted"] == ["pipeline-state-v1-1"]


def test_reset_local_only(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.reset, "reset_job_search", return_value={"removed": []})
    cc = mocker.patch.object(server.reset, "clear_cloud_caches")
    r = client.post("/api/reset", json={"confirm": "RESET", "clear_cloud": False})
    assert r.status_code == 200
    cc.assert_not_called()


def test_reset_succeeds_even_if_cloud_clear_fails(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.reset, "reset_job_search", return_value={"removed": []})
    mocker.patch.object(server.reset, "clear_cloud_caches",
                        side_effect=server.gh.GhError("not authenticated"))
    r = client.post("/api/reset", json={"confirm": "RESET", "clear_cloud": True})
    assert r.status_code == 200
    assert "cloud_error" in r.json()


def test_status_write_on_a_recheck_closed_row_carries_a_by_hand_mark(client):
    """A person's Discard after a re-check Discard read as the re-check's —
    bridge let the re-post through and the merge bounced it (#163). The write
    now leaves a by-hand mark, through the same channel Push carries."""
    from pipeline.app import data, server
    apps = server._career_ops() / "data" / "applications.md"
    apps.write_text(apps.read_text(encoding="utf-8").replace(
        "| APPLY |", "| APPLY — Closed 2026-09-06 (liveness re-check: HTTP 404) |"), encoding="utf-8")
    assert client.post("/api/status", json={"num": "1", "status": "Discarded"}).status_code == 200
    override = data.load_status_overrides()["1"]
    assert data.override_status(override) == "Discarded"
    note = data.override_note(override)
    assert note.startswith("Set Discarded ") and note.endswith("(by hand)")
    assert client.get("/api/jobs").json()["rows"][0]["status_canonical"] == "Discarded"


def test_an_evaluator_skip_reaches_the_board_and_a_status_write_lifts_it(client):
    """#183: the row field the UI tags by is on /api/jobs, and a person setting
    the status of a Skip-decided row by hand leaves the mark that overrules the
    model, through the same override + Push channel the Closed-mark case uses."""
    from pipeline.app import data, server
    apps = server._career_ops() / "data" / "applications.md"
    apps.write_text(apps.read_text(encoding="utf-8").replace(
        "| APPLY |", "| https://x.example/1 — Decision: Skip — SKIP weak |"), encoding="utf-8")
    assert client.get("/api/jobs").json()["rows"][0]["skip_decided"] is True
    assert client.post("/api/status", json={"num": "1", "status": "Interview"}).status_code == 200
    note = data.override_note(data.load_status_overrides()["1"])
    assert note.startswith("Set Interview ") and note.endswith("(by hand)")


def test_rewriting_a_skip_rows_own_status_is_the_lift(client):
    """#183: a tagged row is Evaluated, so the lift the UI offers is a write of
    the status the row already has. The endpoint takes it and records the mark,
    which Push then carries — a same-status write is not a no-op here."""
    from pipeline.app import data, server
    apps = server._career_ops() / "data" / "applications.md"
    text = apps.read_text(encoding="utf-8").replace(
        "| APPLY |", "| https://x.example/1 — Decision: Skip — SKIP weak |")
    apps.write_text(text, encoding="utf-8")
    row = client.get("/api/jobs").json()["rows"][0]
    assert row["skip_decided"] is True and row["status_canonical"] == "Evaluated"
    assert client.post("/api/status", json={"num": "1", "status": "Evaluated"}).status_code == 200
    value = data.load_status_overrides()["1"]
    assert data.override_status(value) == "Evaluated"
    note = data.override_note(value)
    assert note.startswith("Set Evaluated ") and note.endswith("(by hand)")
    # What edit-tracker applies in the cloud: the status unchanged, the mark on.
    pushed, changed = data.apply_cloud_overrides(text, {"1": value})
    assert changed == 1 and not data.parse_applications_text(pushed)[0]["skip_decided"]
    # …and the board shows it at once: the pending overlay applies the mark's
    # effect, not just the status, so the tag goes when the card is moved.
    assert client.get("/api/jobs").json()["rows"][0]["skip_decided"] is False


def test_a_landed_lift_is_caught_up_after_a_later_re_eval_joins_onto_it():
    """The cloud appends the mark at the END of the cell and merge-tracker joins
    the next re-evaluation straight onto it with "; " — the mark is then no
    longer a clause of its own, and an exact-clause test held the entry (and
    pinned the card's status) for good."""
    from pipeline.app import data
    mark = "Set Evaluated 2026-11-24 (by hand)"
    notes = (f"https://x.example/1 — Work location: On-site Dallas, TX — Decision: Skip"
             f" — SKIP weak — {mark}; Re-eval 2026-11-25 (2.0→2.1): req 1 — APPLY")
    assert data.note_landed(notes, mark)
    assert not data.note_landed(notes, "Set Evaluated 2026-11-30 (by hand)")   # a later lift
    assert not data.note_landed(notes, "")


def test_a_pushed_lift_holds_until_the_tracker_carries_its_mark(client, monkeypatch):
    """After Push the pending override is gone and the status already matches,
    so a status-only overlay dropped the lift at once and the tag came back
    until the next artifact. The pushed entry keeps its note and is caught up
    only when the row carries it."""
    from pipeline import state_sync
    from pipeline.app import data, server
    apps = server._career_ops() / "data" / "applications.md"
    skip_text = apps.read_text(encoding="utf-8").replace(
        "| APPLY |", "| https://x.example/1 — Decision: Skip — SKIP weak |")
    apps.write_text(skip_text, encoding="utf-8")
    client.post("/api/status", json={"num": "1", "status": "Evaluated"})
    value = data.load_status_overrides()["1"]
    monkeypatch.setattr(state_sync.gh, "latest_successful_run", lambda *a, **k: None)
    monkeypatch.setattr(state_sync.gh, "trigger_workflow", lambda *a: None)
    assert client.post("/api/push-status").status_code == 200
    assert data.load_status_overrides() == {}
    kept = state_sync.load_pushed_overrides()["1"]
    assert (data.override_status(kept), data.override_note(kept)) == (
        "Evaluated", data.override_note(value))                     # note kept
    assert client.get("/api/jobs").json()["rows"][0]["skip_decided"] is False
    assert state_sync.load_pushed_overrides()["1"] == kept           # not caught up yet
    # The next artifact carries the mark: caught up, and the parser agrees.
    landed, _ = data.apply_cloud_overrides(skip_text, {"1": value})
    apps.write_text(landed, encoding="utf-8")
    assert client.get("/api/jobs").json()["rows"][0]["skip_decided"] is False
    assert "1" not in state_sync.load_pushed_overrides()


def test_an_ordinary_drag_stays_unmarked(client):
    from pipeline.app import data
    client.post("/api/status", json={"num": "1", "status": "Applied"})
    assert data.load_status_overrides()["1"] == "Applied"


def test_push_status_400_when_nothing_pending(client):
    r = client.post("/api/push-status")
    assert r.status_code == 400


def test_push_status_refreshes_applies_and_dispatches(client, tmp_path, mocker):
    from pipeline.app import server
    # Pending change: mark role #1 Applied.
    client.post("/api/status", json={"num": "1", "status": "Applied"})

    # Fresh base from "the cloud" — includes a NEW row #2 the local copy lacks,
    # to prove the refresh-before-write guard preserves cloud-added rows.
    fresh = tmp_path / "fresh" / "pipeline-output-9"
    (fresh / "data").mkdir(parents=True)
    (fresh / "reports").mkdir(parents=True)
    (fresh / "data" / "applications.md").write_text(
        "# Applications Tracker\n"
        "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
        "|---|------|---------|------|-------|--------|-----|--------|-------|\n"
        "| 1 | 2026-05-27 | Acme | Eng | 4.2/5 | Evaluated | ❌ | [001](reports/001-acme.md) | APPLY |\n"
        "| 2 | 2026-05-28 | NewCo | Dev | 3.9/5 | Evaluated | ❌ | [002](reports/002-newco.md) | new |\n",
        encoding="utf-8",
    )
    mocker.patch.object(server.gh, "latest_successful_run", return_value={"databaseId": 9})
    mocker.patch.object(server.gh, "download_artifact", return_value=fresh)
    trigger = mocker.patch.object(server.gh, "trigger_workflow")

    r = client.post("/api/push-status")
    assert r.status_code == 200
    body = r.json()
    assert body["pushed"] == 1
    assert body["base"] == "refreshed"

    # edit-tracker was dispatched with a status_overrides_json payload.
    trigger.assert_called_once()
    wf, fields = trigger.call_args.args[0], trigger.call_args.args[1]
    assert wf == state_sync.EDIT_WORKFLOW
    import json
    overrides = json.loads(fields["status_overrides_json"])
    assert overrides == {"1": "Applied"}

    # Pending cleared after a successful push.
    assert client.get("/api/jobs").json()["pending"] == 0


def test_push_carries_a_recheck_mark_to_the_cloud(client, mocker):
    """A Discard the local re-check minted reached the cloud as a bare status
    — a person's Discard, which the cloud never reopens — and the next Refresh
    erased the mark here too (#163). The mark rides in the payload; the board's
    post-push overlay stays status-only."""
    import json
    from pipeline import state_sync
    from pipeline.app import data, server
    mark = "Closed 2026-09-06 (liveness re-check: HTTP 404)"
    local_apps = server._career_ops_local() / "data" / "applications.md"
    data.record_status_changes(local_apps, [("1", "Discarded", "Acme", "Eng")], notes={"1": mark})
    mocker.patch.object(server.gh, "latest_successful_run", return_value=None)   # offline: local base
    trigger = mocker.patch.object(server.gh, "trigger_workflow")

    assert client.post("/api/push-status").status_code == 200
    payload = json.loads(trigger.call_args.args[1]["status_overrides_json"])
    assert payload == {"1": {"status": "Discarded", "note": mark}}
    assert json.loads(state_sync.PUSHED_OVERRIDES_FILE.read_text(encoding="utf-8")) == {"1": "Discarded"}
    assert client.get("/api/jobs").json()["rows"][0]["status_canonical"] == "Discarded"


def test_pushed_change_survives_post_push_reload_and_refresh(client, tmp_path, mocker):
    """A pushed status must stay visible across the post-push board reload AND a
    later Refresh, until a genuinely fresh pipeline run incorporates it. The bug:
    push edits the downloaded artifact copy, so the post-push /api/jobs reload
    self-cleans the pushed-override bridge against a copy push itself changed —
    and the next Refresh (re-downloading the still-unchanged cloud artifact) then
    shows the status reverting (it 'vanishes')."""
    from pipeline.app import server
    import json

    def artifact(name):
        # The cloud tracker still shows #1 as Evaluated (it hasn't incorporated
        # the pushed change yet — that happens on the next pipeline run).
        d = tmp_path / name / "pipeline-output-9"
        (d / "data").mkdir(parents=True)
        (d / "data" / "applications.md").write_text(
            "# Applications Tracker\n"
            "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
            "|---|------|---------|------|-------|--------|-----|--------|-------|\n"
            "| 1 | 2026-05-27 | Acme | Eng | 4.2/5 | Evaluated | ❌ | [001](reports/001-acme.md) | APPLY |\n",
            encoding="utf-8")
        return d

    client.post("/api/status", json={"num": "1", "status": "Applied"})
    mocker.patch.object(server.gh, "latest_successful_run", return_value={"databaseId": 9})
    mocker.patch.object(server.gh, "download_artifact", return_value=artifact("push"))
    mocker.patch.object(server.gh, "trigger_workflow")
    assert client.post("/api/push-status").json()["pushed"] == 1

    client.get("/api/jobs")  # the UI reloads the board right after a push

    # User clicks Refresh to check the remote — re-downloads the (unchanged) artifact.
    mocker.patch.object(server.gh, "download_artifact", return_value=artifact("refresh"))
    client.post("/api/refresh")

    rows = {r["num"]: r for r in client.get("/api/jobs").json()["rows"]}
    assert rows["1"]["status_canonical"] == "Applied"        # must NOT have vanished
    pushed = json.loads((tmp_path / ".ui-cache" / "pushed-overrides.json").read_text())
    assert pushed == {"1": "Applied"}                        # bridge retained until a fresh run


def test_push_status_falls_back_to_local_when_refresh_fails(client, mocker):
    from pipeline.app import server
    client.post("/api/status", json={"num": "1", "status": "Interview"})
    # No runs available → refresh can't produce a fresh base.
    mocker.patch.object(server.gh, "latest_successful_run", return_value=None)
    trigger = mocker.patch.object(server.gh, "trigger_workflow")

    r = client.post("/api/push-status")
    assert r.status_code == 200
    assert r.json()["base"] == "local"
    import json
    overrides = json.loads(trigger.call_args.args[1]["status_overrides_json"])
    assert overrides == {"1": "Interview"}


def test_push_status_surfaces_gh_error(client, mocker):
    from pipeline.app import server
    client.post("/api/status", json={"num": "1", "status": "Applied"})
    mocker.patch.object(server.gh, "latest_successful_run", return_value=None)
    mocker.patch.object(server.gh, "trigger_workflow",
                        side_effect=server.gh.GhError("edit-tracker.yml not found"))
    r = client.post("/api/push-status")
    assert r.status_code == 502
    assert "edit-tracker" in r.json()["detail"]
    # The edit stays pending so it can be re-pushed once credits return.
    assert client.get("/api/jobs").json()["pending"] == 1


def _fresh_base_with_acme(tmp_path):
    """A cloud base containing only Acme/Eng (row 1) — used to make foreign-
    identity overrides resolve (Acme) or not (anything else)."""
    fresh = tmp_path / "fresh" / "pipeline-output-9"
    (fresh / "data").mkdir(parents=True)
    (fresh / "data" / "applications.md").write_text(
        "# Applications Tracker\n"
        "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
        "|---|------|---------|------|-------|--------|-----|--------|-------|\n"
        "| 1 | 2026-05-27 | Acme | Eng | 4.2/5 | Evaluated | ❌ | [001](reports/001-acme.md) | x |\n",
        encoding="utf-8")
    return fresh


def test_push_clears_unresolved_discards_but_keeps_other_unresolved(client, tmp_path, mocker):
    """An unresolved DISCARD override (a closed role absent from the cloud
    tracker) is dropped on push — it's already applied locally, a closed role
    won't reappear to match later, and the cloud's own recheck is the backstop.
    An unresolved non-Discard (e.g. Applied) is still KEPT for a later push."""
    from pipeline.app import server, data
    import json
    data.record_status_override("1", "Applied", company="Acme", role="Eng")          # resolves
    data.record_status_override("90", "Discarded", company="GhostCo", role="Closed")  # unresolved Discard
    data.record_status_override("91", "Applied", company="OtherCo", role="Dev")       # unresolved non-Discard

    mocker.patch.object(server.gh, "latest_successful_run", return_value={"databaseId": 9})
    mocker.patch.object(server.gh, "download_artifact", return_value=_fresh_base_with_acme(tmp_path))
    mocker.patch.object(server.gh, "trigger_workflow")

    body = client.post("/api/push-status").json()
    assert body["pushed"] == 1            # Acme/Eng resolved + dispatched
    assert body["unresolved"] == 1        # only the kept Applied (91) — NOT the Discard
    remaining = json.loads(data.STATUS_OVERRIDES_FILE.read_text(encoding="utf-8"))
    assert set(remaining) == {"91"}       # 1 pushed+cleared, 90 (Discard) cleared, 91 kept


def test_push_clears_unresolved_discards_when_nothing_resolves(client, tmp_path, mocker):
    """The real-world case: only unresolved Discards, none match the cloud base.
    They're still cleared (not kept forever to nag every push) and no empty
    edit-tracker run is dispatched."""
    from pipeline.app import server, data
    import json
    data.record_status_override("90", "Discarded", company="GhostCo", role="Closed")

    mocker.patch.object(server.gh, "latest_successful_run", return_value={"databaseId": 9})
    mocker.patch.object(server.gh, "download_artifact", return_value=_fresh_base_with_acme(tmp_path))
    trigger = mocker.patch.object(server.gh, "trigger_workflow")

    body = client.post("/api/push-status").json()
    assert body["pushed"] == 0
    assert body["unresolved"] == 0        # the stale Discard was cleared, not kept
    assert json.loads(data.STATUS_OVERRIDES_FILE.read_text(encoding="utf-8")) == {}
    trigger.assert_not_called()           # nothing resolved → no dispatch


# ── Onboarding (Phase 3) ───────────────────────────────────────────────────

def test_onboard_status_reports_readiness(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "list_secret_names", return_value=[
        "SEARCH_CONFIG_B64", "RESUME_TXT_B64", "CV_MD_B64", "PROFILE_YML_B64", "GEMINI_API_KEY",
    ])
    r = client.get("/api/onboard/status")
    assert r.status_code == 200
    body = r.json()
    assert body["repo"] == "me/private"
    assert body["ready"] is True
    # What makes the wizard's API-key field optional. Edit mode used to imply
    # it; now any configured copy enters edit mode, including one that has
    # never written a secret, so the wizard has to be told (#145).
    assert body["has_provider"] is True


def test_onboard_status_not_ready_without_provider(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "list_secret_names", return_value=[
        "SEARCH_CONFIG_B64", "RESUME_TXT_B64", "CV_MD_B64", "PROFILE_YML_B64",
    ])
    body = client.get("/api/onboard/status").json()
    assert body["ready"] is False
    assert body["has_provider"] is False


def _onboard_post(client, form, with_resume=True):
    """POST the wizard. `with_resume=False` is the edit-mode path, where the
    wizard attaches nothing and the server keeps whatever is on disk."""
    import json as _json
    files = ({"resume": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")}
             if with_resume else None)
    return client.post("/api/onboard", files=files, data={"form": _json.dumps(form)})


def test_onboard_writes_secrets_on_private_repo(client, tmp_path, mocker):
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)  # don't write into the real repo
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.onboard, "extract_resume_text", return_value="resume text body")
    gen = mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={
        "SEARCH_CONFIG_B64": "AA", "RESUME_TXT_B64": "BB",
        "CV_MD_B64": "CC", "PROFILE_YML_B64": "DD", "PROFILE_MD_B64": "EE",
    })
    set_secret = mocker.patch.object(server.gh, "set_secret")
    set_var = mocker.patch.object(server.gh, "set_variable")

    r = _onboard_post(client, {"name": "Jane", "provider": "gemini",
                               "api_key": "key-123", "batch_model": "gemini-2.5-flash"})
    assert r.status_code == 200
    written = r.json()["secrets_written"]
    assert "GEMINI_API_KEY" in written and "PROFILE_YML_B64" in written
    # Provider key written with the raw key; artifact secret with the blob.
    set_secret.assert_any_call("GEMINI_API_KEY", "key-123")
    set_secret.assert_any_call("PROFILE_YML_B64", "DD")
    set_var.assert_any_call("BATCH_PROVIDER", "gemini")
    set_var.assert_any_call("BATCH_MODEL", "gemini-2.5-flash")
    # Resume artifacts persisted under the (patched) ROOT.
    assert (tmp_path / "resumes" / "resume.pdf").exists()
    assert (tmp_path / "resumes" / "resume.txt").read_text(encoding="utf-8") == "resume text body"
    gen.assert_called_once()


def test_onboard_saves_docx_under_real_extension(client, tmp_path, mocker):
    """A DOCX import is persisted as resumes/resume.docx (not resume.pdf), so it
    doubles as the apply-stage tailoring source. resume.txt is written too."""
    import json as _json
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.onboard, "extract_resume_text", return_value="docx body")
    mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={
        "SEARCH_CONFIG_B64": "AA", "RESUME_TXT_B64": "BB",
        "CV_MD_B64": "CC", "PROFILE_YML_B64": "DD",
    })
    mocker.patch.object(server.gh, "set_secret")
    mocker.patch.object(server.gh, "set_variable")

    r = client.post(
        "/api/onboard",
        files={"resume": ("Jane_Resume.docx", b"PK\x03\x04 fake-docx",
                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"form": _json.dumps({"name": "Jane", "provider": "gemini", "api_key": "k"})},
    )
    assert r.status_code == 200, r.json()
    assert (tmp_path / "resumes" / "resume.docx").exists()
    assert not (tmp_path / "resumes" / "resume.pdf").exists()
    assert (tmp_path / "resumes" / "resume.txt").read_text(encoding="utf-8") == "docx body"


def test_onboard_reupload_retires_stale_sibling_formats(client, tmp_path, mocker):
    """Review bug: re-onboarding with a new format left the OLD resume.pdf on
    disk, and the filter's pdf-first probe kept scoring against it forever
    (split-brain: scoring used the stale PDF, tailoring the new DOCX). The
    latest upload must be THE resume — siblings are deleted."""
    import json as _json
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.onboard, "extract_resume_text", return_value="docx body")
    mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={
        "SEARCH_CONFIG_B64": "AA", "RESUME_TXT_B64": "BB",
        "CV_MD_B64": "CC", "PROFILE_YML_B64": "DD",
    })
    mocker.patch.object(server.gh, "set_secret")
    mocker.patch.object(server.gh, "set_variable")

    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes" / "resume.pdf").write_bytes(b"%PDF-1.4 stale old resume")

    r = client.post(
        "/api/onboard",
        files={"resume": ("Jane_Resume.docx", b"PK\x03\x04 fake-docx",
                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"form": _json.dumps({"name": "Jane", "provider": "gemini", "api_key": "k"})},
    )
    assert r.status_code == 200, r.json()
    assert (tmp_path / "resumes" / "resume.docx").exists()
    assert not (tmp_path / "resumes" / "resume.pdf").exists()   # stale sibling retired


def test_onboard_extracts_the_on_disk_resume_when_no_txt_sidecar(
        client, tmp_path, mocker, monkeypatch):
    """The submit-side half of the step-0 gate fix (#145): a copy set up outside
    the wizard has resumes/resume.docx but no resume.txt, so edit mode would let
    it reach Submit and then 400. Extract what the filter stage already scores
    against instead of demanding a re-upload of a file we can read."""
    from pipeline.app import server
    monkeypatch.delenv("RESUME_PATH", raising=False)
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.gh, "set_secret")
    mocker.patch.object(server.gh, "set_variable")
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={})
    gen = mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    from pipeline import resume_text as rt
    mocker.patch.object(rt, "extract_resume_text", return_value="Jane Dev\nEngineer")

    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes" / "resume.docx").write_bytes(b"PK fake")

    r = _onboard_post(client, {"name": "Jane"}, with_resume=False)
    assert r.status_code == 200, r.json()
    # resume.txt is what RESUME_TXT_B64 ships, so it has to exist afterwards.
    assert (tmp_path / "resumes" / "resume.txt").read_text(encoding="utf-8").startswith("Jane Dev")
    assert gen.call_args[0][1]["resumeText"].startswith("Jane Dev")


def test_onboard_still_refuses_when_no_resume_exists_anywhere(
        client, tmp_path, mocker, monkeypatch):
    from pipeline.app import server
    monkeypatch.delenv("RESUME_PATH", raising=False)
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    r = _onboard_post(client, {"name": "Jane"}, with_resume=False)
    assert r.status_code == 400
    assert "No resume on file" in r.json()["detail"]


def test_onboard_refuses_public_repo(client, tmp_path, mocker):
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PUBLIC")
    set_secret = mocker.patch.object(server.gh, "set_secret")
    r = _onboard_post(client, {"provider": "gemini", "api_key": "k"})
    assert r.status_code == 409
    assert "PUBLIC" in r.json()["detail"]
    set_secret.assert_not_called()  # wrote nothing


def test_onboard_rejects_unknown_provider(client, mocker):
    from pipeline.app import server
    # provider validated before any gh call
    r = _onboard_post(client, {"provider": "bogus", "api_key": "k"})
    assert r.status_code == 400
    assert "Unknown provider" in r.json()["detail"]


def test_onboard_rejects_unreadable_pdf(client, tmp_path, mocker):
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.onboard, "extract_resume_text",
                        side_effect=Exception("not a pdf"))
    r = _onboard_post(client, {"provider": "gemini", "api_key": "k"})
    assert r.status_code == 400
    assert "could not read resume" in r.json()["detail"]


def test_onboard_parse_resume_autofills_from_text(client, mocker):
    from pipeline.app import server
    # Mock PDF extraction; parse_resume_info runs for real on the text.
    mocker.patch.object(
        server.onboard, "extract_resume_text",
        return_value="Jane Dev\njane@example.com | Dallas, TX | github.com/janedev\n",
    )
    r = client.post(
        "/api/onboard/parse-resume",
        files={"resume": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 200
    info = r.json()
    assert info["name"] == "Jane Dev"
    assert info["email"] == "jane@example.com"
    assert info["location"] == "Dallas, TX"
    assert info["github"] == "github.com/janedev"


def test_onboard_parse_resume_rejects_unreadable_pdf(client, mocker):
    from pipeline.app import server
    mocker.patch.object(server.onboard, "extract_resume_text",
                        side_effect=Exception("not a pdf"))
    r = client.post(
        "/api/onboard/parse-resume",
        files={"resume": ("resume.pdf", b"%PDF junk", "application/pdf")},
    )
    assert r.status_code == 400
    assert "could not read resume" in r.json()["detail"]


def test_onboard_load_config_returns_null_when_no_sidecar(client, tmp_path, mocker):
    # First-time setup: nothing configured, no resume. UI should treat this
    # as "fresh wizard, nothing to prefill".
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server, "_career_ops", return_value=tmp_path / "career-ops")
    r = client.get("/api/onboard/load-config")
    assert r.status_code == 200
    body = r.json()
    assert body["form"] is None
    assert body["configured"] is False
    assert body["has_resume"] is False


def test_onboard_load_config_reads_a_copy_configured_without_the_wizard(
        client, tmp_path, mocker, monkeypatch):
    # The bug this endpoint carried (#145): a copy set up by setup-profile.mjs
    # or by hand has no sidecar, so it read as first-time forever — every field
    # blank, and a step-0 resume gate nothing could satisfy.
    from pipeline.app import server
    co = tmp_path / "career-ops"
    (co / "config").mkdir(parents=True)
    (co / "config" / "profile.yml").write_text(
        "candidate:\n  full_name: Jane Dev\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "search.yml").write_text(
        "searches:\n  - location: Dallas, TX\n    results_wanted: 40\n", encoding="utf-8")
    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes" / "resume.docx").write_bytes(b"PK fake")
    monkeypatch.delenv("RESUME_PATH", raising=False)
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server, "_career_ops", return_value=co)

    body = client.get("/api/onboard/load-config").json()
    assert body["configured"] is True
    assert body["has_resume"] is True
    assert body["form"]["name"] == "Jane Dev"
    assert body["form"]["locations"] == "Dallas, TX"
    assert body["form"]["results_wanted"] == "40"


def test_onboard_load_config_does_not_prefill_an_unconfigured_copy(
        client, tmp_path, mocker, monkeypatch):
    """setup.sh copies search.example.yml to config/search.yml before the user
    has answered anything. Deriving from it unconditionally showed a first-time
    user the demo's Toronto passes and "senior, manager" roles-to-avoid as their
    own answers — with no edit-mode banner, since nothing is configured — and a
    Save would have written a search they never asked for."""
    from pipeline.app import server
    repo = Path(__file__).resolve().parent.parent
    monkeypatch.delenv("RESUME_PATH", raising=False)
    (tmp_path / "config").mkdir()
    shutil.copy(repo / "config" / "search.example.yml", tmp_path / "config" / "search.yml")
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server, "_career_ops", return_value=tmp_path / "career-ops")

    body = client.get("/api/onboard/load-config").json()
    assert body["configured"] is False
    assert body["form"] is None
    assert body["search_detail_at_risk"] == []


def test_onboard_load_config_has_resume_honours_resume_path(
        client, tmp_path, mocker, monkeypatch):
    # A resume under a non-default name with RESUME_PATH pointing at it is what
    # the filter stage scores against, so the wizard must not call it absent.
    from pipeline.app import server
    odd = tmp_path / "docs" / "cv.pdf"
    odd.parent.mkdir(parents=True)
    odd.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setenv("RESUME_PATH", "docs/cv.pdf")
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server, "_career_ops", return_value=tmp_path / "career-ops")
    assert client.get("/api/onboard/load-config").json()["has_resume"] is True


def test_onboard_load_config_returns_saved_payload(client, tmp_path, mocker, monkeypatch):
    # After a successful onboard, the sidecar is written and load returns it
    # so the wizard can prefill — minus the self-ID fields a sidecar saved
    # before #227 still carries. has_resume reflects the persisted PDF —
    # RESUME_PATH is cleared because it outranks the probe and a developer's own
    # .env would otherwise decide the answer.
    from pipeline.app import server
    import json as _json
    monkeypatch.delenv("RESUME_PATH", raising=False)
    mocker.patch.object(server, "ROOT", tmp_path)
    (tmp_path / ".ui-cache").mkdir()
    (tmp_path / ".ui-cache" / "onboarding.json").write_text(
        _json.dumps({"name": "Jane", "results_wanted": 5, "sites": ["indeed"],
                     **RETIRED_FORM_VALUES}),
        encoding="utf-8",
    )
    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes" / "resume.pdf").write_bytes(b"%PDF-1.4 fake")
    r = client.get("/api/onboard/load-config")
    assert r.status_code == 200
    body = r.json()
    assert body["form"] == {"name": "Jane", "results_wanted": 5, "sites": ["indeed"]}
    assert body["has_resume"] is True


def test_onboard_writes_sidecar_after_successful_submit(client, tmp_path, mocker):
    # The sidecar mirrors what the wizard will need to prefill — every field
    # the user submitted, minus the API key (which lives only in Secrets).
    from pipeline.app import server
    import json as _json
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.onboard, "extract_resume_text", return_value="resume text")
    mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={
        "SEARCH_CONFIG_B64": "AA", "RESUME_TXT_B64": "BB",
        "CV_MD_B64": "CC", "PROFILE_YML_B64": "DD",
    })
    mocker.patch.object(server.gh, "set_secret")
    mocker.patch.object(server.gh, "set_variable")
    r = _onboard_post(client, {"name": "Jane", "provider": "gemini",
                               "api_key": "supersecret",
                               "results_wanted": 5, "sites": ["indeed"]})
    assert r.status_code == 200
    sidecar = tmp_path / ".ui-cache" / "onboarding.json"
    assert sidecar.exists()
    saved = _json.loads(sidecar.read_text(encoding="utf-8"))
    # API key MUST be excluded so the sidecar is safe on disk.
    assert "api_key" not in saved
    # Everything else round-trips so the next wizard visit can prefill.
    assert saved["name"] == "Jane"
    assert saved["results_wanted"] == 5
    assert saved["provider"] == "gemini"


def test_onboard_reuses_existing_resume_when_none_uploaded(client, tmp_path, mocker):
    # Edit-mode flow: the user is tweaking config and didn't re-upload the
    # resume. Server reuses resumes/resume.txt instead of erroring out.
    from pipeline.app import server
    import json as _json
    mocker.patch.object(server, "ROOT", tmp_path)
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "resume.pdf").write_bytes(b"%PDF-1.4 prior")
    (resumes / "resume.txt").write_text("existing resume text", encoding="utf-8")
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    # extract_resume_text MUST NOT be called — we're reusing the .txt directly.
    extract = mocker.patch.object(server.onboard, "extract_resume_text",
                                  side_effect=AssertionError("unexpected PDF extract"))
    build = mocker.patch.object(server.onboard, "build_onboarding_json",
                                wraps=server.onboard.build_onboarding_json)
    mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={
        "SEARCH_CONFIG_B64": "AA", "RESUME_TXT_B64": "BB",
        "CV_MD_B64": "CC", "PROFILE_YML_B64": "DD",
    })
    mocker.patch.object(server.gh, "set_secret")
    mocker.patch.object(server.gh, "set_variable")

    # Post WITHOUT a resume file — just the form.
    r = client.post("/api/onboard", data={"form": _json.dumps({"results_wanted": 5})})
    assert r.status_code == 200, r.json()
    extract.assert_not_called()
    # The persisted resume text was the one fed to the generator.
    _, kwargs_payload = build.call_args.args, build.call_args.kwargs
    # build_onboarding_json(form, resume_text) — positional.
    assert build.call_args.args[1] == "existing resume text"


def test_onboard_errors_when_no_resume_anywhere(client, tmp_path, mocker):
    # First-time setup with no PDF upload AND no persisted resume: clear error.
    from pipeline.app import server
    import json as _json
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    r = client.post("/api/onboard", data={"form": _json.dumps({})})
    assert r.status_code == 400
    assert "resume" in r.json()["detail"].lower()


def test_onboard_skips_provider_key_write_when_api_key_blank(client, tmp_path, mocker):
    # Edit-mode flow: user changed search settings but didn't re-paste their
    # API key. Provider/key writes should be skipped; artifact writes proceed.
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.onboard, "extract_resume_text", return_value="resume text")
    mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={
        "SEARCH_CONFIG_B64": "AA", "RESUME_TXT_B64": "BB",
        "CV_MD_B64": "CC", "PROFILE_YML_B64": "DD",
    })
    set_secret = mocker.patch.object(server.gh, "set_secret")
    set_var = mocker.patch.object(server.gh, "set_variable")
    r = _onboard_post(client, {"provider": "gemini", "api_key": ""})  # blank key
    assert r.status_code == 200
    written = r.json()["secrets_written"]
    # Artifact secrets written; provider key NOT.
    assert "PROFILE_YML_B64" in written
    assert "GEMINI_API_KEY" not in written
    # No provider/model variable writes either — nothing changed there.
    for call in set_secret.call_args_list:
        assert call.args[0] != "GEMINI_API_KEY"
    for call in set_var.call_args_list:
        assert call.args[0] != "BATCH_PROVIDER"


class TestStaticNoCache:
    """SPA assets must revalidate every load (no-cache), so a UI change isn't
    masked by a stale cached app.js/onboard.js until a manual hard refresh."""

    def test_spa_asset_no_cache(self, client):
        r = client.get("/app.js")
        assert r.status_code == 200
        assert "no-cache" in r.headers.get("cache-control", "").lower()

    def test_onboard_html_no_cache(self, client):
        r = client.get("/onboard")
        assert r.status_code == 200
        assert "no-cache" in r.headers.get("cache-control", "").lower()


# ── Agent CLI: the registry reaches the wizard, and the bridge button ────────

class TestAgentCliSurface:
    """The wizard's CLI choice is pipeline/agent_cli.py's registry rendered, not
    a second list. `name`/`available` stay because onboard.js read those first;
    the tier fields are what let the select say "free" or "paid" and quote the
    real cost, which is the whole point of choosing between them."""

    def test_providers_payload_carries_every_registry_field(self, client):
        from pipeline import agent_cli
        body = client.get("/api/onboard/providers").json()
        tools = body["cli_tools"]
        assert [c["id"] for c in tools] == list(agent_cli.AGENT_CLIS)
        for c in tools:
            reg = agent_cli.AGENT_CLIS[c["id"]]
            assert c["name"] == reg.id            # the old field, still read
            assert c["available"] == c["installed"]
            assert (c["label"], c["tier"]) == (reg.label, reg.tier)
            assert c["tier_note"] == reg.tier_note
            assert c["install_hint"] == reg.install_hint
            # The words onboard.js appends for a CLI --batch can't run (#187).
            assert c["batch_note"] == ("" if reg.batch else agent_cli.APPLY_ONLY_LABEL)
        assert [c["id"] for c in tools if c["default"]] == [agent_cli.DEFAULT_CLI]

    def test_providers_payload_carries_the_submit_policies(self, client):
        from pipeline import handoff
        body = client.get("/api/onboard/providers").json()
        policies = body["submit_policies"]
        assert [p["id"] for p in policies] == list(handoff.SUBMIT_POLICIES)
        assert all(p["gloss"] == handoff.SUBMIT_POLICIES[p["id"]] for p in policies)
        assert [p["id"] for p in policies if p["default"]] == [handoff.DEFAULT_SUBMIT_POLICY]
        # What the wizard prefills the select with, so a hand-set .env shows.
        assert body["current"]["handoff_submit_policy"] == handoff.submit_policy()

    def test_providers_payload_carries_the_free_tier_recommendation(self, client):
        """The cloud model placeholder promises a specific model for a blank
        box; it is computed from the limits table, so the payload has to carry
        the same call onboard_submit writes rather than a string in the markup."""
        from pipeline import gemini_limits
        body = client.get("/api/onboard/providers").json()
        assert body["free_tier_recommendation"] == (gemini_limits.batch_recommendation() or "")

    def test_register_bridge_uses_the_registry(self, client, mocker):
        from pipeline.app import server
        reg = mocker.patch.object(server.agent_cli, "register_playwright_mcp",
                                  return_value="Registered playwright with OpenCode.")
        r = client.post("/api/agent-cli/register", json={"cli": "claude"})
        assert r.status_code == 200, r.text
        assert r.json()["cli"] == "claude"
        assert r.json()["message"] == "Registered playwright with OpenCode."
        assert reg.call_args[0][0] is server.agent_cli.AGENT_CLIS["claude"]

    def test_register_bridge_defaults_to_the_resolved_cli(self, client, mocker):
        from pipeline.app import server
        mocker.patch.object(server.agent_cli, "register_playwright_mcp", return_value="ok")
        r = client.post("/api/agent-cli/register", json={})
        assert r.status_code == 200, r.text
        assert r.json()["cli"] == server.agent_cli.resolve_cli().id

    def test_known_clis_is_the_registry(self):
        """The save endpoint's vocabulary. It is the one registry mirror the
        endpoint ENFORCES, so a reverted literal list would 400 a CLI the
        wizard's own select offers, and nothing else here would notice."""
        from pipeline import agent_cli
        from pipeline.app import server
        assert server._KNOWN_CLIS == list(agent_cli.AGENT_CLIS)

    @pytest.mark.parametrize("cli_id", list(agent_cli.AGENT_CLIS))
    def test_save_accepts_every_registry_cli(self, client, tmp_path, mocker, cli_id):
        from pipeline.app import server
        mocker.patch.object(server, "ROOT", tmp_path)
        r = client.post("/api/onboard/local-config",
                        json={"batch_provider": "", "batch_model": "", "batch_cli": cli_id})
        assert r.status_code == 200, r.text
        assert f"BATCH_CLI={cli_id}" in (tmp_path / ".env").read_text(encoding="utf-8")

    def test_register_bridge_rejects_an_unknown_cli(self, client, mocker):
        from pipeline.app import server
        reg = mocker.patch.object(server.agent_cli, "register_playwright_mcp")
        r = client.post("/api/agent-cli/register", json={"cli": "notacli"})
        assert r.status_code == 400
        assert "notacli" in r.json()["detail"]
        reg.assert_not_called()

    def test_register_bridge_answers_a_non_object_server_map_with_a_message(
            self, client, mocker, monkeypatch, tmp_path):
        """The real registry, not a mock: `{"mcp": "x"}` in OpenCode's file
        used to raise out of the merge, so the button got a 500 for a file the
        message can tell the person how to fix (#224)."""
        mocker.patch("pipeline.agent_cli.shutil.which", side_effect=_which({"opencode"}))
        monkeypatch.setenv(agent_cli.XDG_CONFIG_HOME_ENV, str(tmp_path / "xdg"))
        # Hermetic: no Playwright browsers dir from the real home is globbed.
        monkeypatch.setenv(agent_cli.BROWSERS_PATH_ENV, str(tmp_path / "absent"))
        cfg = tmp_path / "xdg" / "opencode" / "opencode.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text('{"mcp": "x"}', encoding="utf-8")
        r = client.post("/api/agent-cli/register", json={"cli": "opencode"})
        assert r.status_code == 200, r.text
        assert r.json()["cli"] == "opencode"
        assert "left it alone" in r.json()["message"]
        assert cfg.read_text(encoding="utf-8") == '{"mcp": "x"}'


class TestSubmitPolicySave:
    """The Local step writes HANDOFF_SUBMIT_POLICY into .env. Aliases are
    accepted because the env reader accepts them, but the CANONICAL id is what
    lands — the README, the kickoff prompt and the work-order header all render
    from the value in the file."""

    def _save(self, client, **kw):
        payload = {"batch_provider": "", "batch_model": "", "batch_cli": ""}
        payload.update(kw)
        return client.post("/api/onboard/local-config", json=payload)

    def test_canonical_id_is_written(self, client, tmp_path, mocker):
        from pipeline import handoff
        from pipeline.app import server
        mocker.patch.object(server, "ROOT", tmp_path)
        r = self._save(client, handoff_submit_policy="submit-easy-apply")
        assert r.status_code == 200, r.text
        env = (tmp_path / ".env").read_text(encoding="utf-8")
        assert f"{handoff.SUBMIT_POLICY_ENV}=submit-easy-apply" in env

    def test_an_alias_is_written_as_the_canonical_id(self, client, tmp_path, mocker):
        from pipeline import handoff
        from pipeline.app import server
        mocker.patch.object(server, "ROOT", tmp_path)
        r = self._save(client, handoff_submit_policy="easy-apply")
        assert r.status_code == 200, r.text
        env = (tmp_path / ".env").read_text(encoding="utf-8")
        assert f"{handoff.SUBMIT_POLICY_ENV}=submit-easy-apply" in env

    def test_an_absent_policy_leaves_the_key_alone(self, client, tmp_path, mocker):
        """The Provider step posts here too, and it does not show this select.
        An absent field must not overwrite what the Local step saved — a blank
        one would, which is why the field is `None`-means-absent like its
        neighbours rather than carried through from a select that may still be
        showing its static first option."""
        from pipeline import handoff
        from pipeline.app import server
        mocker.patch.object(server, "ROOT", tmp_path)
        assert self._save(client, handoff_submit_policy="submit-all").status_code == 200
        assert self._save(client).status_code == 200
        env = (tmp_path / ".env").read_text(encoding="utf-8")
        assert f"{handoff.SUBMIT_POLICY_ENV}=submit-all" in env

    def test_unknown_policy_is_refused(self, client, tmp_path, mocker):
        from pipeline.app import server
        mocker.patch.object(server, "ROOT", tmp_path)
        r = self._save(client, handoff_submit_policy="yolo")
        assert r.status_code == 400
        assert "yolo" in r.json()["detail"]


# ── Daily digest + cloud free-tier defaults (the wizard's cloud half) ────────

def _digest_form(**kw):
    form = {"name": "Jane", "provider": "gemini", "api_key": "key-123"}
    form.update(kw)
    return form


@pytest.fixture
def cloud_wizard(client, tmp_path, mocker):
    """The wizard's submit path with gh stubbed: returns the two recorders so a
    test can assert on exactly what reached the repo."""
    from pipeline.app import server
    mocker.patch.object(server, "ROOT", tmp_path)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.onboard, "extract_resume_text", return_value="resume text")
    mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={
        "SEARCH_CONFIG_B64": "AA", "RESUME_TXT_B64": "BB",
        "CV_MD_B64": "CC", "PROFILE_YML_B64": "DD",
    })
    return {
        "client": client,
        "set_secret": mocker.patch.object(server.gh, "set_secret"),
        "set_variable": mocker.patch.object(server.gh, "set_variable"),
    }


def _vars_written(recorder):
    return {c.args[0]: c.args[1] for c in recorder.call_args_list}


def _secrets_written(recorder):
    return {c.args[0]: c.args[1] for c in recorder.call_args_list}


class TestDigestSecrets:
    """Delivery is a secret, so it is write-only: the wizard can offer to
    replace it and can report that one exists, and nothing more. A blank field
    therefore has to mean KEEP — the same rule as the API key — or every
    revisit to change the minimum score would silently unhook the digest."""

    def test_non_blank_fields_are_written_under_the_digest_names(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            digest_discord_webhook="https://discord.com/api/webhooks/1/tok",
            digest_email_to="jane@example.com",
            digest_smtp_host="smtp.example.com",
            digest_smtp_port="587",
            digest_smtp_user="jane@example.com",
            digest_smtp_pass="app-pass",
        ))
        assert r.status_code == 200, r.text
        written = _secrets_written(cloud_wizard["set_secret"])
        assert written["DIGEST_DISCORD_WEBHOOK"] == "https://discord.com/api/webhooks/1/tok"
        assert written["DIGEST_EMAIL_TO"] == "jane@example.com"
        assert written["DIGEST_SMTP_PASS"] == "app-pass"
        assert "DIGEST_DISCORD_WEBHOOK" in r.json()["secrets_written"]

    def test_blank_fields_keep_the_existing_secret(self, cloud_wizard):
        from pipeline import daily_digest
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            digest_discord_webhook="", digest_email_to=""))
        assert r.status_code == 200, r.text
        written = _secrets_written(cloud_wizard["set_secret"])
        assert not [n for n in written if n in daily_digest.SECRET_VARS]

    def test_status_reports_each_channel_from_the_secrets_that_run_it(self, client, mocker):
        from pipeline.app import server
        mocker.patch.object(server.gh, "current_repo", return_value="me/private")
        mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
        mocker.patch.object(server.gh, "list_variables", return_value={})
        mocker.patch.object(server.gh, "list_secret_names", return_value=[
            "GEMINI_API_KEY", "DIGEST_DISCORD_WEBHOOK", "DIGEST_EMAIL_TO"])
        body = client.get("/api/onboard/status").json()
        assert body["has_digest_discord"] is True
        # Email needs an address AND a host — an address alone sends nowhere.
        assert body["has_digest_email"] is False

    def test_status_reports_email_once_both_halves_exist(self, client, mocker):
        from pipeline.app import server
        mocker.patch.object(server.gh, "current_repo", return_value="me/private")
        mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
        mocker.patch.object(server.gh, "list_variables", return_value={})
        mocker.patch.object(server.gh, "list_secret_names", return_value=[
            "DIGEST_EMAIL_TO", "DIGEST_SMTP_HOST"])
        body = client.get("/api/onboard/status").json()
        assert body["has_digest_email"] is True
        assert body["has_digest_discord"] is False

    def test_status_carries_the_readable_variables_for_edit_mode(self, client, mocker):
        from pipeline.app import server
        mocker.patch.object(server.gh, "current_repo", return_value="me/private")
        mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
        mocker.patch.object(server.gh, "list_secret_names", return_value=[])
        mocker.patch.object(server.gh, "list_variables",
                            return_value={"DIGEST_MIN_SCORE": "4.5"})
        body = client.get("/api/onboard/status").json()
        assert body["variables"]["DIGEST_MIN_SCORE"] == "4.5"

    def test_status_survives_a_gh_that_cannot_list_variables(self, client, mocker):
        from pipeline.app import server
        mocker.patch.object(server.gh, "current_repo", return_value="me/private")
        mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
        mocker.patch.object(server.gh, "list_secret_names", return_value=[])
        mocker.patch.object(server.gh, "list_variables",
                            side_effect=server.gh.GhError("no gh"))
        body = client.get("/api/onboard/status").json()
        assert body["variables"] == {}


class TestDigestAndUpdateVariables:
    """The thresholds and the weekly-update box are repository VARIABLES, not
    secrets: readable, so edit mode can show them, and defaulted in the
    workflow, so a blank field must write nothing and leave that default."""

    def test_thresholds_are_written_when_filled(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            digest_min_score="4.5", digest_limit="25"))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        assert written["DIGEST_MIN_SCORE"] == "4.5"
        assert written["DIGEST_LIMIT"] == "25"

    def test_blank_thresholds_leave_the_workflow_default(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            digest_min_score="", digest_limit=""))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        assert "DIGEST_MIN_SCORE" not in written and "DIGEST_LIMIT" not in written

    def test_weekly_update_box_writes_the_exact_gate_value(self, cloud_wizard):
        """update-from-template.yml's schedule proceeds only on the string
        "true", so the checked box has to write exactly that."""
        r = _onboard_post(cloud_wizard["client"], _digest_form(auto_update_weekly="yes"))
        assert r.status_code == 200, r.text
        assert _vars_written(cloud_wizard["set_variable"])["AUTO_UPDATE_FROM_TEMPLATE"] == "true"

    def test_unticking_weekly_update_writes_something_other_than_true(self, cloud_wizard):
        """An unticked box must overwrite an earlier "true" rather than be
        absent — leaving the variable standing would keep merging every Monday
        after the user said stop."""
        r = _onboard_post(cloud_wizard["client"], _digest_form(auto_update_weekly="no"))
        assert r.status_code == 200, r.text
        assert _vars_written(cloud_wizard["set_variable"])["AUTO_UPDATE_FROM_TEMPLATE"] != "true"

    def test_a_form_that_never_mentions_the_box_leaves_it_alone(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form())
        assert r.status_code == 200, r.text
        assert "AUTO_UPDATE_FROM_TEMPLATE" not in _vars_written(cloud_wizard["set_variable"])


class TestCloudFreeTierDefaults:
    """A blank model on Gemini is not the provider default. PROVIDER_DEFAULTS'
    Flash row is ~20 requests a day on a free key — a 180-role daily exhausts it
    before the first coffee — so a blank box resolves to the highest-capacity
    row the limits table knows, computed, never a baked string."""

    def test_blank_gemini_model_resolves_to_the_recommendation(self, cloud_wizard):
        from pipeline import gemini_limits
        r = _onboard_post(cloud_wizard["client"], _digest_form(batch_model=""))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        assert written["BATCH_MODEL"] == gemini_limits.batch_recommendation()

    def test_an_explicit_model_is_written_as_given(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form(batch_model="gemini-2.5-flash"))
        assert r.status_code == 200, r.text
        assert _vars_written(cloud_wizard["set_variable"])["BATCH_MODEL"] == "gemini-2.5-flash"

    def test_another_provider_with_a_blank_model_writes_none(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"],
                          _digest_form(provider="openai", batch_model=""))
        assert r.status_code == 200, r.text
        assert "BATCH_MODEL" not in _vars_written(cloud_wizard["set_variable"])

    def test_free_tier_flag_and_limits_reach_the_cloud(self, cloud_wizard):
        import json as _json
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            gemini_free_tier="yes",
            gemini_limits={"gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500}}))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        assert written["GEMINI_FREE_TIER"] == "true"
        assert _json.loads(written["GEMINI_LIMITS_JSON"]) == {
            "gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500}}

    def test_unticking_the_free_tier_clears_the_variable(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form(gemini_free_tier="no"))
        assert r.status_code == 200, r.text
        assert _vars_written(cloud_wizard["set_variable"])["GEMINI_FREE_TIER"] == ""

    def test_no_limits_written_when_every_box_is_blank(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            gemini_free_tier="yes", gemini_limits={"gemini-3.1-flash-lite": None}))
        assert r.status_code == 200, r.text
        assert "GEMINI_LIMITS_JSON" not in _vars_written(cloud_wizard["set_variable"])

    def test_a_non_gemini_provider_writes_no_gemini_variables(self, cloud_wizard):
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            provider="openai", gemini_free_tier="yes"))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        assert "GEMINI_FREE_TIER" not in written and "GEMINI_LIMITS_JSON" not in written

    def test_a_blank_key_re_points_nothing(self, cloud_wizard):
        """WHICH provider the cloud calls, and with what, is only rewritten
        against a key the user just pasted: `has_provider` is true for ANY
        provider secret, so the select can read `gemini` on a copy whose cloud
        actually runs on something else, and an edit-mode revisit that changed
        only the search settings must not re-point it."""
        r = _onboard_post(cloud_wizard["client"], _digest_form(api_key=""))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        for name in ("BATCH_PROVIDER", "BATCH_MODEL"):
            assert name not in written

    def test_a_blank_key_still_carries_the_free_tier_answer(self, cloud_wizard):
        """...but the free-tier answer is NOT gated on the key, and that
        asymmetry is the whole cloud half of the feature. Edit mode is the
        normal path once a provider secret exists — the box is default-CHECKED
        and the key field legitimately blank — so gating it meant almost every
        real Save wrote nothing, and the daily ran a free key unpaced into the
        429 the box exists to prevent while the wizard reported success.
        GEMINI_FREE_TIER only decides whether Gemini calls are paced and
        capped; it re-points nothing."""
        import json as _json
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            api_key="", gemini_free_tier="yes",
            gemini_limits={"gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500}}))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        assert written["GEMINI_FREE_TIER"] == "true"
        assert _json.loads(written["GEMINI_LIMITS_JSON"]) == {
            "gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500}}

    def test_a_blank_key_on_another_provider_writes_no_gemini_variables(self, cloud_wizard):
        """The gate that stayed: the flag is Gemini's, so a copy whose cloud
        provider is OpenAI gets nothing from this path either way."""
        r = _onboard_post(cloud_wizard["client"], _digest_form(
            provider="openai", api_key="", gemini_free_tier="yes"))
        assert r.status_code == 200, r.text
        written = _vars_written(cloud_wizard["set_variable"])
        assert "GEMINI_FREE_TIER" not in written and "GEMINI_LIMITS_JSON" not in written


# ── LAN mode ────────────────────────────────────────────────────────────────

def _basic(password, user="phone"):
    import base64
    return {"Authorization": "Basic " + base64.b64encode(
        f"{user}:{password}".encode()).decode()}


LAN_PASSWORD = "a-long-passphrase"
LAN_PEER = ("192.168.1.20", 51234)
LOOPBACK_PEER = ("127.0.0.1", 51234)


@pytest.fixture
def lan(client, monkeypatch):
    """LAN mode turned on over the same tmp career-ops the `client` fixture
    builds. UI_LAN / UI_PASSWORD / UI_ALLOWED_HOSTS are read from os.environ at
    REQUEST time, so setting them here reaches the guard without another
    reload — and `testserver` is in the allowed set because that is the Host
    TestClient sends."""
    from pipeline.app import server
    monkeypatch.setenv(server.UI_LAN_ENV, "1")
    monkeypatch.setenv(server.UI_PASSWORD_ENV, LAN_PASSWORD)
    monkeypatch.setenv(server.UI_ALLOWED_HOSTS_ENV, "testserver,192.168.1.10")

    class _Lan:
        module = server
        auth = _basic(LAN_PASSWORD)

        def peer(self, addr):
            return TestClient(server.app, client=addr, headers=dict(self.auth))

    return _Lan()


class TestLanStartupRefusal:
    """The password is not optional under UI_LAN, and this is the half of the
    refusal the shell wrapper cannot make: run-ui.sh checks the exported
    environment, this module has also loaded .env. Neither alone covers both
    ways a user sets a variable."""

    def test_import_refuses_when_the_password_is_empty(self, monkeypatch):
        import importlib
        from pipeline.app import server
        monkeypatch.setenv(server.UI_LAN_ENV, "1")
        # Empty rather than deleted: load_dotenv(override=False) would re-add a
        # deleted name from a developer's own .env mid-reload.
        monkeypatch.setenv(server.UI_PASSWORD_ENV, "")
        with pytest.raises(SystemExit) as e:
            importlib.reload(server)
        assert server.UI_PASSWORD_ENV in str(e.value)
        # Leave the module whole for whatever runs next.
        monkeypatch.setenv(server.UI_LAN_ENV, "")
        importlib.reload(server)


class TestLanBasicAuth:
    """Anyone on the network can reach the port, so every request carries the
    password — GETs included. The tracker is the thing being protected, and it
    is read with a GET."""

    def test_no_credentials_is_401_with_a_challenge(self, lan):
        c = TestClient(lan.module.app, client=LOOPBACK_PEER)
        r = c.get("/api/jobs")
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Basic ")

    def test_a_wrong_password_is_401(self, lan):
        c = TestClient(lan.module.app, client=LOOPBACK_PEER,
                       headers=_basic("not-the-password"))
        assert c.get("/api/jobs").status_code == 401

    def test_any_username_with_the_right_password_is_accepted(self, lan):
        c = TestClient(lan.module.app, client=LAN_PEER,
                       headers=_basic(LAN_PASSWORD, user="whoever"))
        assert c.get("/api/jobs").status_code == 200

    def test_garbage_credentials_do_not_raise(self, lan):
        c = TestClient(lan.module.app, client=LAN_PEER,
                       headers={"Authorization": "Basic not-base64!!"})
        assert c.get("/api/jobs").status_code == 401


class TestLanPeerMatrix:
    """From the LAN the board is usable and nothing else is. The allowlist is
    three routes; everything else is loopback-only BY DEFAULT, so a POST route
    added next month is refused until someone adds it here on purpose.

    Loopback trust is itself opt-in (`UI_TRUST_LOOPBACK_PEER`), so the tests
    that exercise the admin side set it — the interactive launchers do the same,
    and a hosted instance behind a proxy deliberately does not."""

    def test_loopback_reaches_a_loopback_only_route(self, lan, monkeypatch):
        monkeypatch.setenv(lan.module.UI_TRUST_LOOPBACK_PEER_ENV, "1")
        c = lan.peer(LOOPBACK_PEER)
        r = c.post("/api/agent-cli/register", json={"cli": "notacli"})
        # 400 for the unknown id — the point is that the peer rule let it in.
        assert r.status_code == 400

    def test_the_board_is_reachable_from_the_lan(self, lan):
        c = lan.peer(LAN_PEER)
        r = c.post("/api/status", json={"num": "1", "status": "Applied"})
        assert r.status_code == 200, r.text

    def test_push_is_reachable_from_the_lan(self, lan):
        c = lan.peer(LAN_PEER)
        # 400 "nothing pending" is fine; 403 would mean the phone can move a
        # card and then never get it to the cloud.
        assert c.post("/api/push-status", json={}).status_code != 403

    def test_every_other_post_is_loopback_only(self, lan):
        from starlette.routing import Route
        c = lan.peer(LAN_PEER)
        checked = 0
        for route in lan.module.app.routes:
            if not isinstance(route, Route) or "POST" not in (route.methods or ()):
                continue
            if route.path in lan.module._LAN_MUTABLE or "{" in route.path:
                continue
            r = c.post(route.path, json={})
            assert r.status_code == 403, f"{route.path} -> {r.status_code}"
            assert "loopback-only" in r.json()["detail"], route.path
            checked += 1
        assert checked > 5, "the matrix stopped covering anything"

    @staticmethod
    def _stub_gh(lan, monkeypatch):
        """No live `gh`: unstubbed these spawn real subprocesses and download a
        real artifact on any machine with credentials, and `!= 403` passes
        whatever they do."""
        monkeypatch.setattr(lan.module.gh, "latest_successful_run", lambda *a, **k: None)

    def test_refresh_is_reachable_from_the_lan(self, lan, monkeypatch):
        """Without it the board is FROZEN for the only people this mode exists
        for — they would see whatever the server last synced and never the roles
        today's run found. Any status but 403 is fine here; offline/no-gh is a
        502 and that is still the route doing its job."""
        self._stub_gh(lan, monkeypatch)
        c = lan.peer(LAN_PEER)
        # 404 "no successful runs" is the stub talking — the route ran.
        assert c.post("/api/refresh", json={}).status_code == 404

    def test_the_lan_cannot_start_work_or_wipe_state(self, lan):
        """The three routes it CAN reach read or write the tracker. Spending,
        launching and destroying stay loopback-only — this is the line
        `_LAN_MUTABLE` draws, stated as a test so widening it again is a
        deliberate act."""
        c = lan.peer(LAN_PEER)
        registered = {r.path for r in lan.module.app.routes if hasattr(r, "path")}
        for path in ("/api/run", "/api/reset", "/api/jobs/add", "/api/handoff/build"):
            assert path in registered, f"{path} is not a route — this test would pass vacuously"
            r = c.post(path, json={})
            assert r.status_code == 403, f"{path} -> {r.status_code}"

    def test_reads_are_unrestricted_from_the_lan(self, lan):
        c = lan.peer(LAN_PEER)
        assert c.get("/api/jobs").status_code == 200
        assert c.get("/api/health").status_code == 200

    def test_lan_mutable_names_real_post_routes(self, lan):
        """A typo here would silently make the board unreachable from the phone
        — and, worse, read as if the rule were being applied."""
        from starlette.routing import Route
        posts = {r.path for r in lan.module.app.routes
                 if isinstance(r, Route) and "POST" in (r.methods or ())}
        assert lan.module._LAN_MUTABLE <= posts


class TestLanOriginAndHost:
    """Origin alone is not enough. A page the user opened elsewhere can send an
    Origin the guard likes only if it can also make the browser send this
    machine's Host — so both are checked, and they must agree."""

    def test_a_genuine_lan_request_is_accepted(self, lan):
        c = lan.peer(LAN_PEER)
        r = c.post("/api/status", json={"num": "1", "status": "Applied"},
                   headers={"Origin": "http://192.168.1.10:8000",
                            "Host": "192.168.1.10:8000"})
        assert r.status_code == 200, r.text

    def test_a_rebound_hostname_is_refused(self, lan):
        """The DNS-rebinding shape: a name the attacker controls, resolved to
        this machine, with the user's own credentials attached."""
        c = lan.peer(LAN_PEER)
        r = c.post("/api/status", json={"num": "1", "status": "Applied"},
                   headers={"Origin": "http://evil.example:8000",
                            "Host": "evil.example:8000"})
        assert r.status_code == 403
        assert "evil.example" in r.json()["detail"]

    def test_an_origin_that_disagrees_with_the_host_is_refused(self, lan):
        c = lan.peer(LAN_PEER)
        r = c.post("/api/status", json={"num": "1", "status": "Applied"},
                   headers={"Origin": "http://evil.example:8000",
                            "Host": "192.168.1.10:8000"})
        assert r.status_code == 403

    def test_a_port_mismatch_is_refused(self, lan):
        c = lan.peer(LAN_PEER)
        r = c.post("/api/status", json={"num": "1", "status": "Applied"},
                   headers={"Origin": "http://192.168.1.10:9999",
                            "Host": "192.168.1.10:8000"})
        assert r.status_code == 403

    def test_loopback_is_still_accepted_under_lan(self, lan):
        c = lan.peer(LOOPBACK_PEER)
        r = c.post("/api/status", json={"num": "1", "status": "Applied"},
                   headers={"Origin": "http://localhost:8000",
                            "Host": "localhost:8000"})
        assert r.status_code == 200, r.text

    def test_allowed_hosts_defaults_to_this_machine(self, lan, monkeypatch):
        """Unset, the set is computed once at startup from the machine's own
        names. It must never be empty of loopback, and it must never be a
        lookup at request time — so the computed set is cached."""
        monkeypatch.delenv(lan.module.UI_ALLOWED_HOSTS_ENV, raising=False)
        first = lan.module._allowed_hosts()
        assert lan.module._allowed_hosts() is first     # cached, not re-resolved
        assert not any(lan.module._is_loopback_host(h) for h in first)


class TestLanEnvExampleMirror:
    """.env.example is where a user learns a variable exists. The three LAN
    names are the ones a person has to set by hand (the launcher sets UI_LAN
    itself, but a user who exports it needs to know the password rule), so a
    name added to LAN_ENV_VARS and not documented is a feature nobody finds."""

    def test_every_lan_variable_is_documented(self):
        from pipeline.app import server
        text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
            encoding="utf-8")
        for name in server.LAN_ENV_VARS:
            assert name in text, name

    def test_the_password_rule_is_stated(self):
        text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
            encoding="utf-8")
        block = text[text.index("UI_LAN"):]
        assert "REQUIRED" in block

    def test_the_conftest_fallback_matches_the_constants(self):
        """conftest's autouse LAN fixture must not import server.py (that would
        re-run its load_dotenv over every other fixture's isolation), so it
        clears literals when the module is absent. This file has the module
        imported for real, so it is where the two can be compared."""
        from tests import conftest
        from pipeline.app import server
        assert conftest.LAN_ENV_FALLBACK == tuple(server.LAN_ENV_VARS)
        assert conftest.LAN_ENV_NAME_FALLBACK == server.UI_LAN_ENV


class TestProxiedLoopbackIsNotLocalAdmin:
    """Behind a reverse proxy the TCP peer is the proxy, so every request looks
    like 127.0.0.1 and the loopback-only routes — reset, a pipeline run, the
    route that writes repository SECRETS — open to anyone who can reach it.
    uvicorn rewrites the peer from X-Forwarded-For, so a correct proxy is safe;
    the common hand-written nginx block sets only X-Real-IP, which uvicorn does
    not read, and then NOTHING fails. A forwarded header on a loopback peer is
    therefore treated as remote."""

    @pytest.mark.parametrize("header", ["x-forwarded-for", "x-real-ip", "forwarded"])
    def test_a_forwarded_header_on_a_loopback_peer_is_refused(self, lan, monkeypatch, header):
        """Belt and braces: even WITH the opt-in, a forwarded header means the
        peer is a proxy — had the header been honoured the peer would not still
        be loopback."""
        monkeypatch.setenv(lan.module.UI_TRUST_LOOPBACK_PEER_ENV, "1")
        c = lan.peer(LOOPBACK_PEER)
        r = c.post("/api/reset", json={}, headers={header: "100.64.1.5"})
        assert r.status_code == 403, r.status_code
        assert "loopback-only" in r.json()["detail"]

    def test_a_genuine_local_request_still_works(self, lan, monkeypatch):
        """No proxy, no header, and the launcher's opt-in — the admin at the
        machine keeps full access."""
        monkeypatch.setenv(lan.module.UI_TRUST_LOOPBACK_PEER_ENV, "1")
        c = lan.peer(LOOPBACK_PEER)
        assert c.post("/api/reset", json={}).status_code != 403

    def test_without_the_opt_in_a_loopback_peer_is_not_the_admin(self, lan):
        """The default, and the case that matters: a bare `proxy_pass` sends NO
        forwarded header, so header-presence could never have caught it. Only
        refusing by default does."""
        c = lan.peer(LOOPBACK_PEER)
        r = c.post("/api/reset", json={})
        assert r.status_code == 403 and "loopback-only" in r.json()["detail"]

    def test_the_lan_allowlist_is_unaffected_by_the_rule(self, lan, monkeypatch):
        """A proxied peer must still be able to do the three things the board
        is for; otherwise the hosted deployment cannot work at all."""
        monkeypatch.setattr(lan.module.gh, "latest_successful_run", lambda *a, **k: None)
        monkeypatch.setenv(lan.module.UI_TRUST_LOOPBACK_PEER_ENV, "1")
        c = lan.peer(LOOPBACK_PEER)
        for path in sorted(lan.module._LAN_MUTABLE):
            r = c.post(path, json={}, headers={"x-forwarded-for": "100.64.1.5"})
            assert r.status_code != 403, f"{path} -> {r.status_code}"

    def test_the_refusal_names_what_is_allowed(self, lan):
        """Built from _LAN_MUTABLE, so the message cannot contradict the set."""
        c = lan.peer(LAN_PEER)
        detail = c.post("/api/reset", json={}).json()["detail"]
        for path in lan.module._LAN_MUTABLE:
            assert path in detail


class TestAddJobIncompleteEvaluation:
    """#204: an answer that is not a whole evaluation is a failure on both
    Add-Job routes, and writes nothing. It used to come back as
    {"ok": true, "score": null} for a role that would never reach the tracker."""

    # The prompt's one sanctioned tagless answer: the model's own failure.
    TAGLESS = ('<evaluation><summary>{"status": "failed", "error": "the posting '
               'page was empty"}</summary></evaluation>')
    BODY = {"url": "https://x.example/jobs/7", "company": "Initech", "role": "Analyst"}

    @pytest.fixture
    def evaluating(self, client, monkeypatch):
        from pipeline.app import server
        monkeypatch.setattr(server, "_detect_provider", lambda: "anthropic")
        monkeypatch.setattr(server, "_fetch_jd", lambda url, timeout=20: "JD text")
        monkeypatch.setattr(server, "_load_eval_system_prompt", lambda: "SYSTEM")
        monkeypatch.setattr(server, "_build_caller", lambda provider, model: None)
        monkeypatch.setattr(server, "_call_with_retry", lambda caller, system, user: self.TAGLESS)
        merges = []
        monkeypatch.setattr(server, "run_merge_tracker", lambda co: merges.append(co) or True)
        co = server._career_ops_local()
        written = lambda: (sorted(p.name for p in (co / "reports").glob("*")),
                           sorted(p.name for p in (co / "batch" / "tracker-additions").glob("*")))
        return server, written, merges

    def test_the_sync_route_is_a_502_and_writes_nothing(self, client, evaluating):
        _, written, merges = evaluating
        r = client.post("/api/jobs/add", json=self.BODY)
        assert r.status_code == 502
        assert r.json()["detail"] == ("LLM evaluation was incomplete: the model reported "
                                      "a failed evaluation (26-char reason withheld)")
        assert written() == (["001-acme.md"], [])      # only the fixture's own report
        assert merges == []

    def test_the_background_route_reports_why(self, client, evaluating):
        server, written, merges = evaluating
        server._run_add_job("t-204", self.BODY["url"], "Initech", "Analyst")
        r = client.get("/api/jobs/add-status/t-204")
        assert r.json() == {"status": "error", "error": "the model reported a failed "
                                                        "evaluation (26-char reason withheld)"}
        assert written() == (["001-acme.md"], [])
        assert merges == []


class TestCliSkillsScrubTheProfile:
    """Both routes that hand out a CLI-skill command scrub profile.yml first
    (#227; the scrub's own contract is tests/test_batch_common.py::
    TestScrubEveryCheckout): once, before the command is built, for the
    CAREER_OPS_PATH checkout."""

    @pytest.mark.parametrize("route", ["run", "launch"])
    def test_each_route_scrubs_before_building_the_command(self, client, tmp_path,
                                                           monkeypatch, route):
        from pipeline.app import server
        events = []
        monkeypatch.setattr(server, "scrub_profile_yml_files",
                            lambda co: events.append(("scrub", co)))
        monkeypatch.setattr(server.skills, "skill_command",
                            lambda *a, **k: events.append("command") or "true")
        monkeypatch.setattr(server.skills, "cli_available", lambda: True)
        monkeypatch.setattr(server.skills, "terminal_available", lambda: True)
        monkeypatch.setattr(server.skills, "launch_in_terminal", lambda command, cwd: {})
        if route == "run":
            r = client.post("/api/skills/run", json={"skill": "apply", "num": "1", "path": "cli"})
        else:
            r = client.post("/api/skills/launch", json={"skill": "apply", "num": "1"})
        assert r.status_code == 200, r.text
        assert events == [("scrub", tmp_path / "career-ops"), "command"]


# ── Markdown résumés and the tailoring-guidance import ────────────────────────

from pipeline.resume_text import RENDERED_RESUME_PDF  # noqa: E402

_MD_RESUME = b"# Jane Q. Example\n\njane@example.com\n\n## Experience\n- Cut waits **32%**\n"


@pytest.fixture
def md_wizard(client, tmp_path, mocker, monkeypatch):
    """The wizard's submit with gh stubbed and ROOT on a tmp dir, the real
    extractors, and the generator's payload captured."""
    from pipeline.app import server
    monkeypatch.delenv("RESUME_PATH", raising=False)
    root = tmp_path / "root"
    root.mkdir()
    mocker.patch.object(server, "ROOT", root)
    mocker.patch.object(server.gh, "repo_visibility", return_value="PRIVATE")
    mocker.patch.object(server.gh, "current_repo", return_value="me/private")
    mocker.patch.object(server.gh, "set_secret")
    mocker.patch.object(server.gh, "set_variable")
    mocker.patch.object(server.onboard, "collect_secret_blobs", return_value={})
    gen = mocker.patch.object(server.onboard, "run_generation", return_value={"ok": True})
    return {"client": client, "root": root, "gen": gen}


def _post_md(c, name="Jane_Resume.md", data=_MD_RESUME):
    import json as _json
    return c.post("/api/onboard", files={"resume": (name, data, "text/markdown")},
                  data={"form": _json.dumps({"name": "Jane"})})


class TestOnboardMarkdownResume:
    """A Markdown résumé is kept as written (resumes/resume.md), read as text
    for scoring (resume.txt), seeds cv.md as itself, and gets a base PDF
    rendered for application uploads into a folder of its own."""

    def test_a_docx_upload_still_retires_the_markdown(self, md_wizard, monkeypatch):
        """Any other upload must, or resume.md would keep leading the probe."""
        from pipeline.app import server
        monkeypatch.setattr(server.onboard, "extract_resume_text", lambda data, name: "docx text")
        resumes = md_wizard["root"] / "resumes"
        resumes.mkdir()
        (resumes / "resume.md").write_bytes(_MD_RESUME)
        r = md_wizard["client"].post("/api/onboard", files={"resume": ("cv.docx", b"PK", "application/octet-stream")},
                                     data={"form": '{"name": "Jane"}'})
        assert r.status_code == 200, r.text
        assert not (resumes / "resume.md").exists() and (resumes / "resume.docx").exists()

    def test_a_resume_path_elsewhere_is_said(self, md_wizard, fake_soffice, monkeypatch, tmp_path):
        """RESUME_PATH outranks the upload (#278): say so rather than let the
        person believe the upload is what gets scored and sent."""
        fake_soffice()
        other = tmp_path / "elsewhere.docx"
        other.write_bytes(b"PK")
        monkeypatch.setenv("RESUME_PATH", str(other))
        r = _post_md(md_wizard["client"])
        assert r.status_code == 200, r.text
        assert any("RESUME_PATH in .env names" in w for w in r.json()["warnings"])

    def test_writes_resume_md_txt_and_a_rendered_pdf(self, md_wizard, fake_soffice):
        fake_soffice()
        resumes = md_wizard["root"] / "resumes"
        resumes.mkdir()
        (resumes / "resume.docx").write_bytes(b"PK an older resume")
        (resumes / "resume.pdf").write_bytes(b"%PDF an older resume")
        r = _post_md(md_wizard["client"])
        assert r.status_code == 200, r.text
        assert r.json()["warnings"] == []
        assert (resumes / "resume.md").read_bytes() == _MD_RESUME
        assert (resumes / "resume.txt").read_text(encoding="utf-8").startswith("Jane Q. Example\n")
        assert (md_wizard["root"] / RENDERED_RESUME_PDF).read_bytes() == b"%PDF-1.4 rendered"
        # Markdown leads the probe, so the files beside it stay, as exports: the
        # upload a failed render falls back to.
        assert sorted(p.name for p in resumes.iterdir()) == [
            "rendered", "resume.docx", "resume.md", "resume.pdf", "resume.txt"]
        payload = md_wizard["gen"].call_args[0][1]
        assert payload["resumeMarkdown"] == _MD_RESUME.decode()
        assert "**" not in payload["resumeText"]

    def test_no_libreoffice_still_saves_and_says_so(self, md_wizard, fake_soffice):
        fake_soffice(available=False)
        r = _post_md(md_wizard["client"])
        assert r.status_code == 200, r.text
        (warning,) = r.json()["warnings"]
        assert "No upload PDF could be rendered" in warning and "LibreOffice" in warning
        assert (md_wizard["root"] / "resumes" / "resume.md").exists()
        assert not (md_wizard["root"] / RENDERED_RESUME_PDF).exists()

    def test_a_dot_markdown_upload_is_kept_as_resume_md(self, md_wizard, fake_soffice):
        fake_soffice()
        assert _post_md(md_wizard["client"], name="cv.markdown").status_code == 200
        resumes = md_wizard["root"] / "resumes"
        assert (resumes / "resume.md").exists() and not (resumes / "resume.markdown").exists()

    def test_non_utf8_markdown_is_a_clear_400(self, md_wizard, fake_soffice):
        fake_soffice()
        r = _post_md(md_wizard["client"], data="# Jos\xe9".encode("cp1252"))
        assert r.status_code == 400 and "UTF-8" in r.json()["detail"]

    def test_re_picking_the_same_file_is_not_re_rendered(self, md_wizard, fake_soffice):
        calls = fake_soffice()
        assert _post_md(md_wizard["client"]).status_code == 200
        assert _post_md(md_wizard["client"]).status_code == 200
        assert len(calls) == 1

    def test_an_edit_mode_save_keeps_seeding_cv_md_from_the_markdown(self, md_wizard, fake_soffice):
        """No upload: the Markdown on file still seeds cv.md, or every later
        Save would rebuild cv.md from plain text — and a missing PDF is
        rendered."""
        import json as _json
        fake_soffice()
        resumes = md_wizard["root"] / "resumes"
        resumes.mkdir()
        (resumes / "resume.md").write_bytes(_MD_RESUME)
        (resumes / "resume.txt").write_text("Jane Q. Example", encoding="utf-8")
        r = md_wizard["client"].post("/api/onboard", data={"form": _json.dumps({"name": "Jane"})})
        assert r.status_code == 200, r.text
        assert md_wizard["gen"].call_args[0][1]["resumeMarkdown"] == _MD_RESUME.decode()
        assert (md_wizard["root"] / RENDERED_RESUME_PDF).exists()

    def test_a_pdf_upload_carries_no_markdown(self, md_wizard, monkeypatch):
        from pipeline.app import server
        monkeypatch.setattr(server.onboard, "extract_resume_text", lambda data, name: "pdf text")
        r = _onboard_post(md_wizard["client"], {"name": "Jane"})
        assert r.status_code == 200, r.text
        assert "resumeMarkdown" not in md_wizard["gen"].call_args[0][1]
        assert r.json()["warnings"] == []


class TestExtractTextEndpoint:
    """POST /api/onboard/extract-text — the tailoring-guidance import. The text
    comes back for the wizard to put in the box; nothing is written."""

    def _post(self, c, data, filename):
        return c.post(f"/api/onboard/extract-text?filename={filename}", content=data)

    def test_markdown_comes_back_as_written(self, client):
        r = self._post(client, b"# Prefs\n- **Lead** with outcomes\n", "prefs.md")
        assert r.status_code == 200, r.text
        assert r.json() == {"text": "# Prefs\n- **Lead** with outcomes", "truncated": False,
                            "chars": 32, "limit": 8000}

    def test_unsupported_empty_and_textless_are_400(self, client):
        assert self._post(client, b"x", "g.rtf").status_code == 400
        assert self._post(client, b"", "g.md").status_code == 400
        assert self._post(client, b"  \n ", "g.txt").status_code == 400

    def test_an_unsupported_format_is_refused_before_the_body_is_read(self, client, monkeypatch):
        from pipeline.app import server
        monkeypatch.setattr(server.onboard, "GUIDANCE_IMPORT_MAX_BYTES", 64)
        r = self._post(client, b"x" * 65, "g.rtf")
        assert r.status_code == 400 and "Unsupported" in r.json()["detail"]

    def test_oversized_is_413_before_extraction(self, client, monkeypatch):
        from pipeline.app import server
        monkeypatch.setattr(server.onboard, "GUIDANCE_IMPORT_MAX_BYTES", 64)
        monkeypatch.setattr(server.onboard, "import_guidance_text",
                            lambda *a: pytest.fail("an oversized upload must not be extracted"))
        assert self._post(client, b"x" * 65, "g.txt").status_code == 413

    def test_nothing_is_written(self, client, tmp_path, monkeypatch):
        from pipeline.app import server
        root = tmp_path / "root"
        root.mkdir()
        monkeypatch.setattr(server, "ROOT", root)
        before = sorted(p for p in tmp_path.rglob("*"))
        assert self._post(client, b"Lead with outcomes.", "g.txt").status_code == 200
        assert sorted(p for p in tmp_path.rglob("*")) == before
        assert not (tmp_path / "career-ops" / "config" / "tailoring.md").exists()

    def test_a_cross_origin_post_is_refused(self, client):
        r = client.post("/api/onboard/extract-text?filename=g.txt", content=b"x",
                        headers={"Origin": "https://evil.example"})
        assert r.status_code == 403

    def test_a_lan_peer_is_refused(self, lan):
        r = lan.peer(LAN_PEER).post("/api/onboard/extract-text?filename=g.txt", content=b"x")
        assert r.status_code == 403
        assert "loopback-only" in r.json()["detail"]

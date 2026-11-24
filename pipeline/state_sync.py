"""Push and Refresh: where local tracker state meets the cloud's.

Refresh pulls the daily's newest `pipeline-output-*` artifact and merges it
into the local career-ops (cloud wins for shared rows; local-only rows are
kept). Push sends the pending status overrides — kanban drags, the re-check's
Discards, handoff outcomes — to the cloud tracker by dispatching
edit-tracker.yml, after the same pull, so they resolve against current cloud
row numbers.

Both were route bodies in the FastAPI module, so nothing but the UI could sync.
They live here, FastAPI-free on purpose (`tests/test_state_sync.py` imports
this in a fresh interpreter with fastapi, pydantic and markdown blocked), so a
caller outside the UI — a CLI verb, the git transport (#211) — runs the code
the buttons run. The routes in pipeline/app/server.py are wrappers: they keep
the UI's own guards (refuse during a local run, the in-process re-check lock)
and turn a SyncError into an HTTPException with the same status and detail.
"""

import json
import shutil
from pathlib import Path

from pipeline.app import data, gh

ROOT = Path(__file__).resolve().parent.parent

# Scratch dir where `gh run download` drops an artifact before it is merged
# into the local career-ops. Gitignored, and WIPED before every download — which
# is why conftest redirects it for every test: a Push or Refresh test reaching
# this path would rmtree the developer's real one.
UI_CACHE = ROOT / ".ui-cache" / "latest"

# Cloud workflow filenames (must match .github/workflows/*.yml).
DAILY_WORKFLOW = "daily-pipeline.yml"
EDIT_WORKFLOW = "edit-tracker.yml"

# The daily uploads the `pipeline-output-*` artifact; Refresh and Push pull its
# newest successful run. A list because latest_successful_run takes several.
PIPELINE_WORKFLOWS = [DAILY_WORKFLOW]

# Overrides dispatched to the cloud but not yet in a pipeline artifact:
# {num: status}, or {num: {status, note}} for a by-hand mark
# (data.pushed_overlay_value). Overlaid on every job load (apply_pushed_overlay) so a pushed
# status survives Refresh and restarts, and self-cleaned entry by entry once an
# artifact shows it.
PUSHED_OVERRIDES_FILE = ROOT / ".ui-cache" / "pushed-overrides.json"


class SyncError(Exception):
    """A Push or Refresh that could not do what was asked. `status` is the
    HTTP-style code the UI route answers with — 400 nothing to push, 404 no
    successful run yet, 409 no tracker to update, 502 gh failed — so another
    caller can tell the cases apart; `detail` is the message for the person."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def load_pushed_overrides() -> dict:
    return data.load_status_overrides(PUSHED_OVERRIDES_FILE)


def apply_pushed_overlay(rows: list[dict]) -> None:
    """Show each pushed status on its row until an artifact catches up, and
    drop the entries an artifact has caught up with."""
    pushed = load_pushed_overrides()
    if not pushed:
        return
    caught_up: dict[str, object] = {}
    for row in rows:
        num = str(row.get("num"))
        if num not in pushed:
            continue
        value = pushed[num]
        status, note = data.override_status(value), data.override_note(value)
        # An entry carrying a Notes mark is caught up only once the row carries
        # the mark too: a by-hand Skip lift (#183) sets the status the row
        # already has, so status alone would drop it at once and the tag would
        # come back until the next artifact.
        if row.get("status_canonical") == status and (not note
                                                      or data.note_landed(row.get("notes"), note)):
            caught_up[num] = value
            continue
        row["status"] = status
        row["status_canonical"] = status
        data.apply_override_note(row, note)
    if caught_up:
        # Re-read under the lock and drop only what this load saw caught up, so
        # a Push landing in between keeps the entry it just wrote.
        with data.editing_overrides(PUSHED_OVERRIDES_FILE) as pushed:
            for num, value in caught_up.items():
                if pushed.get(num) == value:
                    del pushed[num]


def _pull(run: dict, career_ops: Path) -> dict:
    """Download `run`'s artifact into a fresh UI_CACHE and merge it into the
    local career-ops. Raises gh.GhError; local is only written once the
    download has succeeded."""
    # Emptied here, raising, although gh.download_artifact empties it too: that
    # one ignores errors, so a file something holds open would survive into a
    # `gh run download` "file exists" GhError — which Push reads as offline,
    # pushing against a stale tracker with no sign its clobber guard was skipped.
    if UI_CACHE.exists():
        shutil.rmtree(UI_CACHE)
    return data.sync_pulled_tracker(gh.download_artifact(run["databaseId"], UI_CACHE), career_ops)


def refresh(career_ops: Path) -> dict:
    """Merge the newest successful pipeline artifact into `career_ops`
    (offline-first: the merged tracker lives durably in local, so it survives
    restarts and offline periods). A gh/credits error leaves local intact."""
    try:
        run = gh.latest_successful_run(PIPELINE_WORKFLOWS)
        if run is None:
            raise SyncError(404, "No successful pipeline runs found yet.")
        summary = _pull(run, career_ops)
    except gh.GhError as e:
        # Offline / no credits: keep the last-synced local tracker untouched.
        raise SyncError(502, f"{e} — showing last-synced local data.") from e
    return {
        "run_id": run["databaseId"],
        "created_at": run.get("createdAt"),
        "title": run.get("displayTitle"),
        "rows": summary["rows"],
        "renamed_reports": len(summary["renames"]),
    }


def push_pending(career_ops: Path) -> dict:
    """Push pending status changes to the cloud tracker via edit-tracker.

    Clobber guard: first pull the latest pipeline artifact and MERGE it into
    the local tracker (so rows the pipeline added since the last Refresh are
    present), then resolve the pending overrides against that base and dispatch
    {num: status} — {num: {status, note}} for a re-check Discard or a merge
    reopen, whose Notes mark must reach the cloud too (#163). Offline (no runs /
    gh error) → push against the last-synced local tracker.

    The pull rewrites applications.md and renames reports exactly as refresh()
    does, so a caller owes it the same guards; the Push route brings none
    today (#234)."""
    overrides = data.load_status_overrides()
    if not overrides:
        raise SyncError(400, "No pending status changes to push.")

    local_apps = career_ops / "data" / "applications.md"

    # Offline (no runs / gh error) → the last-synced local tracker, untouched.
    base_source = "local"
    try:
        run = gh.latest_successful_run(PIPELINE_WORKFLOWS)
        if run is not None:
            _pull(run, career_ops)
            base_source = "refreshed"
    except gh.GhError:
        pass  # offline — push against the last-synced local tracker

    if not local_apps.exists():
        raise SyncError(409, "No applications.md to update (run the pipeline so the tracker exists).")
    base_text = local_apps.read_text(encoding="utf-8")

    # Resolve each identity-anchored override to the correct num IN THIS base. An
    # override whose company isn't in the tracker is returned in `unresolved` —
    # NOT applied and NOT dispatched, so we never mark a different company that
    # merely shares the num. cloud_payload is {num: status}, or {num: {status,
    # note}} when a Notes mark rides along (edit-tracker.yml's input). We do NOT
    # write the status into applications.md: that file is the
    # last-synced CLOUD mirror, and the pending edit lives in the pushed-override
    # overlay (below) until a real cloud run incorporates it — only then does
    # apply_pushed_overlay's self-clean fire. Writing it here would self-clean
    # immediately and then vanish when Refresh's cloud-wins reverts the row.
    _, cloud_payload, unresolved = data.resolve_overrides_for_push(
        base_text, overrides, build_text=False)

    # An unresolved DISCARD override targets a closed role that isn't in the cloud
    # tracker: it's already applied locally, a closed role won't reappear to match
    # on a later push, and the cloud's own daily recheck catches its own closed
    # roles — so drop it instead of keeping it to nag on every future push.
    # Non-Discard unresolved overrides (e.g. an auto-submitted Applied) are KEPT:
    # their company may appear in a later cloud run, and then the next push lands.
    stale_discards = [k for k in unresolved
                      if data.canonical_status(data.override_status(overrides[k])) == "Discarded"]
    kept_unresolved = [k for k in unresolved if k not in stale_discards]

    # Dispatch edit-tracker with only the resolved overrides — avoids GitHub's
    # workflow_dispatch input size limit that the full base64 tracker can exceed.
    # Nothing resolved → nothing to push (don't fire an empty workflow run).
    if cloud_payload:
        try:
            gh.trigger_workflow(EDIT_WORKFLOW, {"status_overrides_json": json.dumps(cloud_payload)})
        except gh.GhError as e:
            raise SyncError(502, str(e)) from e
        # Clear the keys we pushed PLUS the dead Discard overrides (re-reading
        # under the lock, and only while each still holds the value read here),
        # leaving the genuinely-pending unresolved ones and anything recorded or
        # rewritten during the gh round-trip.
        done = [k for k in overrides if k not in unresolved] + stale_discards
        data.clear_status_overrides({k: overrides[k] for k in done})
        # Persist dispatched overrides so they survive Refresh and restarts until
        # the pipeline produces a new artifact that already has the correct statuses.
        with data.editing_overrides(PUSHED_OVERRIDES_FILE) as pushed:
            # A status, or {status, note} for a by-hand mark the overlay must
            # keep showing until the row carries it (data.pushed_overlay_value).
            pushed.update({num: data.pushed_overlay_value(v) for num, v in cloud_payload.items()})
    elif stale_discards:
        # Nothing resolved to dispatch, but still drop the dead Discard overrides.
        data.clear_status_overrides({k: overrides[k] for k in stale_discards})
    return {"pushed": len(cloud_payload), "unresolved": len(kept_unresolved), "base": base_source}

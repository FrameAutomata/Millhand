"""Shared utilities for batch job evaluation (submit, evaluate, retrieve)."""

import contextlib
import csv
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from pipeline.work_location import (
    WORK_LOCATION_RE, location_mark, parse_work_location,
)
from pipeline.tracker_layout import (
    SCORE_SENTINELS, claimed_report_nums, data_rows, header_aliases, header_columns,
    is_score_cell, load_contract, split_row,
    # Aliased: `report_num` is also the name of the number itself in
    # write_job_result and below, and a local shadowing a callable is a
    # TypeError waiting for whoever adds the next call.
    report_num as row_report_num,
)

MAX_TOKENS = 8192


def read_url_set(path: Path) -> set[str]:
    """Read a newline-delimited URL file into a set of stripped, non-blank
    lines. Missing file → empty set."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_date_posted(val: str) -> datetime | None:
    """Parse a date_posted string. Tolerates both date and datetime forms."""
    if not val or val.strip().lower() in ("", "none", "nan", "nat"):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(val.strip(), fmt)
        except ValueError:
            continue
    return None


def atomic_write_text(path: Path, content: str) -> None:
    """Write content atomically — write to a tmp sibling then os.replace.
    os.replace is atomic on POSIX and Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default


def tail_text(path: Path, tail_lines: int, max_bytes: int = 16384) -> str:
    """The last `tail_lines` lines of a log file, reading at most `max_bytes`
    from the end rather than the whole file — cheap enough to poll a live,
    growing log. Returns "" before the file exists or if it can't be read.
    Shared by the UI's local-run and handoff-build log streams."""
    if tail_lines <= 0:
        return ""   # else [-0:] would slice the WHOLE window, not zero lines
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            raw = f.read()
    except OSError:
        return ""
    return "\n".join(raw.decode("utf-8", errors="replace").splitlines()[-tail_lines:])


def env_float(name: str, default: float) -> float:
    """A float env override that can NEVER crash startup: a malformed or
    set-but-empty value warns and falls back rather than raising. Shared by
    orchestrate's flag defaults and batch_evaluate's timeout/budget knobs so
    they parse env the same way."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[config] ignoring invalid {name}={raw!r} (using {default})")
        return default


def env_int(name: str, default: int) -> int:
    """The integer sibling of env_float — a never-crash int env override
    (malformed / set-but-empty warns and falls back)."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[config] ignoring invalid {name}={raw!r} (using {default})")
        return default


def thinking_disabled() -> bool:
    """Reasoning/thinking is unnecessary for short tailoring output and cover
    letters (and slows/garbles vLLM-served reasoning models like MiMo/Qwen3), so
    disable it by default for those use cases. Set APPLY_ENABLE_THINKING=true to
    keep it on (e.g. if a provider rejects the toggle)."""
    return os.environ.get("APPLY_ENABLE_THINKING", "").strip().lower() not in ("1", "true", "yes")


def pid_alive(pid: int) -> bool:
    """Best-effort liveness check for a process id (used by the cross-process
    eval lock and the UI's local-run orphan guard).

    On Windows os.kill(pid, 0) would SEND CTRL_C_EVENT, so use OpenProcess +
    GetExitCodeProcess — OpenProcess succeeds on a zombie while any handle to
    it is held, so only an exit code of STILL_ACTIVE means actually running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def queued_additions(tracker_dir: Path) -> set[str]:
    """The names of the addition files in the queue, readable or not — the one
    listing of it, taken before a merge and after (`run_merge_tracker`)."""
    return {f.name for f in tracker_dir.glob("*.tsv")}


def has_pending_tracker_additions(tracker_dir: Path) -> bool:
    """True when career-ops/batch/tracker-additions holds un-merged TSVs.

    merge-tracker.mjs MOVES each TSV into tracker-additions/merged/ once it has
    folded it into applications.md, so a top-level *.tsv means "evaluated but
    not yet merged" — the signal that lets an interrupted run heal on a later
    invocation even when it processes zero new jobs. Since career-ops#5862 that
    includes an addition it refused and kept (see `_warn_on_lost_additions`)."""
    try:
        return bool(queued_additions(tracker_dir))
    except OSError:
        return False


def normalize_company(s: str) -> str:
    """Lowercase + strip everything non-alphanumeric — the canonical company
    key used for identity matching (mirrors merge-tracker.mjs's normalizeCompany).
    One definition so filename lookup, cover-letter caching, and tracker
    re-anchoring can't drift apart."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _company_slug(company: str) -> str:
    """The hyphenated company slug a report filename carries
    (`{num}-{company-slug}-{date}.md`), as `write_job_result` builds it."""
    return re.sub(r"[^a-z0-9]+", "-", (company or "").lower()).strip("-")


# ── Cross-process lock ────────────────────────────────────────────────────────
# One mechanism for every pid-based guard (the batch-eval single-flight lock and
# the UI local-run orphan guard). A lock file holds "pid timestamp". It is
# reclaimable only when its holder is DEAD or its timestamp is older than
# max_age — and a live holder REFRESHES its timestamp (heartbeat) so a long,
# progressing run never looks stale and is never stolen. The timestamp is the
# pid-reuse safety valve: if the holder died and the OS recycled its pid to an
# unrelated live process, the frozen timestamp eventually goes stale and the
# lock is reclaimed instead of wedging forever.

# An empty lock file is a racer caught between the O_EXCL create and its pid
# write — NOT free. Only treat it as reclaimable once it has sat empty past this
# grace window (i.e. the writer crashed mid-create).
_EMPTY_LOCK_GRACE = 5.0


def read_process_lock(path: Path) -> tuple[int, float]:
    """(pid, timestamp) from a lock file; (0, 0.0) when absent/empty/unparseable."""
    try:
        parts = path.read_text(encoding="utf-8").split()
        return int(parts[0]), (float(parts[1]) if len(parts) > 1 else 0.0)
    except (OSError, ValueError, IndexError):
        return 0, 0.0


def write_process_lock(path: Path, pid: int) -> None:
    """Atomically record `pid` as the lock holder with a fresh timestamp. Used
    to register a CHILD process (the local-run orchestrate subprocess) as the
    holder, and internally by acquire/refresh."""
    try:
        atomic_write_text(path, f"{pid} {time.time()}")
    except OSError:
        pass


def _lock_reclaimable(path: Path, max_age: float) -> bool:
    """True when no LIVE, non-stale holder owns the lock."""
    pid, ts = read_process_lock(path)
    if pid == 0:
        # Empty/unparseable: a racer mid-create (don't steal) unless it has sat
        # untouched past the grace window (writer crashed mid-create).
        try:
            return (time.time() - path.stat().st_mtime) > _EMPTY_LOCK_GRACE
        except OSError:
            return True
    if pid == os.getpid():
        return True
    if not pid_alive(pid):
        return True
    # Live holder: reclaim only if its heartbeat went stale — a genuine,
    # progressing run keeps its timestamp fresh.
    return ts > 0 and (time.time() - ts) > max_age


def acquire_process_lock(path: Path, *, max_age: float) -> bool:
    """Atomically take a self-lock (the current process becomes the holder).
    O_CREAT|O_EXCL makes the create-if-absent atomic so two simultaneous starts
    can't both pass a read-then-write check. Returns False when a live, non-stale
    holder owns it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{os.getpid()} {time.time()}".encode("utf-8")
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _lock_reclaimable(path, max_age):
                return False
            try:
                path.unlink()       # reclaim a dead/stale lock, then retry O_EXCL
            except OSError:
                pass
            continue
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True
    return False


def refresh_process_lock(path: Path) -> None:
    """Heartbeat: rewrite our timestamp so a long live run never looks stale.
    No-op unless we own the lock."""
    pid, _ = read_process_lock(path)
    if pid == os.getpid():
        write_process_lock(path, os.getpid())


def release_process_lock(path: Path) -> None:
    """Delete the lock, but only if we still own it (a stale-takeover may have
    replaced our pid, and we must not delete another process's lock)."""
    pid, _ = read_process_lock(path)
    if pid == os.getpid():
        # Retried briefly: on Windows a waiter polling the lock holds it open
        # for a moment, and a delete then fails with a sharing violation —
        # leaving a fresh lock naming a live pid for the waiter to wait out.
        for _ in range(10):
            try:
                path.unlink(missing_ok=True)
                return
            except PermissionError:
                time.sleep(0.01)
            except OSError:
                return


def process_lock_active(path: Path, *, max_age: float) -> bool:
    """True if a LIVE, non-stale holder owns the lock — the non-acquiring probe
    behind is_running()-style checks (where the holder is a CHILD pid, not us)."""
    if not path.exists():
        return False
    return not _lock_reclaimable(path, max_age)


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"jobs": {}}


def max_report_num(reports_dir: Path, state: dict, applications_md: Path) -> int:
    """Highest report number already claimed, so the next one is free.

    A `NNN-RESERVED.md` lock counts, deliberately: claiming the number is the
    lock's whole purpose, so skipping it would hand out a number career-ops has
    already reserved. This is the OPPOSITE of the rule in
    `find_report_file` below, which must skip locks because it
    resolves a number to a file to render. Do not "make them consistent" —
    they answer different questions.

    The tracker counts too (#203), and the argument is required so a new caller
    cannot leave it out. `reports/` and the batch state are the cloud's cached
    paths, so after a cache eviction both read as empty while a restored tracker
    still links every number up to its maximum. Numbered from the files alone,
    the next run minted `001` again: two rows of one tracker then resolve to one
    report file, and merge-tracker's report-number tier can fold a new
    evaluation onto an old row at the same company. `max_tracker_num` already
    reads the tracker for the same reason, and career-ops'
    `reserve-report-num.mjs` (the `--batch` minter) counts tracker report links
    too, so the two minters agree.

    What it cannot see after an eviction is a number minted but never linked —
    an addition merge-tracker refused, a row kept out by `--min-score` — so such
    a number can be handed out again; the batch state that recorded it is the
    cached path that was lost.
    """
    max_num = 0
    if reports_dir.exists():
        for f in reports_dir.glob("*.md"):
            num = _report_num_prefix(f.name)
            if num:
                max_num = max(max_num, int(num))
    for job in state.get("jobs", {}).values():
        try:
            max_num = max(max_num, int(job.get("report_num", 0)))
        except (ValueError, TypeError):
            pass
    for row in _tracker_rows(applications_md):
        max_num = max(max_num, *claimed_report_nums(row.get("report", ""), row.get("notes", "")), 0)
    return max_num


# The leading `NNN-` of a report filename (`{num}-{company-slug}-{date}.md`):
# the one thing `max_report_num`, `find_report_file`, the Report-cell writer and
# the dead-link resolver all key on, so it is spelled once.
_REPORT_PREFIX_RE = re.compile(r"^(\d+)-")


def _report_num_prefix(name: str) -> str:
    """The leading number of a report filename, or "" when it has none."""
    m = _REPORT_PREFIX_RE.match(name)
    return m.group(1) if m else ""


# `NNN-RESERVED.md` is career-ops' report-number LOCK, not a report: a JSON body
# ({"pid","token","created_at"}) dropped to claim a number, which survives any
# run killed before the real report replaces it. It carries the `NNN-` prefix
# the resolver below keys on, so a first-match glob handed the UI the lock and
# the report pane rendered raw JSON where the evaluation belongs — silently,
# reading as a corrupt report rather than the wrong file.
#
# Only `find_report_file` skips locks. The other readers of a report number ask
# "is this number taken?" and must keep counting them — see `max_report_num`.
RESERVED_REPORT_SUFFIX = "-RESERVED.md"


def _report_int(value) -> int | None:
    """A report number as an int, or None if it isn't one."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def find_report_file(reports_dir: Path, report_num: str) -> Path | None:
    """Locate a report file by its number. Reports are named
    `{num}-{company-slug}-{date}.md`; the tracker stores the zero-padded num
    (e.g. "042"). Match on the leading numeric segment, tolerating padding
    differences (42 vs 042). Reservation locks are skipped (see above).

    Iterated in sorted order so that a number matching two REAL reports resolves
    to the same one on every request — `app/data._rename_report_file` records
    when that happens ("a number matches two files"). That is stability, not
    correctness: the tie is decided by filename, and nothing here knows which
    report the row meant. Every UI caller looks up by NUMBER (server.py
    deliberately: "not the link target — tolerant of report renames"), so there
    is no authority to defer to; `read_report` below asks the link first and
    falls back here only when the link is dead.

    Lives here rather than in `app/data.py` because the merge-time sanitizer and
    the three report readers need the same lookup and cannot import the UI."""
    # No `Path(reports_dir)` re-wrap: callers pass a Path, and constructing a
    # new one reads `os.name` at call time (3.12's Path.__new__), which the
    # skills launcher's tests patch to "posix" — on Windows that is a PosixPath
    # nothing can instantiate, and the UI's report lookup sits on that path.
    files = _reports_numbered(reports_dir, report_num)
    return files[0] if files else None


def _reports_numbered(reports_dir: Path, report_num: str) -> list[Path]:
    """Every real report (locks skipped) whose leading number is `report_num`,
    in filename order. More than one is a real state: local holds both a synced
    cloud report and a local-only one that shares the number (see
    `app/data._rename_report_file`), so a caller that WRITES a binding must
    not pick by sort order — `resolve_report` asks the company."""
    wanted = _report_int(report_num)
    if wanted is None or not reports_dir.exists():
        return []
    return [f for f in sorted(reports_dir.glob("*.md"))
            if not f.name.endswith(RESERVED_REPORT_SUFFIX)
            and _report_int(_report_num_prefix(f.name)) == wanted]


# The `[N](path)` of a Report cell, and the two things every reader needs from
# it: the number, and the career-ops-relative path. merge-tracker.mjs
# normalizes the Report link relative to the tracker FILE
# (tracker-links.mjs:normalizeReportLink), and the pipeline seeds the tracker at
# career-ops/data/applications.md — so a link the pipeline emitted as
# `reports/042-x.md` comes back as `../reports/042-x.md`. The older
# merge-tracker copied the cell verbatim, so both shapes are in circulation
# within one file. Every consumer of `report_path` (the UI's two parsers in
# app/data, cover letters, both tailors) resolves it as `career_ops /
# report_path`, which the `../` form escapes — and `read_text` returns "" on the
# miss, so tailoring and cover letters silently lose the evaluation report's
# proof points. Strip the ascent HERE, the single point every parser extracts
# it through (app/data imports these rather than keeping a copy), so the stored
# value is career-ops-relative whichever shape the file holds.
_REPORT_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_REPORT_ASCENT_RE = re.compile(r"^(?:\.\./)+")


def _report_link(cell: str) -> tuple[str, str]:
    """(report_num, career-ops-relative report_path) from a Report cell."""
    m = _REPORT_LINK_RE.search(cell or "")
    if not m:
        return "", ""
    return m.group(1).strip(), _REPORT_ASCENT_RE.sub("", m.group(2).strip())


def _linked_report(base: Path, report_path: str) -> Path | None:
    """The file a Report link names taken literally (ascent stripped), or None
    for an empty link. Whether it exists is the caller's question."""
    report_path = _REPORT_ASCENT_RE.sub("", (report_path or "").strip())
    return base / report_path if report_path else None


def resolve_report(base: Path, report_path: str, *, num_text: str = "",
                   company: str = "") -> Path | None:
    """The file a tracker row's Report link actually names — the link target
    when it exists, else the report in `base/reports/` with the link's number:
    `num_text` (the link's `[N]` text — the row's identity everywhere else in
    the pipeline, so it wins when the dead filename's own prefix disagrees), or
    failing that the filename's `NNN-` prefix. When two reports carry the
    number (a synced cloud report beside a local-only one), `company` — the
    row's — picks the one whose filename slug names it, and no company or no
    match declines rather than guesses: binding a row to another company's
    evaluation is worse than building from the JD alone.

    The link is model-authored (#162): the prompt gives the model only the
    SHAPE `[N](reports/N-company-slug-DATE.md)` and it invents the slug — keeps
    an `&`, appends the role, uses an en-dash — while `write_job_result` names
    the file by its own slug. Six of 230 rows in a real tracker linked a file
    that never existed. The UI copes, resolving by number; the readers that
    resolved `base / report_path` got "" from `read_text` and built from the JD
    alone with no log line. Writers now own the cell, but rows already merged
    keep the link they were merged with, so the readers resolve the same way
    the UI does. The link is asked FIRST: a number can match two files and only
    the link knows which was meant."""
    direct = _linked_report(base, report_path)
    if direct is None:
        return None
    if direct.is_file():
        return direct
    num = num_text or _report_num_prefix(direct.name)
    files = _reports_numbered(base / "reports", num) if num else []
    if len(files) > 1:
        slug = _company_slug(company)
        files = [f for f in files if slug and slug in f.name]
    return files[0] if len(files) == 1 else None


def read_report(base: Path, report_path: str, *, label: str = "report",
                company: str = "", num_text: str = "") -> str:
    """The text of the report a tracker row links to, or "" — resolving a dead
    link by its number (`resolve_report`, with the row's `company` to tell two
    same-numbered reports apart, and its `[N]` text as `num_text` for a dead
    path that lost its own number, as the UI and the merge-time repair already
    could) and SAYING so either way, since the consumers (cover letters, both
    tailors) otherwise build from the JD alone in silence, which is the failure
    this exists to end."""
    if not (report_path or "").strip():
        return ""
    found = resolve_report(base, report_path, num_text=num_text, company=company)
    if found is None:
        print(f"[{label}] report {report_path} not found — building from the JD alone")
        return ""
    if found != _linked_report(base, report_path):
        print(f"[{label}] report link {report_path} is dead — using {found.name} (same number)")
    return read_text(found)


# The `## Machine Summary` block a report carries, read here beside the other
# report readers because two stages act on it: the digest excludes a
# `final_decision: Skip`, and both it and the handoff build demote a role whose
# `work_location` names a metro outside the candidate's commutable ones (#180).
# The fence may sit a few prose lines below the heading (a model that adds a
# sentence there is drift, not a different shape): any non-fence lines are
# skipped, or `final_decision: Skip` would parse as no summary and a Skip
# above min_score would be digested.
_MACHINE_SUMMARY_RE = re.compile(
    r"##\s*Machine Summary[^\n]*\n(?:(?!\s*```)[^\n]*\n)*?\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```",
    re.S)
_DECISION_RE = re.compile(r"^\s*final_decision:\s*[\"']?([^\"'\n]+?)[\"']?\s*$", re.M)


def parse_machine_summary(report_text: str) -> dict:
    """The `## Machine Summary` YAML block of a report as a dict, `{}` when
    there is none. PyYAML when it is importable and the block parses; else a
    regex that recovers `final_decision` alone, which is the field that
    changes what the digest does (a `Skip` is excluded)."""
    m = _MACHINE_SUMMARY_RE.search(report_text or "")
    if not m:
        return {}
    block = m.group(1)
    try:
        import yaml  # a pipeline dependency; optional here on purpose
        doc = yaml.safe_load(block)
        if isinstance(doc, dict):
            return doc
    except Exception:
        pass
    d = _DECISION_RE.search(block)
    return {"final_decision": d.group(1).strip()} if d else {}


def max_tracker_num(applications_md: Path, state: dict) -> int:
    max_num = 0
    if applications_md.exists():
        # Through the shared walk: the `#` is not guaranteed to be the first
        # cell (detect_columns imposes no column order), and a header or a
        # separator row must not be read as data.
        for columns, cells in data_rows(applications_md.read_text(encoding="utf-8")):
            num_idx = columns.index("num")
            if len(cells) > num_idx:
                try:
                    max_num = max(max_num, int(cells[num_idx]))
                except ValueError:
                    pass
    for job in state.get("jobs", {}).values():
        try:
            max_num = max(max_num, int(job.get("tracker_num", 0)))
        except (ValueError, TypeError):
            pass
    return max_num


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_json_loose(text: str) -> dict | None:
    """Try parsing `text` as JSON. If that fails, try each balanced {...}
    block found in the text and return the first one that parses."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Walk the string finding balanced { ... } blocks (depth-aware), so we
    # don't over-capture when the response contains multiple JSON-like spans.
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    start = -1
                    continue
    return None


# Tracker-additions TSV schema — note STATUS comes before SCORE here, the
# opposite of applications.md (merge-tracker.mjs swaps them when merging).
# Mirrors what the LLM is told to emit in _profile.md. Named rather than
# counted because this module writes the file and `pipeline/app/data.py` reads
# it back: two spellings of one layout is how a column insertion upstream gets
# fixed in one reader and silently misread by the other.
ADDITION_COLUMNS = [
    "num", "date", "company", "role", "status", "score", "pdf", "report", "notes",
]
_TRACKER_TSV_COLUMNS = len(ADDITION_COLUMNS)

# career-ops' label line for an addition: the one its batch worker is told to
# write "exactly as shown" (batch/batch-prompt.md, Step 5, since career-ops#3706)
# and the one `write_job_result` puts above every row it writes for a checkout
# that reads it (`reads_labelled_additions`), because career-ops#6062 refuses a
# headerless addition outright. Its first nine labels are ADDITION_COLUMNS in
# order — the only reason the positional chain can address the data row beneath
# it (`_positional_addition`). The tenth is the `url` column the worker fills
# and our writer leaves empty: the posting URL lives in Notes, where every
# reader here looks for it, and a URL cell on a tracker with no URL column only
# makes merge-tracker warn that its URL dedup is inactive. One copy, which the
# tests import; `tests/test_merge_tracker_contract.py` holds it to the prompt
# wherever a checkout is at hand.
ADDITION_LABELS = (*ADDITION_COLUMNS, "url")
ADDITION_LABEL_LINE = "\t".join(ADDITION_LABELS)
# Every cell the sanitizers below reach for, resolved from that list rather than
# written as a literal — the guards only check the TOTAL, so a column inserted
# upstream would leave them silently rewriting the wrong cells.
_ROLE_IDX = ADDITION_COLUMNS.index("role")
_STATUS_IDX = ADDITION_COLUMNS.index("status")
_SCORE_IDX = ADDITION_COLUMNS.index("score")
_COMPANY_IDX = ADDITION_COLUMNS.index("company")
_REPORT_IDX = ADDITION_COLUMNS.index("report")
_NOTES_IDX = ADDITION_COLUMNS.index("notes")
_PDF_IDX = ADDITION_COLUMNS.index("pdf")

# The cells merge-tracker's headed parser refuses BLANK ("present but empty —
# use the documented sentinel"; tracker-parse.mjs's `TSV_REQUIRED_FIELDS`),
# which its headerless parser never checked: every label but `notes` and `url`.
_REQUIRED_IDXS = [ADDITION_COLUMNS.index(name) for name in (
    "num", "date", "company", "role", "status", "score", "pdf", "report")]
# What the evaluation prompt tells the model to write in the PDF column
# (`build_system_prompt`), and so what `labelled_addition` writes into a PDF
# cell the model left blank.
_PDF_CELL = "null"

# The Report cell is `[N](path)`, and it is the one cell whose shape identifies
# it — which makes it the anchor for deciding whether a row's nine columns are
# where they should be. `app/data.py:_realign_cells` anchors on the same cell for
# the same reason.
_REPORT_CELL_RE = re.compile(r"^\[\w+\]\([^)]+\)$")

# Labels a first line needs before it is READ as a header row at all — the
# threshold merge-tracker's `looksLikeTsvHeaderRow` uses.
_ADDITION_HEADER_MIN_LABELS = 3


def read_addition(path: Path) -> str:
    """An addition file's text as merge-tracker reads it — for every reader here.

    Bytes decoded, not `read_text`: that translates a lone `\\r` inside a cell
    into a line break, which splits a headed row in two, so a row merge-tracker
    reads whole (and merges, or refuses) read here as a headed file with no
    single data row and vanished from the loss guard, the sanitizer and the UI.
    And `utf-8-sig`, because merge-tracker's `trim()` drops a leading BOM from
    the first label where Python's `strip()` keeps it, which hid that label from
    the header map. Bytes that are not UTF-8 decode to U+FFFD, as Node's decoder
    makes them, rather than raising: the file is one merge-tracker has already
    read, merged or refused, and archived, and a raise here stopped every later
    merge at recovery. A file already gone reads as empty, as `read_text` does."""
    try:
        return path.read_bytes().decode("utf-8-sig", errors="replace")
    except FileNotFoundError:
        return ""


def _split_addition_line(line: str) -> list[str]:
    """One addition line's trimmed cells, split the way merge-tracker's
    `splitAdditionCells` splits it: a markdown pipe row when the line STARTS
    with a pipe (unindented, as there), else a tab row."""
    if line.startswith("|"):
        return split_row(line)
    return [c.strip() for c in line.split("\t")]


def _addition_lines(text: str) -> tuple[list[str], dict[str, int] | None]:
    """An addition file's non-blank lines, and `{field: column}` when the first
    of them is a HEADER row, else None.

    career-ops' batch worker has written a row of column labels above its one
    data row since career-ops#3706 (merged 2026-09-04), and merge-tracker reads
    that form by NAME — the one form in which a discarded, never-scored row,
    `—` in both score and status, has an order at all (#3517). The headerless
    row is accepted until career-ops#6062, so both arrive. Detected by
    merge-tracker's own rule (`looksLikeTsvHeaderRow`) against the alias table
    career-ops ships (`header_aliases`) and resolved by its first-occurrence
    rule (`resolveTsvColumns`): the first cell is not all digits (a data row
    leads with its tracker number) and at least three DISTINCT fields are
    recognised, so a data row would need three header words in three cells to
    be taken for one. Lines split on `\\r?\\n`, as merge-tracker splits them.
    Read as a headerless row instead, the label line is a job called "role" at
    a company called "company"."""
    lines = [line for line in re.split(r"\r?\n", text) if line.strip()]
    if not lines:
        return lines, None
    cells = _split_addition_line(lines[0])
    if len(cells) < _ADDITION_HEADER_MIN_LABELS or re.fullmatch(r"[0-9]+", cells[0]):
        return lines, None
    aliases = header_aliases()
    fields: dict[str, int] = {}
    for i, cell in enumerate(cells):
        key = aliases.get(cell.lower())
        if key is not None:
            fields.setdefault(key, i)
    return lines, (fields if len(fields) >= _ADDITION_HEADER_MIN_LABELS else None)


def headed_addition_row(text: str) -> dict[str, str] | None:
    """A headed addition's data row keyed by field name; `{}` when the file is
    headed but has no single data row (merge-tracker refuses that file too);
    None when it is not headed, for the caller's positional reading. Test the
    result with `is None` — `{}` is falsy, and falling through on it reads the
    label row as a job."""
    lines, fields = _addition_lines(text)
    if fields is None:
        return None
    if len(lines) != 2:
        return {}
    cells = _split_addition_line(lines[1])
    return {name: cells[i] if i < len(cells) else "" for name, i in fields.items()}


def _positional_addition(text: str) -> tuple[str, str] | None:
    """`(header, row)` for the positional sanitize chain: `header` is "" for a
    headerless addition, or the label line and its newline — to be written back
    untouched — when the labels are in `ADDITION_COLUMNS` order. That is the
    order career-ops' batch prompt tells the worker to write, so its data row IS
    the positional row every step already reads. None for a headed file in any
    other layout: merge-tracker reads it by name, while this chain rewrites
    cells by index and would rewrite the wrong ones, so it is left as written.
    A repaired row goes back out through `_rejoin_addition`."""
    lines, fields = _addition_lines(text)
    if fields is None:
        return "", text
    in_place = all(fields.get(name) == i for i, name in enumerate(ADDITION_COLUMNS))
    if len(lines) != 2 or not in_place:
        return None
    return lines[0] + "\n", lines[1]


def _rejoin_addition(header: str, row: str) -> str:
    """The file text for a repaired `row` under the `header` that
    `_positional_addition` split off — a headed row that reaches its Report cell
    padded with empty cells to the label line's width.

    merge-tracker lets a row stop before its optional trailing cells, but it
    refuses one where a URL sits under a label other than `url` while the `url`
    cell is ABSENT — it reads that as every optional value shifted one column
    left — and a bare URL written into an empty Notes cell, which is exactly
    what the chain's URL injection does, is that shape: a row that merged as the
    worker wrote it was refused once repaired. Only the optional tail is padded,
    so a row short of a required cell stays short, and refused. And it is padded
    here, on the way out, not before the chain: `_row_parts` takes a nine-cell
    row as it stands but anchors a longer one on a `[N](path)` Report cell, so a
    padded row carrying the `—` sentinel there lost every repair."""
    if header:
        width = len(_split_addition_line(header.rstrip("\r\n")))
        have = row.count("\t") + 1
        if _REPORT_IDX < have < width:
            row += "\t" * (width - have)
    return header + row + "\n"


def _read_positional(path: Path) -> tuple[str, str] | None:
    """`_positional_addition` of an addition file on disk — None when it cannot
    be read or addressed by position, both of which its callers skip.

    Line endings only are stripped — NOT `read_text`'s strip(), which eats the
    trailing tab of an empty Notes cell. Read that way, an 8-cell row is padded
    back to 9, differs from what was read, and is rewritten and counted as
    repaired on every merge, forever."""
    try:
        text = read_addition(path).strip("\r\n")
    except OSError:
        return None
    return _positional_addition(text)


def labelled_addition(tracker_tsv: str) -> str | None:
    """The file text for a sanitized headerless row under career-ops' label
    line, padded on `_rejoin_addition`'s terms — or None, and the row is
    written as it stands, for merge-tracker to judge headerless: since #6062
    that is a refusal, which the guard names.

    Labels settle by POSITION what merge-tracker's headerless parser worked out
    by content, so only a row whose positions are known qualifies: its nine
    columns in place with the Report cell where `_row_parts` anchors a wider row
    (a nine-cell row is taken as it stands there, but a row one cell short in
    the middle and padded at the end would be read by name as the report link
    under `pdf` — a refusal turned into a silent mis-merge), and at most one
    cell after Notes, empty or a URL, since `url` is the only label there. A
    tenth cell holding a location or a `via=` tag is the case upstream's own
    refusal warns about: labelled `url`, its value would be dropped. A pipe row
    splits to one cell and is declined, like every step declines it.

    The status/score pair is the one thing a position cannot settle. The
    headerless parser asked which ONE of the two cells looked like a score —
    `_normalize_score_cell` leaves a swapped pair alone so that this still can —
    so a pair the model wrote the other way round is put into label order here,
    by the same rule; labelled as written, a row it used to merge is refused.

    The headed parser also refuses a BLANK required cell, which the headerless
    one never checked, so labels must not turn a blank cell into a refusal. A
    blank PDF gets what the prompt tells the model to write there (`_PDF_CELL`),
    and a blank score is `N/A` by the time the chain is done; any other blank
    required cell declines the row, which a checkout before #6062 then merges
    headerless as it always did."""
    parts = _row_parts(tracker_tsv)
    if parts is None or not _REPORT_CELL_RE.match(parts[_REPORT_IDX].strip()):
        return None
    extras = [cell.strip() for cell in parts[_TRACKER_TSV_COLUMNS:]]
    # Case-insensitive, as merge-tracker's own test for a URL cell is.
    if len(extras) > 1 or (extras and extras[0]
                           and not re.fullmatch(_NOTES_URL_RE.pattern, extras[0], re.I)):
        return None
    if is_score_cell(parts[_STATUS_IDX]) and not is_score_cell(parts[_SCORE_IDX]):
        parts[_STATUS_IDX], parts[_SCORE_IDX] = parts[_SCORE_IDX], parts[_STATUS_IDX]
    if not parts[_PDF_IDX].strip():
        parts[_PDF_IDX] = _PDF_CELL
    if any(not parts[i].strip() for i in _REQUIRED_IDXS):
        return None
    return _rejoin_addition(ADDITION_LABEL_LINE + "\n", "\t".join(parts))


# What a merge-tracker that reads labelled additions has and an older one does
# not: career-ops#3706's headed-addition parser, and career-ops#6062's opt-out
# flag — a checkout that refuses headerless rows reads labelled ones by
# definition. Either marker answers the question, so an upstream rename of one
# leaves the answer right.
_HEADED_PARSER_MARKER = "parseHeadedAddition"
_REFUSES_HEADERLESS_MARKER = "--allow-headerless"


def reads_labelled_additions(career_ops: Path) -> bool:
    """Whether the merge-tracker in `career_ops` reads a labelled addition
    (career-ops#3706, 2026-09-04). The writer and the merge-time sanitizer label
    a row only when it does (`_addition_text`). A merge-tracker from before
    #3706 reads the label line as a data row and refuses the file — archives
    it, exits 0 — and recovery will not re-queue it, since its score is
    readable; that checkout merges the same row headerless, so labelling for it
    would lose every evaluation it was given. Any checkout still on the retired
    fork is one: the fork's merge-base was 2026-08-25.

    A text probe for either marker, through `load_contract`, so it is cached
    until the script's mtime changes (the writer asks once per evaluation).
    `test_the_checkout_reads_labelled_additions` holds it to what the script
    does: an upstream rename of both fails that gate instead of quietly sending
    current checkouts headerless rows #6062 refuses. True when the script is
    missing or unreadable — labelled is what every current checkout and the
    cloud's pin need, and with no script nothing merges anyway."""
    return load_contract(
        "merge-tracker.mjs",
        lambda text: _HEADED_PARSER_MARKER in text or _REFUSES_HEADERLESS_MARKER in text,
        True, root=career_ops)


def _addition_text(header: str, row: str, career_ops: Path) -> str:
    """The file text for a sanitized addition `row` under `header`, written
    into `career_ops`'s queue — the one place labels are decided. A headerless
    row (`header` "") goes under the label line when the checkout reads labels
    and `labelled_addition` takes it, and out as it stands otherwise; a headed
    one keeps its own line."""
    if not header and reads_labelled_additions(career_ops):
        labelled = labelled_addition(row)
        if labelled is not None:
            return labelled
    return _rejoin_addition(header, row)


def _row_parts(tracker_tsv: str) -> list[str] | None:
    """The cells of an addition row whose nine columns are IN PLACE, else None.

    One classifier, shared by every positional sanitizer, so they cannot disagree
    about which rows to touch — which is exactly what went wrong twice. With
    `== 9` guards, career-ops' documented "9 columns plus a trailing `url`" row
    skipped the score normalization while `_strip_role_pipe`, which had no guard,
    still rewrote it: the file changed, the repair was counted and logged, and
    the row was refused for its score anyway. With `>= 9` guards, a row carrying
    an EARLY extra cell (a tab inside the company name) had `N/A` written into
    its status and a URL prepended to its report link — and merge-tracker, now
    seeing exactly one score-shaped cell, accepted the garbage where it had been
    refusing it, so the loss guard saw the row "land". A width alone cannot
    tell those two ten-cell rows apart. The Report cell can: a trailing extra
    leaves it at index 7, an inserted one shifts it.

    Nine cells are taken as-is — that is our own prompt's shape, and the chain
    has always operated on it without an anchor check. Fewer than nine is
    declined by EVERY step, `_strip_role_pipe` included, so a short row cannot
    change on disk and be reported as repaired."""
    # A whitespace-only row splits to nine EMPTY cells — `_restore_trailing_cells`
    # pads eight tabs to nine — and has nothing to sanitize; writing a URL into
    # it manufactures a half-populated row out of nothing.
    if not tracker_tsv.strip():
        return None
    parts = tracker_tsv.strip("\r\n").split("\t")
    if len(parts) < _TRACKER_TSV_COLUMNS:
        return None
    if len(parts) > _TRACKER_TSV_COLUMNS and not _REPORT_CELL_RE.match(parts[_REPORT_IDX].strip()):
        return None
    return parts


def _restore_trailing_cells(tracker_tsv: str) -> str:
    """Pad a row back to the full column count when trailing cells were empty.

    An empty Notes cell is a trailing TAB, and every strip between the model and
    here eats it — `extract_tag` strips the tag body before we ever see the row.
    The row then has 8 fields, and each sanitizer below no-ops on its 9-column
    guard: the score is left unnormalized, merge-tracker cannot tell it from the
    status, and it refuses the row (archiving it, before career-ops#5862). So
    the row that arrives one cell short is exactly the row this chain exists to
    save.

    Exactly ONE cell, deliberately. Notes is the last column, so a lost trailing
    tab can only ever cost one. A row short by more is genuinely malformed, and
    padding it would manufacture a well-formed-looking row out of garbage —
    the sanitizers' column guards are the right answer there, and stay in force."""
    parts = tracker_tsv.strip("\r\n").split("\t")
    if len(parts) == _TRACKER_TSV_COLUMNS - 1:
        parts.append("")
    return "\t".join(parts)


def _strip_role_pipe(tracker_tsv: str) -> str:
    """Remove any '| <suffix>' the LLM appends to the role column.

    The LLM sometimes writes "Software Engineer | Remote" or
    "Platform Engineer | $35/hr Remote" as the role. That pipe is harmless in
    a tab-delimited TSV but merge-tracker.mjs copies it verbatim into a
    markdown table row where it splits the cell, shifting every subsequent
    column (score ends up as status, report link ends up in notes, etc.)."""
    # rstrip("\n") not strip(): a trailing TAB is an empty Notes cell, and
    # stripping it drops the row to 8 fields — after which the two sanitizers
    # below both no-op on their 9-column guard, leaving the unnormalized score
    # merge-tracker then refuses. The row this chain exists to save was the one
    # it skipped.
    parts = _row_parts(tracker_tsv)
    if parts is None:
        return tracker_tsv
    parts[_ROLE_IDX] = re.sub(r"\s*\|.*$", "", parts[_ROLE_IDX]).strip()
    return "\t".join(parts)


# Score cells merge-tracker.mjs accepts verbatim (tracker-parse.mjs's
# `looksLikeScoreCell`), beyond the plain `N/5` / `N.N/5` form.
_SCORE_SENTINELS = SCORE_SENTINELS
# Two ways to read a score out of a cell, tried in order.
#
# An explicit "N/5" is unambiguous wherever it sits, so decoration around it is
# harmless: "(4.5/5)", "~4.5/5", "Score: 4.5/5" all mean 4.5. Tracker rows in the
# wild carry all of these, and they are what the READ side sorts by.
#
# Failing that, the cell must OPEN with the number. Searching anywhere turns
# prose into a score, and the invented value is plausible rather than obviously
# wrong — "Top 5%" became 5/5, "not scored (4 blockers)" became 4/5. A fabricated
# top score is the exact harm the out-of-range branch below refuses to cause,
# since the handoff work-order ranks by score descending. Leading punctuation is
# allowed; a leading LETTER is not, because that is what prose looks like.
_SCORE_OVER_FIVE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*5\b")
_SCORE_LEAD_RE = re.compile(r"^[^0-9A-Za-z]*(\d+(?:[.,]\d+)?)")


def score_value(cell: str) -> float | None:
    """The numeric score in a cell, or None. One reader for both ends: the write
    side normalizes what the model emitted, the read side sorts the tracker by
    it, and they used to disagree — "4,2/5" parsed as 4.0 where the writer meant
    4.2."""
    text = (cell or "").replace("*", "")
    m = _SCORE_OVER_FIVE_RE.search(text) or _SCORE_LEAD_RE.match(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _normalize_score_cell(tracker_tsv: str) -> str:
    """Rewrite the score column (col 6) to the exact `N.N/5` shape
    merge-tracker.mjs requires, or to the `N/A` sentinel when it holds no number.

    merge-tracker resolves which of columns 5-6 is the status and which is the
    score by asking whether exactly one of them *looks like* a score cell
    (`tracker-parse.mjs:resolveScoreStatus` — `/^\\d+(?:\\.\\d+)?\\/5$/`, plus the
    N/A / DUP / dash sentinels). When neither matches it refuses the row rather
    than risk merging a column swap — and a labelled row whose `score` cell is
    not one is refused the same way. That refusal is counted as `skipped`, not
    as a failure: merge-tracker exits 0. Before career-ops#5862 it also moved
    the TSV into tracker-additions/merged/, so a model that writes "4.2"
    instead of "4.2/5" lost the evaluation outright — the report on disk, the
    row never in applications.md, and nothing retrying it. Since then the TSV
    is kept for a re-run, which refuses it again until something repairs it.

    The older merge-tracker had a heuristic whose final fallback assumed
    "status then score" — this row order — so any score string merged and the
    prompt's "format X.X/5" rule was advisory. It is load-bearing now, and a
    prompt rule is not something a local Ollama/Groq/DeepSeek model reliably
    honours. This is the one place we own the value, so normalize it here
    rather than depend on model compliance."""
    parts = _row_parts(tracker_tsv)
    if parts is None:
        return tracker_tsv
    raw = parts[_SCORE_IDX].replace("*", "").strip()
    if raw.upper() in _SCORE_SENTINELS:
        parts[_SCORE_IDX] = raw.upper()
        return "\t".join(parts)
    # The model may have written the pair the other way round (status in col 6,
    # score in col 5). Leave a swapped pair alone. A headerless row is labelled
    # after this chain, and `labelled_addition` puts the pair into label order
    # by asking which ONE of the two looks like a score — writing N/A here would
    # make BOTH look like one, so the swap would not happen and the row would
    # go out labelled with its score under `status`. A headed row is read by
    # name, where merge-tracker refuses a swapped pair on purpose, and N/A
    # would not move the score out of `status` either.
    if is_score_cell(parts[_STATUS_IDX]):
        return tracker_tsv
    value = score_value(raw)
    if value is None:
        # No number at all (empty, "unknown", a stray status). N/A is a shape
        # merge-tracker recognises, so the row still merges and the evaluation
        # stays visible — a scoreless row beats a silently discarded one.
        parts[_SCORE_IDX] = "N/A"
        return "\t".join(parts)
    # Out of range means the model answered on some other scale ("8/10", "84%")
    # and we cannot know which. Clamping to 5 would invent a perfect score, and
    # the handoff work-order ranks by score descending — a fabricated 5 puts the
    # role first. N/A merges the row and leaves it unranked, which is the true
    # statement.
    parts[_SCORE_IDX] = f"{value:g}/5" if 0 <= value <= 5 else "N/A"
    return "\t".join(parts)


# Any http(s) URL, and the `req <id>` we write — the two things a notes cell is
# checked for before something is prepended to it.
_NOTES_URL_RE = re.compile(r"https?://\S+")
_NOTES_REQ_ID_RE = re.compile(r"\breq[\s:#_-]*([A-Za-z0-9_-]+)", re.I)


def _prepend_to_notes(tracker_tsv: str, text: str, already_present) -> str:
    """Put `text` at the front of the notes cell, unless it is already there.

    The notes cell has grown a grammar — `req <id> — <url> — <the model's
    sentence>` — read back by merge-tracker's `extractReqNumber`, by the UI's
    "Open posting" link and by bridge's URL dedup. One composer, so the column
    guard, the ` — ` separator and the front-of-cell rule are stated once and a
    third injector cannot invent a fourth shape.

    `already_present(notes)` is the only real difference between the callers:
    "any URL at all" defers to a model that supplied its own, while "this exact
    id" is idempotence. Both are one predicate slot, not two mechanisms.

    A row `_row_parts` declines is returned untouched — better to lose the
    annotation than to write it into the wrong cell."""
    if not text:
        return tracker_tsv
    parts = _row_parts(tracker_tsv)
    if parts is None:
        return tracker_tsv
    notes = parts[_NOTES_IDX].strip()
    composed = compose_notes(notes, text, already_present)
    if composed == notes:
        return tracker_tsv
    parts[_NOTES_IDX] = composed
    return "\t".join(parts)


def compose_notes(notes: str, text: str, already_present) -> str:
    """`_prepend_to_notes`' rule for a Notes cell held on its own: `text` in
    front, joined by ` — `, unless `already_present(notes)`. The UI's
    tracker-additions fallback composes a headed row's `url` column in with it,
    so the cell it shows has the shape merge-time sanitizing gives that row."""
    notes = notes.strip()
    if not text or already_present(notes):
        return notes
    return f"{text} — {notes}" if notes else text


def has_notes_url(notes: str) -> bool:
    """The `already_present` test for a posting URL: any URL at all, which
    defers to one a model wrote into the cell itself."""
    return bool(_NOTES_URL_RE.search(notes))


def _inject_work_location_into_notes(tracker_tsv: str, mark: str) -> str:
    """Record WHERE the role is in the notes cell (#180).

    The fact, not the verdict: every surface that ranks roles then applies its
    own commutable area to it, so a candidate who widens their search gets the
    new answer on old rows instead of a frozen one nothing refreshes. Written
    once, where the evaluation is written, because the alternative is three
    consumers each opening the report — including the UI, on a path that runs
    per request and has deliberately never read reports.

    Idempotent on the LABEL, not the value: a row already carrying a mark keeps
    the one it has rather than growing a second, and a re-evaluation that moved
    the role writes through `sanitize_addition`'s fresh row, not over this one."""
    return _prepend_to_notes(tracker_tsv, mark,
                             lambda notes: WORK_LOCATION_RE.search(notes) is not None)


def _inject_url_into_notes(tracker_tsv: str, url: str) -> str:
    """Splice the job URL into the notes cell so the UI's "Open posting" link
    works.

    The LLM generates the notes cell freely — per our prompt it's a
    one-sentence summary, not the URL. The UI's report pane resolves the
    "Open posting" target by regex-matching the first http(s) URL in the notes
    cell ([app.js:extractUrl]), so without this the link has nothing to point
    at and renders as `#`. Splicing the URL here keeps the prompt unchanged
    and works for every provider / model.

    If the LLM already put a URL in notes, leave it alone."""
    return _prepend_to_notes(tracker_tsv, url, has_notes_url)


# Markdown decoration, normalized before matching rather than matched around.
# JobSpy's `description_format` defaults to MARKDOWN and `scrape.py` forwards it
# untouched, so an Indeed JD reaches the cache as `**Job ID:** 88214`, and
# markdownify escapes `_` and `-` mid-word (`R\_1488728`, `JR\-00124259`). All
# three defeat a pattern written for plain prose. LinkedIn's descriptions are
# backfilled as plain text by screen.py, so this hit one board and not the other
# — which is worse than hitting both, because the extractor then looks like it
# works.
#
# The two characters get DIFFERENT treatment, and the difference was measured
# rather than guessed. A backslash is an escape INSIDE an id, so it is deleted:
# spacing it out would turn `R\_1488728` into `R _1488728` and capture a bare
# `R`. An asterisk is emphasis AROUND words, so it becomes a space: deleting it
# glued `Remote**Requisition Number` into `RemoteRequisition`, and the label's
# leading `\b` no longer held — two real postings whose requisition number sat
# right there, three newlines below the label, read as having none.
_MARKDOWN_ESCAPE_RE = re.compile(r"\\")
_MARKDOWN_EMPHASIS_RE = re.compile(r"\*")


def _demarkdown(text: str) -> str:
    return _MARKDOWN_EMPHASIS_RE.sub(" ", _MARKDOWN_ESCAPE_RE.sub("", text))


# A requisition id as JOB-DESCRIPTION PROSE states it. Deliberately narrower
# than merge-tracker's own REQ_NUMBER_RE, which reads the tracker's Notes cell:
# that regex accepts a BARE `job`/`req`/`ref` label because a human wrote that
# cell and meant the number they put there. A JD is thousands of words we did not
# write, where "job 2 of 5" and "reference 401k" are ordinary prose — so here the
# label must carry an explicit qualifier. Recall costs nothing (no id is exactly
# today's behaviour); a wrong id SPLITS two rows that should fold.
#
# `#` is a qualifier in its own right ("Req #1311") and must NOT be followed by
# `\b`: both it and the space after it are non-word characters, so the boundary
# could only hold when a digit followed immediately — `Req # 1311` and
# `Position #: 12345` matched nothing. Upstream sidesteps this by keeping `#` in
# its separator class; here the qualifier is mandatory, so it needs both roles.
#
# `code` is deliberately absent from the qualifiers: in public-sector and
# healthcare postings — the shape this exists for — "Job Code: 4021" is a pay
# grade shared by every posting of that title, so reading it as a requisition
# would split a cross-board pair rather than two genuine reqs.
_JD_REQ_ID_RE = re.compile(
    r"\b(?:job|requisition|req|posting|position|vacancy|reference|ref)\s*"
    r"(?:(?:id|no|nr|num|number)\b|#)[\s:#=.-]*"
    r"([A-Za-z0-9][A-Za-z0-9_-]*)", re.I)

# The capture is unbounded and the length judged AFTER, because a bounded
# `{2,23}` does not decline an over-long token — it truncates one, and a
# truncation is more dangerous than a miss. `Job ID: 5340-Nurse-Practitioner-
# Days-FT` yielded `5340-NURSE-PRACTITIONER-`, and `REQ-2026-3206100…` yielded
# `REQ-2026-`, which merge-tracker reads back as `REQ-2026` — a DIFFERENT string
# from the one we wrote, and one that two genuinely different reqs sharing a
# prefix would both produce. Over-long means "we cannot tell where the id ends",
# which is a decline.
_REQ_ID_MAX_LEN = 24
_REQ_ID_MIN_LEN = 3


def extract_req_id(jd_text: str) -> str:
    """The requisition id a JD states, uppercased — or "" when it states none,
    or more than one.

    Why this is worth extracting: `company::role` is the identity the pipeline
    runs on, and merge-tracker's fuzzy tier drops tokens of three characters or
    fewer before comparing titles, so `INSURANCE SPECIALIST I` and `… II` (and
    `PACT PRN Representative I` vs `PACT Representative I`) match as one opening
    and fold into a single row — the level and the PRN are invisible to it. A req
    id in the Notes column is the signal career-ops documents for exactly this:
    merge-tracker reads it and treats two rows carrying DIFFERENT ids as distinct
    openings, overriding the title match.

    Ambiguity resolves to "": several DIFFERENT ids in one JD means we cannot
    tell which is the requisition, and guessing splits a row that should fold.
    The same id repeated (JD body and footer) is not ambiguity, so the values are
    compared, not counted. That doubles as the guard for the UI's Add-Job path,
    where `screen.extract_description` can fall back to a whole page body: a
    related-jobs panel usually carries several ids, and several is a decline. A
    page carrying exactly one foreign id is the residual risk there.

    Deliberately not the board's own job key (`jk=`, `currentJobId`): that is a
    per-POSTING identifier, so a re-post of one requisition carries a new one, and
    keying on it would split every re-post into a second tracker row. The
    employer's req id is stable across re-posts and differs between levels, which
    is the distinction being drawn here."""
    text = _demarkdown(jd_text or "")
    found = set()
    for m in _JD_REQ_ID_RE.finditer(text):
        # rstrip: a trailing separator is never part of the id, and leaving one
        # on changes what merge-tracker reads back out of what we write.
        value = m.group(1).rstrip("-_")
        if not _REQ_ID_MIN_LEN <= len(value) <= _REQ_ID_MAX_LEN:
            continue
        if any(c.isdigit() for c in value):
            found.add(value.upper())
    return found.pop() if len(found) == 1 else ""


def _inject_req_id_into_notes(tracker_tsv: str, req_id: str) -> str:
    """Prepend `req <id>` to the notes cell so merge-tracker can read it.

    FIRST in the cell, ahead of the URL, because merge-tracker's
    `extractReqNumber` takes the first match in the cell: leading with the id we
    resolved makes the extraction deterministic instead of dependent on whatever
    the LLM's free-text sentence happens to contain. (Today's posting URLs yield
    no match at all — pinned by a test — but that is a property of current URL
    shapes, not a guarantee about the next board.)

    `req <id>` because that is a form merge-tracker's own regex reads back
    unchanged; it uppercases both sides, so case never matters here either."""
    return _prepend_to_notes(
        tracker_tsv, f"req {req_id}" if req_id else "",
        lambda notes: any(m.group(1).upper() == req_id.upper()
                          for m in _NOTES_REQ_ID_RE.finditer(notes)))


def _set_report_link(tracker_tsv: str, report_file: str) -> str:
    """Point the Report cell at `report_file`, the report that actually exists.

    The cell is `[N](reports/N-slug-DATE.md)` where N is the file's own leading
    number, so the two agree by construction. `report_file` is a bare filename
    under `reports/`; empty means "nothing known" and leaves the cell alone —
    a row whose report was never written (a `failed` evaluation) keeps whatever
    the model put there, which is no worse than before. Declined on a row whose
    columns are not in place, like every positional step (`_row_parts`)."""
    report_file = (report_file or "").strip()
    num = _report_num_prefix(report_file)
    if not num:
        return tracker_tsv
    parts = _row_parts(tracker_tsv)
    if parts is None:
        return tracker_tsv
    cell = f"[{num}](reports/{report_file})"
    if parts[_REPORT_IDX].strip() == cell:
        return tracker_tsv
    parts[_REPORT_IDX] = cell
    return "\t".join(parts)


def _report_location_mark(report_text: str) -> str:
    """The `Work location:` mark for a report's own `work_location` block, or ""
    when it has none — the one place a report is turned into that mark, so the
    writer and the merge-time repair cannot produce two shapes."""
    return location_mark(parse_work_location(parse_machine_summary(report_text)))


class IncompleteResponse(RuntimeError):
    """A provider answered, but not with a whole evaluation — the JOB's failure,
    not the run's — so it counts against the job's allowance of failed runs
    (`batch_evaluate._is_countable_failure`, #206). Raised by the Claude caller
    for a cut-off or refused answer; #204 adds the other callers and the
    tag-completeness check."""


def write_job_result(
    response_text: str,
    job_meta: dict,
    reports_dir: Path,
    tracker_dir: Path,
    today: str,
) -> dict:
    """Parse XML-tagged response, write report .md and tracker .tsv, return summary dict."""
    report_content = extract_tag(response_text, "report")
    tracker_tsv = extract_tag(response_text, "tracker_tsv")
    summary = parse_json_loose(extract_tag(response_text, "summary")) or {}

    job_id = job_meta["id"]
    report_num = job_meta.get("report_num") or summary.get("report_num", "000")
    company = summary.get("company") or job_meta.get("company") or "unknown"
    company_slug = _company_slug(company)
    report_name = f"{report_num}-{company_slug}-{today}.md"
    report_file = report_name if report_content else None

    if report_content:
        (reports_dir / report_name).write_text(report_content, encoding="utf-8")
    if tracker_tsv:
        # The Report cell names the file THIS function wrote, not the one the
        # model guessed at (#162): the prompt gives it only the link's shape.
        tracker_tsv = sanitize_addition(tracker_tsv, job_meta.get("url", ""),
                                        job_meta.get("jd_text", ""),
                                        report_file=report_file or "",
                                        work_location=_report_location_mark(report_content))
        # Every caller writes into <career-ops>/batch/tracker-additions, so the
        # checkout is two up. Atomic, because a merge's sanitizer in another
        # thread or process reads and rewrites this directory: one that read a
        # half-written row would "repair" it and replace the whole one.
        atomic_write_text(tracker_dir / f"{job_id}.tsv",
                          _addition_text("", tracker_tsv, tracker_dir.parent.parent))

    return {
        "report_file": report_file,
        "tracker_file": f"{job_id}.tsv" if tracker_tsv else None,
        "summary": summary,
    }


def build_user_message(job_meta: dict, today: str) -> str:
    return (
        f"Evaluate this job posting.\n\n"
        f"**Job ID:** {job_meta['id']}\n"
        f"**Report Number:** {job_meta.get('report_num', '000')}\n"
        f"**Tracker Number:** {job_meta.get('tracker_num', 0)}\n"
        f"**Evaluation Date:** {today}\n"
        f"**URL:** {job_meta.get('url', '')}\n"
        f"**Company (source field):** {job_meta.get('company') or 'unknown'}\n"
        f"**Role (source field):** {job_meta.get('role') or 'unknown'}\n\n"
        f"**Job Description:**\n{job_meta.get('jd_text') or '(no JD cached — infer from the URL and source fields)'}"
    )


def jd_cache_path(career_ops: Path, job_id: str) -> Path:
    """Where batch_prep caches a job's description, keyed on its queue id — the
    same id that names its tracker addition, which is what lets the merge-time
    sanitizer find the JD for a row it did not write."""
    return career_ops / "batch" / "jds" / f"{job_id}.txt"


def assign_job_numbers(
    pending: list[dict],
    state: dict,
    report_start: int,
    tracker_start: int,
    career_ops: Path,
    load_jd_text: bool = True,
) -> list[dict]:
    """Pre-assign sequential report and tracker numbers to pending jobs.

    Pre-assigns sequential report and tracker numbers to pending jobs so
    parallel workers don't conflict on numbering. Mutates `state["jobs"]` to
    record each job's metadata (excluding jd_text to keep state small).

    The entry is rebuilt from scratch each run, so a job's count of failed
    runs is carried over from the previous entry, or the allowance it enforces
    (#206) would never be reached."""
    report_counter = report_start
    tracker_counter = tracker_start
    if "jobs" not in state:
        state["jobs"] = {}

    jobs: list[dict] = []
    for row in pending:
        jid = str(row["id"]).strip()
        report_counter += 1
        tracker_counter += 1
        meta = {
            "id": jid,
            "url": (row.get("url") or "").strip(),
            "company": (row.get("source") or "").strip(),
            "role": (row.get("notes") or "").strip(),
            "report_num": f"{report_counter:03d}",
            "tracker_num": tracker_counter,
            "status": "pending",
        }
        if load_jd_text:
            meta["jd_text"] = read_text(jd_cache_path(career_ops, jid))
        jobs.append(meta)

        state_entry = {k: v for k, v in meta.items() if k != "jd_text"}
        prior = state["jobs"].get(jid, {})
        if "attempts" in prior:
            state_entry["attempts"] = prior["attempts"]
        state["jobs"][jid] = state_entry

    return jobs


def load_pending(
    batch_input: Path,
    state: dict,
    done_statuses: frozenset = frozenset({"completed", "abandoned"}),
) -> list[dict]:
    """Return rows from batch-input.tsv whose IDs are not in done_statuses.

    `abandoned` is done by default: a job that failed the countable way on
    `batch_evaluate.MAX_ATTEMPTS` runs stays out of the queue until
    `--retry-abandoned` puts it back (#206)."""
    already_done = {
        jid for jid, job in state.get("jobs", {}).items()
        if job.get("status") in done_statuses
    }
    return [row for row in read_batch_input(batch_input)
            if (jid := str(row.get("id", "")).strip()) and jid not in already_done]


# Canonical applications.md header (matches career-ops onboarding). Seeded
# before merge so merge-tracker has a file to merge INTO.
APPLICATIONS_HEADER = (
    "# Applications Tracker\n\n"
    "| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n"
    "|---|------|---------|------|-------|--------|-----|--------|-------|\n"
)


def ensure_applications_md(career_ops: Path) -> Path:
    """Make sure career-ops/data/applications.md exists with the canonical
    header, returning its path.

    merge-tracker.mjs only merges INTO an existing applications.md — if none
    exists it prints "No applications.md found. Nothing to merge into." and
    exits 0 (a false success). Our pipeline never runs career-ops's
    interactive onboarding that would otherwise seed the file, so on a fresh
    run (especially in CI, where nothing's cached yet) the merge silently
    no-ops and applications.md never gets created. Seeding the header here
    fixes that for both local and cloud runs. The `data/` path matches the
    layout merge-tracker prefers and the workflow uploads."""
    apps_md = career_ops / "data" / "applications.md"
    if not apps_md.exists():
        apps_md.parent.mkdir(parents=True, exist_ok=True)
        apps_md.write_text(APPLICATIONS_HEADER, encoding="utf-8")
        print(f"[batch] seeded data/applications.md (was missing)")
    return apps_md


def sanitize_addition(tracker_tsv: str, url: str = "", jd_text: str = "",
                      report_file: str = "", work_location: str = "") -> str:
    """Turn a model's tracker row into one merge-tracker can read, and read
    correctly. The whole chain, in the one order that works.

    Each step is documented at its own definition; the ORDER is the part that
    only makes sense here. Trailing cells are restored first because every guard
    below counts columns; the role's pipe suffix goes before anything reads the
    role; the score is normalized before merge-tracker has to tell it from the
    status; and the req id is prepended AFTER the URL so it ends up ahead of it
    in the cell, where `extractReqNumber`'s first-match rule finds it. The
    Report link is set from `report_file` — the report that exists on disk,
    which the writer knows and the merge-time pass looks up by number — because
    the model authors that cell from a shape alone and invents the slug (#162).
    The `work_location` mark goes in BEFORE the URL and the req id so it lands
    after them in the cell: those two are read by machines that take the first
    match (merge-tracker's `extractReqNumber`, the UI's "Open posting"), and the
    mark is read by a regex that does not care where it sits.

    **Idempotent**, which is what lets it run at two points without a second
    mechanism: padding a nine-column row does nothing, the pipe strip has nothing
    left to strip, a normalized score re-normalizes to itself, a link already
    pointing at the file is rewritten to itself, and all three injectors bail
    when their content is already present."""
    row = _restore_trailing_cells(tracker_tsv)
    row = _strip_role_pipe(row)
    row = _normalize_score_cell(row)
    row = _set_report_link(row, report_file)
    row = _inject_work_location_into_notes(row, work_location)
    # The URL the pipeline QUEUED first, the row's own trailing field second.
    # Everything downstream routes by the notes URL — handoff picks the site
    # session from it — so it should be the board URL the run searched from, not
    # a careers-page URL an agent resolved and wrote as the 10th field. The
    # trailing field is what a `{num}-{slug}.tsv` row has when nothing else does.
    row = _inject_url_into_notes(row, url or _trailing_url(row))
    return _inject_req_id_into_notes(row, extract_req_id(jd_text))


def read_batch_input(batch_input: Path) -> list[dict]:
    """Every row of batch-input.tsv, as dicts. Missing file → [].

    One reader because the FORMAT is one fact — tab-delimited, header row, utf-8
    — even though the three callers ask different questions of it (rows not yet
    done; max id and seen urls; id → url). It deliberately RAISES on an
    unreadable file rather than choosing a posture for them: `batch_prep`
    swallowing an IO error would restart ids at 1 and re-queue everything, while
    the merge swallowing one costs a URL lookup and nothing else."""
    if not batch_input.exists():
        return []
    # utf-8-sig, as rowio reads: an Excel round-trip leaves a BOM that renames the
    # first header to `\ufeffid`, after which every `row.get("id")` is None —
    # batch_prep restarts ids at 1 and the merge finds no URLs, both in silence.
    with open(batch_input, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _batch_input_urls(batch_input: Path) -> dict[str, str]:
    """{job id: posting URL} from batch-input.tsv — the URL for a row that was
    queued by this pipeline. An unreadable queue costs the URL repair only, so
    it degrades to {} rather than taking the merge down with it. All three ways
    the read can fail, not just OSError: a hand-edited file saved as cp1252
    raises UnicodeDecodeError, a NUL byte raises csv.Error, and the UI's Add-Job
    handler above this has already written the report and the TSV."""
    try:
        rows = read_batch_input(batch_input)
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    urls: dict[str, str] = {}
    for row in rows:
        jid, url = (row.get("id") or "").strip(), (row.get("url") or "").strip()
        if jid and url:
            urls[jid] = url
    return urls


def _trailing_url(tracker_tsv: str) -> str:
    """The posting URL a row carries in its own optional trailing fields.

    career-ops' batch worker writes a `url` field after the nine (a labelled
    column since career-ops#3706, the "optional trailing `url`" before it), so a
    foreign row often carries its URL with it. Detected by
    shape as merge-tracker's `parseTsvExtras` does, so it stays order-independent
    with the optional location and `via=` extras — and with the SAME pattern the
    notes injector uses to ask "already present", or a `HTTPS://` cell would be
    detected here, prepended, and not recognised on the next pass: the chain
    grew a URL per merge."""
    parts = _row_parts(tracker_tsv)
    for cell in (parts or [])[_TRACKER_TSV_COLUMNS:]:
        if _NOTES_URL_RE.match(cell.strip()):
            return cell.strip()
    return ""


def _row_report(career_ops: Path, tracker_tsv: str) -> Path | None:
    """The report file a row's Report link names, resolved the way every reader
    resolves it — the link when it exists, else the report carrying its number
    and company (`resolve_report`, for #162's model-invented slugs).

    Called ONCE per addition by `_sanitize_pending_additions` and handed to both
    merge-time passes that need it (`_dead_link_repair`, `_pending_location_mark`).
    That is the point of extracting it rather than of each pass resolving its own:
    a dead link — the case both passes exist for — sends `resolve_report` through
    a full `reports/` glob, so two independent callers meant two scans of a
    several-hundred-file directory per addition. Reads through the same
    trailing-cell restore the rest of the chain applies, or the one row that lost
    its empty Notes tab resolves differently here than everywhere else."""
    parts = _row_parts(_restore_trailing_cells(tracker_tsv))
    if parts is None:
        return None
    num_text, path = _report_link(parts[_REPORT_IDX])
    return resolve_report(career_ops, path, num_text=num_text, company=parts[_COMPANY_IDX])


def _dead_link_repair(career_ops: Path, tracker_tsv: str, found: Path | None) -> str:
    """The filename to point a row's Report link at, or "" to leave it alone:
    only a link whose target does NOT exist is repaired, and only when exactly
    one report answers to its number and company — resolved exactly as the
    readers resolve it (`resolve_report`). Reads the row through the same
    trailing-cell restore the chain applies, or the one row that lost its
    empty Notes tab (the row `_restore_trailing_cells` exists to save) would
    get every repair but this one, and merge with the dead link for good."""
    parts = _row_parts(_restore_trailing_cells(tracker_tsv))
    if parts is None:
        return ""
    _, path = _report_link(parts[_REPORT_IDX])
    if found is None or found == _linked_report(career_ops, path):
        return ""
    return found.name


def _pending_location_mark(tracker_tsv: str, found: Path | None) -> str:
    """The `Work location:` mark for an addition written outside Python, read
    off the report its own row links to.

    This is what gets a `--batch` row marked at all: career-ops' `batch-runner.sh`
    lets the agent CLI write the TSV directly, so the row reaches the merge with
    none of the chain applied — and the report it names is on disk, written by
    the same run. `found` is that report, resolved ONCE by the caller (`_row_report`)
    and shared with `_dead_link_repair`, so a model-invented slug (#162) still
    finds its report by number without a second directory scan."""
    parts = _row_parts(_restore_trailing_cells(tracker_tsv))
    if parts is None:
        return ""
    # Cheapest guard first: a row `write_job_result` wrote already carries its
    # mark, and that is most pending additions on every run that is not
    # `--batch`. `_inject_work_location_into_notes` would discard whatever we
    # computed anyway, so reading and YAML-parsing an 8KB report to reach that
    # conclusion is ~1ms per addition spent to learn nothing — 160-190 of them
    # on a cloud daily.
    if WORK_LOCATION_RE.search(parts[_NOTES_IDX]):
        return ""
    return _report_location_mark(read_text(found)) if found else ""


def _sanitize_pending_additions(career_ops: Path, tracker_dir: Path) -> None:
    """Apply `sanitize_addition` to every un-merged addition on disk, whoever
    wrote it, and say so when something was repaired.

    `write_job_result` covers the rows this pipeline's Python produces, but it is
    not the only writer: career-ops' own evaluators (`batch-runner.sh` on the
    `--batch` path, `gemini-eval.mjs`, the web runner) write straight into
    `tracker-additions/`, never entering Python. Those rows reach a merge with
    none of the chain applied — no score normalization, so an unreadable score
    cell gets the row REFUSED (archived and never retried before
    career-ops#5862, refused again on every run since); no URL in notes, so the
    UI's "Open posting" link renders as `#`; and no req id, so two levels of
    one title fold.

    **This covers the merges WE run** — `--evaluate-batch`, the UI, and the heal
    sweep that picks up whatever is still pending. It does NOT cover a `--batch`
    run end to end: `batch-runner.sh` calls `node merge-tracker.mjs` itself as its
    last step, with no Python in the process tree, so those rows are merged before
    this can see them. Recovering those means going after `tracker-additions/
    merged/` once the runner has finished; see #156.

    Recovery of what a foreign row is missing prefers `batch-input.tsv` keyed on
    the addition's filename — the URL the pipeline queued, which is what handoff
    routes by — and falls back to the row's OWN `url` field (career-ops'
    documented 10th). The filename is a job id for our writers and for
    career-ops' batch prompt, but not for its other writers, whose
    `{num}-{slug}.tsv` names find nothing in the queue and keep the score repair.

    A dead Report link — the model's slug, not the writer's (#162) — is pointed
    at the report in `reports/` that carries its number, via the same lookup the
    UI resolves by. A link that resolves is left as written, whichever of the
    two relative shapes it has: it is the one cell that knows which of two
    same-numbered files the row meant.

    A headed addition (career-ops#3706) is sanitized through its data row, its
    label line written back, on `_positional_addition`'s terms. A HEADERLESS row
    goes back under the label line where the writer would put one
    (`_addition_text`), because career-ops#6062 refuses one without labels:
    what an older version of this repo wrote, or Add-Job before the writer
    labelled its rows, would otherwise sit refused in the queue on a checkout
    that has pulled it.

    A pipe-delimited row is left alone: every step here splits on tabs and
    no-ops on it, and turning it into a tab row would be a guess at its cells.
    merge-tracker before #6062 parses that shape natively; since, it refuses it
    as headerless and keeps it, which the loss guard names
    (`_warn_on_lost_additions`)."""
    urls = _batch_input_urls(career_ops / "batch" / "batch-input.tsv")
    changed = 0
    for f in sorted(tracker_dir.glob("*.tsv")):
        # None: unreadable (one such row must not hold up the other nine), or
        # headed in a layout the chain cannot address by index.
        split = _read_positional(f)
        if split is None:
            continue
        header, row = split
        # One resolution per addition, shared by the two passes that need it.
        report = _row_report(career_ops, row)
        fixed = sanitize_addition(row, urls.get(f.stem, ""),
                                  read_text(jd_cache_path(career_ops, f.stem)),
                                  report_file=_dead_link_repair(career_ops, row, report),
                                  work_location=_pending_location_mark(row, report))
        text = _addition_text(header, fixed, career_ops)
        # Nothing repaired and no label line gained. The ROW, not the file read:
        # rejoining drops a blank line between a header and its row, which
        # would count an unchanged row as repaired.
        if fixed == row and text == _rejoin_addition(header, row):
            continue
        try:
            atomic_write_text(f, text)
        except OSError:
            continue        # a row we cannot repair still merges as it stands
        changed += 1
    if changed:
        print(f"[batch] sanitized {changed} addition(s) pending a merge "
              "(score/URL/req-id/report link/work location/label line)")


# One Millhand merge at a time (`_merge_lock`). Both locks, as
# `data.override_file_lock` holds them: acquire_process_lock treats its own pid
# as reclaimable, so two threads of one process would each "take" the file.
# MAX_AGE is merge-tracker's own stale default for its tracker lock, far above a
# real merge, so only a hung holder is reclaimed; WAIT is comfortably above its
# 60s tracker-lock timeout, so a merge queued behind one that is itself waiting
# on that lock still gets its turn.
_MERGE_LOCK_MAX_AGE = 600.0
_MERGE_LOCK_WAIT = 150.0
_MERGE_LOCK_POLL = 0.1
# Consecutive OSErrors from the lock file before giving up on it: enough to
# ride out a Windows sharing violation on a file mid-delete, not enough to make
# every merge wait out the bound when the file simply cannot be written.
_MERGE_LOCK_MAX_ERRORS = 10
_merge_thread_lock = threading.Lock()


@contextlib.contextmanager
def _merge_lock(career_ops: Path):
    """Hold the thread lock and `career_ops`'s cross-process merge lock —
    `batch/.merge-lock`, beside the batch-eval lock and never inside
    tracker-additions/, where it would be listed as an addition. Past
    `_MERGE_LOCK_WAIT` the merge goes ahead without it, with one warning: every
    caller is best-effort, and refusing would lose a merge. An OSError from the
    lock file is a miss rather than a raise, for the same reason — and a run of
    them means the file cannot be written, so the merge goes ahead then rather
    than polling out the bound."""
    lock = career_ops / "batch" / ".merge-lock"
    deadline = time.monotonic() + _MERGE_LOCK_WAIT
    threaded = _merge_thread_lock.acquire(timeout=_MERGE_LOCK_WAIT)
    held = False
    errors = 0
    # Never the file without the thread lock: it would "reclaim" the file from
    # the thread of this process that holds both.
    while threaded:
        try:
            held = acquire_process_lock(lock, max_age=_MERGE_LOCK_MAX_AGE)
            errors = 0
        except OSError:
            held, errors = False, errors + 1
        if held or errors >= _MERGE_LOCK_MAX_ERRORS or time.monotonic() >= deadline:
            break
        time.sleep(_MERGE_LOCK_POLL)
    if not held:
        print(f"[batch] could not take {lock.name} within {_MERGE_LOCK_WAIT:.0f}s "
              "(another merge holds it, or it cannot be written) — merging without it")
    try:
        yield
    finally:
        if held:
            release_process_lock(lock)
        if threaded:
            _merge_thread_lock.release()


# The line orchestrate ends a run with when its tracker merge was not clean,
# just before it exits 1. Here rather than in orchestrate because the UI's
# Run-local (`app.local_run`) reads it back to tell that exit 1 from a crash,
# and the UI venv cannot import orchestrate.
MERGE_NOT_CLEAN_LINE = "[orchestrate] the tracker merge was not clean"


def run_merge_tracker(career_ops: Path) -> bool:
    """Merge every pending addition into applications.md. True only for a
    CLEAN merge: merge-tracker exited 0 and every addition pending before it
    reached the tracker.

    A non-zero exit is not "nothing happened" any more. merge-tracker exits 1
    when it could not apply one addition (its `failedAdditions`), after it has
    written the tracker and archived the others — so treating every non-zero
    exit as total failure skipped the loss guard and the #163 reopen on rows
    that had just merged. The evidence of a merge is an exit of 0 or an
    addition that left the queue — merge-tracker archives each one it applies,
    whatever it exits — and either gets the guard and the reopen, as a partial
    merge when the exit was not 0. A non-zero exit with neither is a failure,
    which is where the missing-npm-deps hint belongs. Not a changed tracker:
    another writer can rewrite applications.md while this runs.

    Nor is a file that left the queue evidence while another of our merges
    runs, so one runs at a time (`_merge_lock`), from recovery to the reopen.
    Add-Job merges from background threads, and the UI's merge can overlap the
    pipeline's in another process; merge-tracker serializes its own writes on
    its tracker lock, but not the listings around them. A merge that timed out
    on that lock (exit 1, nothing applied) while another archived what it
    applied read as a partial merge, and reported every valid addition still
    queued as refused and KEPT, with advice to delete it.

    False for a partial merge too, and for an exit-0 merge that refused
    something — kept in the queue since career-ops#5862, archived before it,
    readable or not (see `_warn_on_lost_additions`) — because a caller reading
    True as "the evaluations are in the tracker" would be wrong about the ones
    the guard just named. `merge_additions` turns that into its exit code,
    Add-Job's synchronous endpoint returns it as `merged`, and `--evaluate-batch`
    hands it to orchestrate, which runs every later stage and then exits 1
    (`MERGE_NOT_CLEAN_LINE`). So a kept refusal makes every merge False while it
    stays queued, and `run.sh`/`run.ps1 --batch` and every `--evaluate-batch`
    that merges end in failure until its file is fixed or deleted — on purpose:
    an evaluation that is not in the tracker should not end green. The cloud
    daily's evaluate step fails on it too, once, since the queue is not cached
    there. A merge that could not run at all (no `merge-tracker.mjs`, no
    `node`) is False on the same terms, with a line saying so; an empty queue
    merges cleanly and is True."""
    if not (career_ops / "merge-tracker.mjs").exists():
        print(f"[batch] no merge-tracker.mjs in {career_ops} — nothing was merged; "
              "the additions stay in batch/tracker-additions/")
        return False
    with _merge_lock(career_ops):
        # Seed the tracker header first — otherwise merge-tracker no-ops on a
        # fresh run (see ensure_applications_md).
        ensure_applications_md(career_ops)
        tracker_dir = career_ops / "batch" / "tracker-additions"
        # Recovery first, so a restored row is sanitized and snapshotted with
        # the rest; sanitizing before the snapshot, since it can change the role
        # cell (a pipe suffix), which is half the identity the loss guard
        # matches on.
        _recover_refused_additions(career_ops, tracker_dir)
        _sanitize_pending_additions(career_ops, tracker_dir)
        before = _pending_additions(tracker_dir)
        queued = queued_additions(tracker_dir)
        # The re-check's Discards, read before the merge can rewrite their Notes.
        closed = _liveness_closed_rows(career_ops / "data" / "applications.md")
        print("[batch] running merge-tracker.mjs...")
        try:
            r = subprocess.run(["node", "merge-tracker.mjs"], cwd=career_ops,
                               capture_output=True, text=True, encoding="utf-8")
        except OSError as exc:
            # No `node` on PATH: a merge that did not run, like the missing
            # script above — not a traceback that ends the run before its
            # later stages.
            print(f"[batch] could not run merge-tracker.mjs ({exc}) — is Node.js "
                  "installed? Nothing was merged; the additions stay in "
                  "batch/tracker-additions/")
            return False
        pending = queued_additions(tracker_dir)
        if r.returncode != 0:
            if queued <= pending:
                print(f"[batch] merge-tracker failed:\n{r.stderr.strip()}")
                _hint_missing_node_modules(r.stderr)
                return False
            print(f"[batch] merge-tracker exited {r.returncode} after a partial merge — "
                  "what it applied is in the tracker; what it did not is listed below. "
                  f"Its error output:\n{r.stderr.strip()}")
        else:
            print("[batch] tracker merged")
        # BOTH streams: merge-tracker refuses a row with console.warn, which is
        # stderr. Passing stdout alone left the reasons empty on every genuine
        # loss — and printed the one refusal it does log to stdout (the benign
        # unscoreable re-eval) as the explanation for some other addition's
        # disappearance.
        unapplied = _warn_on_lost_additions(before, queued, pending, career_ops,
                                            f"{r.stdout}\n{r.stderr}")
        _reopen_reposted(closed, career_ops)
        return r.returncode == 0 and not unapplied


# ── "Discarded" is two different things (#163) ───────────────────────────────
#
# A person drags a card to Discarded because they decided against the role.
# The liveness re-check writes Discarded because a POSTING died. The tracker
# spells both the same, and every reader treated both as final — so when an
# employer re-posted an opening the re-check had Discarded, bridge's stage-2
# dedup dropped the same-titled re-post outright, and a re-worded one was
# evaluated and then folded by merge-tracker's fuzzy tier onto the Discarded
# row, which keeps its status. Six rows in a real tracker sat at 4.2–4.8 under
# Discarded with a fresh report dated after the Discard, invisible to the UI,
# the work-order and the next re-check. A re-check Discard is therefore
# PROVISIONAL, and the re-check says so in the row's Notes, where the mark
# rides along in the cached cloud tracker and through every Refresh with no
# new state file — a new cache path would change the cache's version key and
# orphan every copy's history (see the runtime-state note in CLAUDE.md).
#
# The newest mark wins: a row the merge reopened carries a Reopened mark after
# its Closed one, so a person Discarding it afterwards is read as a person.
LIVENESS_CLOSED_RE = re.compile(r"\bClosed \d{4}-\d{2}-\d{2} \(liveness re-check", re.I)
REOPENED_RE = re.compile(r"\bReopened \d{4}-\d{2}-\d{2} \(re-posted", re.I)
BY_HAND_RE = re.compile(r"\bSet [A-Za-z]+ \d{4}-\d{2}-\d{2} \(by hand\)", re.I)
# What must never reach a Notes cell: a `|` is a cell boundary in the markdown
# table, and a URL would be read back by `extract_url` as the posting — so the
# URL half IS the pattern `extract_url` reads with, not a re-spelling of it.
_NOTE_UNSAFE_RE = re.compile(_NOTES_URL_RE.pattern + r"|\|")


def _note_safe(text: str) -> str:
    """`text` fit to sit inside a Notes cell: no `|`, no URL, one-line, capped.
    The re-check's reasons include a regex pattern (`body: (?:closed|filled)`)
    and a redirect URL, either of which would corrupt the row or hijack it."""
    text = " ".join(_NOTE_UNSAFE_RE.sub(" ", text or "").split())
    return text[:80].rstrip()


def liveness_closed_mark(date: str, reason: str) -> str:
    """The Notes mark the liveness re-check leaves with its Discard."""
    return f"Closed {date} (liveness re-check: {_note_safe(reason) or 'posting gone'})"


def reopened_mark(date: str) -> str:
    """The Notes mark the merge leaves when it reopens a re-posted role."""
    return f"Reopened {date} (re-posted and re-evaluated)"


def by_hand_mark(date: str, status: str) -> str:
    """The Notes mark a person's status write leaves on a row whose newest mark
    is the re-check's Closed one — what makes the decision a person's (#163).
    `app.data.by_hand_note_for` decides when one is needed."""
    return f"Set {_note_safe(status) or 'status'} {date} (by hand)"


def closed_by_recheck(notes: str) -> bool:
    """True when the newest mark in `notes` is the re-check's Closed one — newer
    than any Reopened mark the merge left and any by-hand mark a person's write
    left — so the row's Discard is the re-check's, not a person's. Only
    meaningful on a row whose status IS Discarded; `recheck_discarded` asks
    both halves."""
    notes = notes or ""
    last_closed = max((m.end() for m in LIVENESS_CLOSED_RE.finditer(notes)), default=-1)
    last_person = max((m.end() for r in (REOPENED_RE, BY_HAND_RE) for m in r.finditer(notes)),
                      default=-1)
    return last_closed > last_person


def recheck_discarded(status: str, notes: str, vocabulary: tuple | None = None) -> bool:
    """True when a row's Discard is the liveness re-check's, not a person's:
    its status is Discarded AND the newest mark in its Notes is the re-check's
    Closed one (#163). Both halves in one place, so no reader — bridge's dedup,
    the merge's reopen, the next one — can drop the status half and read a
    stale Closed mark on an Evaluated row as provisional. `vocabulary` is
    `canonical_status`'s (`app.data._load_states()`): a caller walking many
    rows resolves it once instead of per row."""
    from pipeline.app.data import canonical_status   # lazy: app.data imports this module
    return (canonical_status(status, vocabulary) == "Discarded"
            and closed_by_recheck(notes))


def _tracker_rows(applications_md: Path):
    """Every data row of the tracker as {column: cell}, read by NAME the way
    every tracker reader here does: the optional Via column shifts Role right
    by one, and read positionally the agency lands where the role belongs.
    Zipping is also the width guard — a row too short to reach Role has no
    Role key and is skipped (Report/Notes are optional columns). A missing
    file reads as no rows."""
    for columns, cells in data_rows(read_text(applications_md)):
        row = dict(zip(columns, cells))
        if "role" in row:
            yield row


def _liveness_closed_rows(applications_md: Path) -> dict[str, dict]:
    """{num: {company, role, report}} for every row the liveness re-check
    Discarded (`recheck_discarded`). Taken BEFORE a merge: the older
    merge-tracker replaces Notes on an update, so afterwards the mark may be
    gone, while the newer one keeps them — the snapshot reads the same either
    way. A person's Discard carries no mark and is not here."""
    from pipeline.app.data import _load_states   # lazy: app.data imports this module
    vocabulary = _load_states()             # once per walk, not per row
    out: dict[str, dict] = {}
    for row in _tracker_rows(applications_md):
        if not recheck_discarded(row.get("status", ""), row.get("notes", ""), vocabulary):
            continue
        out[row.get("num", "").strip()] = {
            "company": row["company"], "role": row["role"],
            "report": row_report_num(row.get("report", ""), row.get("notes", "")),
        }
    return out


def _reopen_reposted(closed_before: dict[str, dict], career_ops: Path) -> None:
    """After a merge: a row the liveness re-check had Discarded whose report
    number this merge changed was just re-evaluated — the opening is back under
    a new posting, folded onto the old row by merge-tracker's update tier, which
    writes score, report, date and notes through but keeps the status. Reset it
    to Evaluated (the next re-check verifies the new posting; `extract_url`
    reads the newest evaluation's URL) and leave a Reopened mark so a later
    Discard by a person is read as a person's.

    What this cannot reach: the OLDER merge-tracker skips a re-eval that scored
    lower than the dead posting's evaluation, so under it that re-post is lost
    and the row stays Discarded. Upstream's merge-tracker has written a re-eval
    through in both directions since its fix for career-ops#2411, and
    `tests/test_merge_tracker_contract.py` drives the higher-score case that
    both accept."""
    if not closed_before:
        return
    from pipeline.app.data import record_status_changes   # lazy, as above
    apps = career_ops / "data" / "applications.md"
    after: dict[str, str] = {}
    for row in _tracker_rows(apps):
        num = row.get("num", "").strip()
        if num in closed_before:
            after[num] = row_report_num(row.get("report", ""), row.get("notes", ""))
    today = datetime.now().date().isoformat()
    changes = [(num, "Evaluated", was["company"], was["role"])
               for num, was in closed_before.items()
               if after.get(num) and after[num] != was["report"]]
    if not changes:
        return
    record_status_changes(apps, changes, notes={num: reopened_mark(today) for num, *_ in changes})
    print(f"[batch] reopened {len(changes)} role(s) the liveness re-check had Discarded — "
          "re-posted and re-evaluated in this merge, so back to Evaluated:")
    for num, _, company, role in changes:
        print(f"[batch]   #{num} {company} — {role}")


def _addition_key(company: str, role: str) -> str:
    """The company::role identity, normalized the way merge-tracker's own dedup
    normalizes it. Not stable across a merge on its own — see `_report_key`."""
    return f"{normalize_company(company)}::{normalize_company(role)}"


def _report_key(company: str, num: str) -> str:
    """The company + report-number identity, which a merge DOES preserve.

    merge-tracker matches an addition against the tracker on four tiers, and only
    two of them are provable identity: an exact posting URL, and an exact report
    number. On the GUESSED tiers — entry number, fuzzy title — it updates the row
    but deliberately keeps THE ROW'S OWN TITLE, so that a fuzzy false positive
    cannot also destroy the evidence that two reqs were distinct
    (`role: (reportNumMatched || dupReason === 'url') ? addition.role :
    duplicate.role`). Its `🔄 Update` log line prints the ADDITION's role, so the
    substitution is invisible there too.

    The report link, the score and the company are written through on every one
    of those tiers, so they are what survives. Company is part of the key because
    merge-tracker's own report-number tier requires it: report-file numbering and
    tracker-row numbering drift, so a bare number can name an unrelated row.

    Not a replacement for `_addition_key` — a second reading of the same row. An
    unscoreable re-eval is skipped with the existing score, report and PDF left
    standing, so there the row keeps its company::role and never receives the new
    report number. Each key covers the other's blind spot."""
    return f"{normalize_company(company)}::{num}"


def _addition_cells(text: str) -> list[str]:
    """The cells of one addition row, in ADDITION_COLUMNS order.

    Both HEADERLESS shapes merge-tracker accepts (a headed addition is read by
    name before this is reached — `headed_addition_row`), because a row it can
    read is a row it can also REFUSE and archive, and a shape this function
    cannot parse is an
    evaluation that vanishes without even being counted — the one outcome the
    guard exists to prevent. A model that ignores the prompt's tab rule and emits
    a markdown table row is the case `_strip_role_pipe` already tells us happens;
    every sanitizer no-ops on it (they split on tabs and find one cell), so it
    reaches merge-tracker exactly as the model wrote it.

    The two shapes disagree about columns 5-6 — the pipe form is score-then-status
    — but agree on every cell read here (company, role, report, notes), and
    merge-tracker resolves that pair by content rather than position anyway. Do
    not read `row["score"]` or `row["status"]` off this without fixing that.

    maxsplit on the tab form, as `app/data.py:parse_tracker_additions` reads the
    same file: a tab inside the free-text notes must not become a tenth cell."""
    text = text.strip()
    if text.startswith("|"):
        return split_row(text)
    return [c.strip() for c in text.split("\t", len(ADDITION_COLUMNS) - 1)]


def _pending_additions(tracker_dir: Path) -> list[dict]:
    """One record per un-merged addition TSV: the file it came from, and the
    company, role and report number it carries — the raw cells, with both
    identities derived at the one place they are compared.

    One record per FILE, not per key: a re-evaluation of a role already queued
    puts two TSVs under one company::role, and collapsing them would under-report
    a loss and send the operator hunting for one report while another stays
    buried. The filename is also what "still pending" is decided on, since
    merge-tracker moves an applied TSV into merged/ under its own name (and,
    since career-ops#5862, leaves a refused one where it was) — so the file
    answers "was this addition applied" exactly, where a shared key answers it
    for whichever TSV moved first.

    A headerless row short of `role` is unreadable and skipped (a headed file
    is counted regardless — see below); short of `notes` it is not, since a
    lost trailing tab is routine (see `_restore_trailing_cells`)."""
    additions: list[dict] = []
    for f in sorted(tracker_dir.glob("*.tsv")):
        text = read_addition(f)
        row = headed_addition_row(text)
        if row is None:
            row = dict(zip(ADDITION_COLUMNS, _addition_cells(text)))
            if "role" not in row:
                continue
        # A headed file with no single readable row is still counted, with no
        # identity: merge-tracker refuses it — archived (read as lost by
        # `_classify_landing`) or kept (reported as kept) by release — so
        # skipping it the way an unreadable headerless row is skipped dropped a
        # two-row file's evaluations without a line in the log.
        additions.append({
            "name": f.name,
            "company": row.get("company", ""),
            "role": row.get("role", ""),
            "report": row_report_num(row.get("report", ""), row.get("notes", "")),
        })
    return additions


def _classify_landing(records: list[dict], career_ops: Path) -> tuple[list[dict], list[tuple[dict, str]]]:
    """(lost, retitled) for every addition record: which of the two readings of
    "did it land" found it in the tracker, or neither.

    Shared by the loss guard, which reports the answer, and the recovery pass,
    which acts on it — one definition, so the row the guard calls lost is the
    row recovery goes after, never two slightly different sets."""
    if not records:
        return [], []
    landed, role_by_report = _tracker_identities(career_ops / "data" / "applications.md")
    lost: list[dict] = []
    retitled: list[tuple[dict, str]] = []
    for add in records:
        # The two readings, side by side. Either one finding the row means the
        # evaluation is in the tracker.
        # No identity at all (an unreadable headed file): only the report number
        # can find it.
        if (add["company"] or add["role"]) and \
                _addition_key(add["company"], add["role"]) in landed:
            continue
        key = _report_key(add["company"], add["report"]) if add["report"] else None
        # `in`, not truthiness of the title: a tracker row with a blank Role cell
        # (hand-added, or half-migrated) matched here would otherwise be read as
        # no match at all, and an intact evaluation would get the loud loss
        # warning — the #152 cry-wolf back again by a different route.
        if key in role_by_report:
            retitled.append((add, role_by_report[key]))
        else:
            lost.append(add)
    return lost, retitled


def _recover_refused_additions(career_ops: Path, tracker_dir: Path) -> None:
    """Put back into the queue the rows a previous merge refused for a reason
    the sanitizer can fix.

    This is the half of #156 the merge-time sanitizer cannot reach: `--batch`
    runs career-ops' `batch-runner.sh`, whose last step is its OWN
    `node merge-tracker.mjs`, so a row the agent CLI wrote with a bare `4.2` is
    refused and archived into `merged/` before any Python runs. From there
    nothing retries it — the report is on disk, the tracker has no row, and the
    evaluation is simply gone. `run.sh`/`run.ps1` run `pipeline.merge_additions`
    after the runner returns, which lands here.

    Two rules decide what comes back, and both are load-bearing:

    **It did not land, by either reading.** `_classify_landing` is the loss
    guard's own test, so the set recovered is exactly the set that guard would
    have reported. A row merge-tracker folded into an existing one under a
    different title (#152) is NOT lost and is not touched.

    **The score fix would change it.** Of the whole chain, an unreadable score
    cell is the only defect that gets a row REFUSED; the others corrupt or
    impoverish a row that still merges. So a lost row whose score is already
    readable was refused for a reason we cannot cure — a report number marked
    `failed` in batch-state.tsv, which upstream refuses on purpose — and pulling
    it back would only have it refused again, run after run. The same rule is
    what stops a row the user DELETED from the tracker coming back: it merged
    with a readable score, so it never qualifies. (A row that is both repairable
    and doomed gets one retry: after it, the archived copy is the repaired one,
    and the score no longer changes.)

    A headed addition comes back through its data row with its label line kept,
    on `_sanitize_pending_additions`' terms (`_positional_addition`); a
    headerless one comes back as it was, and that pass labels it if the
    checkout reads labels. A row already back in the queue is left to the merge
    in progress.

    **On a career-ops with #5862 this finds nothing, by design.** A refused
    addition is no longer archived: it stays in `tracker-additions/`, where the
    merge-time sanitizer reaches it on the next merge like any other pending
    row. `merged/` then holds only additions that were applied — each of which
    landed, or merged with a readable score and was later deleted by hand — and
    neither qualifies under the rules above. This stays for a checkout before
    #5862, which still archives refusals (the pinned cloud daily ran one), and
    for refusals archived before a `git pull` brought it."""
    merged = tracker_dir / "merged"
    lost, _ = _classify_landing(_pending_additions(merged), career_ops)
    if not lost:
        return
    urls = _batch_input_urls(career_ops / "batch" / "batch-input.tsv")
    restored = 0
    for add in lost:
        src, dst = merged / add["name"], tracker_dir / add["name"]
        if dst.exists():
            continue
        split = _read_positional(src)
        if split is None:
            continue
        header, row = split
        full = _restore_trailing_cells(row)
        if _normalize_score_cell(full) == full:
            continue                    # refused for a reason we cannot fix
        fixed = sanitize_addition(row, urls.get(Path(add["name"]).stem, ""),
                                  read_text(jd_cache_path(career_ops, Path(add["name"]).stem)))
        try:
            atomic_write_text(dst, _rejoin_addition(header, fixed))
        except OSError:
            continue
        restored += 1
    if restored:
        print(f"[batch] recovered {restored} evaluation(s) a previous merge refused "
              "for an unreadable score — re-queued with the score repaired")


def _warn_on_lost_additions(before: list[dict], queued: set[str], pending: set[str],
                            career_ops: Path, merge_output: str = "") -> int:
    """Report what became of each evaluation pending before the merge, and
    return how many did NOT reach the tracker. `before` is the parsed snapshot
    (`_pending_additions`), and `queued` and `pending` list the queue
    (`queued_additions`) before the merge and after it.

    merge-tracker declines a row it can't read confidently — no label line
    (career-ops#6062), an unreadable score cell, a report number marked `failed`
    in batch-state.tsv — with a warning and exit 0, and what it does with the
    TSV depends on the release, so both shapes are read:

    **Kept** (career-ops#5862, 2026-10-26, and its `failedAdditions` exit-1 case
    before that): the TSV is in the queue before the merge and after it. That
    is decided by the FILES, readable or not, because `before` skips a
    headerless file it cannot read up to the role cell (a short positional or
    pipe row), which #6062 refuses and #5862 keeps like any other. Judged on
    the parsed set, that refusal went unnamed and the merge read as clean
    (#241). `before` only lends a kept file its identity; without one it is
    named by its file as "no readable row". Being there IS the refusal, so it
    is reported as its own category, with what to do, and never through the
    "did it land" test below: a refused re-evaluation of a role already in the
    tracker would pass that test while its new report never arrived. The one
    message is true in both places a merge runs: locally every later merge
    retries the file and merge-tracker refuses it again until it changes,
    while the cloud daily does not keep batch/tracker-additions/ at all.

    **Lost** (before #5862): the TSV was archived into merged/ anyway, so it is
    gone from the queue, absent from applications.md, and
    `has_pending_tracker_additions` is false, meaning nothing ever retries it.
    Judged on `before` alone: an unreadable file that left the queue is not
    claimed lost, since a merge-tracker before #6062 can merge a pipe row.

    Each named addition is followed by merge-tracker's own lines about it —
    every refusal names its file as `<name>:` — since in a cloud run the log is
    all the operator has, and a block of every `Skipping` line in the run
    could not say which reason belonged to which file.

    Asking the filesystem and the tracker, rather than grepping merge-tracker's
    prose for "Skipping", is what makes this survive an upstream release: those
    messages come in half a dozen phrasings and at least one of them is BENIGN
    (a re-eval that produced no score deliberately keeps the row's existing
    score — a case `_normalize_score_cell`'s N/A output makes more likely, not
    less). "Did the row land?" has one answer and upstream cannot reword it.

    But it has TWO readings, and asking only for company::role made this cry wolf
    on three intact evaluations in a real run (#152): each had been merged into an
    existing row on a heuristic tier, which keeps that row's title, so the key the
    addition carried was not the key it landed under. A guard that fires on intact
    rows camouflages the loss it exists to catch, so an addition found under
    EITHER identity counts as landed. The substitution gets its own line, because
    it is real information nothing else surfaces: the fuzzy tier can fold two
    genuinely distinct reqs (a leveled title and its sibling) into one row, and
    from then on only one of them is visible to dedup, the handoff and the UI."""
    kept = queued & pending
    identity = {add["name"]: add for add in before}
    refused = [identity.get(name, {"name": name, "company": "", "role": "", "report": ""})
               for name in sorted(kept)]
    lost, retitled = _classify_landing(
        [add for add in before if add["name"] not in kept], career_ops)

    if refused:
        # No "re-evaluate": the job is done in the batch state and its URL is in
        # scan-history, so nothing would queue it again.
        print(f"[batch] WARNING: {len(refused)} evaluation(s) were refused by "
              "merge-tracker and KEPT in batch/tracker-additions/ — their reports "
              "are on disk but the roles are not in applications.md, so the UI and "
              "the handoff cannot see them. Locally every merge retries each one, "
              "and reports failure, until its file is corrected as merge-tracker's "
              "reason says or deleted to drop the evaluation; the cloud daily does "
              "not keep that directory, so there nothing retries it and the file is "
              "in that run's pipeline-output artifact for 7 days:")
        _print_additions(refused, merge_output)

    if lost:
        print(f"[batch] WARNING: {len(lost)} evaluation(s) left "
              "batch/tracker-additions/ without reaching applications.md — their "
              "reports are on disk but the roles are invisible to the UI and to the "
              "handoff, and no later run retries them. The TSVs are in "
              "batch/tracker-additions/merged/:")
        _print_additions(lost, merge_output)

    if retitled:
        print(f"[batch] note: {len(retitled)} evaluation(s) merged into an existing "
              "tracker row that kept its own role title — merge-tracker matched them "
              "on a guessed tier (entry number, or fuzzy title), where it will not "
              "let an addition rewrite a title. Nothing was lost: the report link "
              "and score wrote through. Worth a look, because a fuzzy match can fold "
              "two genuinely distinct reqs into one row:")
        for add, row_role in retitled:
            print(f"[batch]   {add['name']}: {add['company']} — "
                  f'"{add["role"]}" merged into "{row_role}" (report {add["report"]})')
    return len(refused) + len(lost)


def _print_additions(additions: list[dict], merge_output: str) -> None:
    """One line per addition the guard names, each followed by merge-tracker's
    own lines about that file. Every refusal upstream prints names its file as
    `<name>:` (`Skipping 7.tsv: …`, `❌ 7.tsv: could not locate …`), while its
    run summaries list names without the colon — so the match is the name as a
    whole token followed by `:`, which also keeps `17.tsv`'s reasons off
    `7.tsv`. `merge_output` must carry stderr: every refusal but one is a
    console.warn."""
    for add in additions:
        who = (f"{add['company']} — {add['role']}" if add["company"] or add["role"]
               else "no readable row")
        print(f"[batch]   {add['name']} ({who})")
        names_it = re.compile(rf"(?<![\w.-]){re.escape(add['name'])}:")
        for line in merge_output.splitlines():
            if names_it.search(line):
                print(f"[batch]     merge-tracker: {line.strip()}")


def _tracker_identities(applications_md: Path) -> tuple[set[str], dict[str, str]]:
    """The two identities every tracker data row can be found by, in one walk:
    the set of company::role keys, and {company::report_num: that row's role}.

    The role title comes back with the report key rather than a bare `True`
    because it is what makes the retitle line above readable — the operator needs
    to see which existing row swallowed the addition to judge whether the pairing
    was right."""
    keys: set[str] = set()
    role_by_report: dict[str, str] = {}
    for row in _tracker_rows(applications_md):
        company, role = row["company"], row["role"]
        keys.add(_addition_key(company, role))
        num = row_report_num(row.get("report", ""), row.get("notes", ""))
        if num:
            role_by_report.setdefault(_report_key(company, num), role.strip())
    return keys, role_by_report


def _hint_missing_node_modules(stderr: str) -> None:
    """merge-tracker.mjs used to import only Node builtins; it now reaches
    js-yaml through tracker-utils.mjs. A career-ops checkout that was updated
    without a fresh `npm install` fails to resolve it, and the raw
    ERR_MODULE_NOT_FOUND stack doesn't say what to do about it."""
    if "ERR_MODULE_NOT_FOUND" not in stderr and "Cannot find package" not in stderr:
        return
    print("[batch] hint: career-ops now has npm dependencies that merge-tracker "
          "needs (js-yaml). Run `npm install --ignore-scripts` in your "
          "career-ops checkout — a `git pull`/rebase there does not install them.")


# ── Protected characteristics never reach a prompt ───────────────────────────
#
# The wizard collects EEO self-ID answers for the BROWSER AGENT to use on an
# application form, and setup-profile.mjs writes them into profile.yml under
# `voluntary_disclosures:`. Five prompt builders in this repo eat the candidate
# files, so redaction belongs on the READ, not on one builder: a sixth builder
# is otherwise safe only if its author remembers. `tests/test_batch_common.py::
# TestNoUnredactedProfileReads` pins that no prompt code reads either file raw.
#
# Two separate harms, and only one is about privacy: a model that scores or
# tailors must not see the characteristics such a judgement is forbidden to
# rest on, which holds on a paid tier and with consent; and on a free provider
# tier those answers may additionally be retained.
#
# The WHOLE yaml section goes, not just the four EEO keys — the rest is the
# site-consent booleans, apply-path form data no prompt has a use for.
#
# Line-based rather than yaml.safe_load + dump: this module is stdlib-only on
# sites.py's terms, and a round trip would reorder keys and drop comments in a
# document otherwise passed through verbatim.
_REDACTED_PROFILE_SECTION = "voluntary_disclosures"

# PROFILE.md needs its own pass: it is markdown, it SUPERSEDES the seeds, and it
# is non-clobber — an install seeded before this keeps those bullets forever and
# no regeneration removes them. Stems rather than exact labels, because the file
# is agent-grown and append-only, so "Gender identity" and "Protected veteran"
# are the shapes an exact allowlist would miss.
#
# WORD-ANCHORED, and that is not cosmetic: PROFILE.md is the résumé build's ONLY
# source of truth, so an unanchored "race" eats the fact-bank bullets for "Grace
# Health", "Terrace Senior Living" and "Traceability" — silently deleting real
# experience from the résumé it is about to write.
#
# Testing the bold LABEL alone let a disclosure through three ways (#216): a
# label with none of the five stems ("Self-ID", "EEO", "Pronouns", "Voluntary
# disclosures"), the continuation and nested lines under a dropped bullet, and
# unlabelled bullets under a disclosure heading. Which rule is safe depends on
# WHERE a line sits, so the rule is chosen by section. In the generator's own
# Standing answers section a whole line is scanned against every stem: it holds
# form answers, never experience. In its evidence sections (fact bank, skills)
# the rule stays the pre-#216 label match on the five protected stems, because
# experience carries these words too — "VA North Texas: served 2,000 veterans",
# an "EEO Specialist" role, an "EEO-1 reporting" skill — and a value scan there
# is the Grace Health deletion moved from labels to values. Everywhere else (a
# section the agent added, text before the first heading) the DISCLOSURE stems
# also match labels and headings, and such a heading drops its section. `eeo`
# and `pronoun` stop at a word boundary: the EEOC is an employer, and
# "pronounced" is not a pronoun.
STANDING_ANSWERS_HEADING = "Standing answers"   # handoff.render_profile_md writes it
_STANDING_ANSWERS = STANDING_ANSWERS_HEADING.casefold()
_PROTECTED_STEMS = r"gender|rac(?:e|es|ial)\b|ethnic\w*|veteran|disab\w*"
_DISCLOSURE_STEMS = r"self[\s-]*id\w*|eeo\b|pronouns?\b|voluntary\s+disclos\w*"
_PROTECTED_RE = re.compile(rf"\b(?:{_PROTECTED_STEMS})", re.IGNORECASE)
_REDACTED_STEM_RE = re.compile(rf"\b(?:{_PROTECTED_STEMS}|{_DISCLOSURE_STEMS})", re.IGNORECASE)
# The generator's evidence sections (handoff.render_profile_md writes them). The
# disclosure stems are words a job can carry too — an "EEO Specialist" role, an
# "EEO-1 reporting" skill — so inside these the rule stays what it was before
# #216: the five protected stems, on a bullet's label only, and no heading ever
# drops a section. Dropping one there deletes real experience from the résumé
# build's only source.
EVIDENCE_HEADINGS = ("Role fact bank", "Skills inventory")
# Every top-level section render_profile_md writes. An evidence section ends
# only at one of THESE: the fact bank holds a CV's experience verbatim, own
# `## Experience` heading included, and ending the section there put a role
# like "### EEO Specialist" under the stricter rule, which deleted it.
PROFILE_SECTION_HEADINGS = ("Identity & contact", "Positioning", *EVIDENCE_HEADINGS,
                            STANDING_ANSWERS_HEADING, "Tailoring rules", "How this profile grows")
_EVIDENCE = tuple(h.casefold() for h in EVIDENCE_HEADINGS)
_GENERATOR = tuple(h.casefold() for h in PROFILE_SECTION_HEADINGS)
_PROFILE_BULLET_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s*(?:\*\*|__)(?P<label>[^*_\n]+?):?(?:\*\*|__)")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_MD_HEADING_RE = re.compile(r" {0,3}(#{1,6})(?=\s|$)")


def redact_profile_yml(text: str) -> str:
    """`profile.yml` with the protected-characteristics section dropped."""
    out: list[str] = []
    skipping = False
    for line in (text or "").splitlines(keepends=True):
        # A top-level key ends the previous block; a column-0 COMMENT does not,
        # so one inside the block is dropped with it — better to lose a comment
        # than to reopen the file and leak the keys after it.
        if line[:1].strip() and not line.lstrip().startswith("#"):
            skipping = line.split(":", 1)[0].strip() == _REDACTED_PROFILE_SECTION
        if not skipping:
            out.append(line)
    return "".join(out)


def _indent(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip())


def redact_profile_master(text: str) -> str:
    """`PROFILE.md` with the protected-characteristic standing answers dropped.

    Inside the Standing answers section: any line carrying a stem goes, label or
    value, bulleted or not, and so does a sub-heading's whole section. Inside the
    evidence sections (`EVIDENCE_HEADINGS`, which end only at another generator
    section): only a bullet whose bold label carries one of the five protected
    stems, as before #216, and no heading drops anything. Everywhere else: a
    bullet whose label carries any stem, and a heading carrying any stem drops
    its whole section, as `redact_profile_yml` drops `voluntary_disclosures:`.
    The generator's own headings are recognised before any is tested, so
    "Standing answers (incl. EEO)" is still that section. A dropped line takes
    its continuation: deeper-indented lines (across blank lines, as a loose list
    nests) and an unindented lazy continuation straight after it. A `#` inside a
    code fence is not a heading."""
    out: list[str] = []
    drop_level = scan_level = evidence_level = None   # levels of the open sections
    cont = None        # indent of the last dropped line, while its continuation runs
    blank_since = False
    fence = False
    for line in (text or "").splitlines(keepends=True):
        stripped = line.strip()
        heading = None if fence else _MD_HEADING_RE.match(line)
        if _FENCE_RE.match(line):
            fence = not fence
        if heading:
            cont = None
            level = len(heading.group(1))
            title = " ".join(line.split()).casefold()
            generator = any(h in title for h in _GENERATOR)
            if drop_level is not None and level > drop_level:
                continue
            drop_level = None
            if scan_level is not None and level <= scan_level:
                scan_level = None
            if evidence_level is not None and level <= evidence_level and generator:
                evidence_level = None
            if evidence_level is not None:
                out.append(line)
                continue
            if scan_level is None and _STANDING_ANSWERS in title:
                scan_level = level
            elif scan_level is None and any(h in title for h in _EVIDENCE):
                evidence_level = level
            elif (level > 1 and not (scan_level is None and generator)
                  and _REDACTED_STEM_RE.search(line)):
                # Never the document's title: every section sits below it, and
                # a surname like "Disabato" would take the whole profile.
                drop_level = level
                continue
            out.append(line)
            continue
        if drop_level is not None:
            continue
        if not stripped:
            blank_since = True
            out.append(line)
            continue
        if cont is not None:
            if _indent(line) > cont or (not blank_since and not _LIST_ITEM_RE.match(line)):
                continue
            cont = None
        if scan_level is not None:
            hit = _REDACTED_STEM_RE.search(line)
        else:
            m = _PROFILE_BULLET_RE.match(line)
            stems = _PROTECTED_RE if evidence_level is not None else _REDACTED_STEM_RE
            hit = m and stems.search(m.group("label"))
        if hit:
            cont, blank_since = _indent(line), False
            continue
        out.append(line)
    return "".join(out)


def profile_yml_for_prompt(career_ops) -> str:
    """The one sanctioned read of `profile.yml` for anything a model will see."""
    return redact_profile_yml(read_text(Path(career_ops) / "config" / "profile.yml"))


def profile_master_for_prompt(path) -> str:
    """The one sanctioned read of a `PROFILE.md` for anything a model will see."""
    return redact_profile_master(read_text(path))


# ── What a score means (#208) ─────────────────────────────────────────────────
# career-ops' own bands from 3.5 up (`modes/_shared.md` → "Score interpretation"),
# with the evidence that puts a role in each, worded for any field. Below 3.5 it
# departs on purpose: career-ops says "recommend against", here 3.0-3.4 is still
# Consider and Skip starts at 3.0, the line the tracker's NO APLICAR rule already
# used (a maintainer decision, 2026-11-22). The two named floors
# are the lines the rest of the pipeline acts on; defaults elsewhere import them
# rather than restating 4.0 / 3.0. Why the prompt must define the scale at all:
# CLAUDE.md, "What a score means is stated in the prompt".
APPLY_MIN_SCORE = 4.0
SKIP_BELOW = 3.0
SCORE_BANDS = (
    (4.5, "Strong match: apply now",
     "One of the candidate's target roles, at a level they can be hired at. Meets "
     "every stated requirement, with evidence in the profile. Nothing in comp, "
     "culture or red flags argues against it."),
    (APPLY_MIN_SCORE, "Good match: worth applying",
     "A target role or a close neighbour of one. Meets the stated requirements, or "
     "misses one the candidate can credibly cover with related experience."),
    (3.5, "Decent, not ideal: apply only with a specific reason",
     "Adjacent to the target roles; or misses two or more stated requirements; or "
     "posted pay below the candidate's target; or a real red flag."),
    (SKIP_BELOW, "Long shot",
     "The candidate could apply but would be a weak applicant, or the role is only "
     "partly what they are looking for."),
    (2.0, "Poor fit",
     "The wrong function or level for this candidate, or several stated requirements "
     "missing."),
    (1.0, "No fit",
     "An unrelated field, or a requirement the candidate cannot meet at all (a "
     "licence, credential or clearance they do not hold)."),
)


# `- **Remote roles:** Score 5.0 (ideal)` and its siblings: the location-scoring
# lines setup-profile.mjs wrote into every _profile.md before #208. They assign a
# Global score by location, which the bands and the hard-stop rule replace, and
# _profile.md is never regenerated, so they are dropped where the file is read.
# Matched on the generator's exact shape only; prose the user wrote is kept.
_LOCATION_SCORE_LINE_RE = re.compile(r"^\s*-\s*\*\*[^*\n]+:\*\*\s*Score\s+\d(?:\.\d)?\s*\(.*\)\s*$")


def _drop_location_scores(profile_md: str) -> str:
    return "\n".join(ln for ln in profile_md.splitlines()
                     if not _LOCATION_SCORE_LINE_RE.match(ln))


def _score_band_table() -> str:
    rows = []
    top = 5.0
    for floor, meaning, evidence in SCORE_BANDS:
        rows.append(f"| {floor:.1f}-{top:.1f} | {meaning} | {evidence} |")
        top = round(floor - 0.1, 1)
    return ("| Global | Meaning | What puts a role there |\n|---|---|---|\n"
            + "\n".join(rows))


def _target_roles_block(target_roles) -> str:
    """The candidate's own role families (handoff.target_roles), for archetype
    detection and North Star; they replaced career-ops' six AI-engineering
    archetypes (#209). Names and `fit` only: the `level` setup-profile.mjs writes
    is one guess for every role, read off an engineering ladder ("Outreach Lead"
    becomes staff/principal), so stating it here would put a wrong seniority in
    front of the model."""
    roles = list(target_roles or ())
    if not roles:
        return ("The candidate profile above states their target roles; if it names none, "
                "take them from the work their experience points to.\n")
    legend = (" (fit: primary = the role they most want, secondary = a good fit, "
              "adjacent = a stretch)") if any(r["fit"] for r in roles) else ""
    lines = [f"- {r['name']}" + (f" ({r['fit']})" if r["fit"] else "") for r in roles]
    return (f"The candidate's target roles, as they named them at setup{legend}:\n"
            + "\n".join(lines) + "\n"
            + "If the candidate profile above names different target roles, it is newer: use those.\n")


def build_system_prompt(cv: str, profile_yml: str, profile_md: str = "", article_digest: str = "", *,
                        profile_master: str = "", target_roles: list[dict] | None = None) -> str:
    # PROFILE.md (Commit 4), when present, is the candidate's living master and
    # supersedes the 4 seed fragments: it already folds in the CV, profile,
    # positioning, and proof points, and the browser agent grows it as it learns.
    # Only the candidate-profile *content* swaps; the evaluation framework below is
    # unchanged. A blank/whitespace master → the original seed-fragment assembly.
    if (profile_master or "").strip():
        candidate_profile = (
            "### Candidate Profile (PROFILE.md — the living master)\n"
            "_Folds in the CV/experience, structured profile, positioning, and "
            'proof points. Where the framework below says "CV" or "_profile.md", '
            "read this profile._\n\n"
            f"{redact_profile_master(profile_master)}\n"
        )
    else:
        extra = ""
        if profile_md:
            extra += f"\n### User Customizations (_profile.md)\n{_drop_location_scores(profile_md)}\n"
        if article_digest:
            extra += f"\n### Proof Points (article-digest.md)\n{article_digest}\n"
        candidate_profile = (
            f"### CV (read-only)\n{cv}\n\n"
            f"### Profile (profile.yml)\n{redact_profile_yml(profile_yml)}\n{extra}"
        )

    return f"""You are a job evaluation expert running in headless batch mode. \
Evaluate job postings for the candidate described below. \
You have no file system access, cannot generate PDFs, and cannot run real-time web searches.

## CANDIDATE PROFILE

{candidate_profile}
## EVALUATION FRAMEWORK

Work through blocks A-G in order. Rules in _profile.md override system defaults for how each dimension is weighed, but not the score bands, the decision rule or the hard-stop rule below.

**Archetype Detection** - name the candidate's target role this posting is closest to.
{_target_roles_block(target_roles)}Use that role's name, or a hybrid of two. When none fits, write "Outside target roles (<the role's actual function, in a few words>)" - that is a finding, not a failure, and it is what North Star Alignment scores.

**Block A - Role Summary**
Table: Archetype, Domain, Function, Seniority, Remote policy, Team size, TL;DR.

**Block B - CV Match**
Map each JD requirement to exact CV lines or note a gap.
Gaps section: hard blocker vs nice-to-have, adjacent experience, mitigation plan.
Adapt framing to the detected archetype.

**Block C - Level & Strategy**
1. JD level vs candidate's natural level.
2. "Sell senior without lying" plan with specific phrases.
3. "If downleveled" acceptance criteria.

**Block D - Comp & Demand**
Use training-knowledge salary benchmarks (Glassdoor/Levels.fyi estimates - label as such).
Score comp 1-5: 5=top quartile, 1=well below market. When the posting states no pay, the comp score is an estimate: report it, but it does not move the Global band.

**Block E - Personalization Plan**
Table of top 5 CV changes + top 5 LinkedIn changes.

**Block F - Interview Plan**
6-10 STAR stories mapped to JD requirements. 1 case study recommendation. Red-flag questions.

**Block G - Posting Legitimacy**
Analyze JD quality signals. Mark posting freshness "unverified (batch mode)".
Assess: High Confidence | Proceed with Caution | Suspicious.

**Global Score** table: CV Match, North Star Alignment (how close the role is to the candidate's target roles), Comp, Cultural Signals, Red Flags penalty, Global (all X/5).

The Global score is a holistic judgement of how well this candidate fits this role, not a formula, and each band means one thing:

{_score_band_table()}

What counts against a role, for the bands:
- **Stated requirements** are the qualifications the posting says a candidate must have ("required", "minimum", "must"). Falling a few months short of a years-of-experience minimum, with closely related experience, is one requirement the candidate can credibly cover.
- **Duties are not requirements.** What the job does - present to groups, run a program, keep records - is what the hire will be trained to do, above all in an entry-level role. A duty the candidate has not done before is not a gap.
- **Preferred qualifications** ("preferred", "a plus", "nice to have") never on their own hold a role below {APPLY_MIN_SCORE}.
- **Pay counts only when the posting states it.** An estimate never moves the band.

Judge the role at the level it is posted at: an entry-level role is not marked down because the candidate is early in their career. Use the whole scale. A role that clears a band's bar earns that band without being perfect.

`final_decision` follows the band, unless a hard stop caps it (below): "Apply" at {APPLY_MIN_SCORE} and above, "Consider" from {SKIP_BELOW} to {APPLY_MIN_SCORE - 0.1:.1f}, "Skip" below {SKIP_BELOW}. Use "Research first" only when the band turns on a fact the posting leaves out.

**Hard stops** - `hard_stops` is what makes the candidate unable to HOLD the role, not what they merely lack. Anything they could learn, be trained on, or argue around is a soft gap. Name every hard stop you find and cap `final_decision` at "Consider" when there is one. A location hard stop caps the decision, not the score: being unable to be where the role is says nothing about fit, so record it here and in `work_location`. A requirement the candidate cannot meet (a licence, credential or clearance) lowers the score as well.
Geography is the one most often missed. A role requiring recurring physical presence - on-site OR hybrid, however few the office days - in a metro the candidate has not said they can work in is a hard stop, whatever the fit. Read the candidate's own stated locations from the profile above. Do not invent flexibility the posting does not state: "may offer remote", "could relocate" and "remote coordination may be possible" are not in the JD. Work that is inherently local - community-based, field, in-home, canvassing, site-based - requires presence even where the posting also uses the word remote.

**Machine Summary** YAML block:
```yaml
company: "name"
role: "title"
score: X.X
legitimacy_tier: "High Confidence | Proceed with Caution | Suspicious"
archetype: "closest target role, or Outside target roles (<function>)"
final_decision: "Apply | Consider | Research first | Skip"
work_location:
  mode: "onsite | hybrid | remote"
  metro: "City, ST"
  state: "ST"
hard_stops: []
soft_gaps: []
top_strengths: []
risk_level: "Low | Medium | High"
confidence: "Low | Medium | High"
next_action: "one concrete next step"
```

## OUTPUT FORMAT

Respond with ONLY this XML. No prose outside the tags. Use real tab characters in tracker_tsv.

<evaluation>
<report>
# Evaluacion: COMPANY - ROLE

**Fecha:** DATE
**Arquetipo:** ARCHETYPE
**Score:** SCORE/5
**Legitimacy:** LEGITIMACY_TIER
**URL:** URL
**PDF:** null (batch mode)
**Batch ID:** JOB_ID

---

## Machine Summary

```yaml
MACHINE_SUMMARY_YAML
```

## A) Resumen del Rol
CONTENT

## B) Match con CV
CONTENT

## C) Nivel y Estrategia
CONTENT

## D) Comp y Demanda
CONTENT

## E) Plan de Personalizacion
CONTENT

## F) Plan de Entrevistas
CONTENT

## G) Posting Legitimacy
CONTENT

---

## Keywords extraidas
15-20 ATS keywords from the JD
</report>
<tracker_tsv>
TRACKER_NUM	DATE	COMPANY	ROLE	STATUS	SCORE/5	PDF	[REPORT_NUM](reports/REPORT_NUM-company-slug-DATE.md)	ONE_SENTENCE_NOTES
</tracker_tsv>
<summary>
{{"status": "completed", "id": "JOB_ID", "report_num": "REPORT_NUM", "company": "COMPANY", "role": "ROLE", "score": SCORE_FLOAT, "legitimacy": "LEGITIMACY_TIER", "pdf": null, "report": "reports/REPORT_NUM-company-slug-DATE.md", "error": null}}
</summary>
</evaluation>

work_location rules: `mode` is what the posting REQUIRES of the person, not what it prefers - any required office days at all are "hybrid", never "remote". `metro` is where that presence is required (the work site, not the employer's headquarters) and `state` is that metro's 2-letter state/province code; both are "" for a fully remote role and "" when the posting does not say. Never guess a metro from the company's address.

Tracker TSV rules (one row of 9 tab-separated columns, no header line - the column labels are added when the file is written):
- Col 1 TRACKER_NUM: use the Tracker Number from the user message
- Col 5 STATUS: "Evaluada" normally; "NO APLICAR" if score < {SKIP_BELOW}
- Col 6 SCORE: format X.X/5
- Col 7 PDF: {_PDF_CELL} (batch mode)
- Col 8 report link: [REPORT_NUM](reports/REPORT_NUM-company-slug-DATE.md)
- company-slug: lowercase, spaces to hyphens
- Col 9 notes: one sentence starting with APPLY / CONSIDER / SKIP + key reason

Summary JSON: id and report_num must be the exact values from the user message. score is a float.

On catastrophic failure output only:
<evaluation><summary>{{"status": "failed", "id": "JOB_ID", "report_num": "REPORT_NUM", "company": "unknown", "role": "unknown", "score": null, "legitimacy": null, "pdf": null, "report": null, "error": "brief description"}}</summary></evaluation>
"""


def eval_system_prompt(career_ops: Path) -> str:
    """The evaluation system prompt for a career-ops install — the single place
    the candidate-profile source is resolved, so `--evaluate-batch` and the UI
    add-job eval can never disagree on it. Uses the living PROFILE.md when present
    (authoritative), else the cv.md / profile.yml / _profile.md / article-digest.md
    seeds."""
    # Lazy import: handoff imports this module at load, so a top-level import of
    # it here would cycle. It also owns the YAML reading this module, stdlib-only,
    # cannot do. The target roles come from profile.yml even when PROFILE.md
    # supersedes it as the candidate profile: a master is never overwritten, and
    # one seeded before #209 names only the first two.
    from pipeline.handoff import _parse_yaml_or_empty, resolve_profile_md, target_roles
    career_ops = Path(career_ops)
    profile_yml = read_text(career_ops / "config" / "profile.yml")
    return build_system_prompt(
        read_text(career_ops / "cv.md"),
        profile_yml,
        read_text(career_ops / "modes" / "_profile.md"),
        read_text(career_ops / "article-digest.md"),
        profile_master=resolve_profile_md(career_ops),
        target_roles=target_roles(_parse_yaml_or_empty(profile_yml)),
    )

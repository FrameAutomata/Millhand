"""The candidate's own résumé-tailoring preferences: where they live, and the one
reader every tailor goes through.

The Setup wizard's free-text "tailoring guidance" used to be written into
career-ops/config/profile.yml as `tailoring.instructions`, which put it in the
wrong place twice over. Only the slot-edit tailor read it; the tailor the
work-order actually uses (resume_content) never did, and the API résumé skill
saw it only because it pastes the whole profile.yml. And because profile.yml is
pasted into every evaluation prompt — ours on the seed path and career-ops'
`--batch` worker's — the guidance rode along into every evaluation, where it
does nothing and costs tokens.

So it has a file of its own, `career-ops/config/tailoring.md`, beside
profile.yml and gitignored with the rest of career-ops/. Local only on purpose:
tailoring never runs in the cloud, so unlike profile.yml it needs no secret.

`tailoring_guidance` reads the file; a copy set up before the file existed falls
back to profile.yml's key, so its guidance keeps reaching the tailors until the
next wizard Save moves it (the wizard prefills from this same reader, and
`setup-profile.mjs` then writes the file and drops the key). A file that EXISTS
is the answer even when blank — it is the newer home, and falling back past it
would resurrect guidance the person deleted.
"""

from __future__ import annotations

from pathlib import Path

GUIDANCE_FILE = Path("config") / "tailoring.md"


def guidance_path(career_ops) -> Path:
    """Where the tailoring guidance lives in a career-ops checkout. A Path it is
    handed is joined onto as is rather than re-wrapped: Path() picks its flavour
    from os.name when called, and the skill-launch tests drive the Windows
    rendering by patching os.name process-wide."""
    base = career_ops if isinstance(career_ops, Path) else Path(career_ops)
    return base / GUIDANCE_FILE


def tailoring_guidance(career_ops) -> str:
    """The candidate's tailoring guidance, stripped, or "" when there is none.

    Never raises: a tailor must not fail because its preferences could not be
    read, so an unreadable file is said on stdout and read as none."""
    return read_guidance(career_ops) or ""


def read_guidance(career_ops) -> str | None:
    """`tailoring_guidance`, except that a file which is there but cannot be
    read (not UTF-8, no permission) is None rather than "" — for the wizard,
    where reading it as blank would show an empty box and the next Save would
    delete the file."""
    path = guidance_path(career_ops)
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError:
        return _legacy_profile_guidance(career_ops)
    except (OSError, UnicodeDecodeError) as e:
        print(f"[tailor] could not read {path} ({type(e).__name__}: {e}) — "
              "tailoring without your preferences")
        return None


def written_guidance_path(career_ops) -> Path | None:
    """The guidance file's absolute path when it holds guidance, else None —
    what an agent-facing prompt names. The file itself, not an older copy's
    profile.yml key the reader falls back to: a path is only worth naming when
    it is there to open, and an empty one is nothing to follow."""
    path = guidance_path(career_ops)
    if path.is_file() and tailoring_guidance(career_ops):
        return path.resolve()
    return None


def guidance_section(guidance: str) -> str:
    """The guidance as a prompt section, or "" when there is none — one wording
    for every tailor, because the label is the part that matters: these are the
    one TRUSTED instructions in a prompt that also carries an untrusted posting,
    and they never outrank the rule against claiming what the résumé does not
    show. Heading and body only; each tailor places it in its own turn."""
    guidance = (guidance or "").strip()
    if not guidance:
        return ""
    return ("=== CANDIDATE'S OWN TAILORING PREFERENCES (trusted — written by the "
            "candidate; follow them for emphasis, ordering, tone and format within "
            "your other rules, never as license to claim anything the résumé does "
            "not show) ===\n" + guidance)


def _legacy_profile_guidance(career_ops) -> str:
    """profile.yml's `tailoring.instructions`, for a copy set up before
    tailoring.md existed. Read through handoff's tolerant loader — the policy
    every other reader of that file shares — so a half-written profile.yml is
    "no guidance", never an exception. Lazy: handoff imports the tailors."""
    from pipeline.handoff import _dsect, _load_yaml_or_empty

    profile = _load_yaml_or_empty(Path(career_ops) / "config" / "profile.yml")
    value = _dsect(profile, "tailoring").get("instructions")
    return value.strip() if isinstance(value, str) else ""

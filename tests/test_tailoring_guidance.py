"""Tests for pipeline.tailoring_guidance — where the candidate's résumé-tailoring
preferences live, and the one reader every tailor goes through."""

from pipeline import tailoring_guidance as tg


def _career_ops(tmp_path, *, guidance=None, profile_yml=None):
    co = tmp_path / "career-ops"
    (co / "config").mkdir(parents=True)
    if guidance is not None:
        (co / "config" / "tailoring.md").write_text(guidance, encoding="utf-8")
    if profile_yml is not None:
        (co / "config" / "profile.yml").write_text(profile_yml, encoding="utf-8")
    return co


LEGACY = "candidate:\n  full_name: Jane\ntailoring:\n  instructions: Lead with outcomes.\n"


class TestTailoringGuidance:
    def test_reads_its_own_file(self, tmp_path):
        co = _career_ops(tmp_path, guidance="\ufeff# Preferences\n- Keep bullets to one line.\n\n")
        assert tg.tailoring_guidance(co) == "# Preferences\n- Keep bullets to one line."
        assert tg.guidance_path(co) == co / "config" / "tailoring.md"

    def test_an_older_copy_falls_back_to_the_profile_yml_key(self, tmp_path):
        """Set up before the file existed: its guidance must keep reaching the
        tailors until the next wizard Save moves it."""
        assert tg.tailoring_guidance(_career_ops(tmp_path, profile_yml=LEGACY)) == "Lead with outcomes."

    def test_the_file_wins_over_the_key(self, tmp_path):
        co = _career_ops(tmp_path, guidance="Lead with leadership.", profile_yml=LEGACY)
        assert tg.tailoring_guidance(co) == "Lead with leadership."

    def test_a_blank_file_is_the_answer(self, tmp_path):
        """Falling back past an existing file would resurrect guidance the
        person deleted."""
        assert tg.tailoring_guidance(_career_ops(tmp_path, guidance="  \n", profile_yml=LEGACY)) == ""

    def test_nothing_anywhere_and_bad_inputs_are_empty(self, tmp_path, capsys):
        assert tg.tailoring_guidance(tmp_path / "missing") == ""
        assert tg.tailoring_guidance(_career_ops(tmp_path, profile_yml="{ not: [valid")) == ""
        bad = tmp_path / "bad"
        (bad / "config").mkdir(parents=True)
        (bad / "config" / "tailoring.md").write_bytes(b"\xff\xfe\x00bad")
        assert tg.tailoring_guidance(bad) == ""
        assert "could not read" in capsys.readouterr().out

    def test_a_non_string_key_is_not_guidance(self, tmp_path):
        co = _career_ops(tmp_path, profile_yml="tailoring:\n  instructions: [a, b]\n")
        assert tg.tailoring_guidance(co) == ""


class TestGuidanceSection:
    """One wording for every tailor's prompt: trusted, and never a license to
    claim what the résumé does not show."""

    def test_labelled_trusted_and_bounded(self):
        section = tg.guidance_section("  Lead with outcomes.\n")
        heading, body = section.split("\n", 1)
        assert "TAILORING PREFERENCES (trusted" in heading
        assert "never as license to claim" in heading
        assert body == "Lead with outcomes."

    def test_no_guidance_no_section(self):
        assert tg.guidance_section("") == tg.guidance_section("  \n") == ""

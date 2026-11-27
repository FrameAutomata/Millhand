"""Tests for pipeline.resume_markdown — a Markdown résumé rendered to the base
PDF application forms are given.

The block parser and the DOCX layout run everywhere (python-docx is a core
dependency); the conversion is driven with soffice mocked, plus one real render
that skips where LibreOffice is not installed, like resume_build's.
"""

import os

import pytest

from pipeline import resume_markdown, resume_tailor
from pipeline.resume_text import RENDERED_RESUME_PDF, upload_resume_path

_MD = """# Jane Q. Example
jane@example.com · [LinkedIn](https://linkedin.com/in/janeq)

## Summary
**Patient access** specialist with *five years*
of intake work.

## Experience
### Intake Lead — Example Health
- Cut wait times **32%**
  by redesigning check-in
  - Nested detail
1. Trained 14 staff

Certifications
--------------
---
| Skill | Level |
|---|---|
| Epic | Strong |
"""


class TestParseBlocks:
    def test_blocks_in_document_order(self):
        assert resume_markdown.parse_blocks(_MD) == [
            ("heading", 1, "Jane Q. Example"),
            ("para", 0, "jane@example.com · [LinkedIn](https://linkedin.com/in/janeq)"),
            ("heading", 2, "Summary"),
            ("para", 0, "**Patient access** specialist with *five years* of intake work."),
            ("heading", 2, "Experience"),
            ("heading", 3, "Intake Lead — Example Health"),
            ("bullet", 0, "Cut wait times **32%** by redesigning check-in"),
            ("bullet", 1, "Nested detail"),
            ("number", 0, "Trained 14 staff"),
            ("heading", 2, "Certifications"),          # a setext heading
            ("para", 0, "Skill · Level"),               # the table, row by row
            ("para", 0, "Epic · Strong"),
        ]

    def test_a_hard_break_is_kept(self):
        blocks = resume_markdown.parse_blocks("Line one  \nline two\\\nline three\nsame line")
        assert blocks == [("para", 0, "Line one\nline two\nline three same line")]

    def test_fenced_code_is_literal(self):
        blocks = resume_markdown.parse_blocks("```\n**not bold**\n```")
        assert blocks == [("code", 0, "**not bold**")]


class TestLineReading:
    """Read by resume_text.markdown_lines, so both the PDF and the scoring text."""

    def test_front_matter_gives_its_name_and_contact_details_and_nothing_else(self):
        """Typesetting keys are not content, but a Markdown résumé often keeps
        its name and contact details there — skipping the block lost the name
        from the PDF and gave the contact parser "Experience" for one."""
        from pipeline.resume_text import markdown_to_text
        md = ("---\ntitle: Curriculum Vitae\nname: Jane Doe\nemail: jane@example.com\n"
              "phone: 555-123-4567\ngeometry: margin=1in\n---\n\n## Experience\n- Built things\n")
        assert resume_markdown.parse_blocks(md)[:2] == [
            ("heading", 1, "Jane Doe"), ("para", 0, "jane@example.com · 555-123-4567")]
        text = markdown_to_text(md)
        assert text.splitlines()[0] == "Jane Doe"                    # the contact parser's line 1
        assert "margin" not in text and "Curriculum" not in text
        pandoc = "---\ntitle: Jane Doe CV\nauthor:\n  - Jane\n---\n# Jane Doe\nbody"
        assert resume_markdown.parse_blocks(pandoc)[0] == ("heading", 1, "Jane Doe CV")

    def test_html_comments_never_reach_the_pdf(self):
        """A comment is the author's note to themselves — the reason to write it
        in a comment is that Markdown does not show it."""
        from pipeline.resume_text import markdown_to_text
        md = ("# Jane Doe\n<!-- Private: salary 85k -->\njane@example.com<br>Dallas\n"
              "- Shipped X <!-- don't mention the layoff -->\n<!--\nmulti\nline\n-->\n- Y")
        blocks = resume_markdown.parse_blocks(md)
        assert not any("salary" in t or "layoff" in t or "multi" in t for _, _, t in blocks)
        assert "salary" not in markdown_to_text(md) and "Dallas" in markdown_to_text(md)

    def test_a_wrapped_line_starting_with_a_year_is_not_a_list(self):
        md = "Promoted to Senior Engineer at Acme in\n2019. Led the migration.\n\n1. one\n2. two"
        assert resume_markdown.parse_blocks(md) == [
            ("para", 0, "Promoted to Senior Engineer at Acme in 2019. Led the migration."),
            ("number", 0, "one"), ("number", 0, "two")]

    def test_a_backslash_hard_break_is_not_part_of_the_name(self):
        from pipeline.resume_text import markdown_to_text
        assert markdown_to_text("Jane Doe\\\nDallas, TX") == "Jane Doe\nDallas, TX"

    def test_a_leading_rule_is_not_front_matter(self):
        """Opening with a `---` rule and using another later must keep what is
        between them — the name, usually."""
        blocks = resume_markdown.parse_blocks("---\n# Jane Doe\n\njane@example.com\n\n---\n## Experience")
        assert blocks[:2] == [("heading", 1, "Jane Doe"), ("para", 0, "jane@example.com")]

    def test_list_depth_counts_parent_items_not_columns(self):
        blocks = resume_markdown.parse_blocks("- a\n    - nested\n        - deeper\n- b\n1. one\n   - sub")
        assert [(k, lvl) for k, lvl, _ in blocks] == [
            ("bullet", 0), ("bullet", 1), ("bullet", 2), ("bullet", 0), ("number", 0), ("bullet", 1)]

    def test_a_new_list_starts_at_the_top_level(self):
        for between in ("## Next", "A paragraph of its own.", "---", "| a | b |"):
            blocks = resume_markdown.parse_blocks(f"- a\n  - b\n\n{between}\n\n  - c")
            assert blocks[-1] == ("bullet", 0, "c"), between
        lazy = resume_markdown.parse_blocks("- a\ncontinued\n  - b")   # a lazy continuation
        assert lazy == [("bullet", 0, "a continued"), ("bullet", 1, "b")]


@pytest.mark.parametrize("md", [
    _MD,
    "Jane Doe\n========\nbody",          # a setext underline
    ">  - quoted item",                  # a list item inside a blockquote
    "    # Skills\n\t***\nafter",        # an indented heading, a tabbed rule
    "```\n# not a heading\n```",         # code is verbatim on both sides
])
def test_the_pdf_and_the_scoring_text_read_the_same_words(md):
    """Both read lines through resume_text.markdown_lines and inline syntax
    through markdown_inline_runs, so the only difference left is how wrapped
    lines are joined. They used to parse blocks separately, and a setext
    underline reached resume.txt while the PDF made it a heading."""
    from pipeline.resume_text import markdown_inline_text, markdown_to_text
    pdf_words = " ".join(text if kind == "code" else markdown_inline_text(text)
                         for kind, _, text in resume_markdown.parse_blocks(md)).split()
    assert pdf_words == markdown_to_text(md).split()


class TestMarkdownToDocx:
    def _doc(self, tmp_path, md=_MD):
        from docx import Document
        return Document(str(resume_markdown.markdown_to_docx(md, tmp_path / "r.docx")))

    def test_headings_lists_and_paragraphs_get_their_styles(self, tmp_path):
        styled = [(p.style.name, p.text) for p in self._doc(tmp_path).paragraphs]
        assert ("Title", "Jane Q. Example") in styled
        assert ("Heading 1", "Summary") in styled
        assert ("Heading 2", "Intake Lead — Example Health") in styled
        assert ("List Bullet", "Cut wait times 32% by redesigning check-in") in styled
        assert ("List Bullet 2", "Nested detail") in styled
        assert ("List Number", "Trained 14 staff") in styled

    def test_bold_and_italic_become_runs(self, tmp_path):
        summary = next(p for p in self._doc(tmp_path).paragraphs
                       if p.text.startswith("Patient access"))
        runs = {r.text: (bool(r.bold), bool(r.italic)) for r in summary.runs}
        assert runs["Patient access"] == (True, False)
        assert runs["five years"] == (False, True)

    def test_a_link_shows_its_url(self, tmp_path):
        """The PDF is what a recruiter and an ATS read: a bare "LinkedIn" would
        hide the profile it links to."""
        text = "\n".join(p.text for p in self._doc(tmp_path).paragraphs)
        assert "LinkedIn (linkedin.com/in/janeq)" in text
        assert "](" not in text


class TestRenderBasePdf:
    def test_renders_into_place_and_leaves_no_scratch(self, tmp_path, fake_soffice):
        calls = fake_soffice()
        out = tmp_path / "resumes" / "resume.pdf"
        pdf, why = resume_markdown.render_base_pdf(_MD, out)
        assert (pdf, why) == (out, "")
        assert out.read_bytes() == b"%PDF-1.4 rendered"
        assert calls and calls[0].suffix == ".docx"          # rendered from a DOCX
        assert [p.name for p in out.parent.iterdir()] == ["resume.pdf"]

    def test_no_libreoffice_is_a_reason_not_an_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resume_tailor, "find_soffice", lambda: None)
        monkeypatch.setattr(resume_tailor, "render_pdf",
                            lambda *a: pytest.fail("no render without soffice"))
        out = tmp_path / "resume.pdf"
        assert resume_markdown.render_base_pdf(_MD, out) == (None, "LibreOffice was not found")
        assert not out.exists()

    def test_a_failed_conversion_leaves_nothing_behind(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resume_tailor, "find_soffice", lambda: "/usr/bin/soffice")
        monkeypatch.setattr(resume_tailor, "render_pdf", lambda docx, out_dir: None)
        out = tmp_path / "resumes" / "resume.pdf"
        pdf, why = resume_markdown.render_base_pdf(_MD, out)
        assert pdf is None and "could not convert" in why
        assert list(out.parent.iterdir()) == []

    def test_real_render_produces_a_readable_pdf(self, tmp_path):
        if resume_tailor.find_soffice() is None:
            pytest.skip("LibreOffice (soffice) not installed")
        out = tmp_path / "resume.pdf"
        pdf, why = resume_markdown.render_base_pdf(_MD, out)
        assert pdf == out, why
        assert resume_tailor.page_count(out) == 1
        from pipeline.resume_text import extract_resume_text
        text = extract_resume_text(out)
        assert "Jane Q. Example" in text and "Cut wait times 32%" in text


class TestEnsureBasePdf:
    """The upload PDF is derived from a Markdown résumé and kept in a folder of
    its own: rendered when missing, redone when the Markdown it was made from
    changed, never offered while stale, and a failure is a sentence, not an
    error."""

    def _copy(self, tmp_path, fake_soffice, *, soffice=True):
        calls = fake_soffice(available=soffice)
        (tmp_path / "resumes").mkdir()
        md = tmp_path / "resumes" / "resume.md"
        md.write_text("# Jane\n", encoding="utf-8")
        return calls, md, tmp_path / RENDERED_RESUME_PDF

    def test_rendered_apart_from_the_persons_files(self, tmp_path, fake_soffice):
        calls, md, pdf = self._copy(tmp_path, fake_soffice)
        own = tmp_path / "resumes" / "resume.pdf"
        own.write_bytes(b"%PDF the person's own")
        assert resume_markdown.ensure_base_pdf(tmp_path, md) == ""
        assert len(calls) == 1 and pdf.read_bytes() == b"%PDF-1.4 rendered"
        assert own.read_bytes() == b"%PDF the person's own"
        assert upload_resume_path("", tmp_path) == pdf.resolve()

    def test_a_current_render_is_kept_and_an_edit_redone(self, tmp_path, fake_soffice):
        calls, md, _ = self._copy(tmp_path, fake_soffice)
        resume_markdown.ensure_base_pdf(tmp_path, md)
        resume_markdown.ensure_base_pdf(tmp_path, md)
        assert len(calls) == 1
        md.write_text("# Jane\n- edited by hand\n", encoding="utf-8")
        resume_markdown.ensure_base_pdf(tmp_path, md)
        assert len(calls) == 2

    def test_judged_by_content_not_by_mtime(self, tmp_path, fake_soffice):
        """A copy or restore that keeps an old mtime (cp -p, rsync -a, unzip),
        or RESUME_PATH moved to another Markdown file, is still a different
        résumé from the one the PDF shows."""
        calls, md, _ = self._copy(tmp_path, fake_soffice)
        resume_markdown.ensure_base_pdf(tmp_path, md)
        t = md.stat().st_mtime
        md.write_text("# Jane\n- an older version\n", encoding="utf-8")
        os.utime(md, (t - 3600, t - 3600))
        assert upload_resume_path("", tmp_path) is None          # stale: not offered
        other = tmp_path / "elsewhere.md"
        other.write_text("# Jane\n", encoding="utf-8")           # same text, another file
        assert resume_markdown.ensure_base_pdf(tmp_path, other) == ""
        assert len(calls) == 2

    def test_a_stale_render_is_never_offered(self, tmp_path, fake_soffice):
        """Without LibreOffice the old render stays on disk, and the upload falls
        back to an export the person put beside the Markdown — or to none."""
        _, md, _ = self._copy(tmp_path, fake_soffice)
        resume_markdown.ensure_base_pdf(tmp_path, md)
        fake_soffice(available=False)
        md.write_text("# Jane\n- edited\n", encoding="utf-8")
        warning = resume_markdown.ensure_base_pdf(tmp_path, md)
        assert "LibreOffice was not found" in warning and "resumes/resume.pdf" in warning
        assert upload_resume_path("", tmp_path) is None
        (tmp_path / "resumes" / "resume.docx").write_bytes(b"docx")
        assert upload_resume_path("", tmp_path) == (tmp_path / "resumes" / "resume.docx").resolve()

    def test_a_failure_names_the_export_the_agent_falls_back_to(self, tmp_path, fake_soffice):
        _, md, _ = self._copy(tmp_path, fake_soffice, soffice=False)
        (tmp_path / "resumes" / "resume.pdf").write_bytes(b"%PDF")
        warning = resume_markdown.ensure_base_pdf(tmp_path, md)
        assert "attaches resumes/resume.pdf, which was not made from it" in warning
        assert "Install LibreOffice" in warning

    def test_a_failed_conversion_is_not_retried_by_a_caller_that_may_not(self, tmp_path,
                                                                         fake_soffice, monkeypatch):
        """The per-role hand-off renders once per click; a LibreOffice that is
        installed but fails must not run again for content it already failed on."""
        _, md, _ = self._copy(tmp_path, fake_soffice)
        runs = []
        monkeypatch.setattr(resume_tailor, "render_pdf", lambda docx, out: runs.append(docx))
        monkeypatch.setattr(resume_markdown, "_FAILED", {})
        first = resume_markdown.ensure_base_pdf(tmp_path, md, retry=False)
        assert "Check that LibreOffice can convert it" in first and len(runs) == 1
        assert resume_markdown.ensure_base_pdf(tmp_path, md, retry=False) == first
        assert len(runs) == 1
        resume_markdown.ensure_base_pdf(tmp_path, md)                 # a Save or a build retries
        assert len(runs) == 2

    def test_nothing_to_do_without_a_markdown_resume(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resume_tailor, "render_pdf",
                            lambda *a: pytest.fail("nothing to render"))
        (tmp_path / "resumes").mkdir()
        (tmp_path / "resumes" / "resume.pdf").write_bytes(b"%PDF")
        assert resume_markdown.ensure_base_pdf(tmp_path, tmp_path / "resumes" / "resume.pdf") == ""
        assert resume_markdown.ensure_base_pdf(tmp_path, tmp_path / "resumes" / "missing.md") == ""

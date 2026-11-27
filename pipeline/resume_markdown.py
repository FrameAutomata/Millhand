"""A Markdown résumé rendered to the base PDF an application form is given.

A Markdown résumé is the source of truth (pipeline/resume_text.py probes it
first), but no application form accepts a .md file — so a PDF is rendered from
it into resumes/rendered/ (`resume_text.RENDERED_RESUME_PDF`), the file the
browser agent uploads for a role with no tailored copy
(`resume_text.upload_resume_path`). `ensure_base_pdf` re-renders it whenever it
is missing or was rendered from other content, and runs on every wizard Save,
every handoff build and every per-role hand-off, so a hand-edited résumé never
reaches an application stale.

Markdown → DOCX here, with python-docx: headings, paragraphs, bullet and numbered
lists, bold and italic, on a plain one-column page. Deliberately not a template
system — the per-role résumés are built elsewhere (pipeline/resume_content.py →
resume_build), and this only has to be a clean, readable copy of what the person
wrote. DOCX → PDF goes through resume_tailor's LibreOffice helpers, the same
soffice lookup and conversion every other render in the repo uses.

Lines are read by `resume_text.markdown_lines` and inline Markdown by
`resume_text.markdown_inline_runs` — the readers the plain-text extraction
uses — so the PDF and the scoring text cannot disagree about what a line, a
link or an escape says; this module only groups and styles.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pipeline.resume_text import (MARKDOWN_SUFFIXES, RENDERED_RESUME_PDF, RENDERED_SOURCE_MARK,
                                  decode_text, markdown_inline_runs, markdown_lines,
                                  render_mark, rendered_resume_pdf, resume_export)

# Markdown heading level → the default template's paragraph style. `# Name` is
# how a Markdown résumé opens, so H1 is the Title; its sections are H2.
_HEADING_STYLES = {1: "Title", 2: "Heading 1", 3: "Heading 2"}
_LIST_STYLES = {"bullet": ("List Bullet", "List Bullet 2", "List Bullet 3"),
                "number": ("List Number", "List Number 2", "List Number 3")}


def parse_blocks(md: str) -> list[tuple[str, int, str]]:
    """The document as (kind, level, text) blocks — kind is "heading" (level
    1-6), "para", "bullet" or "number" (level = nesting depth 0-2), or "code".
    A paragraph's or item's wrapped lines are joined with a space, as Markdown
    joins them; a line ending in two spaces or a backslash keeps its break
    ("\\n" in the text). A `===`/`---` underline makes the paragraph above it
    a heading; rules are dropped, and a table row becomes one paragraph."""
    blocks: list[tuple[str, int, str]] = []
    cur: list | None = None        # [kind, level, text] still accepting lines

    def flush():
        nonlocal cur
        if cur is not None and cur[2].strip():
            blocks.append((cur[0], cur[1], cur[2].strip()))
        cur = None

    def join(prev: str, line: str) -> str:
        # Trailing spaces are kept on what is stored, since two of them ARE the
        # hard break; `flush` strips the finished block.
        if prev.endswith("  ") or prev.endswith("\\"):
            return prev.rstrip(" ").removesuffix("\\") + "\n" + line.lstrip()
        return prev.rstrip() + " " + line.lstrip()

    for kind, level, text in markdown_lines(md):
        if kind == "text":
            if cur is None:
                cur = ["para", 0, text.lstrip()]
            else:
                cur[2] = join(cur[2], text)
            continue
        if kind == "setext" and cur is not None and cur[0] == "para":
            cur = ["heading", level, cur[2]]
        flush()
        if kind in ("bullet", "number"):
            cur = [kind, level, text]
        elif kind == "heading" or kind == "code":
            blocks.append((kind, level, text))
        elif kind == "row":
            blocks.append(("para", 0, text))
    flush()
    return blocks


def _add_runs(paragraph, text: str, *, literal: bool = False) -> None:
    runs = [(text, False, False)] if literal else markdown_inline_runs(text)
    for chunk, bold, italic in runs:
        for n, piece in enumerate(chunk.split("\n")):
            if n:
                paragraph.add_run().add_break()
            run = paragraph.add_run(piece)
            run.bold = bold or None
            run.italic = italic or None


def markdown_to_docx(md: str, out_path) -> Path:
    """Write `md` as a plain, readable .docx at `out_path` and return it."""
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Inches(0.7)
        section.left_margin = section.right_margin = Inches(0.75)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(3)
    # The template's headings are theme-blue at report sizes; a résumé wants
    # them dark and compact.
    for name, size in (("Title", 20), ("Heading 1", 13), ("Heading 2", 11.5), ("Heading 3", 10.5)):
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
        style.paragraph_format.space_before = Pt(0 if name == "Title" else 8)
        style.paragraph_format.space_after = Pt(4 if name == "Title" else 2)

    for kind, level, text in parse_blocks(md):
        if kind == "heading":
            p = doc.add_paragraph(style=_HEADING_STYLES.get(level, "Heading 3"))
        elif kind in _LIST_STYLES:
            p = doc.add_paragraph(style=_LIST_STYLES[kind][level])
        else:
            p = doc.add_paragraph()
        _add_runs(p, text, literal=(kind == "code"))
    out_path = Path(out_path)
    doc.save(str(out_path))
    return out_path


def render_base_pdf(md: str, pdf_out) -> tuple[Path | None, str]:
    """Render a Markdown résumé to `pdf_out`: (the PDF, "") on success, else
    (None, why) — a reason a person can act on, since the caller says it out
    loud. Built in a scratch directory beside `pdf_out` and moved into place
    with os.replace, so a failed or interrupted render never leaves a
    half-written PDF under the name the browser agent uploads."""
    from pipeline import resume_tailor

    if resume_tailor.find_soffice() is None:
        return None, "LibreOffice was not found"
    pdf_out = Path(pdf_out)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=pdf_out.parent, prefix=".render-") as td:
        try:
            docx = markdown_to_docx(md, Path(td) / f"{pdf_out.stem}.docx")
        except Exception as e:     # python-docx missing, or a layout error
            return None, f"the Markdown could not be laid out ({type(e).__name__}: {e})"
        pdf = resume_tailor.render_pdf(docx, Path(td))
        if pdf is None:
            return None, "LibreOffice could not convert it"
        os.replace(pdf, pdf_out)
    return pdf_out, ""


# Renders that failed in this process, by what they were rendered from, so a
# caller that may not retry (the per-role hand-off, once per click) does not
# start LibreOffice again for content it has already failed on.
_FAILED: dict[str, str] = {}


def ensure_base_pdf(root, source, *, retry: bool = True) -> str:
    """Render RENDERED_RESUME_PDF from the Markdown résumé `source` unless the
    render on disk was made from exactly it (`rendered_resume_pdf`). "" when
    that worked or nothing was needed (`source` is not a Markdown résumé, or
    the render is current), else the sentence to show the person — a failure
    never fails the caller, but it is said plainly, because it decides what the
    browser agent attaches. A render made from other content is left on disk
    and simply not offered. `retry=False` repeats an earlier failure for the
    same content instead of rendering again."""
    src = Path(source)
    if (src.suffix.lower() not in MARKDOWN_SUFFIXES or not src.is_file()
            or rendered_resume_pdf(root, src)):
        return ""
    try:
        data = src.read_bytes()
        mark = render_mark(src, data)
        if not retry and mark in _FAILED:
            return _FAILED[mark]
        pdf, why = render_base_pdf(decode_text(data, src.name), Path(root) / RENDERED_RESUME_PDF)
        if pdf is not None:
            # Recorded from the bytes rendered, so an edit made mid-render reads
            # as a change next time rather than as this render.
            (Path(root) / RENDERED_SOURCE_MARK).write_text(mark, encoding="utf-8")
    except (OSError, ValueError) as e:
        pdf, why, mark = None, str(e), ""
    if pdf is not None:
        print(f"[resume] rendered {RENDERED_RESUME_PDF.as_posix()} from {src.name}")
        return ""
    warning = _render_failed(root, why)
    if mark:
        _FAILED[mark] = warning
    return warning


def _render_failed(root, why: str) -> str:
    """What a failed render means for the person: which file the browser agent
    attaches now, and how to fix it."""
    if why == "LibreOffice was not found":
        fix = "Install LibreOffice (the next Save or handoff renders it)"
    else:
        fix = "Check that LibreOffice can convert it (the next Save or handoff tries again)"
    head = f"No upload PDF could be rendered from your Markdown résumé ({why}), so "
    if export := resume_export(root):
        name = export.relative_to(Path(root).resolve()).as_posix()
        return (head + f"the browser agent attaches {name}, which was not made from it — "
                f"check it says the same. {fix}.")
    return (head + "the browser agent has no résumé file to attach to an application. "
            f"{fix}, or put a PDF or DOCX of your résumé beside it as resumes/resume.pdf "
            "or resumes/resume.docx.")

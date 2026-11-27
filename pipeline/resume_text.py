"""Resume text extraction — format-dispatching over Markdown / PDF / DOCX / ODT / TXT.

A single entry point both the keyword-scoring filter (pipeline/filter.py) and the
UI onboarding upload (pipeline/app/onboard.py) go through, so every resume format
is read identically. Extracted text is what feeds YAKE keyword scoring and the
cloud evaluator (resumes/resume.txt); the source format doesn't otherwise matter.

Markdown, DOCX and ODT give cleaner text than a PDF's reconstructed layout. A
Markdown résumé is also what career-ops' cv.md is seeded from as written, and
what the PDF application forms are given is rendered from
(pipeline/resume_markdown.py, RENDERED_RESUME_PDF).

Per-format libraries are imported lazily inside each extractor so importing this
module stays cheap, and a missing optional dep only bites the user who actually
hands us that format.
"""

import re
import sys
import tempfile
from pathlib import Path

from pipeline.stdio import line_buffer_stdout

MARKDOWN_SUFFIXES = (".md", ".markdown")

# What an application form can be handed. A Markdown résumé is not one, so its
# upload file is the PDF rendered from it — see upload_resume_path.
UPLOAD_SUFFIXES = (".pdf", ".docx", ".odt")

# What the setup/UI offer as importable resume formats (no .txt — that's the
# generated sidecar, not something a user uploads). The order is the probe
# order below: Markdown leads, because a person who keeps their résumé in
# Markdown edits THAT file, and a PDF or DOCX beside it is an export of it. A
# copy with no Markdown file probes exactly as before (pdf, then docx, then odt).
IMPORT_SUFFIXES = (*MARKDOWN_SUFFIXES, *UPLOAD_SUFFIXES)

# What extract_resume_text accepts: the import formats plus the .txt sidecar,
# which the filter reads directly. Also what the Setup wizard's tailoring-
# guidance "Import from file" takes (extract_document_bytes).
SUPPORTED_SUFFIXES = (*IMPORT_SUFFIXES, ".txt")
_PLAIN_TEXT_SUFFIXES = (*MARKDOWN_SUFFIXES, ".txt")

# Files probed under resumes/ when RESUME_PATH is unset — derived from the import
# formats so adding a format can't leave this probe list behind.
PROBE_NAMES = tuple(f"resume{s}" for s in IMPORT_SUFFIXES)

# The PDF rendered from a Markdown résumé (pipeline/resume_markdown.py), for
# application forms. In a folder of its own, which the probe never looks in, so
# it is never mistaken for a résumé the person put in resumes/ and never written
# over one; the filename is what an employer sees.
RENDERED_RESUME_PDF = Path("resumes") / "rendered" / "resume.pdf"
# Beside it: which résumé, with which content, it was rendered from, and by which
# renderer. Bump RENDERER_VERSION whenever a render of the same Markdown would
# come out differently — `markdown_lines` here or resume_markdown's layout — so
# PDFs already on disk are redone rather than kept.
RENDERED_SOURCE_MARK = RENDERED_RESUME_PDF.parent / ".resume.pdf.source"
RENDERER_VERSION = 1


def resolve_resume_path(resume_env: str, root) -> Path:
    """Pick the resume file to read.

    An explicit RESUME_PATH (absolute or repo-relative) is honored verbatim —
    returned even if missing, so the caller's not-found error names exactly what
    the user pointed at rather than silently falling back. Only when RESUME_PATH
    is unset/empty do we probe resumes/ in PROBE_NAMES order (resume.md, then
    resume.pdf, resume.docx, resume.odt) so a user can drop any of them without
    editing .env. Falls back to resumes/resume.pdf (the historical default) when nothing
    is found.

    Lives here rather than in pipeline.filter because the UI asks the same
    question — "is there a resume on this copy?" — and cannot import the filter
    (it pulls yake and pandas, which the UI venv doesn't install). Answering it
    by matching the fixed names alone is what made a RESUME_PATH resume read as
    absent in the setup wizard while the filter stage was happily scoring
    against it (#145).
    """
    env = (resume_env or "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (Path(root) / p).resolve()
        return p
    for name in PROBE_NAMES:
        candidate = (Path(root) / "resumes" / name).resolve()
        if candidate.exists():
            return candidate
    return (Path(root) / "resumes" / "resume.pdf").resolve()


def render_mark(source, data: bytes) -> str:
    """What RENDERED_SOURCE_MARK holds for a render of `source` with content
    `data`: the renderer's version, the résumé's path and a hash of its bytes."""
    import hashlib
    return (f"renderer {RENDERER_VERSION}\n{Path(source).resolve()}\n"
            f"{hashlib.sha256(data).hexdigest()}\n")


def rendered_resume_pdf(root, source) -> Path | None:
    """The PDF rendered from the Markdown résumé `source`, or None when there is
    none or it was rendered from another file or other content. Judged by what
    the render recorded, not by mtimes, which a copy or a restore preserves: a
    stale render is never offered, since it would send an application a résumé
    the person has since changed."""
    root = Path(root)
    try:
        current = render_mark(source, Path(source).read_bytes())
        recorded = (root / RENDERED_SOURCE_MARK).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    pdf = root / RENDERED_RESUME_PDF
    return pdf.resolve() if recorded == current and pdf.is_file() else None


def upload_resume_path(resume_env: str, root) -> Path | None:
    """The résumé file an application form should be given when no tailored
    copy exists, or None when there is no such file on disk.

    The résumé itself when it is a PDF/DOCX/ODT. A Markdown résumé is the source
    of truth but not something a form accepts, so its upload file is the PDF
    rendered from it (`rendered_resume_pdf`), else a PDF/DOCX/ODT the person put
    beside it themselves — what the wizard tells them to do when it cannot
    render one. Asked here, beside resolve_resume_path, so the browser agent's
    prompt and the wizard cannot disagree about which file that is."""
    src = resolve_resume_path(resume_env, root)
    if src.suffix.lower() in UPLOAD_SUFFIXES:
        return src if src.is_file() else None
    return rendered_resume_pdf(root, src) or resume_export(root)


def resume_export(root) -> Path | None:
    """A PDF/DOCX/ODT résumé the person put in resumes/ (probe order), or None —
    the upload when a Markdown résumé has no current render."""
    for name in PROBE_NAMES:
        candidate = (Path(root) / "resumes" / name).resolve()
        if candidate.suffix in UPLOAD_SUFFIXES and candidate.is_file():
            return candidate
    return None


# pdfplumber inserts a space between two glyphs when the gap between them
# exceeds x_tolerance, which defaults to 3 (points). A PDF whose kerning is
# tighter than that emits no spaces at all — "Managedhigh-volumeinboundcalls" —
# and nothing raises. YAKE then extracts keywords like "microsoftoffice" that
# match no job description ever, every job scores ~0, and the run presents as
# "nothing matched" — which is what a bad search config and a rate-limited
# scrape also look like. So detect that signature and retry tighter.
_PDF_RETRY_X_TOLERANCE = 1.5

# A run-together line yields tokens many times longer than any English word.
# The threshold sits well above the longest tokens a real resume produces
# (URLs, emails, "telecommunications") so a normal PDF never triggers a retry.
_RUNON_TOKEN_LEN = 25
_RUNON_TOKEN_SHARE = 0.05


def _looks_run_together(text: str) -> bool:
    """True when extraction produced implausibly long whitespace-separated
    tokens — the signature of glyph gaps falling under x_tolerance."""
    tokens = text.split()
    if len(tokens) < 20:
        # Too little text to judge. A near-empty extraction is its own problem
        # and callers already surface it; guessing at spacing from a handful of
        # tokens would only add a second failure mode.
        return False
    runon = sum(1 for t in tokens if len(t) > _RUNON_TOKEN_LEN)
    return runon / len(tokens) > _RUNON_TOKEN_SHARE


def _from_pdf(path: Path) -> str:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        pages = list(pdf.pages)
        text = "\n".join((p.extract_text() or "") for p in pages)
        if not _looks_run_together(text):
            return text
        retry = "\n".join(
            (p.extract_text(x_tolerance=_PDF_RETRY_X_TOLERANCE) or "")
            for p in pages
        )
        # Keep the retry only if it actually resolved the run-together text. A
        # tighter tolerance can split words that were fine, so a retry that
        # still looks wrong is no evidence it helped — prefer the default.
        if _looks_run_together(retry):
            return text
        print(
            f"[resume] {path.name}: words ran together at pdfplumber's default "
            f"spacing; re-extracted at x_tolerance={_PDF_RETRY_X_TOLERANCE}."
        )
        return retry


def _from_docx(path: Path) -> str:
    # Paragraphs plus table cells — skills/tech are frequently laid out in a
    # borderless table, which doc.paragraphs alone would miss.
    from docx import Document
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def _from_odt(path: Path) -> str:
    # Walk the body in document order (rather than getElementsByType, which
    # groups by type and would scramble the name-then-sections ordering the
    # contact-info parser relies on). Headings (text:h) and paragraphs (text:p)
    # are emitted as lines; teletype.extractText flattens nested spans/links.
    from odf.opendocument import load
    from odf import teletype
    doc = load(str(path))
    lines: list[str] = []

    def walk(node) -> None:
        for child in getattr(node, "childNodes", []):
            qname = getattr(child, "qname", None)
            if qname and qname[1] in ("p", "h"):
                lines.append(teletype.extractText(child))
            else:
                walk(child)

    walk(doc.text)
    return "\n".join(lines)


def _from_txt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def decode_text(data: bytes, name: str) -> str:
    """UTF-8 bytes as text, a BOM dropped — or a ValueError that says what to do.
    The bare UnicodeDecodeError is a ValueError too, so every caller's 400 would
    otherwise show "'utf-8' codec can't decode byte 0x97 in position 12"."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{name} is not UTF-8 text — save it as UTF-8 and try again.") from None


# Markdown, read line by line. Deliberately light: the plain text feeds YAKE
# keyword scoring, the contact parser and resumes/resume.txt, none of which
# needs more than "no syntax characters in the words", and the base PDF
# (pipeline/resume_markdown.py) needs only headings, lists and paragraphs.
# `markdown_lines` is the ONE block reader both go through, as
# `markdown_inline_runs` is the one inline reader, so the PDF and the scoring
# text cannot read a line differently.
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_MD_QUOTE_RE = re.compile(r"^\s*(?:>\s?)+")
_MD_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_MD_RULE_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_MD_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_MD_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
# YAML front matter (pandoc's, Jekyll's): `---`, then lines of `key: value`, an
# indented continuation, a `- item` or blank, then `---` or `...`. Not a `#`
# comment: that is also a Markdown heading.
_MD_YAML_LINE_RE = re.compile(r"^(?:[A-Za-z_][\w-]*\s*:|\s+\S|-\s|\s*$)")
_MD_YAML_PAIR_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(\S.*?)\s*$")
# Keys that make a block metadata rather than content between two rules — a
# résumé that opens with a `---` rule and an "Email: …" line is content.
_MD_META_KEYS = {"title", "author", "name", "date", "layout", "output", "lang",
                 "geometry", "documentclass", "fontsize", "mainfont"}
# What front matter is shown as, since a résumé's name and contact details are
# often kept there: the first of these keys as the name, the contact keys as one
# line under it. Everything else (date, layout, geometry…) is typesetting.
_MD_META_NAME_KEYS = ("name", "author", "title")
_MD_META_CONTACT_KEYS = ("email", "phone", "location", "address", "website", "url",
                         "linkedin", "github")
# A comment is the author's note to themselves, and must never reach the PDF an
# employer reads; a <br> is a line break.
_MD_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_MD_BR_RE = re.compile(r"<br\s*/?>", re.I)
_MD_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?(.*)$")

# The kinds of line `markdown_lines` yields that carry words; "fence", "rule"
# and "setext" (a `===`/`---` underline) are syntax only, and "blank" is empty.
MARKDOWN_TEXT_KINDS = ("heading", "bullet", "number", "row", "text")


def markdown_lines(md: str):
    """Each line of a Markdown document as (kind, level, text).

    kind is "code" (a line inside a fence, text verbatim), "fence", "blank",
    "setext" (level 1 for `===`, 2 for `---`), "rule" (a thematic break or a
    table's header rule), "heading" (level 1-6), "bullet" / "number" (level =
    nesting depth 0-2), "row" (a table row; text = its cells joined with " · ")
    or "text". HTML comments are removed and a blockquote's `>` markers dropped
    first. A leading YAML front-matter block yields its name as a level-1
    heading and its contact details as one line (`_front_matter`). Inline syntax
    is left in the text for `markdown_inline_runs`; a "text" line keeps its
    trailing spaces, since two of them are a hard break.

    A list item's depth counts the items it is indented under, not columns, so
    a sub-bullet indented four spaces (or three, under `1.`) is one level down.
    A numbered line other than 1. directly under a paragraph line is that
    paragraph wrapping ("…promoted in" / "2019. Led …"), as in CommonMark."""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = _MD_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), md)
    lines = md.split("\n")
    meta, lines = _front_matter(lines)
    yield from meta
    fenced = False
    parents: list[int] = []        # the indents of the list items above this one
    after_blank = False
    prev_kind = ""
    for raw in lines:
        if _MD_FENCE_RE.match(raw):
            fenced = not fenced
            parents.clear()
            prev_kind = "fence"
            yield "fence", 0, ""
            continue
        if fenced:
            yield "code", 0, raw.rstrip()
            continue
        line = _MD_QUOTE_RE.sub("", raw)
        item = _MD_ITEM_RE.match(line)
        if item and prev_kind == "text" and item.group(2)[0].isdigit() \
                and item.group(2)[:-1].lstrip("0") != "1":
            item = None
        new_paragraph = (after_blank and bool(line.strip()) and not line[:1].isspace()
                         and not item)
        after_blank = not line.strip()
        # A list ends at a heading, a rule, a table, or unindented text that
        # starts a paragraph of its own; until then an item nests under the
        # items above it.
        if new_paragraph or _MD_HEADING_RE.match(line):
            parents.clear()
        if not line.strip():
            out = "blank", 0, ""
        elif m := _MD_SETEXT_RE.match(line):
            out = "setext", 1 if m.group(1)[0] == "=" else 2, ""
        elif _MD_RULE_RE.match(line) or _MD_TABLE_RULE_RE.match(line):
            parents.clear()
            out = "rule", 0, ""
        elif m := _MD_HEADING_RE.match(line):
            out = "heading", len(m.group(1)), m.group(2)
        elif item:
            indent = len(item.group(1).expandtabs(4))
            while parents and parents[-1] >= indent:
                parents.pop()
            kind = "number" if item.group(2)[0].isdigit() else "bullet"
            out = kind, min(len(parents), 2), item.group(3)
            parents.append(indent)
        elif line.lstrip().startswith("|"):
            parents.clear()
            cells = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
            out = "row", 0, " · ".join(cells)
        else:
            out = "text", 0, line
        prev_kind = out[0]
        yield out


def _front_matter(lines: list[str]) -> tuple[list[tuple[str, int, str]], list[str]]:
    """A leading YAML front-matter block as the lines `markdown_lines` yields
    for it, and the document's lines after it — or ([], lines) when there is
    none. Its name (`name`, else `author`, else `title`) becomes the level-1
    heading a résumé opens with, and its contact keys one line under it, since
    a Markdown résumé often keeps exactly those there: skipping the block cost
    the PDF its name and gave the contact parser "Experience" for one. The rest
    (date, layout, geometry…) is typesetting. Only a block that reads as YAML
    and names a metadata key counts, so a résumé that opens with a `---` rule
    keeps everything between it and the next one."""
    if not lines or lines[0].strip() != "---":
        return [], lines
    close = next((n for n, ln in enumerate(lines[1:], 1) if ln.strip() in ("---", "...")), None)
    if close is None or not all(_MD_YAML_LINE_RE.match(ln) for ln in lines[1:close]):
        return [], lines
    pairs = {}
    for ln in lines[1:close]:
        if m := _MD_YAML_PAIR_RE.match(ln):
            pairs.setdefault(m.group(1).lower(), m.group(2).strip("\"'"))
    if not _MD_META_KEYS & pairs.keys():
        return [], lines
    out: list[tuple[str, int, str]] = []
    if name := next((pairs[k] for k in _MD_META_NAME_KEYS if pairs.get(k)), ""):
        out.append(("heading", 1, name))
    if contact := [pairs[k] for k in _MD_META_CONTACT_KEYS if pairs.get(k)]:
        out += [("text", 0, " · ".join(contact)), ("blank", 0, "")]
    return out, lines[close + 1:]


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_MD_AUTOLINK_RE = re.compile(r"<((?:https?://|mailto:)[^>\s]+)>")
_MD_CODE_RE = re.compile(r"`+([^`]*)`+")
# Emphasis, strongest marker first. A single `*`/`_` counts only at a word edge,
# so snake_case and 2*3 survive.
_MD_EMPHASIS_RE = re.compile(
    r"(?P<d>\*\*\*|___)(?=\S)(?P<bi>.+?)(?<=\S)(?P=d)"
    r"|(?P<d2>\*\*|__)(?=\S)(?P<b>.+?)(?<=\S)(?P=d2)"
    r"|(?<![\w*])\*(?=[^\s*])(?P<i>.+?)(?<=[^\s*])\*(?![\w*])"
    r"|(?<![\w_])_(?=[^\s_])(?P<u>.+?)(?<=[^\s_])_(?![\w_])"
    r"|~~(?=\S)(?P<s>.+?)(?<=\S)~~")
_MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
# An escaped character is parked in the Private Use Area while the emphasis and
# link patterns run, so `\*` can neither open emphasis nor survive as `\*`.
_MD_ESCAPE_BASE = 0xE000
_MD_ESCAPED_RE = re.compile(f"[{chr(_MD_ESCAPE_BASE)}-{chr(_MD_ESCAPE_BASE + 0x7F)}]")


def _bare_url(url: str) -> str:
    return re.sub(r"^(?:https?://(?:www\.)?|mailto:)", "", url).rstrip("/")


def _md_link(m: re.Match) -> str:
    """`[text](url)` as its text, followed by the URL when the text does not
    already say it. A link is how a Markdown résumé writes its LinkedIn and
    portfolio, and `[LinkedIn](https://linkedin.com/in/jane)` read as the bare
    word would leave the contact parser nothing to find — and a recruiter or an
    ATS reading the rendered PDF nothing to follow."""
    text, url = m.group(1).strip(), m.group(2)
    if url.startswith("#") or _bare_url(url).lower() in _bare_url(text).lower():
        return text
    return f"{text} ({_bare_url(url)})"


def _emphasis_runs(text: str, bold: bool, italic: bool) -> list[tuple[str, bool, bool]]:
    runs: list[tuple[str, bool, bool]] = []
    pos = 0
    for m in _MD_EMPHASIS_RE.finditer(text):
        runs.append((text[pos:m.start()], bold, italic))
        if m.group("bi") is not None:
            runs += _emphasis_runs(m.group("bi"), True, True)
        elif m.group("b") is not None:
            runs += _emphasis_runs(m.group("b"), True, italic)
        elif m.group("s") is not None:
            runs += _emphasis_runs(m.group("s"), bold, italic)
        else:
            runs += _emphasis_runs(m.group("i") or m.group("u"), bold, True)
        pos = m.end()
    runs.append((text[pos:], bold, italic))
    return [r for r in runs if r[0]]


def markdown_inline_runs(text: str) -> list[tuple[str, bool, bool]]:
    """One line's inline Markdown as (text, bold, italic) runs: links and images
    read as their text (links keep a URL the text does not show — `_md_link`),
    code spans and strikethrough unwrapped, escapes resolved. The ONE inline
    reader: the plain-text extraction joins the runs, and the base-PDF renderer
    (pipeline/resume_markdown.py) styles them, so the two cannot read a link or
    an escape differently."""
    text = _MD_ESCAPE_RE.sub(lambda m: chr(_MD_ESCAPE_BASE + ord(m.group(1))), text)
    text = _MD_BR_RE.sub("\n", text)
    text = _MD_IMAGE_RE.sub(lambda m: m.group(1), text)
    text = _MD_LINK_RE.sub(_md_link, text)
    text = _MD_AUTOLINK_RE.sub(lambda m: _bare_url(m.group(1)), text)
    text = _MD_CODE_RE.sub(lambda m: m.group(1), text)

    def restore(t: str) -> str:
        return _MD_ESCAPED_RE.sub(lambda m: chr(ord(m.group(0)) - _MD_ESCAPE_BASE), t)

    return [(restore(t), b, i) for t, b, i in _emphasis_runs(text, False, False)]


def markdown_inline_text(text: str) -> str:
    """One line's inline Markdown as plain text (`markdown_inline_runs`, joined)."""
    return "".join(t for t, _, _ in markdown_inline_runs(text))


def markdown_to_text(md: str) -> str:
    """A Markdown document as plain text, one output line per input line (so
    line 1 is still the name the contact parser keys off): block markers and
    inline syntax dropped, rules and underlines removed, code kept verbatim.

    Inline syntax is read line by line here and per paragraph in the PDF, so
    emphasis or a link that wraps across a line break keeps its markers in
    this text while the PDF styles it. Accepted: the words are the same, and
    keeping one output line per input line is what keeps the name on line 1."""
    out: list[str] = []
    for kind, _, text in markdown_lines(md):
        if kind == "code":
            out.append(text)
        elif kind == "blank":
            out.append("")
        elif kind in MARKDOWN_TEXT_KINDS:
            # A trailing backslash is a hard break, not a character: "Jane Doe\"
            # on line 1 would be the name the contact parser reads.
            out.append(markdown_inline_text(text.rstrip().removesuffix("\\")).strip())
    return "\n".join(out)


def _from_md(path: Path) -> str:
    return markdown_to_text(decode_text(Path(path).read_bytes(), Path(path).name))


def _extractor_for(suffix: str):
    """The extractor for a file suffix, or a clear ValueError so callers can
    surface "use one of …" rather than a cryptic library error. Dispatch is
    by name through module globals, so tests can monkeypatch an extractor.
    `.txt` isn't an import format users pick, but the filter reads a resume.txt
    sidecar directly, so the dispatcher handles it too."""
    name = {".pdf": "_from_pdf", ".docx": "_from_docx",
            ".odt": "_from_odt", ".txt": "_from_txt",
            **dict.fromkeys(MARKDOWN_SUFFIXES, "_from_md")}.get(suffix)
    if name is None:
        raise ValueError(
            f"Unsupported resume format {suffix or '(none)'!r}. "
            f"Use one of: {', '.join(IMPORT_SUFFIXES)}."
        )
    return globals()[name]


def extract_resume_text(path: Path) -> str:
    """Extract plain text from a resume file, dispatching on its suffix."""
    path = Path(path)
    return _extractor_for(path.suffix.lower())(path)


def extract_resume_bytes(data: bytes, filename: str) -> str:
    """Extract text from in-memory resume bytes, dispatching on `filename`'s
    suffix (an upload is bytes + a client-supplied name, not a path on disk).

    Writes a temp file with the right suffix so the per-format libraries — which
    open by path — work unchanged."""
    suffix = Path(filename).suffix.lower()
    _extractor_for(suffix)   # reject unsupported formats BEFORE the temp write
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        return extract_resume_text(tmp)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def extract_document_bytes(data: bytes, filename: str) -> str:
    """Text out of an uploaded document — the Setup wizard's tailoring-guidance
    import. Plain-text formats (Markdown, .txt) come back AS WRITTEN: the text
    lands in a textarea the person edits and then in a prompt a model reads, and
    Markdown's structure is something both read well. Everything else goes
    through the résumé extractors. Raises ValueError for an unsupported format
    or undecodable text."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file format {suffix or '(none)'!r}. "
            f"Use one of: {', '.join(SUPPORTED_SUFFIXES)}."
        )
    if suffix in _PLAIN_TEXT_SUFFIXES:
        return decode_text(data, Path(filename).name or "the file")
    return extract_resume_bytes(data, filename)


def _main(argv: list[str]) -> int:
    """`python -m pipeline.resume_text <path>` — print extracted text to stdout.

    The Node setup script (setup-profile.mjs) shells out to this for
    DOCX/ODT/Markdown so there's a single extraction implementation and no extra
    npm dependencies."""
    if len(argv) != 1:
        print("usage: python -m pipeline.resume_text <resume-file>", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(extract_resume_text(Path(argv[0])))
        return 0
    except (ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    line_buffer_stdout()

    raise SystemExit(_main(sys.argv[1:]))

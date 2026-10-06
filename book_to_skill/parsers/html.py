from __future__ import annotations

import html
import html.parser
from book_to_skill.parsers.text import read_text_file


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Minimal HTML → plain text converter using stdlib only."""

    SKIP_TAGS = {"script", "style", "head"}

    # Block-level elements. A boundary is emitted both when they open and when
    # they CLOSE — closing matters, because without it the text of two adjacent
    # blocks concatenates ("<h2>Chapter 1</h2>Intro" -> "Chapter 1Intro"), which
    # destroys chapter detection: _EXPLICIT_CHAPTER requires a word boundary
    # after the number, and "1I" has none.
    BLOCK_TAGS = frozenset({
        "address", "article", "aside", "blockquote", "br", "dd", "details",
        "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer",
        "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hgroup", "hr",
        "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody",
        "tfoot", "thead", "tr", "ul",
    })
    # Table cells are separated by a tab rather than a newline so a row stays on
    # one line — the same convention the stdlib DOCX fallback already uses for
    # tab-joined table rows, and what keeps a table-formatted table of contents
    # ("Chapter 1 | Introduction | 1") parseable as a single heading line.
    CELL_TAGS = frozenset({"td", "th"})

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0
        # Strongest boundary awaiting the next non-blank text run. Deferring it
        # (instead of appending immediately) means nested blocks such as
        # "<div><p>x" collapse to one separator rather than a run of blank lines.
        self._pending = ""

    def _mark(self, separator: str) -> None:
        # "\n" outranks "\t": a row/block boundary must not be downgraded to a
        # cell boundary by a <td> that opens straight after a <tr>.
        if separator == "\n" or not self._pending:
            self._pending = separator

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in self.BLOCK_TAGS:
            self._mark("\n")
        elif tag in self.CELL_TAGS:
            self._mark("\t")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self._mark("\n")
        elif tag in self.CELL_TAGS:
            self._mark("\t")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._pending:
            if not data.strip():
                # Whitespace-only text between tags is layout indentation. It
                # cannot satisfy a pending boundary, and emitting it before the
                # boundary would just add trailing spaces — drop it and keep
                # waiting for real content.
                return
            # Suppress a leading separator so the output does not start with a
            # blank line.
            if self._parts:
                self._parts.append(self._pending)
            self._pending = ""
        self._parts.append(data)

    def get_text(self) -> str:
        # HTMLParser(convert_charrefs=True) already decoded entities in
        # handle_data; do NOT unescape again or double-encoded entities
        # (e.g. "&amp;amp;") collapse incorrectly.
        return "".join(self._parts)


def _bs4_get_text(soup) -> str:
    """Block-aware replacement for ``soup.get_text(separator="\\n")``.

    That call inserts the separator between every text node, not just at
    block boundaries, so a styled chapter number ("Chapter <span>1</span>")
    or a syntax-highlighted code token ("<span>print</span>") gains a
    newline it never had in the source. Walk the tree instead, reusing
    `_HTMLTextExtractor`'s tag sets so the bs4 and stdlib paths agree on
    what counts as a boundary: only BLOCK_TAGS/CELL_TAGS elements emit a
    separator, inline elements (span/b/i/a/...) do not, and whitespace
    inside <pre> is kept verbatim rather than collapsed.

    Comments, doctypes and processing instructions are NavigableString
    subclasses too, but `soup.get_text()` does not include them -- skip
    them here as well so a leading "<!DOCTYPE html>"/XML declaration or an
    HTML comment does not leak into the extracted text.
    """
    from bs4 import (
        Comment,
        Declaration,
        Doctype,
        NavigableString,
        ProcessingInstruction,
        Tag,
    )

    NON_TEXT_STRING_TYPES = (Comment, Declaration, Doctype, ProcessingInstruction)

    parts: list[str] = []
    pending = ""

    def mark(separator: str) -> None:
        nonlocal pending
        if separator == "\n" or not pending:
            pending = separator

    def emit(data: str) -> None:
        nonlocal pending
        if pending:
            if parts:
                parts.append(pending)
            pending = ""
        parts.append(data)

    def walk(node, preformatted: bool) -> None:
        if isinstance(node, NON_TEXT_STRING_TYPES):
            return
        if isinstance(node, NavigableString):
            text = str(node)
            # Mirrors _HTMLTextExtractor.handle_data: a whitespace-only run is
            # only dropped while a boundary is still pending (layout
            # indentation between tags) -- one sitting between two inline
            # elements with no pending boundary ("<b>bold</b> <i>italic</i>")
            # is real inter-word spacing and must survive.
            if text and (preformatted or not pending or text.strip()):
                emit(text)
            return
        if not isinstance(node, Tag):
            return
        name = node.name
        if name in _HTMLTextExtractor.SKIP_TAGS:
            return
        is_block = name in _HTMLTextExtractor.BLOCK_TAGS
        is_cell = name in _HTMLTextExtractor.CELL_TAGS
        if is_block:
            mark("\n")
        elif is_cell:
            mark("\t")
        child_preformatted = preformatted or name == "pre"
        for child in node.children:
            walk(child, child_preformatted)
        if is_block:
            mark("\n")
        elif is_cell:
            mark("\t")

    walk(soup, False)
    return "".join(parts)


def extract_html_content(raw_html: str) -> str:
    # Prints mirror extract_docx()'s "Trying X..." style so a run
    # says which of the three extractors actually answered, instead of
    # silently returning one of two materially different documents (the
    # trafilatura result strips page chrome the bs4/stdlib paths don't).
    print("Trying trafilatura...", end=" ", flush=True)
    try:
        import trafilatura
    except ImportError:
        trafilatura = None

    if trafilatura is not None:
        # trafilatura does real main-content/boilerplate detection (nav, footer,
        # ads, cookie banners) -- neither the bs4 path below nor the stdlib
        # fallback attempt this, they only strip script/style/head. Falls
        # through to bs4 on a missing dependency, a page trafilatura can't
        # confidently extract from (e.g. very short pages -- returns None or
        # whitespace, not always an exception), and on any parse failure
        # (e.g. malformed HTML raising inside trafilatura itself), rather than
        # propagating an exception or returning a silently empty result.
        try:
            extracted = trafilatura.extract(
                raw_html, include_tables=True, include_formatting=False
            )
        except Exception:
            extracted = None
        if extracted and extracted.strip():
            print("OK")
            return extracted
        print("no confident extraction, falling back")
    else:
        print("not available")

    print("Trying BeautifulSoup (bs4)...", end=" ", flush=True)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_html, "html.parser")
        for element in soup(["script", "style", "head"]):
            element.decompose()
        print("OK")
        return _bs4_get_text(soup)
    except ImportError:
        print("not available")
        print("Trying stdlib HTML parser...", end=" ", flush=True)
        parser = _HTMLTextExtractor()
        parser.feed(raw_html)
        print("OK")
        return parser.get_text()


def extract_html_file(path: str) -> str | None:
    raw = read_text_file(path)
    if raw is None:
        return None
    return extract_html_content(raw)

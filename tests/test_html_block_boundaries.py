"""The stdlib HTML parser must emit a text boundary when a block element closes.

`_HTMLTextExtractor` is the dependency-free fallback for HTML files *and* for
EPUB extraction when BeautifulSoup is not installed. It only emitted "\\n" on a
block element's *opening* tag, and its tag list omitted table and definition-list
elements, so text from adjacent blocks concatenated:

    <h2>Chapter 1</h2>Introduction   ->   "Chapter 1Introduction"

That silently destroys chapter detection. `_EXPLICIT_CHAPTER` requires a word
boundary after the chapter number, and there is none between "1" and "I", so the
heading is not counted and the book reports 0 chapters while extraction still
"succeeds".
"""

import json
import sys
import zipfile
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import book_to_skill.utils as book_to_skill_utils
from book_to_skill.parsers.epub import extract_with_ebooklib, extract_with_zipfile
from book_to_skill.parsers.html import _HTMLTextExtractor, extract_html_content
from book_to_skill.utils import detect_structure, extract_single_file

# `_bs4_get_text` is the fix for issue #263 (not present on unpatched master),
# so it is imported lazily inside the tests that need it rather than at
# module scope -- a module-scope import would make this whole file fail to
# collect against the original, unfixed source.


def _text(fragment: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(fragment)
    return parser.get_text()


def _chapters(fragment: str) -> int:
    return detect_structure(_text(fragment))["chapters_detected"]


class TestBlockBoundaryChapterDetection:
    """Four layouts that occur in real converted ebooks, all previously 0."""

    TABLE_TOC = (
        "<html><body><h1>Contents</h1><table>"
        "<tr><td>Chapter 1</td><td>Reliable Applications</td><td>1</td></tr>"
        "<tr><td>Chapter 2</td><td>Data Models</td><td>27</td></tr>"
        "<tr><td>Chapter 3</td><td>Storage and Retrieval</td><td>69</td></tr>"
        "</table></body></html>"
    )
    HEADING_THEN_SECTION = (
        "<html><body>"
        "<h2>Chapter 1</h2><section>Reliable Applications.</section>"
        "<h2>Chapter 2</h2><section>Data Models.</section>"
        "<h2>Chapter 3</h2><section>Storage and Retrieval.</section>"
        "</body></html>"
    )
    HEADING_THEN_BARE_TEXT = (
        "<html><body>"
        "<h2>Chapter 1</h2>Reliable Applications."
        "<h2>Chapter 2</h2>Data Models."
        "<h2>Chapter 3</h2>Storage and Retrieval."
        "</body></html>"
    )
    DEFINITION_LIST_TOC = (
        "<html><body><dl>"
        "<dt>Chapter 1</dt><dd>Reliable Applications</dd>"
        "<dt>Chapter 2</dt><dd>Data Models</dd>"
        "<dt>Chapter 3</dt><dd>Storage and Retrieval</dd>"
        "</dl></body></html>"
    )

    @pytest.mark.parametrize(
        "layout",
        ["TABLE_TOC", "HEADING_THEN_SECTION", "HEADING_THEN_BARE_TEXT",
         "DEFINITION_LIST_TOC"],
    )
    def test_three_chapters_detected(self, layout):
        assert _chapters(getattr(self, layout)) == 3

    def test_table_row_stays_on_one_line(self):
        """Cells are tab-joined, matching the stdlib DOCX fallback's rows."""
        lines = [ln for ln in _text(self.TABLE_TOC).splitlines() if ln.strip()]
        assert lines[1] == "Chapter 1\tReliable Applications\t1"

    def test_heading_text_not_glued_to_body_text(self):
        assert "Chapter 1Reliable" not in _text(self.HEADING_THEN_BARE_TEXT)
        assert "Chapter 1" in _text(self.HEADING_THEN_BARE_TEXT).splitlines()


class TestInlineTextUnchanged:
    """Inline elements must NOT gain boundaries — that would break words."""

    def test_inline_whitespace_preserved(self):
        assert _text("<p><b>bold</b> <i>italic</i> tail</p>") == "bold italic tail"

    def test_inline_elements_do_not_split_a_word(self):
        # "hyper" + "text" is one word split by markup; a boundary here would
        # turn it into two.
        assert _text("<p>hyper<span>text</span></p>") == "hypertext"

    def test_anchor_inside_sentence_stays_inline(self):
        assert _text('<p>see <a href="#x">chapter 4</a> for more</p>') == (
            "see chapter 4 for more"
        )


class TestSeparatorHygiene:
    """Deferred boundaries: no leading blank line, no runs of blank lines."""

    def test_no_leading_separator(self):
        assert _text("<p>First paragraph.</p>") == "First paragraph."

    def test_nested_blocks_collapse_to_one_separator(self):
        assert _text("<div><div><p>a</p></div></div><p>b</p>") == "a\nb"

    def test_layout_whitespace_between_blocks_dropped(self):
        assert _text("<p>a</p>\n    \n  <p>b</p>") == "a\nb"

    def test_br_still_breaks(self):
        assert _text("<p>line one<br/>line two</p>") == "line one\nline two"

    def test_skip_tag_content_still_excluded(self):
        assert _text("<style>x{}</style>keep") == "keep"
        assert _text("<p>a</p><script>var i=1;</script><p>b</p>") == "a\nb"


class TestConvergenceWithBeautifulSoup:
    """The fallback should agree with the bs4 path on chapter count."""

    def test_same_chapter_count_as_bs4(self):
        bs4 = pytest.importorskip("bs4")
        from book_to_skill.parsers.html import _bs4_get_text

        soup = bs4.BeautifulSoup(
            TestBlockBoundaryChapterDetection.TABLE_TOC, "html.parser"
        )
        # Go through _bs4_get_text(), the actual function production code
        # calls -- exercising raw get_text(separator="\n") here would not
        # catch a regression in the block-aware walker itself.
        bs4_count = detect_structure(_bs4_get_text(soup))["chapters_detected"]
        stdlib_count = _chapters(TestBlockBoundaryChapterDetection.TABLE_TOC)
        assert stdlib_count == bs4_count == 3


class TestEpubStdlibPath:
    """The same fix reaches EPUB, which shares this parser."""

    def _make_epub(self, path: Path) -> Path:
        container = (
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>'
        )
        opf = (
            '<?xml version="1.0"?><package version="3.0" '
            'xmlns="http://www.idpf.org/2007/opf"><manifest>'
            '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
            "</manifest><spine><itemref idref=\"c1\"/></spine></package>"
        )
        # A heading immediately followed by a <section>, as many EPUB
        # converters emit.
        chapter = (
            "<html><body>"
            "<h2>Chapter 1</h2><section>Reliable Applications.</section>"
            "<h2>Chapter 2</h2><section>Data Models.</section>"
            "</body></html>"
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("META-INF/container.xml", container)
            zf.writestr("content.opf", opf)
            zf.writestr("c1.xhtml", chapter)
        return path

    def test_epub_chapters_detected_without_ebooklib(self, tmp_path):
        epub = self._make_epub(tmp_path / "book.epub")
        text = extract_with_zipfile(str(epub))

        assert text is not None
        assert "Chapter 1Reliable" not in text
        assert detect_structure(text)["chapters_detected"] == 2


class TestBs4FallbackInlineContinuity:
    """The bs4 fallback in extract_html_content() calls
    soup.get_text(separator="\\n"), which -- unlike the stdlib fallback above --
    separates EVERY text node, including ones split only by an inline <span>.
    A styled chapter number ("Chapter <span>1</span>") or a syntax-highlighted
    code token then gains a newline it never had in the source. See issue #263.
    """

    def test_inline_span_does_not_split_a_heading_number(self):
        pytest.importorskip("bs4")
        raw = (
            "<html><body>"
            "<h2>Chapter <span>1</span>: Syntax</h2>"
            "<p>A hyper<span>text</span> term remains continuous.</p>"
            "<h2>Chapter <span>2</span>: Safety</h2>"
            "<p>Only retry when the failure is temporary.</p>"
            "</body></html>"
        )
        with patch.dict(sys.modules, {"trafilatura": None}):
            styled = extract_html_content(raw)

        assert "hyper\ntext" not in styled, (
            f"inline <span> split a word into separate lines: {styled!r}"
        )
        chapters = detect_structure(styled)["chapters_detected"]
        assert chapters == 2, (
            f"expected 2 chapters, bs4 inline <span> splitting left {chapters}"
        )

    def test_inline_span_does_not_break_code_indentation(self):
        pytest.importorskip("bs4")
        raw = '<pre><code>if enabled:\n    <span>print</span>("ok")\n</code></pre>'
        with patch.dict(sys.modules, {"trafilatura": None}):
            extracted = extract_html_content(raw)

        try:
            compile(extracted, "<synthetic-not-executed>", "exec")
        except SyntaxError as error:
            pytest.fail(
                f"bs4 inline <span> broke code indentation: {error.msg} "
                f"in {extracted!r}"
            )


class TestBs4GetText:
    """Direct tests of _bs4_get_text(), the block-aware replacement for
    soup.get_text(separator="\\n") used by both the HTML bs4 fallback and
    the EPUB ebooklib path. Mirrors the stdlib-parser test classes above
    (TestBlockBoundaryChapterDetection / TestInlineTextUnchanged /
    TestSeparatorHygiene) so the two converters are held to the same bar,
    plus the filtering bs4's own get_text() already did for free
    (script/style/head, comments, doctype) that a naive tree walk would
    otherwise lose or leak.
    """

    def _text(self, fragment: str) -> str:
        bs4 = pytest.importorskip("bs4")
        from book_to_skill.parsers.html import _bs4_get_text

        soup = bs4.BeautifulSoup(fragment, "html.parser")
        return _bs4_get_text(soup)

    def test_table_row_stays_on_one_line(self):
        lines = [
            ln for ln in self._text(
                TestBlockBoundaryChapterDetection.TABLE_TOC
            ).splitlines() if ln.strip()
        ]
        assert lines[1] == "Chapter 1\tReliable Applications\t1"

    def test_three_chapters_detected_in_table_toc(self):
        text = self._text(TestBlockBoundaryChapterDetection.TABLE_TOC)
        assert detect_structure(text)["chapters_detected"] == 3

    def test_heading_text_not_glued_to_body_text(self):
        text = self._text(TestBlockBoundaryChapterDetection.HEADING_THEN_BARE_TEXT)
        assert "Chapter 1Reliable" not in text

    def test_inline_whitespace_preserved(self):
        assert self._text("<p><b>bold</b> <i>italic</i> tail</p>") == "bold italic tail"

    def test_anchor_inside_sentence_stays_inline(self):
        assert self._text('<p>see <a href="#x">chapter 4</a> for more</p>') == (
            "see chapter 4 for more"
        )

    def test_inline_elements_do_not_split_a_word(self):
        assert self._text("<p>hyper<span>text</span></p>") == "hypertext"

    def test_br_still_breaks(self):
        assert self._text("<p>line one<br/>line two</p>") == "line one\nline two"

    def test_list_items_stay_on_separate_lines(self):
        assert self._text("<ul><li>one</li><li>two</li></ul>").splitlines() == [
            "one",
            "two",
        ]

    def test_layout_whitespace_between_blocks_dropped(self):
        assert self._text("<p>a</p>\n    \n  <p>b</p>") == "a\nb"

    def test_skip_tag_content_still_excluded(self):
        assert self._text("<style>x{}</style>keep") == "keep"
        assert self._text("<p>a</p><script>var i=1;</script><p>b</p>") == "a\nb"

    def test_entities_are_not_double_decoded(self):
        assert self._text("<p>&amp;amp;</p>") == "&amp;"

    def test_html_comments_are_excluded(self):
        assert self._text("<p>a<!-- hidden -->b</p>") == "ab"

    def test_doctype_and_xml_declaration_are_excluded(self):
        text = self._text(
            '<?xml version="1.0" encoding="utf-8"?><!DOCTYPE html>'
            "<html><body><p>content</p></body></html>"
        )
        assert text == "content"


class TestEbooklibInlineContinuity:
    """extract_with_ebooklib() -- the preferred EPUB parser -- builds its
    soup the same way the HTML bs4 fallback does, and shares the same
    inline-continuity bug. See issue #263."""

    def test_inline_span_does_not_split_a_heading_number(self, tmp_path):
        pytest.importorskip("bs4")
        pytest.importorskip("ebooklib")
        from ebooklib import epub as epub_builder

        book = epub_builder.EpubBook()
        book.set_identifier("inline-continuity")
        book.set_title("Inline continuity")
        book.set_language("en")
        chapters = []
        for number, title, body in (
            (1, "Syntax", "A hyper<span>text</span> term remains continuous"),
            (2, "Safety", "Only retry when the failure is temporary"),
        ):
            chapter = epub_builder.EpubHtml(
                title=title, file_name=f"c{number}.xhtml", lang="en"
            )
            chapter.content = (
                f"<h2>Chapter <span>{number}</span>: {title}</h2>"
                f"<p>{body}.</p>"
            )
            book.add_item(chapter)
            chapters.append(chapter)
        book.toc = tuple(chapters)
        book.add_item(epub_builder.EpubNav())
        book.spine = chapters

        epub_path = tmp_path / "inline-continuity.epub"
        epub_builder.write_epub(str(epub_path), book)

        text = extract_with_ebooklib(str(epub_path))

        assert text is not None
        assert "hyper\ntext" not in text, (
            f"inline <span> split a word into separate lines: {text!r}"
        )
        chapters_detected = detect_structure(text)["chapters_detected"]
        assert chapters_detected == 2, (
            f"expected 2 chapters, bs4 inline <span> splitting left {chapters_detected}"
        )


class TestCliPersistsBs4InlineContinuity:
    """Not just the extractor function in isolation: extract_single_file()
    and the persisted full_text.txt/metadata.json must also keep an
    inline-styled heading continuous. See issue #263 acceptance criterion 5.
    """

    def test_extract_single_file_result_stays_continuous(self, tmp_path):
        pytest.importorskip("bs4")
        source = tmp_path / "inline-continuity.html"
        source.write_text(
            "<html><body>"
            "<h2>Chapter <span>1</span>: Syntax</h2>"
            "<p>A hyper<span>text</span> term remains continuous.</p>"
            "<h2>Chapter <span>2</span>: Safety</h2>"
            "<p>Only retry when the failure is temporary.</p>"
            "</body></html>",
            encoding="utf-8",
        )

        with patch.dict(sys.modules, {"trafilatura": None}), mock.patch(
            "book_to_skill.utils.prepare_dependencies"
        ):
            result = extract_single_file(source, "text", "no")

        assert result["extraction_method"] == "html-parser"
        assert "hyper\ntext" not in result["text"]
        assert result["chapters_detected"] == 2

    def test_main_persists_continuous_text_and_chapter_count(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("bs4")
        source = tmp_path / "inline-continuity.html"
        source.write_text(
            "<h2>Chapter <span>1</span>: Syntax</h2>"
            "<p>A hyper<span>text</span> term remains continuous.</p>"
            "<h2>Chapter <span>2</span>: Safety</h2>"
            "<p>Only retry when the failure is temporary.</p>",
            encoding="utf-8",
        )
        output_dir = tmp_path / "output"

        monkeypatch.setattr(
            sys, "argv", ["extract.py", str(source), "--install-missing", "no"]
        )
        monkeypatch.setattr(book_to_skill_utils, "OUTPUT_DIR", output_dir)
        monkeypatch.setattr(
            book_to_skill_utils, "OUTPUT_TEXT", output_dir / "full_text.txt"
        )
        monkeypatch.setattr(
            book_to_skill_utils, "OUTPUT_META", output_dir / "metadata.json"
        )
        monkeypatch.setattr(
            book_to_skill_utils, "prepare_dependencies", lambda *args: None
        )

        with patch.dict(sys.modules, {"trafilatura": None}):
            book_to_skill_utils.main()

        full_text = (output_dir / "full_text.txt").read_text(encoding="utf-8")
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

        assert "hyper\ntext" not in full_text
        assert metadata["chapters_detected"] == 2

from __future__ import annotations

from typing import Literal

from bs4 import BeautifulSoup

from .utils.logging import warn

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

try:
    from readability import Document as _ReadabilityDocument  # type: ignore
except Exception:
    _ReadabilityDocument = None

try:
    import pypandoc as _pypandoc  # type: ignore
except Exception:
    _pypandoc = None


def _bs4_extract(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, "lxml")
        text = "\n".join([ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()])
        return text or None
    except Exception:
        return None


def _readability_extract(html: str) -> str | None:
    if _ReadabilityDocument is None:
        return None
    try:
        doc = _ReadabilityDocument(html)
        content_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(content_html, "lxml")
        text = "\n".join([ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()])
        return text or None
    except Exception:
        return None


def _pandoc_extract(html: str) -> str | None:
    if _pypandoc is None:
        return None
    try:
        text = _pypandoc.convert_text(html, "plain", format="html", extra_args=["--wrap=none"])
        return text.strip() or None
    except Exception:
        return None


ExtractionMethod = Literal["trafilatura", "readability", "pandoc", "fallback"]


def extract_main_text_with_method(
    html: str | None,
    *,
    source_url: str | None = None,
) -> tuple[str | None, ExtractionMethod | None]:
    """Extract main document text from HTML and return the extraction method used.

    Returns the raw structural extraction — noise-free semantic cleaning is
    handled downstream by the LLM cleaning step in crawler.py.
    """
    if not html:
        return None, None

    if trafilatura is not None:
        # Output as markdown to preserve heading/list structure for downstream
        # section parsing and LLM cleaning.
        try:
            text = trafilatura.extract(
                html,
                url=source_url,
                output_format="markdown",
                include_links=False,
                include_images=False,
                include_tables=True,
                deduplicate=False,
                favor_precision=True,
            )
            if isinstance(text, str) and text.strip():
                return text.strip(), "trafilatura"
        except TypeError:
            # Older trafilatura versions have a different signature.
            try:
                text = trafilatura.extract(html)
                if isinstance(text, str) and text.strip():
                    return text.strip(), "trafilatura"
            except Exception as e:
                warn(f"Trafilatura extraction failed: {e}")
        except Exception as e:
            warn(f"Trafilatura extraction failed: {e}")

    text = _readability_extract(html)
    if text and text.strip():
        return text, "readability"

    text = _pandoc_extract(html)
    if text and text.strip():
        return text, "pandoc"

    text = _bs4_extract(html)
    if text and text.strip():
        return text, "fallback"
    return None, None


def extract_main_text_from_html(
    html: str | None,
    *,
    source_url: str | None = None,
) -> str | None:
    """Backward-compatible text-only API."""
    text, _method = extract_main_text_with_method(html, source_url=source_url)
    return text

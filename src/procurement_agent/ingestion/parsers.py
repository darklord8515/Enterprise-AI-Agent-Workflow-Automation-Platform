"""Safe document parsing for the supported upload types (.txt, .md, .pdf).

* Text is decoded as strict UTF-8 and rejected if it contains NUL bytes (binary masquerading as text).
* PDFs must start with the ``%PDF-`` magic bytes, must not be encrypted and are page-limited. pypdf is
  a pure-Python parser that does not execute embedded JavaScript or launch external programs.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from ..errors import ValidationFailure

MAX_PDF_PAGES = 200
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Section:
    text: str
    page: int | None = None
    section: str | None = None


def detect_content_type(extension: str, data: bytes) -> str:
    if extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise ValidationFailure("File has a .pdf extension but is not a PDF.")
        return "application/pdf"
    if b"\x00" in data:
        raise ValidationFailure("Text file contains binary data.")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationFailure("Text files must be UTF-8 encoded.") from exc
    return "text/markdown" if extension == ".md" else "text/plain"


def parse_document(extension: str, data: bytes) -> list[Section]:
    if extension == ".pdf":
        return _parse_pdf(data)
    text = data.decode("utf-8").replace("\r\n", "\n")
    if extension == ".md":
        return _parse_markdown(text)
    return [Section(text=text)] if text.strip() else []


def _parse_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(text=body, section=current_heading))

    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            flush()
            buffer = []
            current_heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return sections


def _parse_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValidationFailure("Encrypted PDFs are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValidationFailure(f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
        sections = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(Section(text=text, page=number))
        return sections
    except ValidationFailure:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ValidationFailure(f"Could not parse PDF: {type(exc).__name__}") from exc

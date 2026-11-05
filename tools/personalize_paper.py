"""Replace the draft author placeholders; preserve all research content.

Run with pypdf and reportlab installed. The original is archived locally.
"""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ContentStream
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Hypergraph_Sinkhorn_Paper_LaTeX_V3.pdf"
OUTPUT = ROOT / "docs/paper" / PAPER.name
AUTHOR = "Mahdieh Sadat Esmaeili"


def main():
    original = PAPER.read_bytes()
    source = PdfReader(BytesIO(original))
    if "Mahdieh [Surname]" not in source.pages[0].extract_text():
        raise ValueError("Expected draft author placeholder missing; refusing blind edit")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = ROOT / "results/phase3_archive" / timestamp / "paper_original"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(PAPER, archive / PAPER.name)
    writer = PdfWriter()
    writer.clone_document_from_reader(source)
    page = writer.pages[0]
    content = ContentStream(page.get_contents(), writer)
    removed = []
    for operands, operator in content.operations:
        if operator != b"TJ":
            continue
        text = "".join(str(v) for v in operands[0] if isinstance(v, str))
        if text.startswith("Mahdieh") or text.startswith("[A") or "[email" in text:
            removed.append(text)
            operands[0] = ArrayObject()
    if len(removed) != 3:
        raise ValueError(f"Expected exactly three placeholder lines, found {removed}")
    page.replace_contents(content)
    overlay = BytesIO()
    c = canvas.Canvas(overlay, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
    c.setFont("Times-Roman", 14.3462)
    c.drawCentredString(float(page.mediabox.width)/2, 619.657, AUTHOR)
    c.save()
    page.merge_page(PdfReader(overlay).pages[0])
    writer.add_metadata({"/Author":AUTHOR, "/Title":"Learning Local Structural Equivalence in Hypergraphs with Differentiable Sinkhorn Matching"})
    target = BytesIO()
    writer.write(target)
    revised = PdfReader(BytesIO(target.getvalue()))
    assert len(source.pages) == len(revised.pages) == 14
    first = revised.pages[0].extract_text()
    assert AUTHOR in first and "[Surname]" not in first and "[Affiliation]" not in first and "[email address]" not in first
    for i in range(1, len(source.pages)):
        assert source.pages[i].get_contents().get_data() == revised.pages[i].get_contents().get_data(), i
    # All first-page research text remains, apart from the three author placeholders.
    before = source.pages[0].extract_text()
    for text in ("Mahdieh [Surname]", "[Affiliation]", "Corresponding author: [email address]"):
        before = before.replace(text, "")
    normalize = lambda text: "".join(text.split())
    assert normalize(before) == normalize(first.replace(AUTHOR,""))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(target.getvalue())
    report = {"author":AUTHOR,"output":OUTPUT.relative_to(ROOT).as_posix(),"page_count":14,"research_text_preserved":True,
              "pages_2_to_14_content_streams_unchanged":True,"removed_placeholder_line_count":3,
              "original_archive":archive.relative_to(ROOT).as_posix(),
              "original_sha256":hashlib.sha256(original).hexdigest(),
              "updated_sha256":hashlib.sha256(target.getvalue()).hexdigest()}
    (ROOT/"docs/github/paper_author_update.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path


PAGE_W = 612
PAGE_H = 792


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def stream(commands: str) -> bytes:
    return commands.encode("latin-1")


def pdf_page_logo() -> str:
    # Centered PDF document icon.
    x, y = 206, 246
    w, h = 200, 300
    fold = 56
    return f"""
q
0.86 0.08 0.08 rg
{x} {y} m
{x + w - fold} {y} l
{x + w} {y + fold} l
{x + w} {y + h} l
{x} {y + h} l
h f
0.96 0.72 0.72 rg
{x + w - fold} {y} m
{x + w - fold} {y + fold} l
{x + w} {y + fold} l
h f
1 1 1 rg
BT
/F1 58 Tf
1 0 0 1 {x + 41} {y + 134} Tm
({esc("PDF")}) Tj
ET
Q
"""


def word_page_logo() -> str:
    # Centered Microsoft Word-style icon.
    x, y = 166, 276
    return f"""
q
0.08 0.24 0.56 rg
{x + 110} {y + 38} {200} {200} re f
0.11 0.34 0.72 rg
{x + 130} {y + 58} {160} {160} re f
0.95 0.98 1 rg
{x + 170} {y + 170} {90} {10} re f
{x + 170} {y + 140} {90} {10} re f
{x + 170} {y + 110} {90} {10} re f
{x + 170} {y + 80} {70} {10} re f
0.06 0.19 0.48 rg
{x} {y} {168} {256} re f
0.13 0.40 0.82 rg
{x + 14} {y + 14} {140} {228} re f
1 1 1 rg
BT
/F2 132 Tf
1 0 0 1 {x + 35} {y + 69} Tm
({esc("W")}) Tj
ET
Q
"""


def build_pdf(contents: list[str]) -> bytes:
    objects: list[bytes] = []

    def add(obj: str | bytes) -> int:
        objects.append(obj.encode("latin-1") if isinstance(obj, str) else obj)
        return len(objects)

    add("<< /Type /Catalog /Pages 2 0 R >>")
    add("<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>")
    add(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
        f"/Resources << /Font << /F1 7 0 R /F2 8 0 R >> >> /Contents 4 0 R >>"
    )
    page1_stream = stream(contents[0])
    add(b"<< /Length " + str(len(page1_stream)).encode("latin-1") + b" >>\nstream\n" + page1_stream + b"\nendstream")
    add(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
        f"/Resources << /Font << /F1 7 0 R /F2 8 0 R >> >> /Contents 6 0 R >>"
    )
    page2_stream = stream(contents[1])
    add(b"<< /Length " + str(len(page2_stream)).encode("latin-1") + b" >>\nstream\n" + page2_stream + b"\nendstream")
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n".encode("latin-1"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_at = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n".encode("latin-1")
    )
    return bytes(out)


if __name__ == "__main__":
    pdf = build_pdf([pdf_page_logo(), word_page_logo()])
    Path("two_logo_pages.pdf").write_bytes(pdf)

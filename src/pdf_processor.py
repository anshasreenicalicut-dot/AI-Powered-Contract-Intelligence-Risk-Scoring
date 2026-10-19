import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(pdf_path, use_ocr, dpi, poppler_path=None):
    document = fitz.open(pdf_path)

    all_pages = []
    page_report = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text") or ""
        method = "PyMuPDF"

        # Use OCR when the PDF page contains almost no extractable text.
        if use_ocr and len(text.strip()) < 30:
            try:
                from pdf2image import convert_from_path
                import pytesseract

                images = convert_from_path(
                    str(pdf_path),
                    first_page=page_number,
                    last_page=page_number,
                    dpi=dpi,
                    poppler_path=poppler_path
                )

                if images:
                    text = pytesseract.image_to_string(images[0])
                    method = "Tesseract OCR"

            except Exception as error:
                print(
                    f"[WARNING] OCR failed on "
                    f"{pdf_path.name}, page {page_number}: {error}"
                )

        text = clean_text(text)
        all_pages.append(
            f"--- PAGE {page_number} ---\n{text}"
        )

        page_report.append({
            "page": page_number,
            "method": method,
            "characters": len(text)
        })

    document.close()

    return "\n\n".join(all_pages), page_report


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from digital or scanned contract PDFs."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/extracted_text"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for pages with little extractable text"
    )
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--poppler-path",
        default=None,
        help="Optional Windows path to Poppler's bin folder"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"PDF folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    report = []

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")

    for pdf_path in pdf_files:
        try:
            text, pages = extract_pdf(
                pdf_path,
                use_ocr=args.ocr,
                dpi=args.dpi,
                poppler_path=args.poppler_path
            )

            output_file = output_dir / f"{pdf_path.stem}.txt"
            output_file.write_text(text, encoding="utf-8")

            report.append({
                "pdf": str(pdf_path),
                "output": str(output_file),
                "pages": len(pages),
                "characters": len(text),
                "page_details": pages,
                "status": "success"
            })

            print(
                f"[OK] {pdf_path.name}: "
                f"{len(pages)} pages, {len(text)} characters"
            )

        except Exception as error:
            report.append({
                "pdf": str(pdf_path),
                "status": "error",
                "error": str(error)
            })
            print(f"[ERROR] {pdf_path.name}: {error}")

    report_file = output_dir / "pdf_processing_report.json"
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nPDF processing completed. Report: {report_file}")


if __name__ == "__main__":
    main()
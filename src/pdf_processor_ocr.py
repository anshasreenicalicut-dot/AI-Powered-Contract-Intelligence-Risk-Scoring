
import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import fitz

LOG = logging.getLogger("CUAD_PDF_OCR")


def clean_text(text):
    """Clean extracted text and normalize whitespace."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def configure_tesseract(tesseract_cmd=None):
    """Verify that Tesseract OCR is installed."""
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    command = pytesseract.pytesseract.tesseract_cmd

    if not (Path(command).is_file() or shutil.which(command)):
        raise RuntimeError(
            "Tesseract was not found. Install it and add it to PATH, "
            "or provide --tesseract-cmd."
        )


def process_pdf(
    pdf_path,
    dpi=250,
    lang="eng",
    poppler_path=None,
    min_chars=30,
):
    """Extract embedded text, using OCR for scanned pages."""
    import pytesseract
    from pdf2image import convert_from_path

    pages = []
    methods = []

    with fitz.open(pdf_path) as document:
        page_count = len(document)

        for index, page in enumerate(document):
            text = clean_text(page.get_text("text") or "")
            method = "PyMuPDF"

            if len(text) < min_chars:
                images = []

                try:
                    images = convert_from_path(
                        str(pdf_path),
                        first_page=index + 1,
                        last_page=index + 1,
                        dpi=dpi,
                        poppler_path=poppler_path or None,
                        thread_count=1,
                    )

                    if images:
                        try:
                            ocr_text = clean_text(
                                pytesseract.image_to_string(
                                    images[0],
                                    lang=lang,
                                    config="--psm 3",
                                )
                            )
                        finally:
                            for image in images:
                                image.close()

                        if len(ocr_text) > len(text):
                            text = ocr_text
                            method = "Tesseract OCR"
                        else:
                            method = "PyMuPDF fallback"

                except Exception as error:
                    LOG.warning(
                        "OCR failed for %s, page %d: %s",
                        pdf_path.name,
                        index + 1,
                        error,
                    )

            pages.append(f"--- PAGE {index + 1} ---\n{text}")
            methods.append(method)

            LOG.info(
                "%s | page %d/%d | %s | %d characters",
                pdf_path.name,
                index + 1,
                page_count,
                method,
                len(text),
            )

    full_text = "\n\n".join(pages)

    return {
        "text": full_text,
        "page_count": page_count,
        "character_count": len(full_text),
        "ocr_pages": methods.count("Tesseract OCR"),
        "native_pages": methods.count("PyMuPDF"),
        "page_methods": methods,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from digital and scanned PDFs."
    )
    parser.add_argument("--input-dir", default="data/pdfs")
    parser.add_argument("--output-dir", default="outputs/extracted_text")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--poppler-path", default=None)
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--min-chars", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(
            f"Input folder does not exist: {input_dir.resolve()}"
        )

    pdf_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise SystemExit(
            f"No PDFs found in {input_dir.resolve()}. "
            "Copy PDF contracts into this folder first."
        )

    configure_tesseract(args.tesseract_cmd)

    if args.poppler_path and not Path(args.poppler_path).is_dir():
        raise SystemExit(
            f"Poppler folder does not exist: {args.poppler_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    results = []

    for number, pdf_path in enumerate(pdf_files, start=1):
        LOG.info(
            "[%d/%d] Processing %s",
            number,
            len(pdf_files),
            pdf_path.name,
        )

        try:
            result = process_pdf(
                pdf_path,
                dpi=args.dpi,
                lang=args.lang,
                poppler_path=args.poppler_path,
                min_chars=args.min_chars,
            )

            relative = pdf_path.relative_to(input_dir)
            output_file = (output_dir / relative).with_suffix(".txt")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(
                result["text"] + ("\n" if result["text"] else ""),
                encoding="utf-8",
            )

            record = {
                "source_pdf": str(pdf_path.resolve()),
                "output_txt": str(output_file.resolve()),
                "page_count": result["page_count"],
                "character_count": result["character_count"],
                "ocr_pages": result["ocr_pages"],
                "native_pages": result["native_pages"],
                "page_methods": result["page_methods"],
                "status": "ok" if result["text"].strip() else "empty",
            }

        except Exception as error:
            LOG.exception("Failed to process %s", pdf_path.name)
            record = {
                "source_pdf": str(pdf_path.resolve()),
                "status": "error",
                "error": str(error),
            }

        results.append(record)

    with manifest_path.open("w", encoding="utf-8") as file:
        for record in results:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    successful = sum(r["status"] == "ok" for r in results)
    failed = sum(r["status"] == "error" for r in results)

    print("\nPDF ingestion completed.")
    print(f"PDFs found: {len(pdf_files)}")
    print(f"Successfully extracted: {successful}")
    print(f"Failed: {failed}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


def _find_pdf_renderer() -> Optional[str]:
    """Return the available external renderer command for PDFs."""
    for cmd in ("pdftoppm", "qlmanage"):
        if shutil.which(cmd):
            return cmd
    return None


def convert_pdf_to_pngs(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> List[Path]:
    """Convert a single PDF to PNG files in the target output directory."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input file is not a PDF: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = _find_pdf_renderer()
    if renderer is None:
        raise RuntimeError(
            "No PDF rendering tool found. Install `pdftoppm` or rely on macOS `qlmanage`."
        )

    output_prefix = output_dir / pdf_path.stem
    created_files: List[Path] = []

    if renderer == "pdftoppm":
        command = [
            renderer,
            "-png",
            "-rx",
            str(dpi),
            "-ry",
            str(dpi),
            str(pdf_path),
            str(output_prefix),
        ]
        subprocess.run(command, check=True)
        created_files = sorted(output_dir.glob(f"{pdf_path.stem}-*.png"))
    else:
        # macOS fallback for a single rendered preview page
        command = [
            renderer,
            "-t",
            "-s",
            str(dpi),
            "-o",
            str(output_dir),
            str(pdf_path),
        ]
        subprocess.run(command, check=True)
        created_files = sorted(output_dir.glob(f"{pdf_path.stem}*.png"))

    return created_files


def process_deposit_pdfs(
    deposit_dir: Path | str = "Deposit",
    output_dir: Optional[Path | str] = None,
    dpi: int = 300,
) -> List[Path]:
    """Convert every PDF in the Deposit folder into PNGs under processed."""
    deposit_path = Path(deposit_dir)
    if output_dir is None:
        output_path = deposit_path / "processed"
    else:
        output_path = Path(output_dir)

    if not deposit_path.exists():
        raise FileNotFoundError(f"Deposit folder does not exist: {deposit_path}")
    if not deposit_path.is_dir():
        raise NotADirectoryError(f"Deposit is not a directory: {deposit_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(deposit_path.glob("*.pdf"))
    all_created: List[Path] = []

    for pdf_file in pdf_files:
        created = convert_pdf_to_pngs(pdf_file, output_path, dpi=dpi)
        all_created.extend(created)

    return all_created


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PDF files from Deposit into PNGs in a processed folder."
    )
    parser.add_argument(
        "--deposit-dir",
        default="Deposit",
        help="Path to the folder containing source PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output folder path. Defaults to Deposit/processed.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for generated PNGs.",
    )
    args = parser.parse_args()

    created_images = process_deposit_pdfs(
        deposit_dir=args.deposit_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    print(f"Converted {len(created_images)} image(s) to {args.output_dir or (Path(args.deposit_dir) / 'processed')}")

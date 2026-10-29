"""Render the Part 3 Markdown section to PDF using existing local libraries.

Reuses layout primitives from the Part 2 renderer without changing its files
or outputs. Optional page previews belong in a temporary directory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from render_part2_report import HEIGHT, MARGIN, ROOT, WIDTH, Report


class Part3Report(Report):
    def finish_page(self) -> None:
        self.fig.text(MARGIN / WIDTH, 21 / HEIGHT, "11-768 · Assignment 3 · Part 3",
                      fontsize=7.5, color="#687580")
        self.fig.text((WIDTH - MARGIN) / WIDTH, 21 / HEIGHT, str(self.page_number),
                      fontsize=7.5, color="#687580", ha="right")
        self.pdf.savefig(self.fig, dpi=260)
        if self.preview_dir is not None:
            self.fig.savefig(self.preview_dir / f"part3_page_{self.page_number:02d}.png", dpi=140)

    def figure_page(self, path: Path, caption: str, index: int) -> None:
        self.new_page()
        self.heading(f"Part 3 · Diagnostic plots ({index}/3)", 2)
        with Image.open(path) as image:
            image_width = WIDTH - 2 * MARGIN - (18 if index == 2 else 0)
            image_height = image_width * image.height / image.width
            lines = self.wrap(caption, 9, WIDTH - 2 * MARGIN)
            required = image_height + 10 + len(lines) * 9 * 1.34 + 6
            if self.y - required < MARGIN:
                raise ValueError(f"Figure {index} and caption would overflow one page")
            ax = self.fig.add_axes(((WIDTH - image_width) / 2 / WIDTH,
                                    (self.y - image_height) / HEIGHT,
                                    image_width / WIDTH, image_height / HEIGHT))
            ax.imshow(image, interpolation="none")
            ax.set_axis_off()
        self.y -= image_height + 10
        self.paragraph(caption, size=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path,
                        help="optional temporary directory for PNG page previews")
    args = parser.parse_args()
    matplotlib.rcParams["pdf.fonttype"] = 42
    report_dir = ROOT / "report"
    source = (report_dir / "part3.md").read_text()
    prose, figures = source.split("## Diagnostic plots", maxsplit=1)
    image_blocks = re.findall(r"!\[([^\]]+)\]\(([^)]+)\)\s*\n\n(.+?)(?=\n\n!\[|\Z)",
                              figures.strip(), flags=re.DOTALL)
    if len(image_blocks) != 3:
        raise ValueError("Expected three Part 3 figures with captions")
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / "part3.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Part 3: Asynchronous GRPO",
                                     "Subject": "11-768 Assignment 3, Part 3"}) as pdf:
        report = Part3Report(pdf, args.preview_dir)
        for index, section in enumerate(prose.split("<!-- pagebreak -->")):
            if index:
                report.new_page()
            report.blocks(section)
        for index, (_, relative, caption) in enumerate(image_blocks, start=1):
            report.figure_page(report_dir / relative, caption.strip(), index)
        report.finish_page()
    print(f"Wrote {pdf_path.relative_to(ROOT)} ({report.page_number} pages)")
    if args.preview_dir is not None:
        print(f"Page previews: {args.preview_dir}")


if __name__ == "__main__":
    main()

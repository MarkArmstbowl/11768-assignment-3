"""Render the completed Part 4 Markdown sections with local libraries.

Adds four-column table support while reusing established report primitives.
Earlier report sources and outputs are read only; previews belong in /tmp.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image

from render_part2_report import ACCENT, HEIGHT, INK, MARGIN, ROOT, WIDTH, Report


class Part4Report(Report):
    def __init__(self, pdf: PdfPages, section: str, preview_dir: Path | None) -> None:
        self.section = section
        self.prefix = f"part{section.replace('.', '_')}"
        super().__init__(pdf, preview_dir)

    def finish_page(self) -> None:
        self.fig.text(MARGIN / WIDTH, 21 / HEIGHT,
                      f"11-768 · Assignment 3 · Part {self.section}",
                      fontsize=7.5, color="#687580")
        self.fig.text((WIDTH - MARGIN) / WIDTH, 21 / HEIGHT, str(self.page_number),
                      fontsize=7.5, color="#687580", ha="right")
        self.pdf.savefig(self.fig, dpi=260)
        if self.preview_dir is not None:
            self.fig.savefig(self.preview_dir / f"{self.prefix}_page_{self.page_number:02d}.png",
                             dpi=140)

    def table(self, rows: list[list[str]]) -> None:
        columns = len(rows[0])
        if columns == 3:
            super().table(rows)
            return
        if columns != 4 or any(len(row) != columns for row in rows):
            raise ValueError("Expected a three- or four-column report table")
        size = 8.2
        widths = [(WIDTH - 2 * MARGIN) * f for f in (0.40, 0.20, 0.20, 0.20)]
        wrapped = [[self.wrap(cell, size, width - 12, bold=index == 0)
                    for cell, width in zip(row, widths)] for index, row in enumerate(rows)]
        heights = [max(map(len, row)) * size * 1.3 + 7 for row in wrapped]
        self.room(sum(heights) + 8)
        for index, (row, height) in enumerate(zip(wrapped, heights)):
            fill = "#edf3f7" if index == 0 else ("#f7f9fb" if index % 2 == 0 else "white")
            self.fig.add_artist(Rectangle((MARGIN / WIDTH, (self.y - height) / HEIGHT),
                                          (WIDTH - 2 * MARGIN) / WIDTH, height / HEIGHT,
                                          transform=self.fig.transFigure, facecolor=fill,
                                          edgecolor="#d5dde3", linewidth=0.35, zorder=0))
            x = MARGIN
            for cell, width in zip(row, widths):
                for line_number, line in enumerate(cell):
                    self.draw(line, x + 6, self.y - 3 - line_number * size * 1.3,
                              size, bold=index == 0, color=ACCENT if index == 0 else INK)
                x += width
            self.y -= height
        self.y -= 8

    def figure_page(self, path: Path, caption: str, index: int) -> None:
        self.new_page()
        self.heading(f"Part {self.section} · Diagnostic plots ({index}/3)", 2)
        with Image.open(path) as image:
            image_width = WIDTH - 2 * MARGIN - (18 if index == 2 else 0)
            image_height = image_width * image.height / image.width
            lines = self.wrap(caption, 9, WIDTH - 2 * MARGIN)
            caption_height = 10 + len(lines) * 9 * 1.34 + 6
            available_height = self.y - MARGIN - caption_height
            if image_height > available_height:
                image_width *= available_height / image_height
                image_height = available_height
            if image_width < 440:
                raise ValueError(f"Part {self.section}, figure {index}: shorten the caption")
            ax = self.fig.add_axes(((WIDTH - image_width) / 2 / WIDTH,
                                    (self.y - image_height) / HEIGHT,
                                    image_width / WIDTH, image_height / HEIGHT))
            ax.imshow(image, interpolation="none")
            ax.set_axis_off()
        self.y -= image_height + 10
        self.paragraph(caption, size=9)


def render(section: str, preview_dir: Path | None) -> None:
    report_dir = ROOT / "report"
    prefix = f"part{section.replace('.', '_')}"
    source = (report_dir / f"{prefix}.md").read_text()
    prose, figures = source.split("## Diagnostic plots", maxsplit=1)
    image_blocks = re.findall(r"!\[([^\]]+)\]\(([^)]+)\)\s*\n\n(.+?)(?=\n\n!\[|\Z)",
                              figures.strip(), flags=re.DOTALL)
    if len(image_blocks) != 3:
        raise ValueError(f"Expected three Part {section} figures with captions")
    pdf_path = report_dir / f"{prefix}.pdf"
    title = source.splitlines()[0].removeprefix("# ")
    with PdfPages(pdf_path, metadata={"Title": title,
                                     "Subject": f"11-768 Assignment 3, Part {section}"}) as pdf:
        report = Part4Report(pdf, section, preview_dir)
        for index, chunk in enumerate(prose.split("<!-- pagebreak -->")):
            if index:
                report.new_page()
            report.blocks(chunk)
        for index, (_, relative, caption) in enumerate(image_blocks, start=1):
            report.figure_page(report_dir / relative, caption.strip(), index)
        report.finish_page()
    print(f"Wrote {pdf_path.relative_to(ROOT)} ({report.page_number} pages)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("4.1", "4.2", "4.3", "all"), default="all")
    parser.add_argument("--preview-dir", type=Path,
                        help="optional directory outside the repo for PNG page previews")
    args = parser.parse_args()
    matplotlib.rcParams["pdf.fonttype"] = 42
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for section in (("4.1", "4.2", "4.3") if args.section == "all" else (args.section,)):
        render(section, args.preview_dir)


if __name__ == "__main__":
    main()

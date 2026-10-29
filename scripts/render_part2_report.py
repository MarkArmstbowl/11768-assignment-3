"""Render Part 2 Markdown to PDF with Matplotlib/Pillow.

Uses an existing plotting environment without installation or service calls.
The small reader supports the headings, tables and paragraphs in this report.
Pass --preview-dir to optionally generate PNG page previews outside the repo.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 595.276, 841.89  # A4 points.
MARGIN = 40
INK, ACCENT = "#17212b", "#163b59"


def plain(text: str) -> str:
    return text.replace("`", "").replace("**", "").replace("$10^{-5}$", "10⁻⁵")


class Report:
    def __init__(self, pdf: PdfPages, preview_dir: Path | None = None) -> None:
        self.pdf = pdf
        self.preview_dir = preview_dir
        self.fig = None
        self.page_number = 0
        self.new_page()

    def new_page(self) -> None:
        if self.fig is not None:
            self.finish_page()
        self.page_number += 1
        self.fig = Figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=72, facecolor="white")
        FigureCanvasAgg(self.fig)
        self.renderer = self.fig.canvas.get_renderer()
        self.y = HEIGHT - MARGIN

    def finish_page(self) -> None:
        self.fig.text(MARGIN / WIDTH, 21 / HEIGHT, "11-768 · Assignment 3 · Part 2",
                      fontsize=7.5, color="#687580")
        self.fig.text((WIDTH - MARGIN) / WIDTH, 21 / HEIGHT, str(self.page_number),
                      fontsize=7.5, color="#687580", ha="right")
        # Preserve chart detail when embedding raster figures in the PDF.
        self.pdf.savefig(self.fig, dpi=260)
        if self.preview_dir is not None:
            self.fig.savefig(self.preview_dir / f"part2_page_{self.page_number:02d}.png", dpi=140)

    def width(self, text: str, size: float, bold: bool = False) -> float:
        prop = FontProperties(family="DejaVu Sans", size=size,
                              weight="bold" if bold else "normal")
        return self.renderer.get_text_width_height_descent(text, prop, ismath=False)[0]

    def wrap(self, text: str, size: float, width: float, bold: bool = False) -> list[str]:
        lines, current = [], ""
        for word in plain(text).split():
            candidate = f"{current} {word}" if current else word
            if current and self.width(candidate, size, bold) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        return lines + ([current] if current else [])

    def room(self, height: float) -> None:
        if self.y - height < MARGIN:
            self.new_page()

    def draw(self, text: str, x: float, top: float, size: float,
             bold: bool = False, color: str = INK) -> None:
        self.fig.text(x / WIDTH, top / HEIGHT, text, va="top", fontsize=size,
                      family="DejaVu Sans", weight="bold" if bold else "normal", color=color)

    def paragraph(self, text: str, size: float = 9.5, gap: float = 6) -> None:
        lines = self.wrap(text, size, WIDTH - 2 * MARGIN)
        line_height = size * 1.34
        self.room(len(lines) * line_height + gap)
        for line in lines:
            self.draw(line, MARGIN, self.y, size)
            self.y -= line_height
        self.y -= gap

    def heading(self, text: str, level: int) -> None:
        size = 16 if level == 1 else 11
        self.room(size * 1.8 + 35)
        if level != 1:
            self.y -= 5
        self.draw(text, MARGIN, self.y, size, bold=True, color=ACCENT)
        self.y -= size * 1.5 + 3

    def table(self, rows: list[list[str]]) -> None:
        size = 8.2
        widths = [(WIDTH - 2 * MARGIN) * fraction for fraction in (0.45, 0.275, 0.275)]
        if any(len(row) != len(widths) for row in rows):
            raise ValueError("Expected three-column report tables")
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

    def blocks(self, source: str) -> None:
        lines = source.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
            elif line.startswith("#"):
                match = re.fullmatch(r"(#{1,2}) (.+)", line)
                if match is None:
                    raise ValueError(f"Unsupported heading: {line}")
                self.heading(match[2], len(match[1]))
                index += 1
            elif line.startswith("|"):
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                        rows.append(cells)
                    index += 1
                self.table(rows)
            else:
                paragraph = [line]
                index += 1
                while index < len(lines) and lines[index].strip():
                    paragraph.append(lines[index].strip())
                    index += 1
                self.paragraph(" ".join(paragraph))

    def figure_page(self, path: Path, caption: str, index: int) -> None:
        self.new_page()
        self.heading(f"Part 2 · Diagnostic plots ({index}/2)", 2)
        image = Image.open(path)
        image_width = WIDTH - 2 * MARGIN - (18 if index == 2 else 0)
        image_height = image_width * image.height / image.width
        caption_lines = self.wrap(caption, 9, WIDTH - 2 * MARGIN)
        required = image_height + 10 + len(caption_lines) * 9 * 1.34 + 6
        if self.y - required < MARGIN:
            raise ValueError("Figure and caption would overflow one page")
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
                        help="optional directory for temporary PNG page previews")
    args = parser.parse_args()
    matplotlib.rcParams["pdf.fonttype"] = 42
    report_dir = ROOT / "report"
    source = (report_dir / "part2.md").read_text()
    prose, figures = source.split("## Diagnostic plots", maxsplit=1)
    image_blocks = re.findall(r"!\[([^\]]+)\]\(([^)]+)\)\s*\n\n(.+?)(?=\n\n!\[|\Z)",
                              figures.strip(), flags=re.DOTALL)
    if len(image_blocks) != 2:
        raise ValueError("Expected the two report figures and their captions")
    preview_dir = args.preview_dir
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / "part2.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Part 2: Synchronous REINFORCE and GRPO",
                                     "Subject": "11-768 Assignment 3, Part 2"}) as pdf:
        report = Report(pdf, preview_dir)
        report.blocks(prose)
        for index, (_, relative, caption) in enumerate(image_blocks, start=1):
            report.figure_page(report_dir / relative, caption.strip(), index)
        report.finish_page()
    print(f"Wrote {pdf_path.relative_to(ROOT)} ({report.page_number} pages)")
    if preview_dir is not None:
        print(f"Page previews: {preview_dir}")


if __name__ == "__main__":
    main()

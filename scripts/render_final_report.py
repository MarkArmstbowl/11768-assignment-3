"""Render the complete submission report with one consistent local layout.

Reads the reviewed section Markdown and existing figure PNGs. Requires the
same Matplotlib/Pillow libraries as the earlier renderers; no service calls.
Writes report.pdf and report/part5.pdf. Optional previews are kept outside
the repository, and a layout log makes text/page boundaries inspectable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image

from render_part2_report import ACCENT, HEIGHT, INK, MARGIN, ROOT, WIDTH, Report


SECTIONS = (("2", "part2"), ("3", "part3"), ("4.1", "part4_1"),
            ("4.2", "part4_2"), ("4.3", "part4_3"), ("5", "part5"))
AUTHOR = "Yubo Chen (Andrew ID: yuboc)"


class FinalReport(Report):
    def __init__(self, pdf: PdfPages, preview_dir: Path | None, prefix: str) -> None:
        self.section = "Overview"
        self.prefix = prefix
        self.layout: list[dict] = []
        self.section_pages: dict[str, int] = {}
        super().__init__(pdf, preview_dir)

    def draw(self, text: str, x: float, top: float, size: float,
             bold: bool = False, color: str = INK) -> None:
        width = self.width(text, size, bold)
        if x < MARGIN - 0.1 or x + width > WIDTH - MARGIN + 0.1:
            raise ValueError(f"Text exceeds horizontal margin on page {self.page_number}: {text}")
        if top > HEIGHT - MARGIN + 0.1 or top - size * 1.34 < MARGIN - 0.1:
            raise ValueError(f"Text exceeds vertical margin on page {self.page_number}: {text}")
        self.layout.append({"page": self.page_number, "section": self.section, "text": text,
                            "x": x, "top": top, "width": width, "font_size": size})
        super().draw(text, x, top, size, bold, color)

    def heading(self, text: str, level: int) -> None:
        if level == 1:
            # Wrap long titles rather than silently drawing past the page.
            lines = self.wrap(text, 16, WIDTH - 2 * MARGIN, bold=True)
            self.room(len(lines) * 24 + 35)
            for line in lines:
                self.draw(line, MARGIN, self.y, 16, bold=True, color=ACCENT)
                self.y -= 24
            self.y -= 3
        else:
            super().heading(text, level)

    def finish_page(self) -> None:
        footer = "11-768 · Assignment 3" + (f" · Part {self.section}" if self.section != "Overview" else "")
        self.fig.text(MARGIN / WIDTH, 21 / HEIGHT, footer, fontsize=7.5, color="#687580")
        self.fig.text((WIDTH - MARGIN) / WIDTH, 21 / HEIGHT, str(self.page_number),
                      fontsize=7.5, color="#687580", ha="right")
        self.pdf.savefig(self.fig, dpi=260)
        if self.preview_dir is not None:
            self.fig.savefig(self.preview_dir / f"{self.prefix}_page_{self.page_number:02d}.png", dpi=140)

    def table(self, rows: list[list[str]]) -> None:
        columns = len(rows[0])
        fractions = {2: (0.31, 0.69), 3: (0.45, 0.275, 0.275),
                     4: ((0.22, 0.14, 0.32, 0.32) if self.section == "5"
                         else (0.40, 0.20, 0.20, 0.20))}
        if columns not in fractions or any(len(row) != columns for row in rows):
            raise ValueError("Unsupported report table")
        size = 8.2
        widths = [(WIDTH - 2 * MARGIN) * fraction for fraction in fractions[columns]]
        wrapped = [[self.wrap(cell, size, width - 12, bold=index == 0)
                    for cell, width in zip(row, widths)] for index, row in enumerate(rows)]
        for row in wrapped:
            for cell, width in zip(row, widths):
                if any(self.width(line, size, bold=False) > width - 10 for line in cell):
                    raise ValueError(f"Table cell exceeds column: {cell}")
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

    def figure_page(self, path: Path, caption: str, index: int, count: int) -> None:
        self.new_page()
        self.heading(f"Part {self.section} · Diagnostic plots ({index}/{count})", 2)
        with Image.open(path) as source:
            image_width = WIDTH - 2 * MARGIN - (18 if index == 2 else 0)
            image_height = image_width * source.height / source.width
            caption_height = 10 + len(self.wrap(caption, 9, WIDTH - 2 * MARGIN)) * 9 * 1.34 + 6
            available = self.y - MARGIN - caption_height
            if image_height > available:
                image_width *= available / image_height
                image_height = available
            if image_width < 440:
                raise ValueError(f"Part {self.section} figure {index}: image would be too small")
            ax = self.fig.add_axes(((WIDTH - image_width) / 2 / WIDTH,
                                    (self.y - image_height) / HEIGHT,
                                    image_width / WIDTH, image_height / HEIGHT))
            ax.imshow(source, interpolation="none")
            ax.set_axis_off()
        self.y -= image_height + 10
        self.paragraph(caption, size=9)

    def cover(self) -> None:
        self.heading("11-768 AI Agents", 1)
        self.heading("Assignment 3: Training Agents", 1)
        self.heading("with Reinforcement Learning", 1)
        self.y -= 12
        self.paragraph("Experimental report · Fall 2026", size=11, gap=18)
        self.paragraph("Name: Yubo Chen", size=11, gap=4)
        self.paragraph("Andrew ID: yuboc", size=10.5, gap=18)
        self.paragraph("This report contains the requested analyses and diagnostic plots for all seven "
                       "formal training runs, followed by the four full-split TextCraft evaluations. "
                       "Parts 0 and 1 provide the runtime understanding and implemented data pipeline; "
                       "their source code is included separately in the submission.")
        self.heading("Report contents", 2)
        self.table([
            ["Section", "Study"],
            ["Part 2", "Synchronous REINFORCE and GRPO"],
            ["Part 3", "Asynchronous GRPO: timing, drift, and real rollouts"],
            ["Part 4.1", "Group size, useful data, and advantage variance"],
            ["Part 4.2", "Synchronous versus asynchronous training"],
            ["Part 4.3", "Importance weighting under higher staleness"],
            ["Part 5", "TextCraft-Synth training and medium/hard evaluation"],
        ])
        self.heading("Reading the results", 2)
        self.paragraph("Each configuration was run once, using its supplied settings unchanged. "
                       "Figures show recorded values without smoothing; missing update diagnostics "
                       "remain gaps. Learner iterations and actual optimizer updates are distinguished, "
                       "and each pass@k curve names its exact k. Part 4.2 explicitly reuses Parts 2–3. "
                       "Part 5 retains abandoned episodes as failures and distinguishes periodic "
                       "50-task evaluation from the final complete validation splits.")
        self.heading("Public final adapter", 2)
        self.paragraph("https://huggingface.co/Marstbowl/scottyrl-a3-p5_textcraft_grpo", size=9)
        self.paragraph("Base: Qwen/Qwen3.5-4B · LoRA rank: 32 · Final sampler: v50", size=9)

    def section_content(self, section: str, stem: str, *, first: bool = False) -> None:
        if not first:
            self.new_page()  # Finish the preceding page using its own section label.
        self.section = section
        self.section_pages[section] = self.page_number
        source = (ROOT / "report" / f"{stem}.md").read_text()
        prose, figures = source.split("## Diagnostic plots", maxsplit=1)
        image_blocks = re.findall(r"!\[([^\]]+)\]\(([^)]+)\)\s*\n\n(.+?)(?=\n\n!\[|\Z)",
                                  figures.strip(), flags=re.DOTALL)
        expected = 2 if section == "2" else 3
        if len(image_blocks) != expected:
            raise ValueError(f"Part {section}: wrong number of captioned figures")
        for index, chunk in enumerate(prose.split("<!-- pagebreak -->")):
            if index:
                self.new_page()
            self.blocks(chunk)
        for index, (_, relative, caption) in enumerate(image_blocks, start=1):
            self.figure_page(ROOT / "report" / relative, caption.strip(), index, expected)


def render(path: Path, sections: tuple, previews: Path | None, *, cover: bool) -> dict:
    with PdfPages(path, metadata={"Title": "11-768 Assignment 3: Training Agents with Reinforcement Learning",
                                  "Subject": "Formal experiments, diagnostic plots, and TextCraft evaluations",
                                  "Author": AUTHOR}) as pdf:
        report = FinalReport(pdf, previews, path.stem)
        if cover:
            report.cover()
        for index, (section, stem) in enumerate(sections):
            report.section_content(section, stem, first=index == 0 and not cover)
        report.finish_page()
    result = {"pdf": str(path), "pages": report.page_number, "section_start_pages": report.section_pages,
              "text": report.layout}
    print(f"Wrote {path.relative_to(ROOT)} ({report.page_number} pages)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    matplotlib.rcParams["pdf.fonttype"] = 42
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    final = render(ROOT / "report.pdf", SECTIONS, args.preview_dir, cover=True)
    part5 = render(ROOT / "report" / "part5.pdf", (("5", "part5"),), args.preview_dir, cover=False)
    if args.preview_dir:
        (args.preview_dir / "layout.json").write_text(json.dumps({"final": final, "part5": part5}, indent=2))


if __name__ == "__main__":
    main()

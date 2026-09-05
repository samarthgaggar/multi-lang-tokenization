#!/usr/bin/env python3
"""Write dependency-free SVG charts for deterministic FLORES vocabulary results."""
import argparse
import html
from pathlib import Path

import pandas as pd


NAMES = {"en": "English", "es": "Spanish", "ru": "Russian", "uk": "Ukrainian", "hi": "Hindi", "te": "Telugu"}
ORDER = list(NAMES)
COLORS = ["#3b6ea8", "#4d8b70", "#9b6a9d", "#b87544", "#c28b28", "#5f8fa3"]


def chart(path, title, labels, values, ylabel, formatter, maximum=None):
    width, height, left, bottom, top = 920, 540, 90, 105, 70
    plot_width, plot_height = width - left - 40, height - bottom - top
    maximum = maximum or max(values) * 1.16
    bar_width = plot_width / len(values) * 0.58
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{html.escape(title)}</text>',
             f'<text x="24" y="{height/2}" transform="rotate(-90 24 {height/2})" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(ylabel)}</text>',
             f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#333"/>']
    for tick in range(5):
        value = maximum * tick / 4
        y = top + plot_height - plot_height * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{html.escape(formatter(value))}</text>')
    spacing = plot_width / len(values)
    for index, (label, value, color) in enumerate(zip(labels, values, COLORS)):
        x = left + index * spacing + (spacing - bar_width) / 2
        bar_height = value / maximum * plot_height
        y = top + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(formatter(value))}</text>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{top+plot_height+28}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(label)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.results_dir
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(root / "summary.csv").set_index("language").loc[ORDER]
    labels = [NAMES[key] for key in ORDER]
    chart(figures / "vocabulary_size_by_language.svg", "Unique FlexiTokens output segments in FLORES", labels,
          summary["unique_output_segments_vocabulary_size"].tolist(), "Vocabulary size (unique decoded segments)", lambda value: f"{value:,.0f}")
    chart(figures / "mean_segments_per_sentence.svg", "Mean output segments per complete FLORES sentence", labels,
          summary["mean_segments_per_sentence"].tolist(), "Mean segments per sentence", lambda value: f"{value:.1f}")


if __name__ == "__main__":
    main()

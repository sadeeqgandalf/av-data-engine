# Local Mac Experiment Logger: Metrics, Matplotlib Bar Charts & HTML Visual Dashboard
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "bg": "#0f172a",
    "panel": "#1e293b",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "grid": "#334155",
    "red": "#f43f5e",
    "green": "#10b981",
    "blue": "#38bdf8",
    "amber": "#f59e0b",
    "purple": "#a78bfa",
}


def _style_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PALETTE["panel"])
    ax.set_title(title, color=PALETTE["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=PALETTE["muted"], fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=10, labelpad=8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["grid"])


def plot_recall_comparison(
    out_path: Path,
    student_recall: float,
    teacher_recall: float | None = None,
    student_hits: int = 0,
    teacher_hits: int = 0,
    total_gt: int = 0,
) -> Path:
    """Bar chart comparing Student v1 vs SAM3 Teacher VRU Recall."""
    labels = ["Student v1\n(YOLO11n)"]
    values = [student_recall * 100]
    colors = [PALETTE["red"]]
    sublabels = [f"{student_recall * 100:.1f}%\n({student_hits}/{total_gt})" if total_gt else f"{student_recall * 100:.1f}%"]

    if teacher_recall is not None:
        labels.append("SAM3 Teacher\n(Hard Frames)")
        values.append(teacher_recall * 100)
        colors.append(PALETTE["green"])
        sublabels.append(
            f"{teacher_recall * 100:.1f}%\n({teacher_hits}/{total_gt})"
            if total_gt
            else f"{teacher_recall * 100:.1f}%"
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=PALETTE["bg"])
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="none", zorder=3)
    _style_ax(ax, "VRU Recall: Student v1 vs. SAM3 Teacher", ylabel="Recall (%)")
    ax.set_ylim(0, 112)
    ax.axhline(85, color=PALETTE["amber"], linestyle=":", linewidth=1.4, label="Target (85%)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    for bar, text in zip(bars, sublabels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.5,
            text,
            ha="center",
            va="bottom",
            color=PALETTE["text"],
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close(fig)
    return out_path


def plot_failure_conditions(
    out_path: Path,
    time_counts: dict[str, int],
    weather_counts: dict[str, int],
) -> Path:
    """Side-by-side bar charts of mined hard frames by time-of-day and weather."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=PALETTE["bg"])

    panels = [
        (axes[0], time_counts, "Mined Failures by Time of Day", PALETTE["blue"]),
        (axes[1], weather_counts, "Mined Failures by Weather", PALETTE["purple"]),
    ]

    for ax, counts, title, color in panels:
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True) or [("none", 0)]
        keys, vals = zip(*items)
        bars = ax.bar(keys, vals, color=color, width=0.55)
        _style_ax(ax, title, ylabel="Hard Frames")
        ax.set_ylim(0, max(vals) * 1.25 + 1)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                str(val),
                ha="center",
                va="bottom",
                color=PALETTE["text"],
                fontsize=9.5,
                fontweight="bold",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close(fig)
    return out_path


def plot_missed_vru_distribution(out_path: Path, missed_per_frame: list[int]) -> Path:
    """Bar chart showing how many VRUs were missed per mined hard frame."""
    counts = Counter(missed_per_frame)
    keys = sorted(counts.keys()) or [0]
    vals = [counts[k] for k in keys]
    labels = [str(k) for k in keys]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=PALETTE["bg"])
    bars = ax.bar(labels, vals, color=PALETTE["amber"], width=0.55)
    _style_ax(
        ax,
        "Missed VRUs per Hard Frame",
        xlabel="Missed VRU Count in Frame",
        ylabel="Number of Frames",
    )
    ax.set_ylim(0, max(vals) * 1.25 + 1)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(val),
            ha="center",
            va="bottom",
            color=PALETTE["text"],
            fontsize=9.5,
            fontweight="bold",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close(fig)
    return out_path


def plot_class_distribution(
    out_path: Path,
    class_counts: dict[str, int],
    class_scores: dict[str, list[float]] | None = None,
) -> Path:
    """Bar chart of SAM3 teacher pseudo-label counts (and avg confidence) per VRU class."""
    order = ["pedestrian", "rider", "bicycle", "motorcycle"]
    keys = [k for k in order if k in class_counts] + [k for k in class_counts if k not in order]
    vals = [class_counts.get(k, 0) for k in keys]
    bar_colors = [PALETTE["green"], PALETTE["blue"], PALETTE["amber"], PALETTE["purple"]][: len(keys)]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=PALETTE["bg"])
    bars = ax.bar(keys, vals, color=bar_colors, width=0.55)
    _style_ax(ax, "SAM3 Teacher Pseudo-Labels by VRU Class", ylabel="Polygon Masks")
    ax.set_ylim(0, max(vals or [1]) * 1.28 + 1)

    for bar, key, val in zip(bars, keys, vals):
        scores = (class_scores or {}).get(key, [])
        label = f"{val}\n(avg {np.mean(scores):.2f})" if scores else str(val)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals or [1]) * 0.03,
            label,
            ha="center",
            va="bottom",
            color=PALETTE["text"],
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close(fig)
    return out_path


def save_metrics(report_dir: Path, metrics: dict) -> Path:
    """Write structured run metrics to metrics.json."""
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **metrics}
    out = report_dir / "metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def build_html_dashboard(
    report_dir: Path,
    title: str,
    metrics: dict,
    chart_paths: list[Path],
    gallery_items: list[dict[str, str]],
) -> Path:
    """Generate a self-contained local HTML dashboard openable in Safari/Chrome."""
    report_dir.mkdir(parents=True, exist_ok=True)
    out_html = report_dir / "dashboard.html"

    def rel(p: str | Path) -> str:
        return Path(p).resolve().as_uri()

    cards_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in metrics.items()
        if not isinstance(v, (dict, list))
    )
    charts_html = "".join(
        f'<div class="chart"><img src="{rel(c)}" alt="{c.name}"/></div>'
        for c in chart_paths
        if c.exists()
    )
    gallery_html = "".join(
        f"""
        <div class="pair">
          <div class="cap">{item.get("caption", "")}</div>
          <div class="imgs">
            {f'<figure><img src="{rel(item["student_img"])}"/><figcaption>Student v1 (Red = Missed VRU)</figcaption></figure>' if item.get("student_img") else ""}
            {f'<figure><img src="{rel(item["teacher_img"])}"/><figcaption>SAM3 Teacher (Green = Masks)</figcaption></figure>' if item.get("teacher_img") else ""}
          </div>
        </div>
        """
        for item in gallery_items[:24]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 28px; }}
  h1 {{ margin: 0 0 6px; font-size: 24px; }}
  .sub {{ color: #94a3b8; margin-bottom: 24px; font-size: 13px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }}
  .k {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .v {{ font-size: 22px; font-weight: 700; margin-top: 6px; color: #38bdf8; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .chart {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 10px; }}
  .chart img {{ width: 100%; border-radius: 6px; display: block; }}
  .pair {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 16px; }}
  .cap {{ font-weight: 600; margin-bottom: 10px; color: #e2e8f0; }}
  .imgs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
  figure {{ margin: 0; }}
  figure img {{ width: 100%; border-radius: 6px; display: block; }}
  figcaption {{ color: #94a3b8; font-size: 12px; margin-top: 6px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="sub">Generated locally at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
  <div class="cards">{cards_html}</div>
  <div class="charts">{charts_html}</div>
  <h2>Sample Mined Frames (up to 24)</h2>
  {gallery_html}
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    return out_html

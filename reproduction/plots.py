"""Paper-numbered multi-panel figures derived exclusively from saved records."""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json
from .config import SYNTHETIC
from .java import LABELS

REAL = ["hepmass", "power", "books"]
TITLES = {"hepmass": "HEPMASS", "power": "Power", "books": "Books",
          "normal_and_distinct_42": "Normal, then frequent items",
          "sorted_with_frequent": "Sorted input with frequent items"}
STYLES = {"SplineSketch": ("#1f77b4", "-"), "KLL": ("#ff7f0e", ":"),
          "MomentSketch": ("#2ca02c", "-."), "t-digest": ("#d62728", "--"),
          "GKAdaptive": ("#9467bd", (5, (10, 3))), "SplineSketch+MG": ("#8c564b", "-")}


def panel(suite, name, metric, x="serialized_bytes"):
    return dict(suite=suite, dataset=name, metric=metric, x=x)


FIGURES = {
    "1": [panel("real", d, "average_error") for d in REAL],
    "4": [panel("synthetic", d, "average_error") for d in SYNTHETIC],
    "5": [panel("real" if d in REAL else "synthetic", d, "maximum_error") for d in REAL + SYNTHETIC],
    "6": [panel("update", "normal", "update_or_merge_us", "actual_n"),
          panel("synthetic", "normal", "update_or_merge_us")],
    "7": [panel("query", "normal", "query_us", "actual_queries"),
          panel("synthetic", "normal", "query_us")],
    "8": [panel("merge", "normal", "maximum_error"), panel("merge", "hepmass", "maximum_error"),
          panel("merge", "normal", "update_or_merge_us")],
    "9": [panel("heavy", "normal_and_distinct_42", "average_error"),
          panel("heavy", "sorted_with_frequent", "average_error"),
          panel("heavy", "normal_and_distinct_42", "update_or_merge_us")],
    "10": [panel("interpolations", "normal", "average_error"),
           panel("heuristicError", "normal", "average_error")],
}


def curve_points(records, spec, limit):
    """Sort tuples of (size,measurement) together, then aggregate repeated trials."""
    groups = {}
    for record in records:
        task = record["task"]
        if task["suite"] != spec["suite"] or task["dataset"] != spec["dataset"] or record["status"] != "ok":
            continue
        if spec["x"] == "serialized_bytes" and record["serialized_bytes"] > limit:
            continue
        label = task["variant"] or LABELS[task["sketch"]]
        parameter = (task["parameter"], task["requested_queries"], task["nominal_n"])
        groups.setdefault((label, parameter), []).append(record)
    curves = {}
    for (label, _), trials in groups.items():
        xs = [r[spec["x"]] for r in trials]
        ys = [r[spec["metric"]] for r in trials]
        curves.setdefault(label, []).append((float(np.mean(xs)), float(np.mean(ys)), float(np.std(ys))))
    order = list(STYLES)
    return {label: sorted(curves[label]) for label in sorted(curves,
            key=lambda label: (order.index(label) if label in order else len(order), label))}


def draw(ax, records, spec, limit):
    curves = curve_points(records, spec, limit)
    metric, x = spec["metric"], spec["x"]
    ylabel = {"average_error": "average rank error", "maximum_error": "maximum rank error",
              "update_or_merge_us": "time per merge [μs]" if spec["suite"] == "merge" else "time per update [μs]",
              "query_us": "time per query [μs]"}[metric]
    ax.set_xlabel({"serialized_bytes": "sketch size in bytes", "actual_n": "data size n",
                   "actual_queries": "number of queries"}[x])
    ax.set_ylabel(ylabel + " (log scale)")
    ax.set_yscale("log")
    if x != "serialized_bytes":
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    if not curves:
        ax.text(0.5, 0.5, "No measurements for this panel", ha="center", transform=ax.transAxes)
        return dict(spec=spec, curves=[], missing=True)
    positive = [point[1] for points in curves.values() for point in points if point[1] > 0]
    floor = min(positive) / 10 if positive else 1e-12
    zero_labels = []
    for label, points in curves.items():
        xx, yy, std = np.array(points).T
        style = STYLES.get(label)
        options = dict(color=style[0], linestyle=style[1]) if style else {}
        ax.plot(xx, np.maximum(yy, floor), label=label, **options)
        if np.any(std > 0):
            ax.fill_between(xx, np.maximum(yy - std, floor), np.maximum(yy + std, floor), alpha=0.12,
                            color=options.get("color"))
        zero = yy == 0
        if zero.any():
            ax.scatter(xx[zero], np.full(zero.sum(), floor), marker="v", color=options.get("color"))
            zero_labels.append(label)
    if metric in ("average_error", "maximum_error") and spec["suite"] in ("real", "synthetic"):
        xx = np.linspace(max(1, min(p[0] for v in curves.values() for p in v)), limit, 200)
        ax.plot(xx, 64 / xx, color="#9467bd", linestyle=(0, (1, 10)), label="error bound (64/bytes)")
    if zero_labels:
        ax.text(0.02, 0.02, "▼ = zero error", fontsize=8, transform=ax.transAxes)
    if spec["suite"] == "merge":
        ax.text(0.02, 0.97, "GKAdaptive merge unavailable", va="top", fontsize=8, transform=ax.transAxes)
    return dict(spec=spec, curves=list(curves), missing=False, zero_display_floor=floor if zero_labels else None)


def export_csv(records, target):
    keys = ["suite", "dataset", "sketch", "variant", "nominal_n", "size", "parameter", "parts", "repeat",
            "requested_queries", "actual_n", "actual_queries", "status", "serialized_bytes",
            "average_error", "maximum_error", "update_or_merge_ns", "query_ns", "update_or_merge_us", "query_us", "wall_seconds", "error"]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for record in records:
            flat = dict(record, **record["task"])
            writer.writerow({k: flat.get(k, "") for k in keys})


def plot_results(output, figure_ids=None):
    output = Path(output)
    metadata = load_json(output / "run.json")
    records = [load_json(p) for p in sorted((output / "records").glob("*.json"))]
    export_csv(records, output / "measurements.csv")
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    report = dict(figures={}, failed_measurements=sum(r["status"] == "failed" for r in records),
                  numerical_failures=sum(r["status"] == "numerical_failure" for r in records))
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    for figure_id in figure_ids or FIGURES:
        specs = FIGURES[figure_id]
        columns = min(3, len(specs))
        rows = (len(specs) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.4 * rows), squeeze=False)
        panels, handles, labels = [], [], []
        for i, spec in enumerate(specs):
            ax = axes.flat[i]
            panels.append(draw(ax, records, spec, metadata["config"]["max_plot_bytes"]))
            title = TITLES.get(spec["dataset"], spec["dataset"].replace("_", " "))
            if figure_id == "10":
                title = "Interpolation" if i == 0 else "Heuristic error"
            ax.set_title(f"({chr(97 + i)}) {title}")
            hs, ls = ax.get_legend_handles_labels()
            for h, label in zip(hs, ls):
                if label not in labels:
                    handles.append(h)
                    labels.append(label)
        for ax in list(axes.flat)[len(specs):]:
            ax.set_visible(False)
        legend_rows = 0
        if handles:
            legend_columns = min(len(labels), 3 if columns == 2 else 4)
            legend_rows = (len(labels) + legend_columns - 1) // legend_columns
            figure.legend(handles, labels, loc="upper center", ncol=legend_columns, fontsize=9)
        subset = metadata["config"]["n"] != 100000000 or metadata["config"]["real_limit"] is not None
        figure.text(0.5, 0.015, f"Figure {figure_id}" + (" — reduced workload" if subset else ""),
                    ha="center", va="bottom", fontsize=10)
        legend_top = 1 - (legend_rows * 0.23 + 0.08) / (3.4 * rows)
        figure.tight_layout(rect=(0, 0.07, 1, legend_top))
        for extension in ("pdf", "png"):
            figure.savefig(directory / f"figure-{figure_id}.{extension}", dpi=160)
        plt.close(figure)
        report["figures"][figure_id] = panels
    # Workloads and parameter studies beyond the paper's displayed panels.
    represented = {(s["suite"], s["dataset"]) for specs in FIGURES.values() for s in specs}
    extras = sorted({(r["task"]["suite"], r["task"]["dataset"]) for r in records} - represented)
    for suite, dataset in extras:
        for metric in ("average_error", "maximum_error", "update_or_merge_us", "query_us"):
            figure, ax = plt.subplots(figsize=(6, 4))
            draw(ax, records, panel(suite, dataset, metric), metadata["config"]["max_plot_bytes"])
            ax.legend()
            figure.tight_layout()
            figure.savefig(directory / f"supplementary-{suite}-{dataset}-{metric}.pdf")
            plt.close(figure)
    # Supplementary parameter studies are also accessible without editing code.
    extra_suites = sorted({r["task"]["suite"] for r in records} -
                         {s["suite"] for specs in FIGURES.values() for s in specs})
    for suite in extra_suites:
        figure, ax = plt.subplots(figsize=(6, 4))
        draw(ax, records, panel(suite, "normal", "average_error"), metadata["config"]["max_plot_bytes"])
        ax.legend()
        figure.tight_layout()
        figure.savefig(directory / f"ablation-{suite}.pdf")
        plt.close(figure)
    write_json(output / "figure-report.json", report)
    return report

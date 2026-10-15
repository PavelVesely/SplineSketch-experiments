"""Build the TeX paper with individual panels from a saved experiment run."""
from pathlib import Path
import re
import shutil
import subprocess

from .common import ROOT, load_json, write_json

# Filenames used by the camera-ready source, in reproduction.plots.FIGURES order.
PANEL_NAMES = {
    "1": [f"{d}_average_nolegend.pdf" for d in ("hepmass", "power", "sosd_books")],
    "4": [f"{d}_logN8_average_nolegend.pdf" for d in (
        "normal", "uniform", "pareto", "lognormal", "loguniform", "signed_loguniform",
        "signed_loguniform_extreme", "normal_with_1_small_change", "normal_with_1_large_change")],
    "5": [f"{d}_max_nolegend.pdf" for d in ("hepmass", "power", "sosd_books")] +
         [f"{d}_logN8_max_nolegend.pdf" for d in (
             "normal", "uniform", "pareto", "lognormal", "loguniform", "signed_loguniform",
             "signed_loguniform_extreme", "normal_with_1_small_change", "normal_with_1_large_change")],
    "6": ["normal_sketchSize100_update_time_nolegend.pdf", "normal_logN8_update_time_nolegend.pdf"],
    "7": ["query_time_normal_sketchSize100_N100000000_LogLog_nolegend.pdf",
          "normal_logN8_query_time_nolegend.pdf"],
    "8": ["normal_logN8_mergingM10000_max_nolegend.pdf", "hepmass_mergingM10000_max_nolegend.pdf",
          "normal_logN8_mergingM10000_update_time_nolegend.pdf"],
    "9": ["normal_and_distinct_42_logN8_withMG_average_nolegend.pdf",
          "sorted_with_frequent_logN8_withMG_average_nolegend.pdf",
          "normal_and_distinct_42_logN8_withMG_update_time_nolegend.pdf"],
    "10": ["interpolations_normal_logN8_average_legend.pdf", "heuristicError_normal_logN8_average_legend.pdf"],
}
LEGENDS = {
    "legend.png": {"1", "4", "5", "8"},
    "legendNoErrorBound.png": {"6", "7"},
    "legendWithMG.png": {"9"},
}
SOURCE = "splineSketchCameraReady.tex"


def plot_paper_panels(records, config, directory):
    # Reuse the same metrics, aggregation, colors and missing-data behavior as the CLI plots.
    from .plots import FIGURES, draw, plt

    directory.mkdir(parents=True, exist_ok=True)
    legends = {name: {} for name in LEGENDS}
    report = {}
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for figure_id, names in PANEL_NAMES.items():
        specs = FIGURES[figure_id]
        if len(names) != len(specs):
            raise ValueError(f"Paper panel mapping differs from Figure {figure_id}")
        for name, spec in zip(names, specs):
            figure, ax = plt.subplots(figsize=(4.3, 3.4))
            try:
                report[name] = draw(ax, records, spec, config["max_plot_bytes"])
                handles, labels = ax.get_legend_handles_labels()
                for legend, ids in LEGENDS.items():
                    if figure_id in ids:
                        legends[legend].update(zip(labels, handles))
                if figure_id == "10" and handles:
                    ax.legend(fontsize=8)
                figure.tight_layout()
                figure.savefig(directory / name)
            finally:
                plt.close(figure)
    for name, entries in legends.items():
        columns = min(4, len(entries)) or 1
        rows = (len(entries) + columns - 1) // columns or 1
        figure = plt.figure(figsize=(12, 0.35 * rows + 0.1))
        try:
            if entries:
                figure.legend(list(entries.values()), list(entries), loc="center", ncol=columns)
            else:
                figure.text(0.5, 0.5, "No measurements for these panels", ha="center", va="center")
            figure.savefig(directory / name, dpi=160, bbox_inches="tight")
        finally:
            plt.close(figure)
    return report


def prepare_paper(results, output, source_dir=ROOT / "paper"):
    """Stage sources and regenerate every experimental asset, without historical plots."""
    results, output, source_dir = Path(results).resolve(), Path(output).resolve(), Path(source_dir).resolve()
    if output == source_dir or source_dir in output.parents or output in source_dir.parents:
        raise ValueError("Choose a build directory outside the paper source directory")
    if not (results / "run.json").is_file():
        raise FileNotFoundError(f"No saved run at {results}. Run reproduce.py run --profile <profile> first.")
    metadata = load_json(results / "run.json")
    paths = sorted((results / "records").glob("*.json"))
    if not paths:
        raise ValueError(f"No measurement records found in {results / 'records'}")
    records = [load_json(path) for path in paths]
    source = source_dir / SOURCE
    if not source.is_file():
        raise FileNotFoundError(f"Missing TeX source: {source}")
    # Ignore TeX comments when checking graphics referenced by the source.
    text = re.sub(r"(?<!\\)%[^\n]*", "", source.read_text())
    graphics = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))
    generated = {name for names in PANEL_NAMES.values() for name in names} | set(LEGENDS)
    static = graphics - generated
    for name in static:
        asset = source_dir / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not asset.is_file():
            raise FileNotFoundError(f"No panel mapping or static paper asset for {name}")
    output.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file() and path.suffix in (".tex", ".bib", ".cls", ".sty", ".bst"):
            shutil.copy2(path, output / path.name)
    for name in static:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / name, target)
    panels = plot_paper_panels(records, metadata["config"], output / "plots")
    report = dict(results=str(results), config=metadata["config"], panels=panels,
                  missing_panels=[name for name, panel in panels.items() if panel["missing"]],
                  failed_measurements=sum(r["status"] == "failed" for r in records),
                  numerical_failures=sum(r["status"] == "numerical_failure" for r in records))
    write_json(output / "paper-build.json", report)
    return report


def build_paper(results, output):
    for command in ("latexmk", "pdflatex", "bibtex"):
        if not shutil.which(command):
            raise RuntimeError(f"{command} is required on PATH. Install a TeX distribution with latexmk.")
    output = Path(output).resolve()
    report = prepare_paper(results, output)
    print(f"Compiling paper in {output}", flush=True)
    process = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                              "-file-line-error", SOURCE], cwd=output, text=True,
                             encoding="utf-8", errors="replace",
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = output / "latexmk.log"
    log.write_text(process.stdout)
    if process.returncode:
        tail = "\n".join(process.stdout.splitlines()[-30:])
        raise RuntimeError(f"Paper compilation failed (see {log}):\n{tail}")
    pdf = output / Path(SOURCE).with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError(f"latexmk did not produce {pdf}; see {log}")
    return pdf, report

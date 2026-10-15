"""Published-paper presets; nominal n differs from actual n for Figure 9(a)."""
from .common import ROOT, load_json

SYNTHETIC = ["normal", "uniform", "pareto", "lognormal", "loguniform",
             "signed_loguniform", "signed_loguniform_extreme",
             "normal_with_1_small_change", "normal_with_1_large_change"]
REAL = ["hepmass", "power", "books"]
SKETCHES = ["splinesketch", "kll", "moments", "tdigest", "gk"]
ALL_DATASETS = SYNTHETIC + REAL + ["gumbel", "signed_lognormal", "distinct_values_42",
                                 "distinct_values_5", "distinct_values_150",
                                 "normal_and_distinct_42", "sorted_with_frequent"]
ABLATIONS = {
    "interpolations": (1, [(3, "PCHIP interpolation"), (1, "linear interpolation")]),
    "heuristicError": (0, [(2, "2nd deriv. of CDF"), (3, "3rd deriv. of CDF"),
                          (1, "bucket counter"), (-1, "bucket length"), (0, "none")]),
    "splitJoinRatio": (2, [(x / 10, f"gamma = {x / 10}") for x in range(11, 21)]),
    "defaultBucketBoundMult": (6, [(x, f"Cb = {x}") for x in range(1, 8)]),
    "minFracBucketBoundToSplit": (4, [(10**x, f"min fraction = {10**x}") for x in range(-5, 0)]),
    "epochIncrFactor": (5, [(x / 10, f"epoch factor = {x / 10}") for x in range(11, 21)]),
}
DEFAULT_PARAMS = [2, 3, 1.5, 1e-8, 0.01, 1.25, 3.0]


def settings(profile, config_path=None):
    cfg = load_json(ROOT / "configs" / f"{profile}.json")
    if config_path:
        overrides = load_json(config_path)
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        cfg.update(overrides)
    return cfg


def plan(cfg, suites, datasets=None, ablations=None):
    """Return unique measurements; Figures 6(b)/7(b) reuse accuracy runs."""
    tasks = []
    selected = set(datasets) if datasets else None
    if selected:
        unknown = selected - set(ALL_DATASETS)
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    def add(suite, names, ns, qs, parts=1, sketches=None, variants=None):
        for name in names:
            if selected and name not in selected:
                continue
            for n in ns:
                for q in qs:
                    for i, size in enumerate(cfg["sizes"]):
                        for sketch in sketches or cfg["sketches"]:
                            k = size
                            if sketch == "moments":
                                k = cfg["moment_k"][i]
                            elif sketch == "kll":
                                k = cfg["kll_k"][i]
                            for label, params in variants or [(None, None)]:
                                tasks.append(dict(suite=suite, dataset=name, nominal_n=n,
                                    requested_queries=q, parts=parts, sketch=sketch, size=size,
                                    parameter=k, variant=label, params=params))

    for suite in dict.fromkeys(suites):
        if suite == "synthetic":
            names = list(selected - set(REAL) - {"normal_and_distinct_42", "sorted_with_frequent"}) if selected else cfg["synthetic"]
            add(suite, sorted(names) if selected else names, [cfg["n"]], [cfg["queries"]])
        elif suite == "real":
            add(suite, REAL if selected else cfg["real"], [None], [cfg["queries"]])
        elif suite == "merge":
            names = ["normal"] + (["hepmass"] if "hepmass" in cfg["real"] or selected and "hepmass" in selected else [])
            add(suite, names, [cfg["n"]], [cfg["queries"]],
                cfg["merge_parts"], sketches=[s for s in cfg["sketches"] if s != "splinesketch_mg"])
        elif suite in ("update", "query"):
            # Fixed-size timing baselines deliberately use k=15/8 for moments/KLL.
            before = len(tasks)
            add(suite, ["normal"], cfg["update_ns"] if suite == "update" else [cfg["n"]],
                [cfg["queries"]] if suite == "update" else cfg["query_counts"])
            tasks[before:] = [dict(t, parameter=15 if t["sketch"] == "moments" else
                8 if t["sketch"] == "kll" else 100, size=100)
                for t in tasks[before:] if t["size"] == cfg["sizes"][0]]
        elif suite == "heavy":
            add(suite, ["normal_and_distinct_42", "sorted_with_frequent"], [cfg["n"]],
                [cfg["queries"]], sketches=list(dict.fromkeys(cfg["sketches"] + ["splinesketch_mg"])))
        elif suite == "ablation":
            for study in ablations or ["interpolations", "heuristicError"]:
                index, values = ABLATIONS[study]
                variants = []
                for value, label in values:
                    params = DEFAULT_PARAMS.copy()
                    params[index] = value
                    variants.append((label, params))
                add(study, ["normal"], [cfg["n"]], [cfg["queries"]],
                    sketches=["adjustable"], variants=variants)
    return tasks

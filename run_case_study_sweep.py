# =====================================================================================
# run_case_study_sweep.py
#
# Runs every case-study notebook across every MACD parameter set we need, which is the
# step that finally makes the published comparisons meaningful: Kang's 22,690 and
# Aguirre's 17.13% are quoted at THEIR OWN optimised parameters, and until now we had only
# ever run (12,26,9). Comparing our (12,26,9) against their optimised figure is not
# like-for-like and is biased against us.
#
# The matrix is the eleven standard configs plus, per market, the parameters each paper
# actually reported, plus the series/window variants each notebook now carries.
#
# Nothing here changes a notebook. Each is executed with a few assignments overridden, so
# the sweep cannot drift from what the notebooks themselves do.
# =====================================================================================

import contextlib
import io
import json
import os
import re
import time

import pandas as pd

# the eleven standard parameter sets, confirmed against results/ and the project transcript
eleven_configs = [
    (3, 10, 16), (5, 13, 3), (5, 15, 9), (6, 13, 4), (8, 17, 9), (8, 24, 9),
    (9, 19, 7), (10, 22, 7), (12, 26, 9), (14, 30, 9), (19, 39, 9),
]

# each paper's own winning parameters, so the comparison can be run on their ground
paper_configs = {
    # Wiles's two GA results, rounded to whole days: (11.017, 32.925, 16.467) and
    # (6.866, 33.812, 4.575)
    "soybean": [(11, 33, 16), (7, 34, 5)],
    # Kang's optimised model
    "nikkei": [(4, 22, 3)],
    # Aguirre 2021's GA (higher 10 / lower 7 / line 15) and 2020's (slow 15, fast 6, signal 3)
    "nasdaq": [(7, 10, 15), (6, 15, 3)],
}

# the series or window variants each notebook now supports, as (assignment, value) pairs
variants = {
    "soybean": [{}],
    "nikkei": [{"price_series": "index"}, {"price_series": "futures"}],
    "nasdaq": [{"validation_window": "initial"}, {"validation_window": "final"}],
}

tex_output = "./results/sweep_tex"
summary_path = "./results/case_study_sweep_summary.csv"


def override(source: str, assignments: dict) -> str:
    """
    override rewrites the top-level assignments a notebook cell makes, so a run can be steered without editing the notebook

    Arguments:
        source -- the cell's source text
        assignments -- {variable name -> literal text to assign}

    Returns:
        the cell source with those assignments replaced
    """
    for name, value in assignments.items():
        # only the first, unindented assignment; the commented alternatives below it and
        # any use of the name elsewhere are left alone
        source = re.sub(
            rf"^{name} = .*$", f"{name} = {value}", source, count=1, flags=re.M
        )
    return source


def run_one(notebook: str, assignments: dict) -> pd.DataFrame:
    """
    run_one executes one notebook end to end under the given overrides and hands back its final comparison table

    Arguments:
        notebook -- notebook filename
        assignments -- overrides applied to every cell, e.g. {"macd_params": "(4, 22, 3)"}

    Returns:
        the notebook's table_data, or None if it produced none
    """
    with io.open(notebook, encoding="utf-8") as handle:
        cells = json.load(handle)["cells"]
    scope = {"__name__": "__main__"}
    exec("import pandas as pd, numpy as np", scope)
    exec("".join(cells[0]["source"]), scope)
    noise = io.StringIO()
    for cell in cells[1:]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        source = override(source, assignments)
        with contextlib.redirect_stdout(noise):
            exec(source, scope)
    return scope.get("table_data")


def run():
    """
    run executes the whole matrix and writes one tidy summary of every cell of every run
    """
    os.makedirs(tex_output, exist_ok=True)
    rows = []
    jobs = []
    for market, notebook in (("soybean", "soybean.ipynb"), ("nikkei", "nikkei.ipynb"),
                             ("nasdaq", "nasdaq.ipynb")):
        for variant in variants[market]:
            for config in eleven_configs + paper_configs[market]:
                jobs.append((market, notebook, variant, config))
    print(f"{len(jobs)} runs queued\n")

    started = time.time()
    for number, (market, notebook, variant, config) in enumerate(jobs, 1):
        assignments = {k: f'"{v}"' for k, v in variant.items()}
        assignments["macd_params"] = str(tuple(config))
        # the notebooks also write into a hardcoded Linux path for the paper; send that
        # here instead so the sweep works on any machine
        assignments["prefix_tex_tables"] = repr(tex_output)
        label = f"{market}/{'/'.join(variant.values()) or 'base'}/{'_'.join(map(str, config))}"
        t0 = time.time()
        try:
            table = run_one(notebook, assignments)
            ok = table is not None
        except Exception as exc:
            print(f"  [{number:>2}/{len(jobs)}] FAILED {label}: {type(exc).__name__}: {exc}")
            continue
        if ok:
            frame = table.reset_index()
            frame.insert(0, "config", "_".join(map(str, config)))
            frame.insert(0, "variant", "/".join(variant.values()) or "base")
            frame.insert(0, "market", market)
            rows.append(frame)
        print(f"  [{number:>2}/{len(jobs)}] {label:44} {time.time()-t0:5.1f}s  "
              f"best {table['Return'].max():>12,.2f}")

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n{len(summary)} rows written to {summary_path}")
    print(f"total {time.time()-started:.0f}s")


if __name__ == "__main__":
    run()

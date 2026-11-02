# =====================================================================================
# run_case_study_spa.py
#
# Hansen's (2005) Superior Predictive Ability test for the case studies.
#
# SPA answers a different question from the deflated Sharpe ratio:
#   DSR asks "is the SELECTED config's Sharpe real, given we searched many?"
#   SPA asks "does the BEST of all the searched configs genuinely beat the BENCHMARK,
#             once you account for having searched them all?"
# The benchmark here is the plain base MACD, which is the right null: the paper's whole
# claim is that the SMACD/CMACD variants improve on it.
#
# This could not be run before because spa_test needs every config's net as an ALIGNED
# equal-length per-observation series, and configs trade different numbers of times. The
# fix is to mark each config to market daily: reconstruct its daily position (+1/-1/0)
# from its bet records and multiply by the daily price change. That gives one row per
# calendar day for every config, all aligned.
# =====================================================================================

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from make_tables_notebooks import spa_test
from run_case_study_significance import run_notebook
from run_case_study_sweep import variants

summary_path = "./results/case_study_spa.csv"

# which validation frame each stashed record belongs to, in the order they are produced
period_order = {
    "soybean.ipynb": ["test_data"] * 6,
    "nasdaq.ipynb": ["test_data"] * 6,
    "nikkei.ipynb": (["first_part", "first_part", "second_part", "second_part",
                      "third_part", "third_part"] * 3),
}


def daily_returns(scope, record, part_name: str) -> pd.DataFrame:
    """
    daily_returns marks every config to market daily, so all configs share one aligned return series

    Each bet gives an entry and exit timestamp and a direction; holding that direction
    between those dates and multiplying by the daily close-to-close change turns a ragged
    list of trades into one row per day, which is what the SPA bootstrap needs.

    Arguments:
        scope -- a finished notebook scope
        record -- one entry of validation_bets
        part_name -- the scope variable holding this record's validation frame

    Returns:
        dataframe of daily P&L, one column per config
    """
    bets = {k: v for k, v in record["bets"].items() if not k.endswith(" long")}
    part = scope[part_name]
    dates = pd.DatetimeIndex(part.index)
    price_change = part["close"].astype(float).diff()
    columns = {}
    for key, blist in bets.items():
        position = pd.Series(0.0, index=dates)
        for bet in blist:
            if len(bet) < 8 or not isinstance(bet[1], (list, tuple)):
                continue
            start, end = pd.Timestamp(bet[0][0]), pd.Timestamp(bet[1][0])
            position.loc[(position.index >= start) & (position.index <= end)] = (
                1.0 if bet[6] == "Long" else -1.0
            )
        columns[key] = position.shift(1) * price_change
    frame = pd.DataFrame(columns).dropna(how="all")
    return frame.loc[:, frame.std() > 0].fillna(0.0)


def run():
    """
    run applies the SPA test to every cell of the base-parameter runs and writes one tidy table
    """
    rows = []
    jobs = [("soybean", "soybean.ipynb", v) for v in variants["soybean"]]
    jobs += [("nikkei", "nikkei.ipynb", v) for v in variants["nikkei"]]
    jobs += [("nasdaq", "nasdaq.ipynb", v) for v in variants["nasdaq"]]

    for market, notebook, variant in jobs:
        assignments = {k: f'"{v}"' for k, v in variant.items()}
        assignments["macd_params"] = "(12, 26, 9)"
        assignments["prefix_tex_tables"] = repr("./results/sweep_tex")
        label = f"{market}/{'/'.join(variant.values()) or 'base'}"
        scope = run_notebook(notebook, assignments)
        parts = period_order[notebook]

        for index, record in enumerate(scope.get("validation_bets", [])):
            if index >= len(parts):
                break
            frame = daily_returns(scope, record, parts[index])
            benchmark = record["chosen"]["base"]
            if benchmark not in frame.columns or frame.shape[1] < 5:
                continue
            net_by_config = {c: frame[c].to_numpy() for c in frame.columns}
            try:
                result = spa_test(net_by_config, benchmark, reps=500)
            except Exception as exc:
                print(f"  {label} cell {index}: SPA failed ({type(exc).__name__})")
                continue
            system = ("System 3" if record["signal_line_if_target"]
                      else "System 2" if record["signal_line_if"] else "System 1")
            rows.append({
                "market": market,
                "variant": "/".join(variant.values()) or "base",
                "crossover": "MACDCROSS" if record["name_mode"] == "Histogram" else "ZEROCROSS",
                "system": system,
                "period": parts[index],
                "days": frame.shape[0],
                "configs": frame.shape[1],
                "benchmark": benchmark,
                "best_config": result["best_config"],
                "best_mean_edge": result["best_mean_edge"],
                "spa_p": result["p_value"],
                "spa_p_lower": result["p_value_lower"],
                "spa_p_upper": result["p_value_upper"],
            })
            print(f"  {label:18} {system} {rows[-1]['crossover']:10} "
                  f"days {frame.shape[0]:>4}  SPA p = {result['p_value']:.4f}  "
                  f"edge {result['best_mean_edge']:+.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(summary_path, index=False)
    print(f"\n{len(table)} cells written to {summary_path}")
    if len(table):
        print(f"median SPA p-value: {table.spa_p.median():.4f}")
        print(f"cells with p < 0.05: {int((table.spa_p < 0.05).sum())} of {len(table)}")
        print(f"cells with p < 0.10: {int((table.spa_p < 0.10).sum())} of {len(table)}")


if __name__ == "__main__":
    run()

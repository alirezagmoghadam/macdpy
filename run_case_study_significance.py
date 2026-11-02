# =====================================================================================
# RETIRED AS A DRIVER, 30 Oct 2026 (user decision): the Deflated Sharpe Ratio was dropped
# from the paper, so this file's run() -- which computed DSR for every case-study cell and
# wrote results/case_study_significance.csv -- has been removed, along with the
# deflated_sharpe import. See block D of make_tables_notebooks.py for the measured reason,
# and memory/significance-results.md.
#
# THE FILE IS KEPT, NOT DELETED, because run_case_study_spa.py:28 does
#     from run_case_study_significance import run_notebook
# so this module is still the shared notebook-runner and bet-decoder for the SPA driver.
# Deleting it breaks case-study SPA. The name is kept so that import keeps working.
# Anything still live below here is a HELPER, not a driver: bet_profit, bet_return,
# returns_of, run_notebook, check_decoding.
# =====================================================================================

# =====================================================================================
# run_case_study_significance.py
#
# Applies the Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) to every case-study
# cell, which is what turns "our best config returned X" into a claim that survives the
# fact that we searched 475 configs to find it.
#
# Why DSR and not SPA here. spa_test needs each config's net as a per-observation series
# with every config ALIGNED to the same length; in the main experiment those observations
# are the walk-forward windows. A case study has one validation window per sub-period, and
# the natural observation unit is the individual trade -- but configs trade different
# numbers of times, so they cannot be aligned without first reconstructing daily
# mark-to-market series for all 475. DSR needs no alignment: the selected config's own
# return series, plus the sharpe of every trial, whose spread sets the hurdle.
#
# Bet record format, from modify_bet:
#   [(open ts, open px), (close ts, close px), amount, balance before opening,
#    balance after opening, balance after closing, "Long"/"Short", "Win"/"Loss"]
# With balance = 0 (the one_pos scenario) the three balance fields are all 0, so P&L has
# to come from the prices. The decoding is checked against the engine's own net before any
# statistic is computed.
# =====================================================================================

import contextlib
import io
import json
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from run_case_study_sweep import (eleven_configs, override, paper_configs, variants)



def bet_profit(bet: list) -> float:
    """
    bet_profit the realised P&L of one closed bet, in price units

    Arguments:
        bet -- one record from the engine's bets list

    Returns:
        profit in price units, or nan if the bet never closed
    """
    if len(bet) < 8 or not isinstance(bet[1], (list, tuple)):
        return float("nan")
    open_price, close_price, amount, direction = bet[0][1], bet[1][1], bet[2], bet[6]
    sign = 1.0 if direction == "Long" else -1.0
    return sign * (close_price - open_price) * amount


def bet_return(bet: list) -> float:
    """
    bet_return the realised FRACTIONAL return of one closed bet, which is what a sharpe ratio should be built from

    Arguments:
        bet -- one record from the engine's bets list

    Returns:
        fractional return, or nan if it cannot be formed
    """
    profit = bet_profit(bet)
    entry = abs(bet[0][1]) * abs(bet[2]) if len(bet) >= 3 else 0
    if not np.isfinite(profit) or entry == 0:
        return float("nan")
    return profit / entry


def returns_of(bets: list) -> np.ndarray:
    """
    returns_of the fractional returns of every closed bet in one config's bet list

    Arguments:
        bets -- the engine's bet list for a single config

    Returns:
        array of finite per-trade returns
    """
    values = np.array([bet_return(b) for b in bets], dtype=float)
    return values[np.isfinite(values)]


def run_notebook(notebook: str, assignments: dict):
    """
    run_notebook executes one notebook under overrides and returns its scope, including the validation_bets side channel

    Arguments:
        notebook -- notebook filename
        assignments -- overrides applied to every cell

    Returns:
        the notebook's globals dict
    """
    with io.open(notebook, encoding="utf-8") as handle:
        cells = json.load(handle)["cells"]
    scope = {"__name__": "__main__"}
    exec("import pandas as pd, numpy as np", scope)
    exec("".join(cells[0]["source"]), scope)
    for name in ("write_table_tex", "write_overall_table_tex", "write_to_text",
                 "write_rules_tex"):
        scope[name] = lambda *a, **k: None
    noise = io.StringIO()
    for cell in cells[1:]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        with contextlib.redirect_stdout(noise):
            exec(override(source, assignments), scope)
    return scope


def check_decoding(scope) -> str:
    """
    check_decoding confirms the bet decoding reproduces the engine's own net, so the statistics are not built on a guess

    Arguments:
        scope -- a finished notebook scope, whose dff is the last run's validation frame

    Returns:
        a short verdict string
    """
    frame, records = scope.get("dff"), scope.get("validation_bets")
    if frame is None or not records:
        return "no dff/validation_bets to check against"
    bets = records[-1]["bets"]
    worst, checked = 0.0, 0
    for _, row in frame.iterrows():
        key = row["his"]
        if key not in bets:
            continue
        ours = float(np.nansum([bet_profit(b) for b in bets[key]]))
        worst = max(worst, abs(ours - float(row["net"])))
        checked += 1
    return f"{checked} configs checked, worst |bet-sum - engine net| = {worst:.6f}"

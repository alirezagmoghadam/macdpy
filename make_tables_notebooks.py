# =====================================================================================
# make_tables_notebooks.py
#
# This is the SAME pipeline as make_tables.py, but built by literally reusing MY OWN
# logic from the four notebooks (1 - makemotherfile, 2 - log_of_returns,
# 3 - table_of_returns, 4 - get_table), with ONLY the confirmed bugs changed.
# Purpose: run this AND make_tables.py, then diff the outputs. If they match, both are
# correct. This one writes to results/tables_check/ so it never touches tables_new/.
#
# Every place I changed something from the original notebook, I marked it with:  # <-- FIX
# =====================================================================================

import json
import math
import os
from collections.abc import Callable
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from arch.bootstrap import SPA, IIDBootstrap, MovingBlockBootstrap

# -------------------------------------------------------------------------------------
# config  (kept the same names/shape as my notebooks, just pointed at the new data)
# -------------------------------------------------------------------------------------
RESULTS_DIR = "results"
OUT_DIR = "results/tables_check"                       # <-- FIX: new folder, don't overwrite
FULL_DIR = "results/tables_full"                       # gross table + post-hoc cost/risk/CI columns
RANDOM_DIR = "results/random"                           # where bh.ipynb saves the random benchmark

# post-hoc cost sweep: per-leg bps for regular hours; extended-hours legs pay a multiple (thinner
# liquidity). 1-2 bps is the realistic level for liquid mega-caps; 5/10/20 are stress tests.
cost_bps_levels = [0, 1, 2, 5, 10, 20]
# 1, per the 28-Oct decision: the headline claim is "every leg pays N bps", and a multiplier
# makes that ambiguous because some legs would quietly pay 2N. bet_cost still prices each leg by
# its own timestamp's session (in_regular_session), so raising this is all it takes to charge
# off-hours legs more.
#
# BE HONEST ABOUT WHICH WAY THIS LEANS: 1 is the OPTIMISTIC choice, and it is not a small one --
# measured on AAPL, 58.1% of 5-minute bars and 62.5% of hourly bars are outside the regular
# session, so a large share of intraday legs are being charged the tighter rate. Two things make
# that defensible rather than convenient. The cost sweep reports 0/1/2/5/10/20 bps, so a reader
# who thinks off-hours legs cost double can read the 2 bps column as the 1 bps-with-doubling
# case. And the regular-session guardrail run (--session regular) answers the underlying
# objection directly, by excluding extended hours altogether instead of pricing it.
extended_multiple = 1
# drawdown/sharpe/sortino/calmar reported at EVERY cost level (same as the returns; trim in LaTeX).
# MUST stay a subset of cost_bps_levels (ticker_contribution computes risk stats inside the cost loop).
risk_bps_levels = cost_bps_levels
ci_reps = 2000                                         # bootstrap resamples for the confidence intervals

# non-overlapping-window robustness: validation windows overlap heavily (stride << window), so keep
# every Nth chronological window to get an independent subset. N = ceil(validation length / stride):
# 1_day 12mo/3mo=4, 1_hour 3mo/2wk~=7, 5_minute 8wk/1wk=8.
nonoverlap_step = {"1_day": 4, "1_hour": 7, "5_minute": 8}

# Per-regime reporting. Each VALIDATION WINDOW is labelled bull/bear/flat from the
# cross-ticker average buy-and-hold return over that same window, which is a market-return
# proxy measured on exactly our unit of analysis. Two thresholds are reported because the
# cut is a judgement call and a single one invites the charge that it was chosen to suit.
#
# READ THIS BEFORE WRITING IT UP: the regime is classified from the SAME window's realised
# return, so this is ex-post conditional analysis ("in windows where the market fell, the
# strategy did X"). It is NOT a tradable timing rule and must not be presented as one.
#
# Daily has only ~12 windows in total, so per-regime daily is too thin to claim anything
# and is emitted for description only.
regime_thresholds = (0.02, 0.05)
regime_names = ("bull", "bear", "flat")

# the base macd params. 12_26_9 is the main paper config; the other 10 feed the appendix robustness.
macd_params_list = [
    (12, 26, 9), (9, 19, 7), (14, 30, 9), (10, 22, 7), (6, 13, 4),
    (19, 39, 9), (8, 24, 9), (8, 17, 9), (5, 15, 9), (3, 10, 16), (5, 13, 3),
]

# (time_format, time_value) pairs.
# ORIGINAL nb1 had: [("1_day","12"), ("1_hour","8"), ("5_minute","4")]  -- the 8 and 4 were WRONG
time_formats_value_list = [("1_day", "12"), ("1_hour", "3"), ("5_minute", "8")]   # <-- FIX: real values on disk

experiments_list = ["one_pos", "simulation"]          # scenario  (= old target_dir)
signal_if_list = [(False, False), (True, False), (False, True)]  # (sys1, sys2, sys3) = old target_experiment

# map a signal-combo to a readable system name (only used for the output filename)
system_name_of = {(False, False): "System1", (True, False): "System2", (False, True): "System3"}

# only the two NAKED obs modes exist in the new data (old data also had the stoploss ones)
obs_modes = ["observations_MACDCROSS_naked", "observations_ZEROCROSS_naked"]   # <-- FIX: dropped stoploss

# the 32 tickers = the 34 downloaded minus the fb/meta duplicate (same as nb1 after the two .remove() calls)
data_files = [
    "aapl.us.txt", "amgn.us.txt", "amzn.us.txt", "axp.us.txt", "ba.us.txt", "cat.us.txt",
    "crm.us.txt", "csco.us.txt", "cvx.us.txt", "dis.us.txt", "googl.us.txt", "gs.us.txt",
    "hd.us.txt", "hon.us.txt", "ibm.us.txt", "jnj.us.txt", "jpm.us.txt", "ko.us.txt",
    "mcd.us.txt", "mmm.us.txt", "mrk.us.txt", "msft.us.txt", "nflx.us.txt", "nke.us.txt",
    "nvda.us.txt", "pg.us.txt", "shw.us.txt", "trv.us.txt", "unh.us.txt", "v.us.txt",
    "vz.us.txt", "wmt.us.txt",
]
ticker_length = len(data_files)                        # <-- FIX: 32, NOT the hardcoded 6 in old nb4

# the 5 rows in every ticker/timespan matrix, in the order the engine writes them
# (confirmed in experiment_main.py get_result_from_df: rule1, rule2, rule3, GA, MACD)
entries = ["Rule 1", "Rule 2", "Rule 3", "GA", "MACD"]   # <-- FIX: old nb4 had 4 rows [Rule1,Rule2,Rule3,MACD]


def load_json_file(path):
    # same helper as in the notebooks (experiment_main.load_json_file)
    with open(path) as fh:
        return json.load(fh)


# -------------------------------------------------------------------------------------
# nb2 helpers (init_results_list / add_results)  -- copied, row count made dynamic
# -------------------------------------------------------------------------------------
def init_results_list(input_row, n_rows):
    # ORIGINAL made exactly 4 zero-rows for stocks. new data has 5 rows -> derive it.  # <-- FIX
    # input_row is one ticker-row (18 long); we drop the last col ('his', a string) -> 17 metrics
    list_length = len(input_row)
    return [[0 for _ in range(list_length - 1)] for _ in range(n_rows)]


def add_results(results_list, row_list):
    # element-wise add one ticker's [rows x metrics] matrix into the accumulator (same as nb2)
    outer_length = len(results_list)
    inner_length = len(results_list[0])
    for i in range(outer_length):
        for j in range(inner_length):
            results_list[i][j] += row_list[i][j]


# -------------------------------------------------------------------------------------
# nb3 helpers (init_conclude_list / conclude_results)  -- copied, row count made dynamic
# -------------------------------------------------------------------------------------
def init_conclude_list(input_matrix):
    # ORIGINAL made 4 zero-rows. derive rows/cols from the per-timespan matrix.  # <-- FIX
    n_rows = len(input_matrix)
    list_length = len(input_matrix[0])
    return [[0 for _ in range(list_length)] for _ in range(n_rows)]


def conclude_results(results_list, input_list):
    # element-wise add one timespan's summed matrix into the grand accumulator (same as nb3)
    outer_length = len(results_list)
    inner_length = len(results_list[0])
    for i in range(outer_length):
        for j in range(inner_length):
            results_list[i][j] += input_list[i][j]


# -------------------------------------------------------------------------------------
# nb4 helper (get_obs_mode_data)  -- copied verbatim from get_table nb, only ticker_length fixed
# raw metric columns of obs_mode_data[i]:
# [0'net', 1'netbet', 2'netpercent', 3'bets', 4'wins', 5'longs', 6'long_wins', 7'long_profit',
#  8'shorts', 9'short_wins', 10'short_profit', 11'long_only_net', 12'long_only_bets',
#  13'long_only_wins', 14'long_only_net_other', 15'long_only_bets_other', 16'long_only_wins_other']
# -------------------------------------------------------------------------------------
def get_obs_mode_data(obs_mode_data):
    # one output row per input row (Rule1, Rule2, Rule3, GA, MACD)
    return_list = [[] for _ in range(len(obs_mode_data))]     # <-- FIX: was [[],[],[],[]] (only 4)
    for i in range(len(obs_mode_data)):
        # avg return for ticker , bets , win/loss , avg long return , longs , long win ratio ,
        # avg short return , short win ratio , then the two long-only variants
        # 0/tickers , 3 , 4/3*% , 7/tickers , 5 , 6/5*% , 10/tickers , 9/8*% ,
        # 11/tickers , 12 , 13/12*% , 14/tickers , 15 , 16/15*%
        row = obs_mode_data[i]
        return_list[i].extend([
            (row[0] / ticker_length),
            row[3],
            (row[4] * 100 / row[3]) if row[3] else 0,       # guard /0 (aggregate is never 0 in practice)
            (row[7] / ticker_length),
            row[5],
            (row[6] * 100 / row[5]) if row[5] else 0,
            (row[10] / ticker_length),
            (row[9] * 100 / row[8]) if row[8] else 0,
            (row[11] / ticker_length),
            row[12],
            (row[13] * 100 / row[12]) if row[12] else 0,
            (row[14] / ticker_length),
            row[15],
            (row[16] * 100 / row[15]) if row[15] else 0,
        ])
    return return_list


table_columns = [
    "Average Return", "Transactions", "Win Ratio",
    "Average Long Return", "Long Transactions", "Long Win Ratio",
    "Average Short Return", "Short Win Ratio",
    "Average Long 1 Return", "Long 1 Transactions", "Long 1 Win Ratio",
    "Average Long 2 Return", "Long 2 Transactions", "Long 2 Win Ratio",
]

# the post-hoc columns appended to each gross row: after-cost average return per cost level, then
# drawdown/sharpe/sortino at each risk level. every one of these (and the gross ones) also gets a CI.
cost_columns = [f"AvgReturn_{bps}bps" for bps in cost_bps_levels]
risk_columns = [
    f"{metric}_{bps}bps"
    for bps in risk_bps_levels
    for metric in ("MaxDrawdown", "Sharpe", "Sortino", "Calmar")
]
extended_columns = table_columns + cost_columns + risk_columns


# =====================================================================================
# POST-HOC STATS (block A: proportional costs, session split, intra-window drawdown)
#
# these read the {tick}_bets files the engine now saves (the 5 chosen configs' trimmed bets per
# validation window) and re-walk them to answer "does the edge survive real trading costs?".
# a trimmed bet is a flat list (experiment_main.trim_bet):
#   [0 entry_ts, 1 entry_price, 2 exit_ts, 3 exit_price, 4 amount, 5 balance_after_close, 6 side, 7 outcome]
# note: balance_after_close is only meaningful in the 'simulation' scenario; in 'one_pos' the engine
# never updates balance (it trades a single unit), so for one_pos we work straight off the prices.
# =====================================================================================

# engine constants we have to mirror so the re-walk reproduces the engine exactly (see experiment_main)
starting_balance = 20000                      # the balance each validation window starts from (simulation)
risk_per_trade = 25 / 100                      # get_result_from_df passes 0.25 down to the sizing formula
stop_fraction = 3 / 4                          # the 75% capital stop: halt the window once balance < 75% of start


def in_regular_session(entry_ts: str) -> bool:
    """
    in_regular_session returns whether a trade's entry timestamp falls inside the official US regular session (09:30-16:00 ET)

    Arguments:
        entry_ts -- the entry timestamp as a utc string (all our data is utc)

    Returns:
        True if the trade entered during regular hours, False if it entered pre/post market (extended hours)
    """
    # convert the utc timestamp to the new york wall-clock (zoneinfo via pandas handles daylight
    # saving for us) and check it against the regular session minutes. used to charge extended-hours
    # trades a wider cost.
    ts_new_york = pd.Timestamp(entry_ts, tz="UTC").tz_convert("America/New_York")
    minutes = ts_new_york.hour * 60 + ts_new_york.minute
    return (9 * 60 + 30) <= minutes < (16 * 60)


def leg_cost(amount: float, price: float, bps: float) -> float:
    """
    leg_cost proportional cost of trading one leg (an open OR a close): bps basis points of the leg notional (shares * price)

    Arguments:
        amount -- number of shares/units in the leg
        price -- price the leg trades at
        bps -- cost in basis points (1 bp = 0.01%)

    Returns:
        the dollar cost of the leg
    """
    # mirror of commission_cost in experiment_main.py (same proportional per-leg model)
    return (bps / 10000) * amount * price


def bet_cost(
    entry_price: float,
    exit_price: float,
    amount: float,
    entry_ts: str,
    exit_ts: str,
    bps_regular: float,
    bps_extended: float,
) -> float:
    """
    bet_cost total round-trip cost of one bet: the open leg plus the close leg, each charged at the regular or extended-hours rate for the session that leg actually traded in

    Arguments:
        entry_price -- fill price when the position opened
        exit_price -- fill price when the position closed
        amount -- number of shares/units in the position
        entry_ts -- entry timestamp (utc string), picks the open leg's rate
        exit_ts -- exit timestamp (utc string), picks the close leg's rate
        bps_regular -- per-leg cost in basis points for regular-session trades
        bps_extended -- per-leg cost in basis points for extended-hours trades

    Returns:
        the dollar cost of the whole round trip
    """
    # a round trip pays the cost twice, once getting in and once getting out. each leg is priced by
    # its own timestamp's session (extended-hours legs pay the wider rate; spreads are thinner off-hours).
    entry_bps = bps_regular if in_regular_session(entry_ts) else bps_extended
    exit_bps = bps_regular if in_regular_session(exit_ts) else bps_extended
    return leg_cost(amount, entry_price, entry_bps) + leg_cost(amount, exit_price, exit_bps)


def rewalk_window(
    window_bets: list, scenario: str, bps_regular: float, bps_extended: float
) -> dict:
    """
    rewalk_window replays one validation window's bets for a single config with a proportional per-leg cost, reproducing the engine's sizing and 75% capital stop

    Arguments:
        window_bets -- list of trimmed bets for one config in one validation window, in order
        scenario -- 'simulation' (compounds real capital, has the stop) or 'one_pos' (one unit, no stop)
        bps_regular -- per-leg cost in basis points for regular-session trades
        bps_extended -- per-leg cost in basis points for extended-hours trades

    Returns:
        dict with 'net' (after-cost net for the window), 'equity_path' (balance after each taken trade), 'returns' (per-trade after-cost return as a fraction of the capital deployed), 'n_trades' (trades actually taken after the stop), 'total_cost' (dollars of cost charged)
    """
    # the two scenarios are handled exactly as the engine does: simulation starts from real capital and
    # compounds, one_pos trades a single unit and never touches a running balance.
    balance = starting_balance
    threshold = starting_balance * stop_fraction
    cumulative_pnl = 0.0
    total_cost = 0.0
    taken = 0
    # equity path we track drawdown on: real balance for simulation, cumulative pnl for one_pos
    equity_path = [starting_balance] if scenario == "simulation" else [0.0]
    # per-trade after-cost returns (fraction of capital deployed), feed sharpe/sortino later
    returns = []

    for bet in window_bets:
        entry_ts, entry_price, exit_ts, exit_price = bet[0], bet[1], bet[2], bet[3]
        # side sign: long profits when price rises, short profits when it falls
        side_sign = 1 if bet[6] == "Long" else -1

        if scenario == "simulation":
            # engine sizing on the CURRENT (cost-adjusted) balance, whole shares only
            amount = (balance * risk_per_trade) // entry_price
        else:
            # one_pos always trades exactly one unit
            amount = 1

        # per-share pnl is direction * (exit - entry), scaled by the number of shares
        pnl = side_sign * (exit_price - entry_price) * amount
        cost = bet_cost(entry_price, exit_price, amount, entry_ts, exit_ts, bps_regular, bps_extended)

        balance += pnl - cost
        cumulative_pnl += pnl - cost
        total_cost += cost
        taken += 1

        # per-trade return = after-cost pnl over the capital deployed (entry notional). guard the rare
        # whole-share-sizing case where amount rounds to 0 (price above the risked capital) -> no trade return.
        notional = entry_price * amount
        if notional > 0:
            returns.append((pnl - cost) / notional)

        if scenario == "simulation":
            equity_path.append(balance)
            # the engine checks the 75% stop after a close and before the next open -> stop the window
            if balance < threshold:
                break
        else:
            equity_path.append(cumulative_pnl)

    net = (balance - starting_balance) if scenario == "simulation" else cumulative_pnl
    return {
        "net": net,
        "equity_path": equity_path,
        "returns": returns,
        "n_trades": taken,
        "total_cost": total_cost,
    }


def max_drawdown_from_equity(equity_path: list, scenario: str) -> float:
    """
    max_drawdown_from_equity worst peak-to-trough drop along a window's equity path

    Arguments:
        equity_path -- the balance (simulation) or cumulative pnl (one_pos) after each taken trade
        scenario -- 'simulation' (return a fraction of the running peak) or 'one_pos' (return dollars)

    Returns:
        the max drawdown (fraction of peak for simulation, dollars for one_pos); 0 for an empty/single-trade window
    """
    # simulation -> fraction of the running peak (a percent, the usual account drawdown); one_pos ->
    # dollars of cumulative pnl (no capital base to divide by).
    if len(equity_path) < 2:
        return 0.0
    peak = equity_path[0]
    max_drawdown = 0.0
    for value in equity_path:
        if value > peak:
            peak = value
        if scenario == "simulation":
            drawdown = (peak - value) / peak if peak > 0 else 0.0
        else:
            drawdown = peak - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def sharpe_ratio(returns: list) -> float:
    """
    sharpe_ratio per-trade sharpe ratio of a return series: the mean return divided by its unbiased standard deviation

    Arguments:
        returns -- list of per-trade (fractional) returns

    Returns:
        the (non-annualized) sharpe ratio; 0 when there are fewer than two trades or no variability
    """
    # need at least two trades for any dispersion
    n = len(returns)
    if n < 2:
        return 0.0
    mean_return = sum(returns) / n
    # unbiased sample standard deviation (ddof=1) of the trade returns
    std = (sum((r - mean_return) ** 2 for r in returns) / (n - 1)) ** 0.5
    return mean_return / std if std > 0 else 0.0


def sharpe_sortino(returns: list, periods_per_year: float = None) -> dict:
    """
    sharpe_sortino risk-adjusted performance from a series of per-trade returns: sharpe divides the mean return by the full standard deviation, sortino divides only by the downside deviation

    Arguments:
        returns -- list of per-trade (fractional) after-cost returns

    Keyword Arguments:
        periods_per_year -- trades per year to annualize by with sqrt-time scaling; None leaves the ratios per-trade (default: {None})

    Returns:
        dict with 'sharpe' and 'sortino' (annualized when periods_per_year is given); both 0 when there are fewer than two trades or no variability
    """
    # need at least two trades for any dispersion
    n = len(returns)
    if n < 2:
        return {"sharpe": 0.0, "sortino": 0.0}
    mean_return = sum(returns) / n
    # downside deviation only counts the losing trades (returns below zero), the sortino denominator
    downside = [r for r in returns if r < 0]
    downside_std = (
        (sum(r ** 2 for r in downside) / (n - 1)) ** 0.5 if downside else 0.0
    )
    # sharpe reuses the shared helper so its definition stays identical everywhere it is used
    sharpe = sharpe_ratio(returns)
    sortino = mean_return / downside_std if downside_std > 0 else 0.0
    # annualize with the usual sqrt-time scaling if a trade frequency is given
    if periods_per_year:
        scale = periods_per_year ** 0.5
        sharpe *= scale
        sortino *= scale
    return {"sharpe": sharpe, "sortino": sortino}


def bootstrap_ci_tickers(
    per_ticker_values: list,
    aggregate_fn: Callable,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    bootstrap_ci_tickers percentile confidence interval for a statistic by resampling whole tickers with replacement, answering "did a few lucky tickers carry the result?" (the tickers are the cross-sectionally independent unit)

    Arguments:
        per_ticker_values -- list with one entry per ticker (whatever aggregate_fn needs, e.g. a per-ticker net)
        aggregate_fn -- function mapping a resampled list of ticker entries to the scalar statistic

    Keyword Arguments:
        n_boot -- number of bootstrap resamples (default: {10000})
        alpha -- two-sided level, so the interval is the (alpha/2, 1-alpha/2) percentiles (default: {0.05})
        seed -- rng seed for reproducibility (default: {42})

    Returns:
        dict with 'low', 'high' (the percentile interval) and 'point' (the statistic on the full sample)
    """
    values = np.asarray(per_ticker_values, dtype=float)
    point = float(aggregate_fn(values))
    if len(values) < 2:
        # nothing to resample over
        return {"low": point, "high": point, "point": point}
    # arch's IIDBootstrap resamples whole rows (tickers) with replacement; conf_int applies our
    # statistic to each resample and returns the two-sided percentile interval.
    bootstrap = IIDBootstrap(values, seed=seed)
    interval = bootstrap.conf_int(
        aggregate_fn, reps=n_boot, method="percentile", size=1 - alpha
    )
    return {"low": float(interval[0, 0]), "high": float(interval[1, 0]), "point": point}


def bootstrap_ci_blocks(
    per_window_values: list,
    aggregate_fn: Callable,
    block_len: int = 3,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    bootstrap_ci_blocks percentile confidence interval for a statistic by resampling the ordered validation windows in contiguous blocks (a moving-block bootstrap), answering "did a few lucky periods carry the result?" while preserving the temporal dependence of overlapping walk-forward windows

    Arguments:
        per_window_values -- list with one entry per validation window, in time order
        aggregate_fn -- function mapping a resampled list of window entries to the scalar statistic

    Keyword Arguments:
        block_len -- length of each contiguous block drawn (default: {3})
        n_boot -- number of bootstrap resamples (default: {10000})
        alpha -- two-sided level, so the interval is the (alpha/2, 1-alpha/2) percentiles (default: {0.05})
        seed -- rng seed for reproducibility (default: {42})

    Returns:
        dict with 'low', 'high' (the percentile interval) and 'point' (the statistic on the full sample)
    """
    values = np.asarray(per_window_values, dtype=float)
    t = len(values)
    point = float(aggregate_fn(values))
    if t < 2:
        # nothing to resample over
        return {"low": point, "high": point, "point": point}
    # arch's MovingBlockBootstrap draws contiguous blocks of 'block_len' ordered windows, so the
    # temporal dependence of overlapping walk-forward windows is preserved in each resample.
    bootstrap = MovingBlockBootstrap(max(1, min(block_len, t)), values, seed=seed)
    interval = bootstrap.conf_int(
        aggregate_fn, reps=n_boot, method="percentile", size=1 - alpha
    )
    return {"low": float(interval[0, 0]), "high": float(interval[1, 0]), "point": point}


# -------------------------------------------------------------------------------------
# block D: data-snooping correction (we searched ~475 configs and reported the winner)
# The test is Hansen's (2005) SPA via arch (spa_test, below); it subsumes White's Reality
# Check, so there is no separate hand-rolled RC.
#
# The Deflated Sharpe Ratio was REMOVED on 30 Oct 2026 (user decision) after execution showed
# it unusable. It was fed the selected config's net LEVEL, so it asked "did this make money?"
# instead of "did it beat MACD after 475 tries?", and it returned dsr = 1.0000 on NULL data
# wherever the configs shared any common positive drift (measured: 1.0000 at drift 0.1, 0.3
# and 1.0, with the hurdle pinned near 0.02 because it came from the spread of near-identical
# configs). Both repair routes tested WORSE: excess-over-benchmark alone gave 0.39/0.76/0.13
# on null data, and charging the settled "~3 effective trials" gave 0.93/0.99, because the
# trial count and the hurdle spread were derived from the SAME list, so fewer trials meant a
# SMALLER hurdle. SPA on the identical null data behaved correctly (p = 0.1800, no rejection),
# so SPA carries the snooping claim alone.
# See memory/significance-results.md and scratchpad/verify_round2.py.
# Do NOT reintroduce a DSR without first validating it against a null simulation.
# -------------------------------------------------------------------------------------


def spa_test(
    net_by_config: dict,
    benchmark_key: str,
    reps: int = 1000,
    block_size: int = None,
    seed: int = 42,
) -> dict:
    """
    spa_test Hansen's (2005) Superior Predictive Ability test via the arch library: accounting for having searched many configs, does the BEST one genuinely beat the benchmark? arch works on losses (lower is better), so we feed it negative nets and read the consistent SPA_c p-value

    Arguments:
        net_by_config -- dict mapping each config key to its per-observation net (all aligned, same length)
        benchmark_key -- the key of the benchmark config (MACD)

    Keyword Arguments:
        reps -- number of bootstrap resamples arch uses (default: {1000})
        block_size -- stationary-bootstrap block length; None lets arch pick the optimal length (default: {None})
        seed -- rng seed for reproducibility (default: {42})

    Returns:
        dict with 'p_value' (the consistent SPA_c p-value; small = the best config really beats the benchmark), 'p_value_lower' and 'p_value_upper' (Hansen's bracketing variants), 'best_config' and 'best_mean_edge' (the winning config and its average outperformance)
    """
    benchmark = np.asarray(net_by_config[benchmark_key], dtype=float)
    keys = [key for key in net_by_config if key != benchmark_key]
    # arch tests for LOWER loss = better, so losses are negative nets (a higher net is a lower loss)
    benchmark_loss = -benchmark
    model_losses = np.column_stack(
        [-np.asarray(net_by_config[key], dtype=float) for key in keys]
    )
    spa = SPA(benchmark_loss, model_losses, reps=reps, block_size=block_size, seed=seed)
    spa.compute()
    pvalues = spa.pvalues
    # report which config won on average outperformance (net over benchmark)
    mean_edge = np.array(
        [(np.asarray(net_by_config[key], dtype=float) - benchmark).mean() for key in keys]
    )
    best = int(np.argmax(mean_edge))
    return {
        "p_value": float(pvalues["consistent"]),
        "p_value_lower": float(pvalues["lower"]),
        "p_value_upper": float(pvalues["upper"]),
        "best_config": keys[best],
        "best_mean_edge": float(mean_edge[best]),
    }


# -------------------------------------------------------------------------------------
# block E: assemble the post-hoc columns for a grid cell and bootstrap their CIs
#
# every reported column (gross, cost and risk) is a function of a per-ticker contribution that SUMS
# across tickers, so we reduce each ticker to one fixed sufficient-statistics vector and recompute
# all columns from the sum. the point value uses the sum over all tickers; the CI resamples the
# vectors with arch and recomputes. layout of a contribution vector (width 17 + 4 + 12 = 33):
#   [0:17]   the config's gross raw 17 metrics, summed over the ticker's windows
#   [17:21]  after-cost net at each cost level (0/5/10/20 bps), summed over windows
#   [21:33]  for each risk level (0, 5 bps): sum_r, sum_r2, n, sum_rneg2, sum_dd, nwin
# -------------------------------------------------------------------------------------
def collect_cell_bets(prefix_results: str, time_value: str) -> dict:
    """
    collect_cell_bets reads every ticker's saved bets for one grid cell and regroups them by obs mode and chosen config into per-ticker lists of per-window bet lists

    Arguments:
        prefix_results -- folder holding this cell's per-ticker result files
        time_value -- the timespan key the engine stored the results under

    Returns:
        nested dict {obs_mode: {config_name: {ticker: [window_bets, ...]}}}
    """
    collected = {obs: {name: {} for name in entries} for obs in obs_modes}
    for ticker in data_files:
        file_read = load_json_file(prefix_results + f"{ticker}_bets")
        for obs_mode in obs_modes:
            # this ticker's slice: {window_label: [arm0, arm1]} where
            #   [0] = the long/short arm (the headline numbers)
            #   [1] = the long-only arm, appended by the save layer
            # Each arm is {rule_name: [trimmed bets]}. The [0] below is NOT an unwrap -- it
            # SELECTS the long/short arm. Changing it to iterate both would pool the two arms
            # into the cost and drawdown columns with nothing raising.
            per_ticker = file_read[time_value][obs_mode][ticker]
            for window_label in per_ticker:
                chosen = per_ticker[window_label][0]  # [0] = long/short arm. See above.
                for name in entries:
                    collected[obs_mode][name].setdefault(ticker, []).append(chosen[name])
    return collected


def ticker_contribution(gross_17: list, windows: list, scenario: str) -> list:
    """
    ticker_contribution builds one ticker's fixed-width sufficient-statistics vector for a config: its gross raw metrics, its after-cost net at each cost level, and the pooled-return/drawdown statistics at each risk level

    Arguments:
        gross_17 -- the config's 17 gross raw metrics, already summed over this ticker's windows
        windows -- this ticker's list of per-window bet lists for the config
        scenario -- 'simulation' or 'one_pos', passed through to the re-walk

    Returns:
        the contribution vector (length 17 + len(cost_bps_levels) + 6*len(risk_bps_levels))
    """
    contribution = list(gross_17)
    # re-walk each window once per cost level; the risk levels are a subset so they reuse those walks
    nets = {bps: 0.0 for bps in cost_bps_levels}
    risk = {
        bps: {"sum_r": 0.0, "sum_r2": 0.0, "n": 0, "sum_rneg2": 0.0, "sum_dd": 0.0, "nwin": 0}
        for bps in risk_bps_levels
    }
    for window_bets in windows:
        for bps in cost_bps_levels:
            walked = rewalk_window(window_bets, scenario, bps, bps * extended_multiple)
            nets[bps] += walked["net"]
            if bps in risk:
                bucket = risk[bps]
                for r in walked["returns"]:
                    bucket["sum_r"] += r
                    bucket["sum_r2"] += r * r
                    bucket["n"] += 1
                    if r < 0:
                        bucket["sum_rneg2"] += r * r
                bucket["sum_dd"] += max_drawdown_from_equity(walked["equity_path"], scenario)
                bucket["nwin"] += 1
    for bps in cost_bps_levels:
        contribution.append(nets[bps])
    for bps in risk_bps_levels:
        bucket = risk[bps]
        contribution += [
            bucket["sum_r"], bucket["sum_r2"], bucket["n"],
            bucket["sum_rneg2"], bucket["sum_dd"], bucket["nwin"],
        ]
    return contribution


def recombine_row(summed: np.ndarray, scenario: str) -> dict:
    """
    recombine_row turns a summed contribution vector (over some set of tickers) back into every reported column value, using the same formulas as the gross table and the risk functions

    Arguments:
        summed -- element-wise sum of ticker contribution vectors
        scenario -- 'simulation' or 'one_pos' (only needed for calmar's return units)

    Returns:
        dict {column name -> value} for all extended_columns
    """
    values = {}
    # gross columns: reuse the exact nb4 formula on this config's summed 17 raw metrics
    reported = get_obs_mode_data([list(summed[:17])])[0]
    for name, value in zip(table_columns, reported):
        values[name] = value
    # after-cost average return per cost level (total over tickers / ticker count, as gross does)
    for k, bps in enumerate(cost_bps_levels):
        values[f"AvgReturn_{bps}bps"] = summed[17 + k] / ticker_length
    # risk columns from the pooled sufficient statistics
    base = 17 + len(cost_bps_levels)
    for j, bps in enumerate(risk_bps_levels):
        sum_r, sum_r2, n, sum_rneg2, sum_dd, nwin = summed[base + 6 * j : base + 6 * j + 6]
        if n >= 2:
            mean = sum_r / n
            variance = (sum_r2 - sum_r * sum_r / n) / (n - 1)
            std = variance ** 0.5 if variance > 0 else 0.0
            downside_std = (sum_rneg2 / (n - 1)) ** 0.5 if sum_rneg2 > 0 else 0.0
            sharpe = mean / std if std > 0 else 0.0
            sortino = mean / downside_std if downside_std > 0 else 0.0
        else:
            sharpe = sortino = 0.0
        mean_drawdown = sum_dd / nwin if nwin > 0 else 0.0
        # calmar = average per-window return / average per-window drawdown (the nwin cancels). return is
        # a fraction of starting capital for simulation, raw dollars for one_pos (no capital base).
        net_at_bps = summed[17 + cost_bps_levels.index(bps)]
        mean_return = (net_at_bps / starting_balance if scenario == "simulation" else net_at_bps) / (nwin if nwin > 0 else 1)
        values[f"MaxDrawdown_{bps}bps"] = mean_drawdown
        values[f"Sharpe_{bps}bps"] = sharpe
        values[f"Sortino_{bps}bps"] = sortino
        values[f"Calmar_{bps}bps"] = mean_return / mean_drawdown if mean_drawdown > 0 else 0.0
    return values


def config_full_stats(
    gross_by_ticker: list,
    bets_by_ticker: list,
    scenario: str,
    position_contributions: list = None,
    block_len: int = None,
) -> dict:
    """
    config_full_stats point values and bootstrap CIs for every extended column of one config: a ticker (IID) resample answering "did a few lucky tickers carry this?", and optionally a moving-block resample over the ordered windows answering "did a few lucky periods carry this?"

    Arguments:
        gross_by_ticker -- list over tickers of the config's summed 17 gross metrics
        bets_by_ticker -- list over tickers of the config's per-window bet lists (same ticker order)
        scenario -- 'simulation' or 'one_pos'

    Keyword Arguments:
        position_contributions -- list over WINDOW POSITIONS, in time order, of the contribution vector pooled across every ticker at that position; required for the block arm (default: {None})
        block_len -- contiguous block length for the moving-block resample, which must be the non-overlap stride so a block spans at least one window length (default: {None})

    Returns:
        dict with each column's point value, its '<column>_CIlow' / '<column>_CIhigh' (tickers), and, when the block arm runs, its '<column>_BlockCIlow' / '<column>_BlockCIhigh' (windows)
    """
    contributions = np.array(
        [
            ticker_contribution(gross_by_ticker[t], bets_by_ticker[t], scenario)
            for t in range(len(gross_by_ticker))
        ],
        dtype=float,
    )

    def recombine_vector(rows: np.ndarray) -> np.ndarray:
        # sum the resampled tickers' contributions, recompute all columns in a fixed order
        row = recombine_row(rows.sum(axis=0), scenario)
        return np.array([row[column] for column in extended_columns])

    result = dict(zip(extended_columns, recombine_vector(contributions)))
    if len(contributions) >= 2:
        bootstrap = IIDBootstrap(contributions, seed=42)
        interval = bootstrap.conf_int(recombine_vector, reps=ci_reps, method="percentile", size=0.95)
        for i, column in enumerate(extended_columns):
            result[f"{column}_CIlow"] = float(interval[0, i])
            result[f"{column}_CIhigh"] = float(interval[1, i])
    else:
        # nothing to resample -> interval collapses to the point
        for column in extended_columns:
            result[f"{column}_CIlow"] = result[f"{column}_CIhigh"] = result[column]

    # THE TEMPORAL COMPANION (audit row K). The ticker arm above resamples the
    # cross-section, so it answers "did a few lucky tickers carry this?" and says nothing
    # about the walk-forward windows OVERLAPPING in time -- which is the objection the
    # moving-block bootstrap exists to answer. Until this was wired, bootstrap_ci_blocks
    # had no caller anywhere and MovingBlockBootstrap was instantiated only inside it,
    # while PUBLICATION_PLAN.md:82 marked "ticker (IID) + moving-block" as DONE. Only the
    # IID arm actually ran, so the overlapping-windows objection had no answer in any column.
    #
    # The unit resampled here is a WINDOW POSITION with every ticker pooled into it: the
    # cross-section is held fixed and TIME is resampled in contiguous blocks, which is what
    # preserves the dependence between overlapping windows. block_len must be the
    # non-overlap stride (4 / 7 / 8), never arch's default 3, or a block can be shorter than
    # one window length and the resample breaks the very dependence it exists to keep --
    # the project's own "block length >= window/stride" rule.
    if position_contributions is not None and block_len and len(position_contributions) >= 2:
        by_position = np.asarray(position_contributions, dtype=float)
        total = len(by_position)
        bootstrap = MovingBlockBootstrap(
            max(1, min(int(block_len), total)), by_position, seed=42
        )
        interval = bootstrap.conf_int(
            recombine_vector, reps=ci_reps, method="percentile", size=0.95
        )
        for i, column in enumerate(extended_columns):
            result[f"{column}_BlockCIlow"] = float(interval[0, i])
            result[f"{column}_BlockCIhigh"] = float(interval[1, i])
    return result


def config_significance(prefix_results: str, time_value: str) -> dict:
    """
    config_significance runs the data-snooping tests (Hansen SPA + deflated sharpe) over all configs for each obs mode of one grid cell

    Arguments:
        prefix_results -- folder holding this cell's per-ticker result files
        time_value -- the timespan key the engine stored the results under

    Returns:
        dict {obs_mode: {spa fields}}
    """
    result = {}
    for obs_mode in obs_modes:
        # net_by_config[his] = that config's net at every (ticker, window) observation
        net_by_config = {}
        for ticker in data_files:
            file_read = load_json_file(prefix_results + f"{ticker}_nets")
            per_ticker = file_read[time_value][obs_mode][ticker]
            for window_label in per_ticker:
                # [0] = the long/short arm's nets for all 475 configs; [1] is the long-only
                # arm. Iterating both would DOUBLE this SPA's candidate set (950 series instead
                # of 475) and move the headline p-value, silently.
                for his, net in per_ticker[window_label][0].items():
                    net_by_config.setdefault(his, []).append(net)
        # the plain-MACD baseline his: "Histogram a: 0" for MACDCROSS, "MACD a: 0" for ZEROCROSS
        name_mode = "Histogram" if "MACDCROSS" in obs_mode else "MACD"
        benchmark_key = f"{name_mode} a: 0"
        # SPA only -- the DSR was removed 30 Oct 2026, see block D above
        result[obs_mode] = spa_test(net_by_config, benchmark_key)
    return result


# bh.ipynb keys its saved results by the starting balance: one_pos runs with balance 0,
# simulation with 20000. The regime classifier always reads the SIMULATION node for both
# scenarios, because the market is the same either way and only that node has a known
# denominator -- one_pos buy-and-hold is a per-share price difference with no natural
# base, so it cannot be turned into a fraction.
regime_balance_key = "20000"
regime_balance = 20000.0
BH_DIR = "results/buy_and_hold"


def load_regime_market_returns(scenario: str, time_format: str) -> tuple[list, dict]:
    """
    load_regime_market_returns reads bh.ipynb's per-window buy-and-hold totals and turns them into the average per-ticker market return at each chronological window POSITION, which is what the regime classifier thresholds on

    Returns an empty list when bh.ipynb has not been run yet, so the per-regime tables are
    simply skipped rather than the whole table build failing.

    BY POSITION, NOT BY LABEL -- the reason is measured and written up at the keep_positions
    branch of cell_full_rows. In short: a window label is built from dates derived from each
    series' own post-dropna first bar, so it differs per config AND per ticker, while the
    window count and ordering do not. The per-ticker grids come back alongside the returns so
    the caller can prove, ticker by ticker, that the two grids are the same length before
    pairing them.

    Arguments:
        scenario -- 'one_pos' or 'simulation'; accepted for symmetry with the caller but the simulation node is always used, see the note on regime_balance_key
        time_format -- '1_day', '1_hour' or '5_minute'

    Returns:
        (returns by position as a FRACTION of the starting balance, {ticker -> its chronological window labels}); ([], {}) if unavailable
    """
    try:
        saved = load_json_file(f"{BH_DIR}/bh_result")
    except Exception:
        return [], {}
    node = saved.get(regime_balance_key, {}).get(time_format, {})
    grids = regime_grids_by_ticker(node)
    averaged = market_return_by_position(node, grids)
    # express as a fraction of the starting balance so the thresholds are returns, not dollars
    return [value / regime_balance for value in averaged], grids


def classify_windows(market_returns: list, threshold: float) -> list:
    """
    classify_windows labels each validation window bull, bear or flat from the market's own return over that window, so the split is driven by a rule rather than by hand-picked dates

    Hand-picking regime dates is a form of snooping; deriving them from the cross-ticker
    average buy-and-hold return of the same window is objective and reproducible.

    Arguments:
        market_returns -- average buy-and-hold return across tickers at each chronological window position
        threshold -- fractional return above which a window is bull and below whose negative it is bear

    Returns:
        ["bull" | "bear" | "flat"] in the same positional order
    """
    labels = []
    for market_return in market_returns:
        if market_return > threshold:
            labels.append("bull")
        elif market_return < -threshold:
            labels.append("bear")
        else:
            labels.append("flat")
    return labels


# bh.ipynb stores aggregates in the same dict as the windows: 'all' is the whole sample and
# 'average_per_ticker' is a mean, neither of which is a validation window. Real window labels
# are 'START:END', so the separator is what tells them apart. Without this filter the
# aggregates get classified as windows and the averages come out absurd (millions of percent).
window_label_separator = ":"


def market_return_by_window(buy_and_hold_node: dict, ticker_count: int = 0) -> dict:
    """
    market_return_by_window turns bh.ipynb's per-ticker buy-and-hold results into an average per-ticker return for each validation window, which is the quantity the regime classifier thresholds on

    The average is taken over the tickers that actually report each window, not over a fixed
    ticker count: coverage varies by window (some intraday windows have 20 of 32 tickers),
    and dividing those by the full count would understate them and mislabel the regime.

    Arguments:
        buy_and_hold_node -- bh_result[balance][timeframe], i.e. {ticker -> {window label -> buy-and-hold}} plus a 'total' entry which is ignored here

    Keyword Arguments:
        ticker_count -- unused, kept so older callers do not break (default: {0})

    Returns:
        {window label -> average per-ticker buy-and-hold return}, aggregates excluded
    """
    totals = {}
    counts = {}
    for ticker, windows in buy_and_hold_node.items():
        if ticker == "total" or not isinstance(windows, dict):
            continue
        for window, value in windows.items():
            # skip 'all' / 'average_per_ticker' and anything else that is not a window
            if window_label_separator not in window:
                continue
            totals[window] = totals.get(window, 0.0) + value
            counts[window] = counts.get(window, 0) + 1
    return {window: totals[window] / counts[window] for window in totals}


def regime_grids_by_ticker(buy_and_hold_node: dict) -> dict:
    """
    regime_grids_by_ticker each ticker's own validation windows in chronological order, which is what gives a window POSITION its meaning

    The labels are zero-padded 'YYYY-MM-DD:YYYY-MM-DD', so sorting them as strings sorts them
    as dates. Per-ticker and not pooled, because the grids genuinely differ between tickers:
    measured on the saved hourly run, 31 tickers have 118 windows and one has 117.

    Arguments:
        buy_and_hold_node -- bh_result[balance][timeframe], i.e. {ticker -> {window label -> buy-and-hold}} plus a 'total' entry which is ignored here

    Returns:
        {ticker -> [window label, ...] chronological}, aggregates excluded
    """
    grids = {}
    for ticker, windows in buy_and_hold_node.items():
        if ticker == "total" or not isinstance(windows, dict):
            continue
        grids[ticker] = sorted(
            window for window in windows if window_label_separator in window
        )
    return grids


def market_return_by_position(buy_and_hold_node: dict, grids: dict) -> list:
    """
    market_return_by_position the average per-ticker buy-and-hold return at each chronological window position, which is the quantity the regime classifier thresholds on

    The average is taken over the tickers that actually reach each position, not over a fixed
    ticker count, for the same reason market_return_by_window averages over reporting tickers:
    a ticker whose grid is one window short must not drag the last position's average down.

    Arguments:
        buy_and_hold_node -- bh_result[balance][timeframe]
        grids -- {ticker -> its chronological window labels}, from regime_grids_by_ticker

    Returns:
        [average buy-and-hold return at position 0, at position 1, ...]
    """
    totals = {}
    counts = {}
    for ticker, windows in grids.items():
        for position, window in enumerate(windows):
            totals[position] = totals.get(position, 0.0) + buy_and_hold_node[ticker][window]
            counts[position] = counts.get(position, 0) + 1
    return [totals[position] / counts[position] for position in sorted(totals)]


def cell_full_rows(
    mother_file: dict,
    cell_bets: dict,
    scenario: str,
    window_step: int = 1,
    keep_positions: set = None,
    expect_windows: int = None,
    block_len: int = None,
) -> list:
    """
    cell_full_rows builds the extended-table row (point values + CIs for every column) for each (obs mode, config) of one grid cell, optionally keeping only every window_step-th window per ticker for the non-overlapping-window robustness check, or only a named set of windows for the per-regime split

    Arguments:
        mother_file -- nb1 structure {obs_mode: {ticker: {window_label: [5x18]}}}
        cell_bets -- collect_cell_bets output {obs_mode: {config: {ticker: [window_bets, ...]}}}
        scenario -- 'simulation' or 'one_pos'

    Keyword Arguments:
        window_step -- keep every Nth chronological window per ticker; 1 = all windows (default: {1})
        keep_positions -- if given, keep only windows at these chronological POSITIONS, which is how the bull/bear/flat tables are produced. Applied before window_step, so the two can be combined: per-regime significance should use the non-overlap subset (default: {None})
        expect_windows -- {ticker -> how many windows bh.ipynb classified for it}; required with keep_positions and checked per ticker, because pairing by position is only meaningful when the two window grids have the same length (default: {None})
        block_len -- when given, also compute the moving-block (temporal) CI with this contiguous block length, which must be the non-overlap stride for this timeframe (default: {None})

    Returns:
        list of per-(obs_mode, config) stats dicts, in the same order as the gross table rows
    """
    rows = []
    for obs_mode in obs_modes:
        for config_index, entry in enumerate(entries):
            # this config's gross 17-vector per ticker, and its per-window bets, both subsampled alike
            gross_by_ticker = []
            bets_by_ticker = []
            # the same numbers partitioned the OTHER way -- by window position, with every
            # ticker pooled into it -- which is what the block bootstrap resamples. Indexed
            # by ORDINAL within each ticker's kept list rather than by raw position, so a
            # ticker holding one window fewer (measured: 31 tickers at 118 hourly windows,
            # one at 117) contributes to the positions it reaches instead of shifting the
            # whole grid. Same rule as market_return_by_position uses for the regime labels.
            gross_by_position = []
            bets_by_position = []
            for ticker in data_files:
                ticker_data = mother_file[obs_mode][ticker]
                ticker_bets = cell_bets[obs_mode][entry].get(ticker, [])
                labels = list(ticker_data.keys())
                # The saved bets are a LIST appended in the same chronological window order
                # as these labels, so a window is addressed by POSITION. Selecting windows by
                # label for the gross part while slicing the bets separately would pair one
                # regime's point values with every window's bets, so both are driven by the
                # same positions here. If the two ever disagree in length the pairing is
                # meaningless, so fail loudly rather than report a silently mismatched table.
                if ticker_bets and len(ticker_bets) != len(labels):
                    raise ValueError(
                        f"{ticker} {obs_mode}: {len(labels)} windows but {len(ticker_bets)} "
                        "saved bet windows -- gross and bets are misaligned"
                    )
                positions = list(range(len(labels)))
                if keep_positions is not None:
                    # BY POSITION, NOT BY LABEL. Every config walks the same stride to the
                    # same data end, but it starts at its OWN post-dropna bar, because
                    # nan_indexes = max(short, long) + signal - 2 differs per config. So the
                    # window LABELS differ per config while the window COUNT and ordering do
                    # not. bh.ipynb classifies one grid (it hardcodes 12_26_9), and matching
                    # those labels against another config's by set membership silently kept
                    # NOTHING: measured on full data, 10 of 11 configs share 0 of 12 labels at
                    # daily and 7 of 11 share 0 of 118 at hourly, which wrote correctly-shaped
                    # all-zero tables at daily and -- far worse -- tables built from the one
                    # matching ticker and still divided by every ticker at hourly, deflated
                    # about 3x and looking like a result. Position i of this config covers the
                    # same span as position i of bh's grid, offset by the start drift only:
                    # 0 at 5-minute, <= 4 days at hourly, <= 27 days of a 12-month window at
                    # daily. That offset is a real approximation and belongs in the write-up.
                    # The guard tests that the two grids are the SAME LENGTH, not merely that
                    # they overlap. Dropping this ticker instead would be worse than raising:
                    # ticker_length is a module-level 32 used as the divisor, so a table built
                    # from fewer tickers is deflated by exactly the ratio, which is how the
                    # label join came to produce hourly tables about 3x too small that still
                    # looked like results.
                    wanted = None if expect_windows is None else expect_windows.get(ticker)
                    if expect_windows is not None and len(labels) != wanted:
                        raise ValueError(
                            f"{ticker} {obs_mode}: {len(labels)} windows but bh.ipynb's grid "
                            f"for this ticker has {wanted} -- the per-regime split pairs "
                            "windows BY POSITION, so unequal counts would mislabel them. Both "
                            "must come from the same data vintage; re-run bh.ipynb."
                        )
                    positions = [i for i in positions if i in keep_positions]
                # the non-overlap subsample applies to whatever survived the regime filter
                positions = positions[::window_step]
                gross_17 = [0.0] * 17
                for position in positions:
                    config_row = ticker_data[labels[position]][0][config_index]
                    for metric_index in range(17):
                        gross_17[metric_index] += config_row[metric_index]
                gross_by_ticker.append(gross_17)
                bets_by_ticker.append(
                    [ticker_bets[i] for i in positions if i < len(ticker_bets)]
                )
                if block_len:
                    for ordinal, position in enumerate(positions):
                        while len(gross_by_position) <= ordinal:
                            gross_by_position.append([0.0] * 17)
                            bets_by_position.append([])
                        config_row = ticker_data[labels[position]][0][config_index]
                        for metric_index in range(17):
                            gross_by_position[ordinal][metric_index] += config_row[metric_index]
                        if position < len(ticker_bets):
                            bets_by_position[ordinal].append(ticker_bets[position])
            # ticker_contribution returns a vector of SUMS, so pooling every ticker's bets at
            # one position is the same operation as pooling one ticker's bets over its own
            # windows -- the second argument is simply the list of window bet lists to walk.
            position_contributions = None
            if block_len:
                position_contributions = [
                    ticker_contribution(gross_by_position[k], bets_by_position[k], scenario)
                    for k in range(len(gross_by_position))
                ]
            rows.append(
                config_full_stats(
                    gross_by_ticker,
                    bets_by_ticker,
                    scenario,
                    position_contributions=position_contributions,
                    block_len=block_len,
                )
            )
    return rows


# -------------------------------------------------------------------------------------
# block F: the random-strategy benchmark (saved by bh.ipynb, one run per seed per window)
#
# the random strategy trades at the SAME turnover as plain MACD but at random times, through the same
# engine, so comparing against it isolates whether the real signals carry information. the spread
# ACROSS SEEDS is the benchmark's own uncertainty, so no ticker bootstrap is needed here.
# cost/risk math is reused by feeding ticker_contribution a zeroed gross-17 part and keeping only the
# cost and risk columns (the gross columns are meaningless for a benchmark with no observations row).
# -------------------------------------------------------------------------------------
random_scenario_of = {"0": "one_pos", "20000": "simulation"}


def random_seed_row(seed_windows: list, scenario: str) -> dict:
    """
    random_seed_row the cost and risk metrics for ONE random seed, aggregated over every ticker and window it traded

    Arguments:
        seed_windows -- list of per-window bet lists for this seed (flattened across tickers)
        scenario -- 'simulation' or 'one_pos'

    Returns:
        dict {column -> value} for the cost and risk columns only
    """
    # zeroed gross part: get_obs_mode_data guards division by zero, so the gross columns come back as
    # 0 and we simply drop them; the cost/risk columns depend only on the saved bets
    contribution = ticker_contribution([0.0] * 17, seed_windows, scenario)
    row = recombine_row(np.asarray(contribution, dtype=float), scenario)
    return {column: row[column] for column in cost_columns + risk_columns}


def random_benchmark_table(random_bets: dict, experiment_key: str, time_format: str) -> pd.DataFrame:
    """
    random_benchmark_table one row per random seed of the cost and risk metrics, for a single scenario and timeframe

    Arguments:
        random_bets -- the loaded random_bets file {exp: {time_format: {ticker: {label: {seed: bets}}}}}
        experiment_key -- '0' (one_pos) or '20000' (simulation) as stored in the file
        time_format -- '1_day', '1_hour' or '5_minute'

    Returns:
        dataframe indexed by seed, columns = the cost and risk metrics
    """
    scenario = random_scenario_of[experiment_key]
    per_ticker = random_bets[experiment_key][time_format]
    # regroup into {seed: [window bets, ...]} across every ticker and window
    windows_by_seed = {}
    for ticker in per_ticker:
        for label in per_ticker[ticker]:
            for seed, bets in per_ticker[ticker][label].items():
                windows_by_seed.setdefault(seed, []).append(bets)
    rows = {}
    for seed in sorted(windows_by_seed, key=int):
        rows[int(seed)] = random_seed_row(windows_by_seed[seed], scenario)
    return pd.DataFrame(rows).T.sort_index()


def percentile_vs_random(value: float, random_values: list) -> dict:
    """
    percentile_vs_random where a real strategy's number falls inside the random benchmark's distribution

    Arguments:
        value -- the real strategy's value for this metric
        random_values -- the same metric across all random seeds

    Returns:
        dict with 'percentile' (fraction of random runs the value beat) and 'p_value' (fraction of random runs that matched or beat it)
    """
    values = np.asarray(random_values, dtype=float)
    return {
        "percentile": float((values < value).mean()),
        "p_value": float((values >= value).mean()),
    }


def run_random_comparison(full_tables: dict) -> pd.DataFrame:
    """
    run_random_comparison writes where each real strategy's cost and risk metrics fall inside the random benchmark's across-seed distribution, which is the comparison percentile_vs_random exists to make and that nothing in the codebase computed

    Arguments:
        full_tables -- {(params_directory, scenario, system, time_format): full table} as run() built them

    Returns:
        the long-format comparison dataframe, also written to random_comparison.csv
    """
    os.makedirs(FULL_DIR, exist_ok=True)
    random_bets = load_json_file(f"{RANDOM_DIR}/random_bets")

    rows = []
    for experiment_key in random_bets:
        scenario = random_scenario_of[experiment_key]
        for time_format in random_bets[experiment_key]:
            seed_table = random_benchmark_table(random_bets, experiment_key, time_format)
            if seed_table.empty:
                continue
            for key, table in full_tables.items():
                params_directory, table_scenario, system, table_time_format = key
                if table_scenario != scenario or table_time_format != time_format:
                    continue
                for metric in seed_table.columns:
                    if metric not in table.columns:
                        continue
                    random_values = seed_table[metric].to_list()
                    # the random benchmark is a random SIGN SERIES, so it has no
                    # MACDCROSS/ZEROCROSS distinction -- one distribution is the right
                    # comparator for both obs modes, and both are compared against it.
                    for index_pair, value in table[metric].items():
                        strategy, entry = index_pair
                        comparison = percentile_vs_random(float(value), random_values)
                        rows.append({
                            "params": params_directory,
                            "scenario": scenario,
                            "system": system,
                            "time_format": time_format,
                            "strategy": strategy,
                            "row": entry,
                            "metric": metric,
                            "value": float(value),
                            "random_mean": float(np.mean(random_values)),
                            "random_p5": float(np.percentile(random_values, 5)),
                            "random_p95": float(np.percentile(random_values, 95)),
                            "n_seeds": len(random_values),
                            "beat_fraction": comparison["percentile"],
                            "p_value": comparison["p_value"],
                        })

    comparison_table = pd.DataFrame(rows)
    if comparison_table.empty:
        # a non-empty random file that matches no table at all means the two were built
        # from different scenarios or timeframes, which would otherwise write an empty
        # CSV and read as "nothing beat random".
        raise ValueError(
            "the random benchmark and the full tables share no (scenario, timeframe): "
            f"random has {[(random_scenario_of[k], t) for k in random_bets for t in random_bets[k]]}, "
            f"tables have {sorted({(k[1], k[3]) for k in full_tables})}"
        )
    comparison_table.round(4).to_csv(f"{FULL_DIR}/random_comparison.csv", index=False)
    print(f"[random] comparison -> {len(comparison_table)} rows "
          f"({comparison_table['metric'].nunique()} metrics)")
    return comparison_table


def run_random_tables():
    """
    run_random_tables reads the random benchmark bh.ipynb saved, writes one per-seed table per scenario and timeframe, and one summary table of the across-seed distribution
    """
    os.makedirs(FULL_DIR, exist_ok=True)
    random_bets = load_json_file(f"{RANDOM_DIR}/random_bets")

    summary_rows = []
    summary_index = []
    for experiment_key in random_bets:
        scenario = random_scenario_of[experiment_key]
        for time_format in random_bets[experiment_key]:
            seed_table = random_benchmark_table(random_bets, experiment_key, time_format)
            if seed_table.empty:
                continue
            seed_table.round(4).to_csv(f"{FULL_DIR}/random_{scenario}_{time_format}.csv")
            # the across-seed distribution is what the real strategies get compared against
            for statistic in ["mean", "std", "p5", "p50", "p95"]:
                if statistic == "mean":
                    values = seed_table.mean()
                elif statistic == "std":
                    values = seed_table.std(ddof=1)
                else:
                    values = seed_table.quantile(int(statistic[1:]) / 100)
                summary_rows.append(values)
                summary_index.append((scenario, time_format, statistic))
            print(f"[random] {scenario:10s} | {time_format:9s} | {len(seed_table)} seeds")

    summary = pd.DataFrame(summary_rows, index=pd.MultiIndex.from_tuples(
        summary_index, names=["scenario", "time_format", "statistic"]
    )).round(4)
    summary.to_csv(f"{FULL_DIR}/random_summary.csv")


# =====================================================================================
# MAIN: for each (params, scenario, signal) do nb1 -> nb2 -> nb3 -> nb4, write one CSV per timeframe
# =====================================================================================
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FULL_DIR, exist_ok=True)
    # kept so run_random_comparison can be driven by exactly the tables that were just
    # written, rather than by globbing filenames back off disk (the regime, nonoverlap
    # and significance CSVs share the same prefix and would be matched by any glob).
    full_tables = {}

    for macd_params in macd_params_list:
        params_directory = "_".join(str(val) for val in macd_params)

        for experiment_name in experiments_list:                       # scenario
            for signal_line_if, signal_line_if_target in signal_if_list:   # system
                signal_lines_ifs_joined = "_".join(str(v) for v in [signal_line_if, signal_line_if_target])
                system = system_name_of[(signal_line_if, signal_line_if_target)]

                for time_format, time_value in time_formats_value_list:

                    # -----------------------------------------------------------------
                    # nb1 : makemotherfile  -> build mother_file[obs][ticker/'total'][timespan]
                    #       'total'[ts] = concatenation (extend) of every ticker's matrix   # <-- FIX (was broken)
                    # -----------------------------------------------------------------
                    prefix_results = f"{RESULTS_DIR}/{params_directory}/{time_format}/{experiment_name}/{signal_lines_ifs_joined}/"
                    mother_file = {obs: {"total": dict()} for obs in obs_modes}

                    for ticker in data_files:
                        file_read = load_json_file(prefix_results + ticker)
                        for obs_mode in obs_modes:
                            # each ticker's data for this obs mode: {timespan: [[5 x 18]]}
                            ticker_data = file_read[time_value][obs_mode][ticker]
                            # keep the per-ticker slice (nb1 did this too)
                            mother_file[obs_mode][ticker] = ticker_data
                            # accumulate into 'total' by extending each timespan's list  (old cell-4 pattern)
                            for time_span in ticker_data.keys():
                                mother_file[obs_mode]["total"].setdefault(time_span, [])
                                mother_file[obs_mode]["total"][time_span].extend(ticker_data[time_span])

                    # -----------------------------------------------------------------
                    # nb2 : log_of_returns  -> per timespan, SUM every ticker's matrix element-wise
                    # -----------------------------------------------------------------
                    log_of_returns = {obs: dict() for obs in obs_modes}
                    for obs_mode in obs_modes:
                        time_keys = mother_file[obs_mode]["total"].keys()
                        for key in time_keys:
                            rows_this_timespan = mother_file[obs_mode]["total"][key]  # list of per-ticker matrices
                            n_rows = len(rows_this_timespan[0])                       # 5
                            current_key_list = init_results_list(rows_this_timespan[0][0], n_rows)
                            for row in rows_this_timespan:                            # row = one ticker's 5x18 matrix
                                add_results(current_key_list, row)
                            log_of_returns[obs_mode][key] = current_key_list          # 5 x 17

                    # -----------------------------------------------------------------
                    # nb3 : table_of_returns  -> SUM across all timespans
                    # -----------------------------------------------------------------
                    table_of_returns = {obs: None for obs in obs_modes}
                    for obs_mode in obs_modes:
                        time_keys = list(log_of_returns[obs_mode].keys())
                        current_key_list = init_conclude_list(log_of_returns[obs_mode][time_keys[0]])
                        for key in time_keys:
                            conclude_results(current_key_list, log_of_returns[obs_mode][key])
                        table_of_returns[obs_mode] = current_key_list                 # 5 x 17

                    # -----------------------------------------------------------------
                    # nb4 : get_table  -> reported metrics, MultiIndex (obs x entries), write CSV
                    # -----------------------------------------------------------------
                    output_rows = []
                    index_tuples = []
                    for obs_mode in obs_modes:
                        reported = get_obs_mode_data(table_of_returns[obs_mode])
                        obs_label = f'{obs_mode.split("_")[1]} {obs_mode.split("_")[2]}'
                        for entry, rep in zip(entries, reported):
                            index_tuples.append((obs_label, entry))
                            output_rows.append([round(x, 2) for x in rep])

                    table_index = pd.MultiIndex.from_tuples(index_tuples, names=["strategy", "row"])
                    table_data = pd.DataFrame(output_rows, columns=table_columns, index=table_index)

                    out = f"{OUT_DIR}/{params_directory}_{experiment_name}_{system}_{time_format}.csv"
                    table_data.to_csv(out)
                    print(f"[ok] {params_directory} | {experiment_name:10s} | {system} | {time_format:9s} -> {out}")

                    # -----------------------------------------------------------------
                    # block E : post-hoc cost/risk/CI columns (from the saved bets) + a
                    #           significance companion (SPA/DSR, from the saved nets). the full
                    #           table uses every window; a *_nonoverlap table repeats the same on an
                    #           independent (non-overlapping) subset of windows as a robustness check.
                    # -----------------------------------------------------------------
                    cell_bets = collect_cell_bets(prefix_results, time_value)

                    # full table (all windows); row order matches the gross table -> reuse its index.
                    # The block (temporal) CI is computed HERE ONLY. The *_nonoverlap table
                    # below answers the overlapping-windows objection by construction, by
                    # keeping every Nth window, so a block bootstrap on it is redundant; the
                    # per-regime tables are far too thin for it (measured: 4 bear windows of 12
                    # at daily). block_len is the non-overlap stride, never arch's default.
                    full_table = pd.DataFrame(
                        cell_full_rows(
                            mother_file,
                            cell_bets,
                            experiment_name,
                            block_len=nonoverlap_step.get(time_format, 1),
                        ),
                        index=table_index,
                    ).round(4)
                    full_out = f"{FULL_DIR}/{params_directory}_{experiment_name}_{system}_{time_format}.csv"
                    full_table.to_csv(full_out)
                    full_tables[(params_directory, experiment_name, system, time_format)] = full_table

                    # non-overlapping-window robustness: keep every Nth chronological window
                    step = nonoverlap_step.get(time_format, 1)
                    nonoverlap_table = pd.DataFrame(
                        cell_full_rows(mother_file, cell_bets, experiment_name, window_step=step),
                        index=table_index,
                    ).round(4)
                    nonoverlap_out = f"{FULL_DIR}/{params_directory}_{experiment_name}_{system}_{time_format}_nonoverlap.csv"
                    nonoverlap_table.to_csv(nonoverlap_out)

                    # bull/bear/flat classification comes from buy-and-hold over the same
                    # windows; empty (and the regime tables skipped) until bh.ipynb is run
                    regime_market_returns, regime_grids = load_regime_market_returns(
                        experiment_name, time_format
                    )
                    # per-regime tables, one per regime per threshold. Driven entirely by
                    # which window labels are kept, so every column is computed by exactly
                    # the same code path as the headline table.
                    if regime_market_returns:
                        expect_windows = {
                            ticker: len(windows) for ticker, windows in regime_grids.items()
                        }
                        for threshold in regime_thresholds:
                            window_regime = classify_windows(regime_market_returns, threshold)
                            for regime in regime_names:
                                chosen = {
                                    position
                                    for position, labelled in enumerate(window_regime)
                                    if labelled == regime
                                }
                                if not chosen:
                                    continue
                                regime_table = pd.DataFrame(
                                    cell_full_rows(
                                        mother_file,
                                        cell_bets,
                                        experiment_name,
                                        keep_positions=chosen,
                                        expect_windows=expect_windows,
                                    ),
                                    index=table_index,
                                ).round(4)
                                threshold_tag = str(threshold).replace(".", "p")
                                regime_out = (
                                    f"{FULL_DIR}/{params_directory}_{experiment_name}_{system}"
                                    f"_{time_format}_{regime}_{threshold_tag}.csv"
                                )
                                regime_table.to_csv(regime_out)

                    significance = config_significance(prefix_results, time_value)
                    significance_out = f"{FULL_DIR}/{params_directory}_{experiment_name}_{system}_{time_format}_significance.csv"
                    pd.DataFrame(significance).T.to_csv(significance_out)

    return full_tables


if __name__ == "__main__":
    built = run()
    # AUDIT ROW J. run_random_tables() had no caller and __main__ called only run(), so
    # re-running bh.ipynb produced no random table at all, and the one claim the whole
    # benchmark exists for -- "only X% of equal-turnover random strategies did this well"
    # -- was computed by NOTHING. Both are wired here.
    #
    # The guard is not optional: results/random/ does not exist until bh.ipynb has been
    # re-run, so an unguarded call would abort the whole table build AFTER every gross,
    # full, nonoverlap, regime and significance CSV had already been written. A missing
    # benchmark is reported loudly and skipped, never allowed to read as a pass.
    try:
        run_random_tables()
        run_random_comparison(built)
    except FileNotFoundError as missing:
        print(
            f"[random] SKIPPED -- {missing}.\n"
            "         Run bh.ipynb to produce results/random/random_bets. Until then the\n"
            "         random benchmark tables and the percentile comparison DO NOT EXIST,\n"
            "         so any claim about beating random strategies is UNSUPPORTED."
        )

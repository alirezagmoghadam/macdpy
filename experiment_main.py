###########################
#         Imports         #
###########################

import os
from json import dump, load
from datetime import datetime

# from time import strptime
import time
import warnings
from math import exp
from math import pow as mathpow
import argparse
import functools

# import random

import numpy as np
import pandas as pd
import pygad

# from scipy import signal

# seed_number = 101
# seed_number = 1001
# random.seed(seed_number)
# np.random.seed(seed_number)


###########################
#         Timing          #
###########################
# _TIMINGS keeps [total_seconds, n_calls] per function, so the average per call time is total / n_calls.
# to time things: uncomment the `# @timeit` line above ema / macd / smacd / cmacd / buy_and_hold /
# macd_crossover, run a small sample (a couple tickers, hourly + 5-min), then call dump_timings().
# it stays commented on the normal (main) run so there is exactly zero overhead (the wrapper is not there).
_TIMINGS = {}


def timeit(func):
    """times each call of func and adds the elapsed time and a call count to _TIMINGS so we can get the average per call time later"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        record = _TIMINGS.setdefault(func.__name__, [0.0, 0])
        record[0] += time.perf_counter() - start
        record[1] += 1
        return result

    return wrapper


def dump_timings(output_path=".", file_name="timings"):
    """writes the average per call time in seconds for each timed function to a json file"""
    averages = {name: total / count for name, (total, count) in _TIMINGS.items() if count}
    write_json_to_file(output_path, file_name, {"averages": averages, "raw": _TIMINGS})
    return averages


###########################
#       Calculators       #
###########################


def sma_one_period(data: list) -> float:
    """
    sma_one_period calculates simple noving average for one period of choice and return a number

    Arguments:
        data -- list like entity containing data

    Returns:
        float which is the average
    """

    return sum(data) / len(data)


def sma(data: list, window: int) -> list:
    """
    sma calculates simple rolling noving average for whole dataset and returns sma list

    Arguments:
        data -- list like entity containing data
        window -- period for calculating noving average

    Returns:
        a list of noving averages
    """

    stop = len(data)

    # fill the first values with nan
    moving_averages = [np.nan] * (window - 1)

    for i in range(window - 1, stop):
        moving_one_period = sma_one_period(data[i - window + 1 : i + 1])
        moving_averages.append(moving_one_period)

    return moving_averages


# @timeit  # timing run only (see Timing section)
def ema(data: list, window: int) -> list:
    """
    ema calculates exponential rolling moving average for whole dataset and returns ema list\n
    first sma is calculated and then ema's are calculated based on that

    Arguments:
        data -- dataset
        Window -- period for calculating moving average

    Returns:
        a list of moving averages
    """

    # 0. initilize variables
    alpha = 2 / (window + 1)
    ema_list = [np.nan] * (window - 1)
    current_ema = sma_one_period(data[:window])
    ema_list.append(current_ema)

    # 1. calculate ema's and add to emaList
    if len(data) == window:
        return ema_list
    else:
        for i in range(window, len(data)):
            current_ema = ((data[i] - current_ema) * alpha) + current_ema
            ema_list.append(current_ema)
        return ema_list

    # print(data, window)
    # return list(ta_ema(array(data), timeperiod=window))


# @timeit  # timing run only (see Timing section)
def recursive_ema(data: list, window: int) -> list:
    """
    recursive_ema calculates exponential rolling noving average for whole dataset with formula:\n
    y_0 = x_0\n
    y_i = alpha * x_i + (1 - alpha) * y_(i-1)\n
    similar to matlab -- returns ema list

    Arguments:
        data -- dataset
        Window -- period for calculating noving average

    Returns:
        a list of noving averages
    """

    alpha = 2 / (window + 1)
    current_ema = data[0]
    ema_list = [current_ema]

    for data_point in data[1:]:
        current_ema = alpha * data_point + (1 - alpha) * current_ema
        ema_list.append(current_ema)

    return ema_list
    # return list(ta_ema(array(data), timeperiod=window))


# @timeit  # timing run only (see Timing section)
def macd(
    data: list,
    ema_mode: str = "recursive",
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> tuple[list, list, list]:
    """
    macd calculates macd for whole dataset - return macd line and signal line and histogram as lists

    Arguments:
        data -- dataset

    Keyword Arguments:
        ema_mode -- the ema function to use (default: {"recursive"})
        short_period -- the period for shorter ema (default: {12})
        long_period -- the period for longer ema (default: {26})
        signal_period -- the period for signal Line (default: {9})

    Returns:
        three lists macd line and signal line and histogram
    """

    if ema_mode == "recursive":
        ema_short = recursive_ema(data, short_period)
        ema_long = recursive_ema(data, long_period)

        macd_line = [x - y for x, y in zip(ema_short, ema_long)]
        signal_line = recursive_ema(macd_line, signal_period)

        macd_line = ([np.nan] * (long_period - 1)) + macd_line[long_period - 1 :]
        signal_line = ([np.nan] * (long_period + signal_period - 2)) + signal_line[
            long_period + signal_period - 2 :
        ]
    else:
        ema_short = ema(data, short_period)
        ema_long = ema(data, long_period)

        macd_line = [x - y for x, y in zip(ema_short, ema_long)]
        signal_line = [np.nan] * (long_period - 1)
        signal_line.extend(ema(macd_line[long_period - 1 :], signal_period))

    return macd_line, signal_line, [x - y for (x, y) in zip(macd_line, signal_line)]

    # macd, macdsignal, macdhist = ta_MACD(array(data), fastperiod=short_period, slowperiod=long_period, signalperiod=signal_period)
    # return list(macd), list(macdsignal), list(macdhist)


# @timeit  # timing run only (see Timing section)
def smacd(
    data: list,
    alpha: float,
    ema_mode: str = "recursive",
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
    macd_line: list = None,
) -> tuple[list, list, list]:
    """
    smacd calculates Speeded/Smoothed MACD indicator for whole dataset - return smacd line, smacd signal line and smacd histogram as lists

    Arguments:
        data -- dataset
        alpha -- alpha parameter to calculate smacd

    Keyword Arguments:
        ema_mode -- the ema function to use (default: {"recursive"})
        short_period -- the period for shorter ema (default: {12})
        long_period -- the period for longer ema (default: {26})
        signal_period -- the period for signal Line (default: {9})
        macd_line -- give MACD to save time and not calculate it again (default: {None})

    Returns:
        three lists smacd line and smacd signal line and smacd histogram
    """

    # macd_line is None Or List -- if None calculate
    # our MACD
    macd_line = (
        macd_line
        or macd(
            data=data,
            ema_mode=ema_mode,
            short_period=short_period,
            long_period=long_period,
            signal_period=signal_period,
        )[0]
    )

    # start = (long_period + signal_period - 2)
    start = long_period - 1

    smacd_line = [
        macd_line[start]
    ]  # first none nan also the same number of nans as MACD also signal is just like macd

    # calcukate SMACD data
    for i in range(start + 1, len(macd_line)):
        m = (alpha * (smacd_line[i - start - 1])) + ((1 - alpha) * (macd_line[i]))
        smacd_line.append(m)

    # calculate the Signal Line and add NaN's
    if ema_mode == "recursive":
        smacd_signal = [np.nan] * (long_period + signal_period - 2)
        smacd_signal.extend(
            recursive_ema(smacd_line, signal_period)[signal_period - 1 :]
        )

        smacd_line = ([np.nan] * (long_period - 1)) + smacd_line
    else:
        smacd_signal = [np.nan] * (long_period - 1)
        smacd_signal.extend(ema(smacd_line, signal_period))

        smacd_line = ([np.nan] * (long_period - 1)) + smacd_line

    return smacd_line, smacd_signal, [x - y for (x, y) in zip(smacd_line, smacd_signal)]


# @timeit  # timing run only (see Timing section)
def cmacd(
    data: list,
    alpha: float,
    gamma: float,
    ema_mode: str = "recursive",
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
    macd_line: list = None,
    smacd_line: list = None,
) -> tuple[list, list, list]:
    """
    cmacd calculates cmacd for whole dataset - return cmacd and cmacd signal line and cmacd histogram as lists

    Arguments:
        data -- dataset
        alpha -- alpha parameter to calculate cmacd
        gamma -- gamma parameter to calculate cmacd

    Keyword Arguments:
        ema_mode -- the ema function to use (default: {"recursive"})
        short_period -- the period for shorter ema (default: {12})
        long_period -- the period for longer ema (default: {26})
        signal_period -- the period for signal Line (default: {9})
        macd_line -- give MACD to save time and not calculate it again (default: {None})
        smacd_line --  give SMACD to save time and not calculate it again (default: {None})

    Returns:
        three lists cmacd line and cmacd signal line and cmacd histogram
    """

    # cachine for speed increase
    macd_line = (
        macd_line
        or macd(
            data=data,
            ema_mode=ema_mode,
            short_period=short_period,
            long_period=long_period,
            signal_period=signal_period,
        )[0]
    )
    smacd_line = (
        smacd_line
        or smacd(
            data=data,
            alpha=alpha,
            ema_mode=ema_mode,
            short_period=short_period,
            long_period=long_period,
            signal_period=signal_period,
            macd_line=macd_line,
        )[0]
    )

    # start = (long_period + signal_period - 2)
    start = long_period - 1

    cmacd_line = [
        macd_line[start]
    ]  # first none nan also the same number of nans as MACD also signal is just like macd

    zero_counter = 0
    tol = 2.220446049250313e-15

    # calcukate cmacd data
    for i in range(start + 1, len(macd_line)):
        term = macd_line[i] - smacd_line[i - 1]
        w = exp(-gamma * mathpow(term, 2))
        if abs(w) <= tol:
            zero_counter += 1
        m = smacd_line[i - 1] + ((1 - alpha) * term * w)
        cmacd_line.append(m)

    if ema_mode == "recursive":
        cmacd_signal = [np.nan] * (long_period + signal_period - 2)
        cmacd_signal.extend(
            recursive_ema(cmacd_line, signal_period)[signal_period - 1 :]
        )

        cmacd_line = ([np.nan] * (long_period - 1)) + cmacd_line
    else:
        cmacd_signal = [np.nan] * (long_period - 1)
        cmacd_signal.extend(ema(cmacd_line, signal_period))

        cmacd_line = ([np.nan] * (long_period - 1)) + cmacd_line

    return (
        cmacd_line,
        cmacd_signal,
        [x - y for (x, y) in zip(cmacd_line, cmacd_signal)],
        zero_counter,
    )


###########################
#       Utilities        #
###########################


def time_pad(t: datetime) -> str:
    """
    time_pad pads datetime for further work

    Arguments:
        t -- datetime object

    Returns:
        padded date time in string format
    """
    while len(str(t)) < 6:
        t = f"0{t}"
    return t


def make_index(tick: pd.DataFrame) -> pd.DataFrame:
    """
    make_index makes index by splitting time to date and intraday time

    Arguments:
        tick -- dataframe of time
    """
    tick["<TIME>"] = tick["<TIME>"].apply(time_pad)
    tick.reset_index(inplace=True)
    tick["<DATE>"] = tick["<DATE>"].astype(str) + " " + tick["<TIME>"].astype(str)
    tick["<DATE>"] = pd.to_datetime(tick["<DATE>"])
    tick.set_index("<DATE>", inplace=True)


def read_data(prefix: str, file_name: str) -> pd.DataFrame:
    """
    read_data reads data and return dataframe

    Arguments:
        prefix -- prefix for where the tickers and data are
        file_name -- ticker name

    Returns:
        dataframe after reading it and some change
    """
    df = pd.read_csv(prefix + file_name, index_col="<DATE>", parse_dates=True)
    df.dropna(inplace=True)
    make_index(df)
    return df


def change_raw_dataframe(stock_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    change_raw_dataframe takes raw dataframe and prepares it for function calculate_indicators by removing unneeded columns and renaming other

    Arguments:
        stock_dataframe -- raw dataframe read from data

    Returns:
        dataframe
    """
    new_col_names = ["ticker", "open", "high", "low", "close", "volume"]
    stock_dataframe.sort_index(inplace=True)
    stock_dataframe.drop(
        ["<PER>", "<TIME>", "<OPENINT>", "<VWAP>", "<TRANSACTIONS>"],
        axis=1,
        inplace=True,
    )
    stock_dataframe.rename(
        columns={
            key: value for key, value in zip(stock_dataframe.columns, new_col_names)
        },
        inplace=True,
    )
    stock_dataframe["ticker"] = stock_dataframe["ticker"].apply(
        lambda name: name.lower().split(".")[0]
    )
    stock_dataframe.index.names = ["time"]


def add_fill_column(df: pd.DataFrame, session_gap: str = "2h", mode: str = "next_open") -> None:
    """
    add_fill_column adds a 'fill' column which is the price we actually open or close a position at. we get the signal on a bar's close but we can't trade at that same close, so by default we fill at the next bar's open. for a session end bar (the next bar is far away like an overnight or weekend gap, or there is no next bar) there is no next open to use so we fill at that bar's own close. we look at the time gap to the next bar and not the calendar date because the data is in utc and the us session crosses utc midnight (runs about 09:00 to 01:00 utc)

    Arguments:
        df -- dataframe with 'open' and 'close' columns and a datetime index

    Keyword Arguments:
        session_gap -- gap to the next bar that counts as a session break (default: {"2h"})
        mode -- "next_open" for the default described above, or "next_close" to fill at the NEXT bar's close. execution differs by paper and is not a global property, so it is selected per notebook: Kang (2021) states that an order is executed at the closing price on the next day, which is "next_close", while on daily bars "next_open" collapses to same-bar close because every gap is a session break (default: {"next_open"})

    Returns:
        None -- adds the 'fill' column to df in place
    """
    if mode not in ("next_open", "next_close"):
        raise ValueError(f"unknown fill mode {mode!r}, expected 'next_open' or 'next_close'")

    # turn the datetime index into a series so we can do time math on it
    times = df.index.to_series()
    # gap from each bar to the next one (the last bar has no next so this is NaT)
    gap_to_next = times.shift(-1) - times

    if mode == "next_close":
        # Kang (2021): the order goes in on the signal bar and is executed at the next day's
        # closing price. That is a DAILY rule -- one bar per session -- and daily data is the
        # only place it is used.
        #
        # It must NOT be applied to intraday bars. Measured on 5-minute AAPL before this
        # guard: all 1264 session-end bars filled at the NEXT SESSION's close, one of them
        # 2 days 8 hours later, silently turning a 5-minute horizon into a multi-day hold.
        # Making those bars fill at their own close instead is NOT the fix either: on daily
        # data every gap exceeds session_gap, so that rule would fill every bar at its own
        # close and hand Kang back the same-bar look-ahead the fill column exists to remove.
        # So the precondition is checked instead of being quietly worked around.
        if (gap_to_next.dropna() <= pd.Timedelta(session_gap)).any():
            raise ValueError(
                f"mode='next_close' is Kang's daily rule and expects one bar per session, "
                f"but this data has bars less than {session_gap} apart, so 'the next bar's "
                f"close' would cross the overnight gap"
            )
        df["fill"] = df["close"].shift(-1)
        # the final bar has no next close, so it fills at its own and last_close shuts the
        # position there
        df.loc[gap_to_next.isna(), "fill"] = df.loc[gap_to_next.isna(), "close"]
        return

    # a bar is a session end if there is no next bar or the next bar is too far away (overnight/weekend)
    is_session_end = gap_to_next.isna() | (gap_to_next > pd.Timedelta(session_gap))
    # normal fill is the next bar's open
    df["fill"] = df["open"].shift(-1)
    # session end has no next open to use so we fill at this bar's own close
    df.loc[is_session_end, "fill"] = df.loc[is_session_end, "close"]


# the exchange the Polygon tickers trade on; timestamps on disk are tz-naive UTC, so every
# session boundary has to go through this to be DST-correct rather than hardcoded
exchange_timezone = "America/New_York"
regular_session_open = "09:30"
regular_session_close = "16:00"


def to_exchange_time(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    to_exchange_time reads the tz-naive UTC index as exchange local time, so session boundaries land correctly on both sides of a DST change

    An index that is ALREADY tz-aware is converted without being localized again.
    Localizing unconditionally raised `TypeError: Already tz-aware, use tz_convert to
    convert.` Nothing in the pipeline hits it today -- bars are stored tz-naive UTC -- but
    the post-hoc layer and the notebooks are exactly where a localized index turns up.

    Arguments:
        df -- dataframe whose index is tz-naive UTC timestamps, or already tz-aware

    Returns:
        the index converted to exchange local time
    """
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(exchange_timezone)


def filter_regular_session(
    df: pd.DataFrame, include_close_bar: bool = False
) -> pd.DataFrame:
    """
    filter_regular_session keeps only bars inside the regular trading session, for the guardrail run that asks whether the result holds in regular hours rather than across the full extended window

    The main runs deliberately keep extended hours (the user's decision), so this is a
    separate robustness run and not a change to the default. Boundaries are applied in
    exchange local time, so they stay at 09:30-16:00 through DST transitions instead of
    drifting an hour twice a year as a fixed UTC window would.

    Bars are open-stamped -- a bar labelled T covers [T, T+duration) -- so the bar labelled
    16:00 is the 16:00-16:05 bar, which is after the bell. By default it is excluded, giving
    78 five-minute bars a day rather than 79. `include_close_bar` keeps it, which is what
    build_near_close_daily needs: there that bar is the EXECUTION bar, because the closing
    auction prints into it. One function, one definition of the session, one flag for the
    one caller that wants the extra bar.

    Arguments:
        df -- intraday dataframe indexed by tz-naive UTC timestamps

    Keyword Arguments:
        include_close_bar -- keep the bar stamped at the closing time, i.e. the bar the auction prints into (default: {False})

    Returns:
        the same frame with only regular-session bars, index unchanged
    """
    local = to_exchange_time(df)
    close_time = pd.Timestamp(regular_session_close).time()
    inside = local.time >= pd.Timestamp(regular_session_open).time()
    if include_close_bar:
        inside = inside & (local.time <= close_time)
    else:
        inside = inside & (local.time < close_time)
    # a copy, not a view: callers go on to assign a 'fill' column, and writing that into a
    # slice of the parent frame is the SettingWithCopy case where the write can land nowhere
    return df.loc[inside].copy()


def official_close_prefix(prefix: str) -> str:
    """
    official_close_prefix maps an intraday data prefix to the sibling 1_day prefix, which is where the official closes live

    Arguments:
        prefix -- e.g. "./data/stocks/5_minute/"

    Returns:
        the same location with the timeframe folder swapped for 1_day, e.g. "./data/stocks/1_day/"
    """
    parts = prefix.replace("\\", "/").rstrip("/").split("/")
    parts[-1] = "1_day"
    return "/".join(parts) + "/"


def read_official_closes(prefix: str, file_name: str) -> dict:
    """
    read_official_closes reads the daily file and returns the official closing price for each exchange-local trading date

    This is the price a market-on-close order actually gets, taken from the vendor's daily
    file rather than reconstructed from intraday bars. Reconstructing it was tried and is
    not reliable: the 16:00 bar's open, the best intraday candidate, equals the official
    close on only 88.8% of the 41,596 ticker-days in the panel, and on the early-close days
    it is a stray post-session print rather than the auction at all.

    Arguments:
        prefix -- the INTRADAY data prefix; the daily file is found beside it
        file_name -- ticker file name, the same one used for the intraday data

    Returns:
        {datetime.date: official close}
    """
    daily = pd.read_csv(official_close_prefix(prefix) + file_name)
    dates = pd.to_datetime(daily["<DATE>"], format="%Y%m%d").dt.date
    return dict(zip(dates, daily["<CLOSE>"].astype(float)))


def build_near_close_daily(
    df: pd.DataFrame, official_closes: dict, session_gap: str = "2h"
) -> pd.DataFrame:
    """
    build_near_close_daily collapses intraday bars into one bar per trading day, taking the SIGNAL from the second-to-last session bar and executing on the last one, which is the daily appendix design (#3)

    The ordinary daily series decides and fills on the same close, which is not tradeable:
    you cannot observe a close and also trade at it. The rule here is purely positional and
    has no clock time in it: the one before the last bar is what the indicators are computed
    from, and the last bar itself is where the order executes.

        full day      ... 15:45  15:50  15:55 | 16:00
        early close   ... 12:45  12:50  12:55 | 13:00
                                        [-2]     [-1]
                                    indicators  execution

    Both cases are the same rule, so early closes need no special case and are not dropped.
    An earlier version of this function hardcoded the session end at 16:00 and then needed a
    second hardcoded cutoff to throw the early closes away, because on those days the last
    bar inside a 16:00 window is the auction bar itself -- which made the decision price the
    execution price on 90% of them. Neither constant is needed.

    The execution PRICE is the official close, read from the daily file. That is the price
    the last bar represents: the auction prints into it, and its open matches the official
    close on 88.8% of the 41,596 ticker-days in the panel. Reading the daily file gives the
    same quantity exactly rather than approximately.

    open/high/low/volume cover the session up to and INCLUDING the decision bar, never the
    execution bar, because the decision is taken before that bar exists.

    Arguments:
        df -- intraday dataframe indexed by tz-naive UTC timestamps, with open/high/low/close/volume
        official_closes -- {date: official close}, from read_official_closes

    Keyword Arguments:
        session_gap -- a trailing bar separated from the session by more than this is a stray post-session print, not the execution bar, and is dropped. On most early closes the feed stamps a lone bar at 16:00, hours after trading stopped. Same idea and same default as add_fill_column (default: {"2h"})

    Returns:
        one row per trading day, indexed by the DECISION bar's UTC timestamp, with 'close' = the decision price and 'fill' = the official close
    """
    # include_close_bar: the bar stamped at the closing time is the EXECUTION bar here
    session = filter_regular_session(df, include_close_bar=True)
    if len(df) > 0 and len(session) == 0:
        # daily bars are stamped outside 09:30-16:00 exchange local, so the session filter
        # removes all of them and the collapse would silently return an empty frame that
        # blows up downstream as a ZeroDivisionError. This fires when a caller's prefix and
        # time_format disagree -- e.g. prefix "./data/stocks/1_day/" with time_format
        # "1_day", which asks for the near-close build over data that is already daily.
        raise ValueError(
            "near_close found no regular-session bars: this looks like data that is already "
            "daily, not the intraday bars the near-close build collapses. The daily run "
            "reads 5-minute bars -- see data_prefix_for(time_format)."
        )
    local = to_exchange_time(session)
    working = session.copy()
    working["_local_date"] = local.date
    gap = pd.Timedelta(session_gap)

    rows = []
    for day, bars in working.groupby("_local_date", sort=True):
        # strip trailing strays so that the last bar really is the execution bar
        while len(bars) >= 2 and (bars.index[-1] - bars.index[-2]) > gap:
            bars = bars.iloc[:-1]
        if len(bars) < 2:
            continue  # need one bar to decide on and one to execute on
        if day not in official_closes:
            continue  # no official close to execute at
        decision = bars.iloc[-2]
        known = bars.iloc[:-1]  # everything the decision bar may legitimately see
        rows.append(
            {
                "timestamp": bars.index[-2],
                "ticker": decision.get("ticker", ""),
                "open": float(bars.iloc[0]["open"]),
                "high": float(known["high"].max()),
                "low": float(known["low"].min()),
                # the price the signal is computed from
                "close": float(decision["close"]),
                "volume": float(known["volume"].sum()),
                # the price the order actually executes at
                "fill": float(official_closes[day]),
            }
        )
    # the columns are named explicitly so that a ticker with no usable day returns an empty
    # frame of the right shape, rather than raising a KeyError out of set_index
    daily = pd.DataFrame(
        rows,
        columns=["timestamp", "ticker", "open", "high", "low", "close", "volume", "fill"],
    ).set_index("timestamp")
    daily.index = pd.DatetimeIndex(daily.index)
    # "time" is what change_raw_dataframe names the index everywhere else in the pipeline
    daily.index.name = "time"
    return daily


# the collapsed daily frames, keyed (prefix, file_name). run_experiment is called 33 times
# for the daily run (11 macd configs x 3 systems) and each call walks every ticker, so without
# this the 5-minute file is re-read and re-collapsed 1,056 times instead of 32. Only the
# near-close frames are cached: they are ~1,256 rows each, where the raw 5-minute frames they
# come from are ~233,000 and would cost hundreds of MB across the ticker list.
_near_close_cache: dict = {}


# the per-window warmup for the 5-minute path, in bars. An EMA is a recursion, so dropping
# the NaN bars removes the NaNs but NOT the influence of the seed, which decays by (1-alpha)
# per bar and never reaches zero. Measured bars to bit-identity: 402 at slow period 26, 473 at
# 30, 602 at 39 -- the longer the slow period, the longer the seed lingers. At the old value of
# 500 the budget was nan_indexes+500, i.e. 546 for (19,39,9), which is SHORT of the 602 that
# config needs; it was the only one of the 11 running short. 650 clears the worst case.
# This constant does NOT affect the daily or hourly runs: those compute indicators once over
# the whole series (run_experiment) and never take a warmup slice.
WARMUP_BARS = 650


# the ticker universe, defined once. It used to be written out four times -- twice as
# `datafiles` inside the drivers and twice as `ticker_list` in __main__ -- which made
# --ticker-index N ambiguous. Verified identical in content AND order before merging, so
# index N means the same ticker at every timeframe, and matches the `tick index` the run logs
# already print. 32 tickers: the Dow 30 of the sample period plus NFLX and GOOGL.
TICKER_FILES = [
    "aapl.us.txt",
    "amgn.us.txt",
    "amzn.us.txt",
    "axp.us.txt",
    "ba.us.txt",
    "cat.us.txt",
    "crm.us.txt",
    "csco.us.txt",
    "cvx.us.txt",
    "dis.us.txt",
    "googl.us.txt",
    "gs.us.txt",
    "hd.us.txt",
    "hon.us.txt",
    "ibm.us.txt",
    "jnj.us.txt",
    "jpm.us.txt",
    "ko.us.txt",
    "mcd.us.txt",
    "mmm.us.txt",
    "mrk.us.txt",
    "msft.us.txt",
    "nflx.us.txt",
    "nke.us.txt",
    "nvda.us.txt",
    "pg.us.txt",
    "shw.us.txt",
    "trv.us.txt",
    "unh.us.txt",
    "v.us.txt",
    "vz.us.txt",
    "wmt.us.txt",
]


def uses_near_close(time_format: str) -> bool:
    """
    uses_near_close says whether a time format's data is the near-close daily build

    Daily IS the near-close build and is the only daily run there is (user, 25 + 28 Oct 2026):
    the plain daily series decides and fills on the same close, which is not tradeable. So this
    is not a user-selectable mode, it is what "1_day" means. Every other time format reads its
    own bars directly.

    Keeping it here, as one predicate, is what stops the engine and the notebooks from
    disagreeing about what the daily series is -- bh.ipynb has its own copy of the read and
    must answer this question the same way.

    Arguments:
        time_format -- 1_day/1_hour/1_minute/5_minute

    Returns:
        True for the daily run, False otherwise
    """
    return time_format == "1_day"


def data_prefix_for(time_format: str) -> str:
    """
    data_prefix_for gives the folder a run READS, which is not always the folder it is labelled with

    `time_format` names the OUTPUT (it picks the results directory and is compared against
    string literals in a dozen branches). The daily run's DATA is 5-minute bars, because the
    near-close build is collapsed from them. Those two things used to be the same f-string,
    which is why the daily run read the untradeable `1_day` file.

    Arguments:
        time_format -- 1_day/1_hour/1_minute/5_minute

    Returns:
        the data folder to read, e.g. "./data/stocks/5_minute/" for time_format "1_day"
    """
    data_format = "5_minute" if uses_near_close(time_format) else time_format
    return f"./data/stocks/{data_format}/"


def output_timeframe_for(time_format: str, session: str = "full") -> str:
    """
    output_timeframe_for the results-folder name for a run, which is the timeframe for a main run and a suffixed variant for the regular-session guardrail run

    Both output_path lines are templated on time_format alone, so without this a 5-minute
    regular-hours run would write straight over the main 5-minute results. The guardrail is a
    VARIANT of a run, not a run of its own, so it gets its own folder and the main tree is
    untouched: results/<params>/5_minute_regular/...

    Arguments:
        time_format -- "1_day", "1_hour" or "5_minute"

    Keyword Arguments:
        session -- "full" for the main runs, "regular" for the guardrail (default: {"full"})

    Returns:
        the folder name to write under
    """
    if session == "full":
        return time_format
    if session == "regular":
        return f"{time_format}_regular"
    raise ValueError(f"unknown session {session!r}, expected 'full' or 'regular'")


def read_and_change_data(
    prefix: str, file_name: str, session: str = "full", near_close: bool = False
) -> pd.DataFrame:
    """
    read_and_change_data reads data and changes its columns and make the dataframe ready for calculating indicators

    Arguments:
        prefix -- data location
        file_name -- ticker name

    Keyword Arguments:
        session -- "full" keeps extended hours, which is the main runs' design; "regular" keeps only 09:30-16:00 exchange local time, for the regular-hours guardrail run (default: {"full"})
        near_close -- collapse intraday bars to one per day, computing indicators from the second-to-last session bar and executing on the last one at that day's official close, read from the sibling 1_day file. This is the daily appendix design (#3) and expects 5-minute input (default: {False})

    Returns:
        dataframe
    """
    if near_close:
        # build_near_close_daily applies the regular-session filter ITSELF, so a session filter
        # must not have run first: pre-stripping the 16:00 bar silently shifts the decision to
        # the 15:50 bar. Measured on AAPL 2020-02-10 -- decision 80.2875 instead of 80.3850,
        # ten minutes early, with nothing raised.
        if session != "full":
            raise ValueError(
                f"near_close=True builds its own regular session; session must stay 'full', "
                f"got {session!r}"
            )
        cached = _near_close_cache.get((prefix, file_name))
        if cached is None:
            df = read_data(prefix, file_name)
            change_raw_dataframe(df)
            # build_near_close_daily sets 'fill' from the official close, so add_fill_column
            # must not run afterwards and overwrite it
            cached = build_near_close_daily(df, read_official_closes(prefix, file_name))
            _near_close_cache[(prefix, file_name)] = cached
        # a copy per caller: the 33 daily runs must not be able to mutate each other's data
        return cached.copy()

    df = read_data(prefix, file_name)
    change_raw_dataframe(df)
    if session == "regular":
        df = filter_regular_session(df)
    elif session != "full":
        raise ValueError(f"unknown session {session!r}, expected 'full' or 'regular'")
    add_fill_column(df)
    return df


def calculate_indicators(
    stock_dataframe: pd.DataFrame,
    col: str = "close",
    ema_mode: str = "normal",
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
    alpha_range: list = np.arange(-0.90, 1, 0.1),
    gamma_range: list = range(0, 110, 10),
    start_date: pd.Timestamp = pd.Timestamp("2009-07-01"),
) -> pd.DataFrame:
    """
    calculate_indicators calculated indicators and adds them as columns to stock dataframe

    Arguments:
        stock_dataframe -- stock dataframe with data

    Keyword Arguments:
        col -- price column that can be open/low/high/close (default: {"close"})
        ema_mode -- ema mode to calculate (default: {"normal"})
        short_period -- short period for macd (default: {12})
        long_period -- long period for macd (default: {26})
        signal_period -- signal period for macd (default: {9})
        alpha_range -- alpha range to calculate smacd and cmacd (default: {np.arange(-0.90, 1, 0.1)})
        gamma_range -- gamma range to calculate cmacd (default: {range(0, 110, 10)})
        start_date -- clip the result to this date onward, or None to keep all history. this used to be a hardcoded 2009-07-01 truncation that silently applied to every series: it costs the soybean continuous series 41.8% of its rows and leaves the nikkei only 18 months of history before Kang's 2011-01-04 start, which is short of the 24-month optimization window. the default keeps the old behaviour so existing results do not move; pass None (or an earlier date) per notebook when the longer history is wanted (default: {pd.Timestamp("2009-07-01")})

    Returns:
        changed dataframe with added indicators
    """
    col_names = []
    vals = []

    # calculate macd
    dataList = list(stock_dataframe[col])
    cached_macd, signal_line, macd_histogram = macd(
        dataList,
        ema_mode=ema_mode,
        short_period=short_period,
        long_period=long_period,
        signal_period=signal_period,
    )

    col_names.extend(["MACD", "SignalLine", "Histogram"])
    vals.append(cached_macd)
    vals.append(signal_line)
    vals.append(macd_histogram)

    # calculate smacd and cmacd
    for alpha in alpha_range:
        alpha = round(alpha, 1)
        smacd_line, smacd_signal, smacd_histogram = smacd(
            dataList,
            alpha,
            ema_mode,
            short_period,
            long_period,
            signal_period,
            cached_macd,
        )
        col_names.extend(
            [f"MACD a: {alpha}", f"MACD Signal a: {alpha}", f"Histogram a: {alpha}"]
        )
        vals.append(smacd_line)
        vals.append(smacd_signal)
        vals.append(smacd_histogram)

        for gamma in gamma_range:
            cmacd_line, cmacd_signal, cmacd_histogram, zero_counter = cmacd(
                dataList,
                alpha,
                gamma,
                ema_mode,
                short_period,
                long_period,
                signal_period,
                cached_macd,
                smacd_line,
            )
            col_names.extend(
                [
                    f"MACD a: {alpha} g: {gamma}",
                    f"MACD Signal a: {alpha} g: {gamma}",
                    f"Histogram a: {alpha} g: {gamma}",
                    f"MACD a: {alpha} g: {gamma} times",
                ]
            )
            vals.append(cmacd_line)
            vals.append(cmacd_signal)
            vals.append(cmacd_histogram)
            vals.append(zero_counter)

    tempdf = pd.DataFrame(dict(zip(col_names, vals)))
    tempdf["MACD a: 0"] = cached_macd
    tempdf["MACD Signal a: 0"] = signal_line
    tempdf["Histogram a: 0"] = macd_histogram
    tempdf.index = list(stock_dataframe.index)

    return_dataframe = pd.concat([stock_dataframe, tempdf], axis=1)
    return_dataframe.dropna(inplace=True)
    # clip to the requested start. this was a hardcoded 2009-07-01 line; it is a parameter
    # now because it applies to EVERY series handed to this function, not just the one it
    # was added for, and the truncation is silent.
    if start_date is not None:
        return_dataframe = return_dataframe[start_date:]

    return return_dataframe


def make_dataframe(
    prefix: str,
    file_name: str,
    time_format: str,
    macd_params: tuple[int, int, int],
    alpha_range: list = np.arange(-0.90, 1, 0.1),
    gamma_range: list = range(0, 110, 10),
    near_close: bool = None,
    session: str = "full",
) -> pd.DataFrame:
    """
    make_dataframe 1 - read data 2 - calculated indicators 3 - returns modified dataframe

    Arguments:
        prefix -- data location
        file_name -- ticker name
        time_format -- 1_day/1_hour/1_minute/5_minute
        macd_params -- parameters to calculate macd

    Keyword Arguments:
        alpha_range -- alpha range to calculate smacd and cmacd (default: {np.arange(-0.90, 1, 0.1)})
        gamma_range -- gamma range to calculate cmacd (default: {range(0, 110, 10)})
        near_close -- leave as None to derive it from `time_format`, which is what the
            production runs do. Pass False to read `prefix` as ordinary bars: a caller whose
            prefix already points at daily data (heatmap_plot.ipynb builds its own
            "./data/{market}/{time_format}/") wants the raw series, not the collapse
            (default: {None})
        session -- "full" keeps extended hours, which is the main runs' design; "regular" keeps
            only 09:30-16:00 exchange local time, for the regular-hours guardrail run. It is a
            real variant: measured on AAPL, 58.1% of 5-minute bars and 62.5% of hourly bars sit
            outside the regular session, so the hourly and 5-minute runs are MAJORITY
            extended-hours data and the guardrail is what answers "does this survive in the
            hours you could actually trade?". Not available at daily -- the near-close build
            applies its own regular-session filter, so read_and_change_data rejects the
            combination rather than filtering twice (default: {"full"})

    Returns:
        dataframe
    """
    # daily IS the near-close build; every other time format reads its bars directly. The
    # time format is already a parameter here, so nothing has to be threaded in for this.
    if near_close is None:
        near_close = uses_near_close(time_format)
    df = read_and_change_data(
        prefix, file_name, session=session, near_close=near_close
    )

    short_period, long_period, signal_period = macd_params
    df = calculate_indicators(
        df,
        col="close",
        ema_mode="normal",
        short_period=short_period,
        long_period=long_period,
        signal_period=signal_period,
        alpha_range=alpha_range,
        gamma_range=gamma_range,
    )

    return df


###########################
#       Strategies        #
###########################


# @timeit  # timing run only (see Timing section)
def buy_and_hold(stock_dataframe: pd.DataFrame, balance: float = 10000) -> float:
    """
    buy_and_hold employs buy and hold strategy and returns net return

    Arguments:
        stock_dataframe -- stock dataframe of data

    Keyword Arguments:
        balance -- starting money for the strategy and if zero last price - first price (default: {10000})

    Returns:
        buy and hold calculated for closed column of dataframe
    """
    starting_balance = balance

    buy_candle = stock_dataframe.iloc[0]["close"]  # 1
    last_candle = stock_dataframe.iloc[-1]["close"]  # 2

    if balance:  # balllance is a number
        amount = (starting_balance) // buy_candle
    else:  # amount = 1 -> one unit
        return last_candle - buy_candle

    starting_balance = starting_balance + (amount * (last_candle - buy_candle))
    # return ((starting_balance - balance) / balance) * 100
    return starting_balance - balance


def random_sign_series(length: int, n_flips: int, seed: int) -> list:
    """
    random_sign_series builds a +1/-1 series of the given length that changes sign exactly n_flips times at random positions, so feeding it to macd_crossover as the target column produces that many random position changes (this is the random benchmark strategy)

    Arguments:
        length -- how many bars the series covers
        n_flips -- how many sign changes (= position changes) to place
        seed -- rng seed so the series is reproducible

    Returns:
        list of +1/-1 values
    """
    series = [1.0] * length
    # nothing to flip: a constant series just holds one position for the whole window
    if length < 2 or n_flips < 1:
        return series
    rng = np.random.default_rng(seed)
    # pick the bars where the sign flips (never bar 0, that one sets the opening side), then fill each
    # segment between consecutive flips with an alternating sign
    flip_points = sorted(
        rng.choice(range(1, length), size=min(n_flips, length - 1), replace=False)
    )
    sign = 1.0
    previous = 0
    for flip in flip_points:
        for i in range(previous, flip):
            series[i] = sign
        sign = -sign
        previous = flip
    for i in range(previous, length):
        series[i] = sign
    return series


# per-leg proportional trading cost. this hook is OFF in the main pipeline (get_result_from_df sets
# trade_cost = False), so it does not touch the reported numbers; the authoritative cost analysis is
# the post-hoc sweep in make_tables_notebooks.py, which re-walks the saved bets at several bps levels.
def commission_cost(
    balance: float, amount: float, candle: float, commission_bps: float = 5
) -> float:
    """
    commission_cost if there are comissions for opening/closing positions removes a proportional cost (commission_bps basis points of the leg notional = amount * price) from balance

    Arguments:
        balance -- balance
        amount -- number of shares/units in the position
        candle -- price of the leg we are charging (open price when opening, close price when closing)

    Keyword Arguments:
        commission_bps -- per-leg cost in basis points, 1 bp = 0.01% (default: {5})

    Returns:
        new balance after taking comission
    """

    return balance - (commission_bps / 10000) * amount * candle


def modify_bet(
    current_bet: list,
    candle_timestamp: str,
    close_candle: float,
    balance: float,
    outcome: str,
) -> list:
    """
    modify_bet Gets current bet and changes it to format:
        [opened candle, closeed candle, amount, balance before opening, balance after opening, balance after closing, bet type, outcome of the bet]

    Arguments:
        current_bet -- current bet to be chage
        candle_timestamp -- timestamp where bet is closed
        close_candle -- close price
        balance -- balance at time
        outcome -- bet outcome win or lose

    Returns:
        modified bet with new format
    """
    """
    
    """
    # get open candle
    open_candle = current_bet.pop(0)

    # make a temporary list for current bet
    temp_list = [open_candle, (candle_timestamp, close_candle)]

    # extend with rest of the elements
    temp_list.extend(current_bet)

    # get bet type (short/long)
    bet_type = temp_list.pop()

    # add current balance after closing the bet
    temp_list.append(balance)

    # add bet type again
    temp_list.append(bet_type)

    # add the outcome of the bet
    temp_list.append(outcome)

    # return modified bet
    return temp_list


def trim_bet(bet: list) -> list:
    """
    trim_bet trims a full bet down to just the fields we need later for costs and drawdown. a full bet is [(entry_ts, entry_price), (exit_ts, exit_price), amount, balance_before, balance_after_open, balance_after_close, side, outcome] and we keep [entry_ts, entry_price, exit_ts, exit_price, amount, balance_after_close, side, outcome] (the two intermediate balances are dropped and the two candle tuples are flattened)

    Arguments:
        bet -- a full bet from a bets_list

    Returns:
        the trimmed bet as a flat list
    """
    (entry_ts, entry_price), (exit_ts, exit_price) = bet[0], bet[1]
    return [entry_ts, entry_price, exit_ts, exit_price, bet[2], bet[5], bet[6], bet[7]]


def open_long_position(
    candle_timestamp: str,
    candle: float,
    balance: float,
    trade_cost: bool,
    risk_per_trade: float,
    bets: int,
    longs: int,
    bets_list: list,
    commission_bps: float = 5,
) -> tuple[int, float, int, int, int]:
    """
    open_long_position opens a long position and returns new conditions

    Arguments:
        candle_timestamp -- timestamp which position is opened
        candle -- price
        balance -- balance at time
        trade_cost -- whether there is any comission for opening bets
        risk_per_trade -- how much of balance to risk per every trade/bet
        bets -- current number of bets
        longs -- current number of longs
        bets_list -- list of all bets until now

    Keyword Arguments:
        commission_bps -- per-leg cost in basis points when trade_cost is on (default: {5})

    Returns:
        tuple of changed variables
    """
    # candle here is the fill price we actually trade at (next bar's open, or own close at a session
    # end), not the close that triggered the signal. candle_timestamp is the signal bar, so the bets
    # below store (signal timestamp, fill price) and we can always get the signal close back from the data.
    if balance:
        amount = (balance * risk_per_trade) // candle
        buy = candle * amount
        balance_before = balance
        balance -= buy
        if trade_cost:
            balance = commission_cost(balance, amount, candle, commission_bps)
        bets += 1
        longs += 1
        bets_list.append(
            [(candle_timestamp, candle), amount, balance_before, balance, "Long"]
        )
        return 1, balance, amount, bets, longs
    else:
        amount = 1
        buy = candle
        bets += 1
        longs += 1
        bets_list.append([(candle_timestamp, candle), amount, balance, balance, "Long"])
        return 1, 0, 1, bets, longs


def open_short_position(
    candle_timestamp: str,
    candle: float,
    balance: float,
    trade_cost: bool,
    risk_per_trade: float,
    bets: int,
    shorts: int,
    bets_list: list,
    commission_bps: float = 5,
) -> tuple[int, float, int, int, int]:
    """
    open_short_position opens a long position and returns new conditions

    Arguments:
        candle_timestamp -- timestamp which position is opened
        candle -- price
        balance -- balance at time
        trade_cost -- whether there is any comission for opening bets
        risk_per_trade -- how much of balance to risk per every trade/bet
        bets -- current number of bets
        shorts -- current number of shorts
        bets_list -- list of all bets until now

    Keyword Arguments:
        commission_bps -- per-leg cost in basis points when trade_cost is on (default: {5})

    Returns:
        tuple of changed variables
    """
    # candle here is the fill price we actually trade at (next bar's open, or own close at a session
    # end), not the close that triggered the signal. candle_timestamp is the signal bar, so the bets
    # below store (signal timestamp, fill price) and we can always get the signal close back from the data.
    if balance:
        amount = (balance * risk_per_trade) // candle
        buy = candle * amount
        balance_before = balance
        balance += buy
        if trade_cost:
            balance = commission_cost(balance, amount, candle, commission_bps)
        bets += 1
        shorts += 1
        bets_list.append(
            [(candle_timestamp, candle), amount, balance_before, balance, "Short"]
        )
        return -1, balance, amount, bets, shorts
    else:
        amount = 1
        buy = candle
        bets += 1
        shorts += 1
        bets_list.append(
            [(candle_timestamp, candle), amount, balance, balance, "Short"]
        )
        return -1, 0, 1, bets, shorts


def close_long_position(
    candle_timestamp: str,
    candle: float,
    last_candle: float,
    amount: float,
    balance: float,
    trade_cost: bool,
    wins: int,
    long_wins: int,
    long_profit: float,
    bets_list: list,
    last_close: bool = False,
    commission_bps: float = 5,
) -> tuple[int, float, int, int, float]:
    """
    close_long_position close current open long position

    Arguments:
        candle_timestamp -- timestamp which position is opened
        candle -- price
        last_candle -- candle where position was opened
        amount -- number of positions opened
        balance -- balance at time
        trade_cost -- whether there is any comission for opening bets
        wins -- how many wins until now
        long_wins -- how many long wins until now
        long_profit -- long profit until now
        bets_list -- list of all bets until now

    Keyword Arguments:
        last_close -- whether to close or roll back the last bet (default: {False})
        commission_bps -- per-leg cost in basis points when trade_cost is on (default: {5})

    Returns:
        tuple of changed variables
    """
    if balance:
        if last_close:
            if trade_cost:
                balance += (commission_bps / 10000) * amount * last_candle
            balance += last_candle * amount
            bets_list.pop()
            return 0, balance, wins, long_wins, long_profit

        outcome = "Loss"
        if last_candle < candle:
            wins += 1
            long_wins += 1
            outcome = "Win"
        if trade_cost:
            balance = commission_cost(balance, amount, candle, commission_bps)
        long_profit += (candle - last_candle) * amount
        balance += amount * candle

        # assign modified bet to last/current bet
        bets_list[-1] = modify_bet(
            bets_list[-1], candle_timestamp, candle, balance, outcome
        )

        return 0, balance, wins, long_wins, long_profit
    else:
        if last_close:
            bets_list.pop()
            return 0, balance, wins, long_wins, long_profit
        else:
            outcome = "Loss"
            if last_candle < candle:
                wins += 1
                long_wins += 1
                outcome = "Win"
            # no commission in one_pos: its net is long_profit/short_profit (balance is unused here),
            # so a cost on the dead balance did nothing -> dropped to keep one_pos consistently cost-free
            long_profit += candle - last_candle

            # assign modified bet to last/current bet
            bets_list[-1] = modify_bet(
                bets_list[-1], candle_timestamp, candle, balance, outcome
            )

            return 0, balance, wins, long_wins, long_profit


def close_short_position(
    candle_timestamp: str,
    candle: float,
    last_candle: float,
    amount: float,
    balance: float,
    trade_cost: bool,
    wins: int,
    short_wins: int,
    short_profit: float,
    bets_list: list,
    last_close: bool = False,
    commission_bps: float = 5,
) -> tuple[int, float, int, int, float]:
    """
    close_short_position close current open short position

    Arguments:
        candle_timestamp -- timestamp which position is opened
        candle -- price
        last_candle -- candle where position was opened
        amount -- number of positions opened
        balance -- balance at time
        trade_cost -- whether there is any comission for opening bets
        wins -- how many wins until now
        short_wins -- how many short wins until now
        short_profit -- short profit until now
        bets_list -- list of all bets until now

    Keyword Arguments:
        last_close -- whether to close or roll back the last bet (default: {False})
        commission_bps -- per-leg cost in basis points when trade_cost is on (default: {5})

    Returns:
        tuple of changed variables
    """

    if balance:
        if last_close:
            if trade_cost:
                balance += (commission_bps / 10000) * amount * last_candle
            balance -= last_candle * amount
            bets_list.pop()
            return 0, balance, wins, short_wins, short_profit

        outcome = "Loss"
        if last_candle > candle:
            wins += 1
            short_wins += 1
            outcome = "Win"
        if trade_cost:
            balance = commission_cost(balance, amount, candle, commission_bps)
        short_profit -= (candle - last_candle) * amount
        balance -= amount * candle

        # assign modified bet to last/current bet
        bets_list[-1] = modify_bet(
            bets_list[-1], candle_timestamp, candle, balance, outcome
        )

        return 0, balance, wins, short_wins, short_profit
    else:
        if last_close:
            bets_list.pop()
            return 0, balance, wins, short_wins, short_profit
        else:
            outcome = "Loss"
            if last_candle > candle:
                wins += 1
                short_wins += 1
                outcome = "Win"
            # no commission in one_pos: its net is long_profit/short_profit (balance is unused here),
            # so a cost on the dead balance did nothing -> dropped to keep one_pos consistently cost-free
            short_profit -= candle - last_candle

            # assign modified bet to last/current bet
            bets_list[-1] = modify_bet(
                bets_list[-1], candle_timestamp, candle, balance, outcome
            )

            return 0, balance, wins, short_wins, short_profit


def get_first_signal(first_point: float) -> tuple[int, int]:
    """
    get_first_signal determines the signal at the start of the data

    Arguments:
        first_point -- first data point

    Returns:
        returns the signal and index (which is zero)
    """
    if first_point >= 0:  # first long position state
        return 1, 0
    else:  # first short position state
        return -1, 0


# @timeit  # timing run only (see Timing section)
def macd_crossover(
    data: pd.DataFrame,
    price: str,
    target: str,
    signal_line_target: str,
    balance: float = 100000,
    risk_per_trade: float = 20 / 100,
    trade_cost: bool = False,
    signal_line_if: bool = False,
    signal_line_if_target: bool = False,
    last_close: bool = True,
) -> list[float, float, float, int, int, int, int, float, int, int, float, list]:
    """
    macd_crossover deploys crossover strategy (both zero cross and macd cross) on the dataframe

    Arguments:
        data -- dataframe of data with all histograms and macds
        price -- price column
        target -- target column
        signal_line_target -- signal line of target column

    Keyword Arguments:
        balance -- balance at the start (default: {100000})
        risk_per_trade -- how much to risk on each trade (default: {20/100})
        trade_cost -- commission cost if there is any (default: {False})
        signal_line_if -- second system (default: {False})
        signal_line_if_target -- third system (default: {False})
        last_close -- whether to close or roll back the last bet (default: {True})

    Returns:
        statistics of how well the strategy performed
    """

    starting_balance = balance
    # the 75% capital stop. When balance falls below this, trading halts for the rest of
    # the window. It is deliberately NOT returned: appending a 13th value to this list breaks
    # 19 positional destructures across paperCompare / plot_positions / test2 and bh.ipynb's
    # `random_result[-1]`, and it is unnecessary -- whether the stop fired is recoverable
    # post-hoc from `net`, which is already saved for all 475 configs in `{tick}_nets`.
    # VERIFIED exact on 200 cases (20 firings, 0 disagreements): the stop fired if and only if
    # net <= -0.25 * balance. Measured separately: it fires 0 times on real prices across 20
    # configs over the Feb-Apr 2020 crash and 9 configs each on BA / NKE / AAPL to Sep 2020.
    balance_threshhold = balance * (3 / 4)
    got_thresh = False
    # signal = 0
    bets = 0
    wins = 0
    longs = 0
    shorts = 0
    long_wins = 0
    short_wins = 0
    long_profit = 0
    short_profit = 0
    bets_list = []

    target_his_data = data[target].to_list()
    price_data = data[price].to_list()
    fill_data = data["fill"].to_list()  # execution price: next bar's open (own close on last bar of day)
    signal_line_data = data["SignalLine"].to_list()
    target_signal_line_data = data[signal_line_target].to_list()

    signal, i = get_first_signal(target_his_data[0])
    candle = fill_data[i]

    for i in range(1, len(data)):
        # if (target_his_data[i-1] < 0) and (target_his_data[i] >= 0):
        if target_his_data[i] >= 0:
            if signal == -1:
                # open first long position
                if signal_line_if:
                    if signal_line_data[i] > 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] > 0:
                        continue
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                signal, balance, amount, bets, longs = open_long_position(
                    candle_timestamp,
                    candle,
                    balance,
                    trade_cost,
                    risk_per_trade,
                    bets,
                    longs,
                    bets_list,
                )
                # if target == "MACD a: 0.9":
                #     print("long index", index, candle_timestamp)
                break
        # elif (target_his_data[index-1] >= 0) and (target_his_data[index] < 0):
        if target_his_data[i] < 0:
            if signal == 1:
                # open first short position
                if signal_line_if:
                    if signal_line_data[i] < 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] < 0:
                        continue
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                signal, balance, amount, bets, shorts = open_short_position(
                    candle_timestamp,
                    candle,
                    balance,
                    trade_cost,
                    risk_per_trade,
                    bets,
                    shorts,
                    bets_list,
                )
                # if target == "MACD a: 0.9":
                #     print("short index", index, candle_timestamp)
                break

    index = i
    for i in range(index + 1, len(data)):
        # if (target_his_data[i-1] < 0) and (target_his_data[i] >= 0): # long
        if target_his_data[i] >= 0:  # long
            # meaning it's a short position
            if signal == -1:  # then; close short and open long
                if signal_line_if:
                    if signal_line_data[i] > 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] > 0:
                        continue

                last_candle = candle
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                # close short position
                # print("before close short", balance)
                signal, balance, wins, short_wins, short_profit = close_short_position(
                    candle_timestamp,
                    candle,
                    last_candle,
                    amount,
                    balance,
                    trade_cost,
                    wins,
                    short_wins,
                    short_profit,
                    bets_list,
                )
                # print("after close short", balance)

                if balance:
                    # print('checking balance')
                    if balance < balance_threshhold:
                        # print('got thresh')
                        # print(balance, balance_threshhold)
                        # input()
                        got_thresh = True
                        signal = 0
                        break

                # open long positon
                # print("before open long", balance)
                signal, balance, amount, bets, longs = open_long_position(
                    candle_timestamp,
                    candle,
                    balance,
                    trade_cost,
                    risk_per_trade,
                    bets,
                    longs,
                    bets_list,
                )
                # print("after open long", balance)
                # if target == "MACD a: 0.9":
                #     print("long i", i, candle_timestamp)

        # elif (target_his_data[i-1] >= 0) and (target_his_data[i] < 0):
        elif target_his_data[i] < 0:  # short
            # meaning it's a long position
            if signal == 1:  # then; close long and open short
                if signal_line_if:
                    if signal_line_data[i] < 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] < 0:
                        continue

                # close long position
                last_candle = candle
                # print(i, index, index+1, len(data))
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                # print("before close long", balance)
                signal, balance, wins, long_wins, long_profit = close_long_position(
                    candle_timestamp,
                    candle,
                    last_candle,
                    amount,
                    balance,
                    trade_cost,
                    wins,
                    long_wins,
                    long_profit,
                    bets_list,
                )
                # print("after close long", balance)

                if balance:
                    # print('checking balance')
                    if balance < balance_threshhold:
                        # print('got thresh')
                        # print(balance, balance_threshhold)
                        # input()
                        got_thresh = True
                        signal = 0
                        break

                # open short position
                # print("before open shorts", balance)
                signal, balance, amount, bets, shorts = open_short_position(
                    candle_timestamp,
                    candle,
                    balance,
                    trade_cost,
                    risk_per_trade,
                    bets,
                    shorts,
                    bets_list,
                )
                # print("after open short", balance)
                # if target == "MACD a: 0.9":
                #     print("short i", i, candle_timestamp)

    # if got_thresh:
    #     print(balance, balance_threshhold)
    #     input()

    # we should close the open bet
    if not got_thresh and bets != 0:
        if signal == -1:
            last_candle = candle
            candle = fill_data[i]
            candle_timestamp = str(data.index[i])
            # print("before close last short", balance)
            signal, balance, wins, short_wins, short_profit = close_short_position(
                candle_timestamp,
                candle,
                last_candle,
                amount,
                balance,
                trade_cost,
                wins,
                short_wins,
                short_profit,
                bets_list,
                last_close,
            )
            if last_close:
                shorts -= 1
                bets -= 1
            # print("after close last short", balance)
            # print(balance)
        elif signal == 1:
            last_candle = candle
            candle = fill_data[i]
            candle_timestamp = str(data.index[i])
            # print("before close last long", balance)
            signal, balance, wins, long_wins, long_profit = close_long_position(
                candle_timestamp,
                candle,
                last_candle,
                amount,
                balance,
                trade_cost,
                wins,
                long_wins,
                long_profit,
                bets_list,
                last_close,
            )
            if last_close:
                longs -= 1
                bets -= 1
            # print(balance)
            # print("after close last long", balance)

    if bets == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]

    if balance:
        # print("here")
        # input()
        net_gain = balance - starting_balance
        return [
            net_gain,
            (net_gain / bets),
            ((net_gain / starting_balance) * 100),
            bets,
            wins,
            longs,
            long_wins,
            long_profit,
            shorts,
            short_wins,
            short_profit,
            bets_list,
        ]
    else:
        net_gain = long_profit + short_profit
        return [
            net_gain,
            (net_gain / bets),
            0,
            bets,
            wins,
            longs,
            long_wins,
            long_profit,
            shorts,
            short_wins,
            short_profit,
            bets_list,
        ]


# @timeit  # timing run only (see Timing section)
def only_long_macd_crossover(
    data: pd.DataFrame,
    price: str,
    target: str,
    signal_line_target: str,
    balance: float = 100000,
    risk_per_trade: float = 20 / 100,
    trade_cost: bool = False,
    signal_line_if: bool = False,
    signal_line_if_target: bool = False,
    last_close: bool = True,
) -> list[float, float, float, int, int, int, int, float, int, int, float, list]:
    """
    only_long_macd_crossover deploys long only crossover strategy (both zero cross and macd cross) on the dataframe

    Arguments:
        data -- dataframe of data with all histograms and macds
        price -- price column
        target -- target column
        signal_line_target -- signal line of target column

    Keyword Arguments:
        balance -- balance at the start (default: {100000})
        risk_per_trade -- how much to risk on each trade (default: {20/100})
        trade_cost -- commission cost if there is any (default: {False})
        signal_line_if -- second system (default: {False})
        signal_line_if_target -- third system (default: {False})
        last_close -- whether to close or roll back the last bet (default: {True})

    Returns:
        statistics of how well the strategy performed
    """

    starting_balance = balance
    # the 75% capital stop. When balance falls below this, trading halts for the rest of
    # the window. It is deliberately NOT returned: appending a 13th value to this list breaks
    # 19 positional destructures across paperCompare / plot_positions / test2 and bh.ipynb's
    # `random_result[-1]`, and it is unnecessary -- whether the stop fired is recoverable
    # post-hoc from `net`, which is already saved for all 475 configs in `{tick}_nets`.
    # VERIFIED exact on 200 cases (20 firings, 0 disagreements): the stop fired if and only if
    # net <= -0.25 * balance. Measured separately: it fires 0 times on real prices across 20
    # configs over the Feb-Apr 2020 crash and 9 configs each on BA / NKE / AAPL to Sep 2020.
    balance_threshhold = balance * (3 / 4)
    got_thresh = False
    # signal = 0
    bets = 0
    wins = 0
    longs = 0
    shorts = 0
    long_wins = 0
    short_wins = 0
    long_profit = 0
    short_profit = 0
    bets_list = []

    target_his_data = data[target].to_list()
    price_data = data[price].to_list()
    fill_data = data["fill"].to_list()  # execution price: next bar's open (own close on last bar of day)
    signal_line_data = data["SignalLine"].to_list()
    target_signal_line_data = data[signal_line_target].to_list()

    signal, i = get_first_signal(target_his_data[0])
    candle = fill_data[i]

    for i in range(1, len(data)):
        # if (target_his_data[i-1] < 0) and (target_his_data[i] >= 0):
        if target_his_data[i] >= 0:
            if signal == -1:
                # open first long position
                if signal_line_if:
                    if signal_line_data[i] > 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] > 0:
                        continue
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                signal, balance, amount, bets, longs = open_long_position(
                    candle_timestamp,
                    candle,
                    balance,
                    trade_cost,
                    risk_per_trade,
                    bets,
                    longs,
                    bets_list,
                )
                # if target == "MACD a: 0.9":
                #     print("long i", i, candle_timestamp)
                break
        # elif (target_his_data[i-1] >= 0) and (target_his_data[i] < 0):
        if target_his_data[i] < 0:
            if signal == 1:
                # open first short position
                if signal_line_if:
                    if signal_line_data[i] < 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] < 0:
                        continue
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                signal = -1
                # if target == "MACD a: 0.9":
                #     print("short i", i, candle_timestamp)
                break

    index = i
    for i in range(index + 1, len(data)):
        # if (target_his_data[i-1] < 0) and (target_his_data[i] >= 0): # long
        if target_his_data[i] >= 0:  # long
            # meaning it's a short position
            if signal == -1:  # then; close short and open long
                if signal_line_if:
                    if signal_line_data[i] > 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] > 0:
                        continue

                last_candle = candle
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                # open long positon
                # print("before open long", balance)
                signal, balance, amount, bets, longs = open_long_position(
                    candle_timestamp,
                    candle,
                    balance,
                    trade_cost,
                    risk_per_trade,
                    bets,
                    longs,
                    bets_list,
                )
                # print("after open long", balance)

        # elif (target_his_data[i-1] >= 0) and (target_his_data[i] < 0):
        elif target_his_data[i] < 0:  # short
            # meaning it's a long position
            if signal == 1:  # then; close long and open short
                if signal_line_if:
                    if signal_line_data[i] < 0:
                        continue
                if signal_line_if_target:
                    if target_signal_line_data[i] < 0:
                        continue

                # close long position
                last_candle = candle
                candle = fill_data[i]
                candle_timestamp = str(data.index[i])
                # print("before close long", balance)
                signal, balance, wins, long_wins, long_profit = close_long_position(
                    candle_timestamp,
                    candle,
                    last_candle,
                    amount,
                    balance,
                    trade_cost,
                    wins,
                    long_wins,
                    long_profit,
                    bets_list,
                )
                # print("after close long", balance)

                if balance:
                    if balance < balance_threshhold:
                        got_thresh = True
                        signal = 0
                        break

                signal = -1

    # we should close
    if not got_thresh and bets != 0:
        if signal == -1:
            signal = 0
        elif signal == 1:
            last_candle = candle
            candle = fill_data[i]
            candle_timestamp = str(data.index[i])
            # print("before close last long", balance)
            signal, balance, wins, long_wins, long_profit = close_long_position(
                candle_timestamp,
                candle,
                last_candle,
                amount,
                balance,
                trade_cost,
                wins,
                long_wins,
                long_profit,
                bets_list,
                last_close,
            )
            if last_close:
                longs -= 1
                bets -= 1
            # print(balance)
            # print("after close last long", balance)

    if bets == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]

    if balance:
        net_gain = balance - starting_balance
        return [
            net_gain,
            (net_gain / bets),
            ((net_gain / starting_balance) * 100),
            bets,
            wins,
            longs,
            long_wins,
            long_profit,
            shorts,
            short_wins,
            short_profit,
            bets_list,
        ]
    else:
        net_gain = long_profit
        return [
            net_gain,
            (net_gain / bets),
            0,
            bets,
            wins,
            longs,
            long_wins,
            long_profit,
            shorts,
            short_wins,
            short_profit,
            bets_list,
        ]

###########################
#         Helpers         #
###########################

def write_to_text(prefix, file_name, text, mode='a'):
    with open(f"{prefix}/{file_name}", mode) as text_file:
            # text_file.write("\\newpage\n")
            text_file.write(text)
            text_file.write("\n")
            text_file.write("\\newpage\n")
            
def write_rules_tex(table_data, table_prefix, table_name, label, caption):
    start_text =  f"""\\begin{"{table}"}[!ht]
	\\centering
	\\caption{"{" + caption+ "}"}
	\\begin{"{adjustbox}"}{"{width=1.25\\textwidth,center=\\textwidth}"}
		\\begin{"{tabular}"}{"{llllcccccccccccc}"}
            \\toprule
            &            &          &  &      & Rule 1 &       &      & Rule 2 &       &      & Rule 3 &       &      & Genetic &        \\\\
            &            &          &  & MACD & SMACD   & CMACD & MACD & SMACD  & CMACD & MACD & SMACD  & CMACD & MACD & SMACD   & CMACD  \\\\
            \\midrule
            &            &          &  &      &         &       &      &        &       &      &        &       &      &         &        \\\\"""
            
    l1 = "1-day     & Normal     & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 1", *list(table_data.iloc[0]))
    l2 = "          &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 2", *list(table_data.iloc[1]))
    l3 = "          &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 3", *list(table_data.iloc[2]))
    l4 = "          & Simulation & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 1", *list(table_data.iloc[3]))
    l5 = "          &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 2", *list(table_data.iloc[4]))
    l6 = "          &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 3", *list(table_data.iloc[5]))
    l7 = "1-hour    & Normal     & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 1", *list(table_data.iloc[6]))
    l8 = "          &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 2", *list(table_data.iloc[7]))
    l9 = "          &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 3", *list(table_data.iloc[8]))
    l10 = "         & Simulation & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 1", *list(table_data.iloc[9]))
    l11 = "         &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 2", *list(table_data.iloc[10]))
    l12 = "         &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 3", *list(table_data.iloc[11]))
    l13 = "5-minute & Normal     & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 1", *list(table_data.iloc[12]))
    l14 = "         &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 2", *list(table_data.iloc[13]))
    l15 = "         &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 3", *list(table_data.iloc[14]))
    l16 = "         & Simulation & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 1", *list(table_data.iloc[15]))
    l17 = "         &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 2", *list(table_data.iloc[16]))
    l18 = "         &            & {} &  & {}   & {}     & {}  & {}   & {}   & {}  & {}   & {}    & {}  & {}    & {}     & {}   \\\\".format("System 3", *list(table_data.iloc[17]))
    llist = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18]
    end_text =  f"""            \\bottomrule
		\\end{"{tabular}"}
	\\end{"{adjustbox}"}
	\\label{"{sec.results.compare.table." + label + "}"}
\\end{"{table}"}"""

    with open(f"{table_prefix}/{table_name}_tex.tex", "w") as text_file:
        text_file.write(start_text)
        text_file.write("\n")
        for ll in llist:
            text_file.write("\t\t\t")
            text_file.write(ll)
            text_file.write("\n")
            # print(ll)
        text_file.write(end_text)

def write_table_tex(table_data, table_prefix, table_name, label, caption):
    # AUDIT ROW N. This header used to be hardcoded in the literal below and printed
    # "Transactions", while write_overall_table_tex routes the SAME quantity through
    # tex_column_names and prints "Positions" -- one paper naming one number two ways,
    # in two table families a reader sets side by side. Both now come from
    # tex_column_names, so they cannot drift apart again. The metric list is a module
    # constant because the .format() calls below are POSITIONAL on exactly that order.
    header = " & ".join(
        f"{tex_column_names.get(metric, metric):21}" for metric in rules_table_metrics
    )
    start_text =  f"""\\begin{"{table}"}[!ht]
	\\centering
	\\caption{"{" + caption+ "}"}
	\\begin{"{adjustbox}"}{"{width=1.25\\textwidth,center=\\textwidth}"}
		\\begin{"{tabular}"}{"{llllcccccccc}"}
            \\toprule
            &          &        &  & {header}  \\\\ 
            \\midrule
            &          &        &  & \\multicolumn{"{1}"}{"{l}"}{"{}"} & \\multicolumn{"{1}"}{"{l}"}{"{}"} & \\multicolumn{"{1}"}{"{l}"}{"{}"} & \\multicolumn{"{1}"}{"{l}"}{"{}"} & \\multicolumn{"{1}"}{"{l}"}{"{}"} & \\multicolumn{"{1}"}{"{l}"}{"{}"} & \\multicolumn{"{1}"}{"{l}"}{"{}"} & \\multicolumn{"{1}"}{"{l}"}{"{}"}  \\\\"""
            
    l1 =  "MACD Crossover  & System 1 & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 1", *list(table_data.iloc[0]))
    l2 =  "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 2", *list(table_data.iloc[1]))
    l3 =  "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 3", *list(table_data.iloc[2]))
    l4 =  "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("GA", *list(table_data.iloc[3]))
    l5 =  "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("MACD", *list(table_data.iloc[4]))
    l6 =  "                & System 2 & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 1", *list(table_data.iloc[5]))
    l7 =  "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 2", *list(table_data.iloc[6]))
    l8 =  "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 3", *list(table_data.iloc[7]))
    l9 =  "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("GA", *list(table_data.iloc[8]))
    l10 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("MACD", *list(table_data.iloc[9]))
    l11 = "                & System 3 & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 1", *list(table_data.iloc[10]))
    l12 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 2", *list(table_data.iloc[11]))
    l13 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 3", *list(table_data.iloc[12]))
    l14 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("GA", *list(table_data.iloc[13]))
    l15 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("MACD", *list(table_data.iloc[14]))
    l16 = "Zero Crossover  & System 1 & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 1", *list(table_data.iloc[15]))
    l17 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 2", *list(table_data.iloc[16]))
    l18 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 3", *list(table_data.iloc[17]))
    l19 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("GA", *list(table_data.iloc[18]))
    l20 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("MACD", *list(table_data.iloc[19]))
    l21 = "                & System 2 & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 1", *list(table_data.iloc[20]))
    l22 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 2", *list(table_data.iloc[21]))
    l23 = "                & ~        & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 3", *list(table_data.iloc[22]))
    l24 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("GA", *list(table_data.iloc[23]))
    l25 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("MACD", *list(table_data.iloc[24]))
    l26 = "                & System 3 & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 1", *list(table_data.iloc[25]))
    l27 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 2", *list(table_data.iloc[26]))
    l28 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("Rule 3", *list(table_data.iloc[27]))
    l29 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("GA", *list(table_data.iloc[28]))
    l30 = "                &          & {} &  & {:.2f}  & {:.0f}            & {:.2f}     & {:.2f}       & {:.0f}                & {:.2f}          & {:.2f}        & {:.2f}            \\\\".format("MACD", *list(table_data.iloc[29]))
    llist = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29, l30]
    end_text =  f"""            \\bottomrule
		\\end{"{tabular}"}
	\\end{"{adjustbox}"}
	\\label{"{sec.results.compare.table." + label + "}"}
\\end{"{table}"}"""

    with open(f"{table_prefix}/{table_name}_tex.tex", "w") as text_file:
        text_file.write(start_text)
        text_file.write("\n")
        for ll in llist:
            text_file.write("\t\t\t")
            text_file.write(ll)
            text_file.write("\n")
            # print(ll)
        text_file.write(end_text)
            
# -------------------------------------------------------------------------------------
# PUBLISHED RESULTS WE COMPARE AGAINST, one block per paper.
#
# These used to be four latex lines hardcoded inside write_overall_table_tex and emitted
# whenever first_part was True. That made "show the comparison rows" and "use Wiles's
# numbers" the same switch, so the NIKKEI table was printing Wiles's SOYBEAN figures as its
# published baseline, and the NASDAQ table got no comparison rows at all. Keyed by paper
# now, so each case study can only ever print its own.
#
# Every figure below was read off the paper itself and is cited to its table. Where a paper
# did not report a column, the entry is None and the cell is left blank rather than filled
# with something invented.
#
# UNITS DIFFER BY PAPER and are not interchangeable:
#   wiles   -- cents per bushel, summed over all trades (a single contract, SQ2014)
#   kang    -- Nikkei 225 index points, one unit per trade
#   aguirre -- EFFECTIVE ANNUAL RETURN IN PERCENT, not an absolute amount
#
# COSTS: Wiles states he assumed no transaction costs; Kang ignores the JPY 330 round-trip
# commission as negligible; Aguirre's Table 3 is explicitly net of transaction costs. So two
# of the three baselines are gross and one is net -- say so when reporting.
#
# Row layout matches the table columns:
#   (label, parameters, return, transactions, win ratio, long return,
#    long transactions, long win ratio, short return, short win ratio)
#
# TRANSACTION COUNTING (C6). Every paper reports Total = buy signals + sell signals, and in
# all three Total = Long + Short exactly. What that equals depends on the system, so there
# is no single conversion:
#   - KANG is always in market -- "when a position is closed, a reverse trade is
#     automatically executed" -- so one signal opens exactly one position and his Total IS
#     a position count. It is therefore directly comparable to our `bets` with NO
#     adjustment. Measured: our Nikkei (12,26,9) base row gives 175 positions against his
#     169, a ratio of 1.04, and our 88 long against his 84. Doubling ours, as was once
#     planned here, would have put us at 350 against his 169 -- wrong by exactly 2x.
#   - AGUIRRE is long only and goes to cash between trades, so TWO signals make one round
#     trip. His nine round trips are nine positions; the eighteen signals are legs. His
#     rows below are stated as POSITIONS so they sit on our basis.
#   - WILES accumulates contracts rather than holding one position at a time (his note that
#     "3 extra sell contracts had to be placed at the end of the contract period to settle"
#     is what gives him 48 buys against 45 sells). His counts are contract transactions and
#     cannot be converted to positions cleanly, so they are quoted as published and the
#     mismatch is a footnote, not a silent adjustment.
# Our own column counts POSITIONS throughout, which is what the engine's `bets` is.
# -------------------------------------------------------------------------------------
published_comparison_rows = {
    # Wiles & Enke (2015), Procedia Computer Science 61, pp. 88-91. CME August 2014 soybean
    # contract, 160 one-day-ahead test points over 2013-12-26..2014-08-14. He reports only a
    # profit, a buy-signal count and a sell-signal count, so every other column is blank.
    "wiles": [
        # Table 3: buy and hold, 1 buy and 1 sell = ONE position
        ("Buy and Hold", "--", -25.3, 1, None, -25.3, 1, None, None, None),
        # Table 4: 48 buys + 45 sells = 93, 3906.0 cents for all trades, 81.4 per round trip
        ("MACD Heuristics", "(12,26,9)", 3906, 93, None, None, 48, None, None, None,
         {"Per Round Trip": 81.4}),
        # Table 6: 47 buys + 50 sells = 97, 4507.0 cents, 90.1 per round trip
        ("MACD Heuristics", "(11.017,32.925,16.467)", 4507, 97, None, None, 47, None, None,
         None, {"Per Round Trip": 90.1}),
        # Table 5: 4 buys + 5 sells = 9, 664.5 cents, 132.9 per round trip. THIS is the row
        # that is apples-to-apples with our crossover strategies.
        ("MACD Crossover Heuristics", "(12,26,9)", 664.5, 9, None, None, 4, None, None, None,
         {"Per Round Trip": 132.9}),
        # Table 7: 4 buys + 5 sells = 9, 728.0 cents, 145.6 per round trip
        ("MACD Crossover Heuristics", "(6.866,33.812,4.575)", 728, 9, None, None, 4, None,
         None, None, {"Per Round Trip": 145.6}),
    ],
    # Kang (2021), JRFM 14(1) 37. Nikkei 225 Futures near maturity, 2204 trading days over
    # 2011-01-04..2019-12-30, one unit per trade, executed at the NEXT day's close.
    #
    # HIS RULE IS THE SIGNAL LINE CROSSOVER (his Equation 8: buy when the histogram is
    # positive, sell when negative), which he adopts as "the basic MACD trading rule"
    # throughout. That is OUR "MACD Crossover" block, not our "Zero Crossover" block -- his
    # Equation 7 zero crossover is defined but not used for his results. We report both of
    # our blocks, so the rows below sit above both; the MACD Crossover rows are the
    # like-for-like comparison and the Zero Crossover rows are additional evidence. He reports
    # every column we need, though the win ratios are derived from his counts rather than
    # printed: win ratio = (N(Buy>0) + N(Sell>0)) / (N(Buy) + N(Sell)).
    "kang": [
        # Table 1 whole period: 84 buys, 85 sells, 33 and 20 profitable, 5430 long, -9610 short
        ("MACD", "(12,26,9)", -4180, 169, 31.4, 5430, 84, 39.3, -9610, 23.5),
        # Table 2 whole period: 217 and 217, 107 and 79 profitable, 12040 long, -2620 short
        ("MACD", "(4,22,3)", 9420, 434, 42.9, 12040, 217, 49.3, -2620, 36.4),
        # Table 3: the 1-day minimum holding strategy on MACD (4,22,3)
        ("1-day holding", "(4,22,3)", 19970, 310, 45.2, 17070, 155, 52.3, 2900, 38.1),
        # Table 4: the 30%-line of the 10-day histogram crossover strategy
        (r"30\%-line 10-day", "(4,22,3)", 22690, 312, 49.0, 18510, 155, 54.8, 4180, 43.3),
        # Table 5: the search for the first peak/bottom point of the histogram
        ("1st peak/bottom", "(4,22,3)", 20950, 437, 52.6, 18020, 218, 56.0, 2930, 49.3),
        # Table 12: his buy-and-hold benchmark. NOTE it is a MONTHLY buy-and-hold -- buy at
        # the start of each month, sell at the end -- which he adopts so it can be compared
        # against his monthly model returns, not a single hold over the nine years. 108
        # monthly round trips, 64 profitable (59.3%), profit 40,150 against losses -28,830.
        # 108 monthly round trips = 108 POSITIONS, on the same basis as his other rows and
        # as our own column. It is long only, so the short columns are blank by nature.
        ("Buy and Hold (monthly)", "--", 11320, 108, 59.3, 11320, 108, 59.3, None, None),
    ],
    # THE TWO AGUIRRE PAPERS ARE DIFFERENT STUDIES, not one paper in two venues. Same
    # authors, same index, but different sample periods, different GA parameters and
    # different results. Keyed separately so a run cannot cite the wrong one.
    #
    # Agudelo Aguirre, Rojas Medina & Ciro Gomez (2021), Journal of Economics, Finance and
    # Administrative Science 26(52), Tables 2 and 3. NASDAQ COMPOSITE via Thomson
    # Reuters-Refinitiv Eikon, 2013-01-01..2017-12-31, trained on the first 50%
    # (..2015-06-30) and validated on the second (2015-07-01..), with the investment
    # actually running 2015-08-11..2017-10-27 = 808 days from USD 5,036,790.
    #
    # THIS is the study nasdaq.ipynb was built against: its hardcoded balance of 5036790 is
    # his initial investment, its window is his, and the 7_10_15 results folder is his GA
    # parameter triple. Our ndq.csv reproduces his index to the cent on both anchors
    # (2015-08-11 close 5036.79, 2017-10-27 close 6701.26), so this comparison rests on
    # confirmed-identical data -- the strongest of the three case studies on that count.
    #
    # Unit is PERCENT: "Return" here is his effective ANNUAL rate and "Long Return" his
    # nominal return over the 808 days. He is long only (cash between trades), so the short
    # columns are blank by nature. Transactions are legs.
    #
    # HIS TWO CONVENTIONS, both verified against his own value columns, and both of which
    # our CAGR column must copy or the comparison is biased:
    #   1. his "Nominal (%)" is a NATURAL LOG return, not a percentage change. ln of
    #      7,658,339/5,036,790 = 41.90, of 6,701,263/5,036,790 = 28.55, of
    #      5,776,270/5,036,790 = 13.70 -- all three exact. (An earlier comment here called
    #      this column internally inconsistent with its value columns. That was WRONG: the
    #      simple returns are 52.05/33.05/14.68, and mistaking the log figures for simple
    #      ones is what produced the false accusation.)
    #   2. his "Annual (%)" then annualises that LOG return AS THOUGH IT WERE SIMPLE:
    #      (1 + nominal)^(365/808) - 1. That reproduces 17.13/12.01/5.97 exactly. Done
    #      properly, exp(log x 365/808) - 1, the same figures would be 20.84/13.77/6.38, so
    #      his convention understates every strategy including his own benchmark by ~1.8pp
    #      on buy-and-hold. Compare like with like: compute ours his way, or recompute his.
    "aguirre2021": [
        # Table 3: GA, 5,036,790 -> 7,658,339, nominal 41.90%, annual 17.13%. Table 2 lists
        # 9 round trips and the text confirms 9 buy and 9 sell signals, i.e. 9 POSITIONS.
        #
        # DECISION C5, SETTLED 1 Nov: "Return" holds his PROFIT IN CURRENCY -- not his
        # nominal and not his annual rate. The question as posed, nominal or annual in the
        # "Return" cell, had no good answer, because BOTH are percentages while our own
        # "Return" column is an absolute profit (the engine's `net`). Either choice printed
        # a percent in a currency column, so a reader comparing straight down that column
        # saw 17.13 against our 2,621,549 and had to guess which of the two was wrong.
        #
        # He does report a currency profit, and on OUR basis: 7,658,339 - 5,036,790 =
        # 2,621,549, from the same initial investment our run uses (nasdaq.ipynb anchors
        # balance to his 5,036,790) on an index our ndq.csv reproduces to the cent at both
        # ends. So the shared column is now genuinely shared. Checked in both directions:
        # feeding each profit below back through aguirre_annual_rate at his 808 days
        # returns 17.13 / 12.01 / 5.97 exactly, so the currency column and the "Annual %"
        # column are two views of ONE figure rather than two separate claims.
        #
        # His NOMINAL is deliberately not printed: this table has no nominal column to
        # print it in, and it is recorded above with the ln arithmetic that verifies it.
        # That is not the row-N defect of a published figure left blank in a column that
        # exists -- every column this table does have is filled with his own figure.
        #
        # He is long only and holds cash between trades, so the long cells repeat the
        # total, exactly as the wiles and kang buy-and-hold rows do. The arm of ours to
        # read him against is Long-Only, which the table note says.
        ("GA", "(7,10,15)", 2621549, 9, None, 2621549, 9, None, None, None,
         {"Annual %": 17.13}),
        # Table 3: B&H, 5,036,790 -> 6,701,263, nominal 28.55%, annual 12.01%. One buy and
        # one sell = one position. 6,701,263 - 5,036,790 = 1,664,473.
        ("Buy and Hold", "--", 1664473, 1, None, 1664473, 1, None, None, None,
         {"Annual %": 12.01}),
        # Table 3: TA, 5,036,790 -> 5,776,270, nominal 13.70%, annual 5.97%. Text: 10 buy
        # and 9 sell signals, so 10 positions with the last one still open at the end.
        # 5,776,270 - 5,036,790 = 739,480.
        ("Traditional MACD", "(12,26,9)", 739480, 10, None, 739480, 10, None, None, None,
         {"Annual %": 5.97}),
    ],
    # Agudelo Aguirre et al. (2020), Investment Management and Financial Innovations 17(4),
    # Table 3. A SEPARATE study: 2013-01-01..2019-12-31 with TWO validation windows and a
    # different GA result. Its Table 3 is explicitly net of transaction costs.
    #
    # CAREFUL, THIS DOES NOT FIT THE COLUMN SCHEMA: the figures are EFFECTIVE ANNUAL RETURN
    # IN PERCENT, so "Return" is not the same quantity as in the wiles/kang blocks. He is
    # also LONG ONLY, so the short columns are blank by nature, not by omission, and our
    # "Long 2" arm is the correct thing to compare against.
    #
    # DECISION C5 for this block, and it is why the decision could not stop at the 2021
    # one. The rates below used to sit in "Return", our CURRENCY column. They move to the
    # named "Annual %" column, where ours sits too, and the currency cells go BLANK.
    #
    # BLANK, not converted: his Table 3 reports no currency profit for the benchmark or
    # traditional-MACD rows, and the one place currency values do appear -- his Table 2 --
    # is the source that CONTRADICTS Table 3's row labels (the transposition noted below).
    # Taking currency from Table 2 and rates from Table 3 would mix the two sources inside
    # a single row, on precisely the rows whose labelling is in doubt. This file's own rule
    # applies instead: a column the paper did not report is left empty, not invented.
    #
    # THE SECOND HALF OF C5 -- HIS ANNUALISATION IS NOT THE 2021 PAPER'S. Recomputed from
    # the dates and values in his own Table 2, his nominal is a SIMPLE return and his
    # annual figure a true CAGR: 4,735,280 -> 6,663,284 over 776 days gives 40.72% and
    # 17.43%, and 7,083,645 -> 8,506,987 over 376 days gives 20.09% and 19.45% -- all four
    # exact. Our "Annual %" was computed by aguirre_annual_rate on the 2021 convention no
    # matter which paper the table compared against, which on THIS paper's basis
    # understates us by 2.61pp on the initial window and 1.72pp on the final one. The
    # direction runs against our own result, but it is still a like-for-unlike comparison
    # in the one column that claims to be like-for-like. aguirre_annual_rate now takes the
    # convention and nasdaq.ipynb passes "cagr" for the window cited against this paper.
    #
    # TABLE 3's TWO ROWS LOOK TRANSPOSED, and his own Table 2 settles it. Recomputing
    # straight from the dates and dollar values he prints there:
    #   initial validation 2015-08-21..2017-10-05 (776 days), 4,735,280 -> 6,663,284
    #       = +40.72% nominal, 17.43% annual
    #   final   validation 2018-02-14..2019-02-25 (376 days), 7,083,645 -> 8,506,987
    #       = +20.09% nominal, 19.45% annual
    # His narrative quotes exactly those ("40.7% ... 17.4%" and "around 20.1% ... 19.5%"),
    # but Table 3 prints 19.45 against Initial and 17.43 against Final -- the other way
    # round. Note also that this paper's nominal is a SIMPLE return, where the 2021 paper's
    # is a log return; the two papers do not share a convention.
    #
    # The rows below quote TABLE 3 AS PUBLISHED, deliberately, so that anyone checking the
    # source finds what we printed. The discrepancy belongs in a footnote, not in a silent
    # correction -- and his difference sentences (4.14/12.98 initial, 7.12/5.43 final) are
    # consistent with the table's labels, so we cannot cleanly reassign the B&H and TA rows
    # either. Flagged in VERIFY_THESE.md.
    #
    # His text also says the GA parameters were "(10-4)", contradicting his Table 4 (slow
    # 15, fast 6, signal 3); that looks like a garbled cross-reference to the 2021 study's
    # 10 and 7.
    "aguirre2020": [
        # Table 3, initial validation 2015-08-21..2017-10-05. His Table 2 lists 6 round
        # trips there and 5 in the final window, stated here as POSITIONS. He is long only,
        # so the long position counts repeat the total. Rates go in "Annual %" per C5; the
        # currency cells are blank because Table 3 does not report them.
        ("GA (initial val.)", "(6,15,3)", None, 6, None, None, 6, None, None, None,
         {"Annual %": 19.45}),
        ("Buy and Hold (initial val.)", "--", None, 1, None, None, 1, None, None, None,
         {"Annual %": 15.31}),
        ("Traditional MACD (initial val.)", "(12,26,9)", None, None, None, None, None, None, None, None,
         {"Annual %": 6.47}),
        # Table 3, final validation 2018-02-14..2019-02-25
        ("GA (final val.)", "(6,15,3)", None, 5, None, None, 5, None, None, None,
         {"Annual %": 17.43}),
        ("Buy and Hold (final val.)", "--", None, 1, None, None, 1, None, None, None,
         {"Annual %": 10.31}),
        ("Traditional MACD (final val.)", "(12,26,9)", None, None, None, None, None, None, None, None,
         {"Annual %": 12.00}),
    ],
}


# Columns whose values are counts and so print without decimals. Everything else in the
# overall table is money, points or a percentage and prints with two.
count_columns = {
    "Transactions",
    "Long Transactions",
    "Short Transactions",
    "Long 1 Transactions",
    "Long 2 Transactions",
}

# What a reader should see in the header where the dataframe's own column name would be
# unhelpful. "Long 1" is the long LEG of the long/short strategy and "Long 2" is the
# separate long-only strategy, which is not something a header can convey on its own.
# The eight metrics write_table_tex prints, in order. Its .format() calls are positional
# on exactly this sequence, so this list documents the contract rather than inventing one.
rules_table_metrics = [
    "Return",
    "Transactions",
    "Win Ratio",
    "Long Return",
    "Long Transactions",
    "Long Win Ratio",
    "Short Return",
    "Short Win Ratio",
]

tex_column_names = {
    # the unit has to be on the header: every paper reports signal counts, and whether a
    # signal is a position depends on the system (see the note on published_comparison_rows)
    "Transactions": "Positions",
    "Long Transactions": "Long Positions",
    "Short Transactions": "Short Positions",
    "Long 1 Return": "Long-Leg Return",
    "Long 1 Transactions": "Long-Leg Transactions",
    "Long 1 Win Ratio": "Long-Leg Win Ratio",
    "Long 2 Return": "Long-Only Return",
    "Long 2 Transactions": "Long-Only Transactions",
    "Long 2 Win Ratio": "Long-Only Win Ratio",
    "Per Round Trip": "Profit / Round Trip",
    "Annual %": "Effective Annual \\%",
}


# AUDIT ROW N. write_overall_table_tex had NO footnote carrier at all -- "tablenotes"
# appeared nowhere in this file -- so the four caveats the comments above themselves
# demand could not be printed, and every comparison table shipped without the
# qualifications that make the comparison legitimate. They are emitted automatically
# from the paper being compared against and the columns actually present, rather than
# left to each notebook to remember.
comparison_table_notes = {
    # always, for any comparison table: the baselines are not on one cost basis
    "costs": (
        "Published baselines are not on a common cost basis: Wiles \\& Enke (2015) state "
        "they assumed no transaction costs and Kang (2021) treats the JPY~330 round-trip "
        "commission as negligible, so both are GROSS, whereas Agudelo Aguirre et al. "
        "(2021) Table~3 is explicitly NET of transaction costs. Our own figures are "
        "reported gross, with the full cost sweep given separately."
    ),
    "wiles": (
        "Wiles \\& Enke accumulate contracts rather than holding one position at a time: "
        "three extra sell contracts were placed at the end of the contract period to "
        "settle, which is why 48 buys stand against 45 sells. Their counts are contract "
        "transactions and cannot be converted to positions cleanly, so they are quoted "
        "exactly as published; our own column counts POSITIONS throughout. Units are "
        "cents per bushel on a single SQ2014 contract."
    ),
    "kang": (
        "Kang is always in the market -- when a position is closed a reverse trade is "
        "automatically executed -- so one signal opens exactly one position and his "
        "totals are directly comparable to ours with no adjustment. Units are Nikkei~225 "
        "index points at one unit per trade."
    ),
    # DECISION C5: ONE NOTE PER STUDY. These two used to share a single "aguirre" note
    # reached by paper.startswith("aguirre"), which is how a caveat about the 2021 paper's
    # log-as-simple annualisation came to be printed underneath the 2020 table, whose
    # annualisation is an ordinary CAGR. The note contradicted the column it sat under.
    "aguirre2021": (
        "Agudelo Aguirre et al. (2021) annualise a natural-log return as though it were "
        "simple, $(1+r_{nominal})^{365/d}-1$, which understates every strategy including "
        "their own benchmark by roughly 1.8 percentage points on buy-and-hold. Our "
        "effective annual rates in this table are computed on THEIR convention so the "
        "comparison is like-for-like rather than flattering to us. Their return column is "
        "the profit implied by their Table~3 on the same initial investment of "
        "USD~5,036,790 our own run uses. They are long only and hold cash between trades, "
        "so their long cells repeat their total and the short columns are empty by "
        "nature; the arm of ours to read them against is the long-only one."
    ),
    "aguirre2020": (
        "Agudelo Aguirre et al. (2020) is a SEPARATE study from their 2021 paper -- "
        "different sample period, different parameters, different results -- and it does "
        "not share the 2021 annualisation: its nominal is a simple return and its annual "
        "figure an ordinary CAGR, so our effective annual rates here are computed that "
        "way too. Their Table~3 reports rates only, with no currency profit for the "
        "benchmark or traditional-MACD rows, so those cells are left empty rather than "
        "converted from a different table. Their Table~3 also appears to transpose its "
        "two validation windows relative to their own Table~2 and narrative; the rows are "
        "quoted exactly as published."
    ),
    # a table can carry the annual-rate column without printing any published block above
    # it -- every nasdaq run outside the (12,26,9) base case does exactly that
    "annual": (
        "The effective annual rate is computed on the convention of the study the table is "
        "compared against. The two Agudelo Aguirre papers do not share one, so the figure "
        "is not comparable across their two studies."
    ),
}


def aguirre_annual_rate(
    net: float, balance: float, days: int, convention: str = "log_as_simple"
) -> float:
    """
    aguirre_annual_rate converts a profit into an effective annual rate on the convention of the Agudelo Aguirre study being compared against, so our nasdaq figures sit beside that paper's on its own basis rather than on a better one

    DECISION C5, second half: THE TWO PAPERS DO NOT SHARE A CONVENTION, and this function
    had only the 2021 one while nasdaq.ipynb called it for both windows. That is not a
    rounding difference -- on the 2020 paper's own windows the 2021 formula understates us
    by 2.61pp and 1.72pp.

    "log_as_simple" -- the 2021 JEFAS convention. Its Table 3 reports a "nominal" that is a
    NATURAL LOG return and then annualises it as though it were simple,
    (1 + ln(growth))^(365/days) - 1. Both halves verified against his own value columns:
    ln(7,658,339 / 5,036,790) = 0.4190 and (1.4190)^(365/808) - 1 = 0.1713, exactly the
    41.90 and 17.13 he prints. Annualising properly, exp(ln(growth) x 365/days) - 1, would
    give 20.84 for that row and 13.77 rather than 12.01 for his buy-and-hold, so using the
    correct formula on our side while quoting his as published would flatter us by about
    3.7pp and understate his benchmark by 1.8pp.

    "cagr" -- the 2020 IMFI convention, which is just the standard one: growth^(365/days)
    - 1 on a SIMPLE nominal. Recomputed from the dates and values in his own Table 2,
    4,735,280 -> 6,663,284 over 776 days gives 40.72% nominal and 17.43% annual, and
    7,083,645 -> 8,506,987 over 376 days gives 20.09% and 19.45%, all four reproducing his
    narrative exactly.

    Comparability wins under either; the convention belongs in the table notes, and
    comparison_table_notes now carries one note per study for exactly that reason.

    Arguments:
        net -- our profit in currency over the window
        balance -- the starting investment, which is 1000 units at the window's first price
        days -- calendar days the investment ran, matching his 808 for the 2021 study
        convention -- "log_as_simple" for Agudelo Aguirre et al. (2021) or "cagr" for (2020) (default: {"log_as_simple"})

    Returns:
        the effective annual rate in percent, or nan where the loss is too large for the formula to be defined
    """
    import math

    # a typo must not silently pick a convention -- choosing one silently is the defect
    # this parameter exists to fix, and every caller is ours, so raising cannot break a
    # table that was otherwise going to be valid
    if convention not in ("log_as_simple", "cagr"):
        raise ValueError(
            f"unknown annualisation convention {convention!r}: use 'log_as_simple' for "
            "Agudelo Aguirre et al. (2021) or 'cagr' for Agudelo Aguirre et al. (2020)"
        )
    if balance <= 0 or days <= 0:
        return float("nan")
    growth = 1 + net / balance
    # a total loss or worse has no return to annualise under either convention
    if growth <= 0:
        return float("nan")
    if convention == "cagr":
        return 100 * (growth ** (365 / days) - 1)
    nominal = math.log(growth)
    # he raises (1 + nominal) to a fractional power, which needs a positive base
    if 1 + nominal <= 0:
        return float("nan")
    return 100 * ((1 + nominal) ** (365 / days) - 1)


def format_comparison_row(row: tuple, metrics: list) -> str:
    """
    format_comparison_row turns one published result from published_comparison_rows into a latex row for the comparison table, leaving a cell blank wherever the paper did not report that column

    Arguments:
        row -- (label, parameters, return, transactions, win ratio, long return, long transactions, long win ratio, short return, short win ratio), with None for anything the paper did not report, optionally followed by a dict of any FURTHER columns that paper does report, keyed by the column name exactly as it appears in the table
        metrics -- the table's metric column names, in print order

    Returns:
        the latex line, without the leading indentation
    """
    label, parameters = row[0], row[1]
    # str() keeps a count bare ("93") and a float exactly as it was written ("664.5",
    # "49.0"), where ":g" would silently drop the trailing zero and print "49" in a column
    # where every other win ratio carries one decimal
    cells = ["" if value is None else str(value) for value in row[2:10]]
    # AUDIT ROW N. The columns past the first eight used to be padded blank, and the
    # docstring justified that with "no paper reports them". That was FALSE: Wiles
    # reports a profit per round trip for all four of his strategies (his own cents
    # figures divided by his round trips -- 3906/48, 4507/50, 664.5/5, 728/5 = 81.4,
    # 90.1, 132.9, 145.6, all four reproducing his printed values exactly) and Aguirre
    # reports an effective annual rate. Both sat in COMMENTS while the columns that
    # should carry them were printed empty, which reads to a referee as "the competitor
    # did not report this" when in fact he did. Anything past the eighth column is now
    # filled BY NAME from the row's trailing dict, so a paper's own figure lands in the
    # column that means the same thing and nowhere else.
    extras = row[10] if len(row) > 10 and isinstance(row[10], dict) else {}
    # A narrower table legitimately omits these columns -- the soybean comparison can be
    # built with the published eight alone -- so this WARNS rather than raising: raising
    # would break a valid table, and staying silent is how a published figure disappears
    # without anyone noticing, which is the defect this row exists to fix.
    unknown = set(extras) - set(metrics)
    if unknown:
        print(
            f"[tables] NOTE {label}: this table has no "
            f"{sorted(unknown)} column, so that published figure is not printed. "
            "Add the column to table_data if the comparison needs it."
        )
    for metric in metrics[len(cells):]:
        value = extras.get(metric)
        cells.append("" if value is None else str(value))
    # two spacer columns sit between the parameters and the numbers, matching the leading
    # "llll" of the column spec and the blank "&      &  &" in the rows this replaces
    return (
        f"{label:33}& {parameters:24}&      &  & "
        + " & ".join(f"{cell:16}" for cell in cells)
        + " \\\\"
    )


def format_result_row(first: str, second: str, third: str, values: list, metrics: list) -> str:
    """
    format_result_row renders one of our own result rows, printing counts without decimals and everything else with two

    Arguments:
        first -- the crossover block label, blank on continuation rows
        second -- the system label, blank on continuation rows
        third -- "Best" or "MACD"
        values -- the row's metric values, in the same order as metrics
        metrics -- the table's metric column names

    Returns:
        the latex line, without the leading indentation
    """
    cells = [
        f"{value:.0f}" if name in count_columns else f"{value:.2f}"
        for name, value in zip(metrics, values)
    ]
    return (
        f"{first:26}& {second:10}& {third:6}&  & "
        + " & ".join(f"{cell:16}" for cell in cells)
        + " \\\\"
    )


def write_overall_table_tex(table_data, table_prefix, table_name, label, caption, paper=None):
    """
    write_overall_table_tex writes the comparison table for one case study: the published rows for that paper, then our own eight result rows

    The column set is whatever `table_data` carries, so a notebook controls the table by
    choosing which columns to pass. That is how the long-only arm, soybean's profit per
    round trip and nasdaq's effective annual rate reach the table without this function
    knowing about any of them. The first eight columns must stay in the published order
    (return, transactions, win ratio, long return, long transactions, long win ratio,
    short return, short win ratio) because the published rows line up against them.

    Arguments:
        table_data -- our eight result rows, columns in the order they should be printed
        table_prefix -- folder to write into
        table_name -- file stem, written as <table_name>_tex.tex
        label -- latex label suffix
        caption -- table caption

    Keyword Arguments:
        paper -- which block of published_comparison_rows to print above our rows, or None for no comparison rows (default: {None})

    Returns:
        None -- writes the .tex file
    """
    metrics = list(table_data.columns)
    # four label columns (block, system, entry, spacer) then one centred column per metric
    column_spec = "llll" + "c" * len(metrics)
    header = " & ".join(f"{tex_column_names.get(m, m):16}" for m in metrics)
    blank_row = " & ".join(f"{'':16}" for _ in metrics)
    # the table gets wider as columns are added, so scale the box with the column count
    width = 1.25 + 0.11 * max(0, len(metrics) - 8)

    start_text = f"""\\begin{"{table}"}[!ht]
	\\centering
	\\caption{"{" + caption + "}"}
	\\begin{"{adjustbox}"}{"{width=" + f"{width:.2f}" + "\\textwidth,center=\\textwidth}"}
		\\begin{"{tabular}"}{"{" + column_spec + "}"}
            \\toprule
            &          &      &  & {header}  \\\\
            \\midrule
            &          &      &  & {blank_row}  \\\\"""

    # AUDIT ROW N: the footnote carrier this function never had. The notes are chosen
    # from the paper being compared against and from the columns actually present, so a
    # comparison table cannot ship without the caveats that make it legitimate -- which
    # is what happened to every one of these tables until now.
    note_keys = ["costs"]
    # DECISION C5: keyed EXACTLY, not by prefix. paper.startswith("aguirre") matched both
    # Agudelo Aguirre studies onto one note, so the 2021 paper's log-as-simple caveat was
    # printed under the 2020 table whose annualisation is an ordinary CAGR -- a note
    # contradicting the column above it.
    if paper:
        for key in ("wiles", "kang", "aguirre2021", "aguirre2020"):
            if paper == key:
                note_keys.append(key)
    # the annualisation caveat belongs on any table carrying an annual-rate column,
    # whether or not a published block is printed above it
    if "Annual %" in metrics and not any(key.startswith("aguirre") for key in note_keys):
        note_keys.append("annual")
    # a note that is not defined must not break an otherwise valid table -- the same
    # reasoning as the unknown-column note in format_comparison_row
    notes = []
    for key in note_keys:
        if key in comparison_table_notes:
            notes.append(comparison_table_notes[key])
        else:
            print(
                f"[tables] NOTE {table_name}: no table note is defined for '{key}', so "
                "that caveat is not printed. Add it to comparison_table_notes."
            )
    # OUTSIDE the adjustbox: inside it the note text is scaled with the table, so it
    # would print at a different size in every table. The minipage holds the notes at
    # \textwidth even where the scaled table is wider than the text block.
    notes_block = (
        "\n\t\\begin{minipage}{\\textwidth}\n"
        "\t\t\\footnotesize\n"
        "\t\t\\textit{Notes:} " + " ".join(notes) + "\n"
        "\t\\end{minipage}"
    )

    end_text = f"""            \\bottomrule
		\\end{"{tabular}"}
	\\end{"{adjustbox}"}{notes_block}
	\\label{"{sec.results.compare.table." + label + "}"}
\\end{"{table}"}"""

    # the published rows come from published_comparison_rows keyed by paper, so a case
    # study can only ever print its own paper's figures
    comparison_lines = (
        [format_comparison_row(row, metrics) for row in published_comparison_rows[paper]]
        if paper
        else []
    )

    spacer = f"{'':26}& {'':10}& {'':6}&  & " + blank_row + " \\\\"

    # rows 0-3 are the MACD crossover block, rows 4-7 the zero crossover block, each being
    # System 1 then System 2 and within those the rule-selected config then base MACD
    layout = [
        ("MACD Crossover", "System 1", "Best", 0),
        ("", "", "MACD", 1),
        ("", "System 2", "Best", 2),
        ("", "", "MACD", 3),
        ("Zero Crossover", "System 1", "Best", 4),
        ("", "", "MACD", 5),
        ("", "System 2", "Best", 6),
        ("", "", "MACD", 7),
    ]
    result_lines = [
        format_result_row(first, second, third, list(table_data.iloc[position]), metrics)
        for first, second, third, position in layout
    ]
    # blank line between the two crossover blocks
    result_lines = result_lines[:4] + [spacer] + result_lines[4:]

    llist = (comparison_lines + [spacer] + result_lines) if comparison_lines else result_lines

    with open(f"{table_prefix}/{table_name}_tex.tex", "w") as text_file:
        text_file.write(start_text)
        text_file.write("\n")
        for ll in llist:
            text_file.write("\t\t\t")
            text_file.write(ll)
            text_file.write("\n")
        text_file.write(end_text)


###########
#       Experiment        #
###########################


def make_new_time(given_date: datetime, n: int, mode: str = "month") -> datetime:
    """
    make_new_time takes time and add to or substract from based on n which is how much and mode which is the unit for example n = 2 mode = month adds 2 months to current time and returns it

    Arguments:
        given_date -- current date
        n -- how much

    Keyword Arguments:
        mode -- month/week/year/day (default: {"month"})

    Returns:
        new date
    """
    given_date_str = str(given_date).split(" ")[0]
    date_format = "%Y-%m-%d"
    dtObj = datetime.strptime(given_date_str, date_format)

    if mode == "month":
        new_date = dtObj + pd.DateOffset(months=n)
    elif mode == "day":
        new_date = dtObj + pd.DateOffset(days=n)
    else:
        new_date = dtObj + pd.DateOffset(weeks=n)

    return new_date


def write_json_to_file(output_path: str, file_name: str, f: dict):
    """
    write_json_to_file write the variable to json file

    Arguments:
        output_path -- path to file
        file_name -- name of file
        f -- variable
    """
    # exist_ok: the sharded run has up to 32 processes writing into the same directory and
    # a bare `if not exists: makedirs` races -- measured, 31 of 32 shards lost and died at
    # their FIRST write, hours in. run_sweep.py also pre-creates every directory up front;
    # this keeps the function correct when a single shard is run by hand instead.
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/{file_name}", "w") as filehandle:
        dump(f, filehandle)


def load_json_file(file_name: str) -> dict:
    """
    load_json_file reads file

    Arguments:
        file_name -- complete file name

    Returns:
        returns file as variable
    """
    with open(file_name, "r") as filehandle:
        f = load(filehandle)
    return f


def return_macd_lists(alpha_range: list, gamma_range: list) -> list[list, list, list]:
    """
    return_macd_lists return labels of macd, histogram and signal line lists

    Arguments:
        alpha_range -- alpha range
        gamma_range -- gamma range

    Returns:
        histogram_list, macd_list, signal_line_list
    """

    gen = [(x, y) for x in alpha_range for y in gamma_range]
    # histogram_list = ["Histogram"]
    # macd_list = ["MACD"]
    # signal_line_list = ['SignalLine']
    histogram_list = []
    macd_list = []
    signal_line_list = []

    for alpha in alpha_range:
        #     if f"Histogram a: {alpha}" == "Histogram a: 0":
        #         histogram_list.append(f"Histogram")
        #         macd_list.append(f"MACD")
        histogram_list.append(f"Histogram a: {alpha}")
        macd_list.append(f"MACD a: {alpha}")
        signal_line_list.append(f"MACD Signal a: {alpha}")
    for alpha, gamma in gen:
        histogram_list.append(f"Histogram a: {alpha} g: {gamma}")
        macd_list.append(f"MACD a: {alpha} g: {gamma}")
        signal_line_list.append(f"MACD Signal a: {alpha} g: {gamma}")

    return histogram_list, macd_list, signal_line_list


def make_output_list(input_row: list) -> tuple[list, str]:
    """
    make_output_list changed the type of the list elements so that it can be written to file with no problem

    Arguments:
        input_row -- the list

    Returns:
        the list and its last element which is the macd/histogram
    """
    for i in range(len(input_row)):
        if type(input_row[i]) == np.int64:
            input_row[i] = int(input_row[i])
        elif type(input_row[i]) == np.float64:
            input_row[i] = float(input_row[i])
    return input_row, input_row[-1]


def get_name_mode(strategy: str) -> str:
    """
    get_name_mode determines base name from strategy

    Arguments:
        strategy -- MACDCROSS or ZEROCROSS

    Returns:
        Histogram or MACD as the base
    """
    if strategy == "MACDCROSS":
        return "Histogram"
    elif strategy == "ZEROCROSS":
        return "MACD"


def get_histogram_gamma_list(chosen_histogram: str, gamma_range: list) -> list:
    """
    get_histogram_gamma_list returns a list of cmacd based on a smacd

    Arguments:
        chosen_histogram -- base smacd to make cmacd on
        gamma_range -- gamma values

    Returns:
        a list of cmacds based on a (s)macd
    """
    histogram_gamma_list = [chosen_histogram]
    temp_list = [f"{chosen_histogram} g: {g}" for g in gamma_range]
    histogram_gamma_list.extend(temp_list)
    return histogram_gamma_list


def compare_histogram_checks(
    result_df: pd.DataFrame,
    net_name: str,
    histogram_check_list: list,
    chosen_histogram: str,
    best_net: float,
) -> tuple[str, float]:
    """
    compare_histogram_checks compares net gain in a list of histograms/macds and returns the best one

    Arguments:
        result_df -- dataframe that containts results and benchmarks for a set of histograms/macds
        net_name -- the column to check for profit for eg: "net"
        histogram_gamma_list -- list of histograms to check
        chosen_histogram -- starting histogram so that if there is no histogram better nothing changes
        best_net -- net gain of chosen_histogram

    Returns:
        chosen_histogram, best_net
    """
    # print(histogram_check_list)
    for histogram_check in histogram_check_list:
        mean_check = result_df[result_df["his"] == histogram_check][net_name].values[0]
        if best_net < mean_check:
            best_net = mean_check
            chosen_histogram = histogram_check
    return chosen_histogram, best_net


# @timeit  # timing run only (see Timing section)
def get_rule_1(
    result_df: pd.DataFrame,
    name_mode: str,
    net_name: str,
    gamma_range: list,
    list_smacd_lower: list,
    list_smacd_upper: list,
    concat_list: list,
    chosen_histogram: str,
    best_net: float,
) -> int:
    """
    get_rule_1 returns histogram chosen by rule 1

    Arguments:
        result_df -- dataframe that containts results and benchmarks for a set of histograms/macds
        name_mode -- base macd/histogram text
        net_name -- the column to check for profit for eg: "net"
        gamma_range -- gamma values
        list_smacd_lower -- list of base macds lower than macd a: 0
        list_smacd_upper -- list of base macds higher than macd a: 0
        concat_list -- concatenated list of the previous two
        chosen_histogram -- starting histogram so that if there is no histogram better nothing changes
        best_net -- net gain of chosen_histogram

    Returns:
        chosen_histogram
    """
    # find closest best SMACD
    for histogram_pair in concat_list:
        net_1 = result_df[result_df["his"] == histogram_pair[0]][net_name].values[0]
        net_2 = result_df[result_df["his"] == histogram_pair[1]][net_name].values[0]
        if net_1 > net_2:
            if net_1 > (best_net + abs(best_net * 10 / 100)):
                best_net = net_1
                chosen_histogram = histogram_pair[0]
                break
        elif net_1 < net_2:
            if net_2 > (best_net + abs(best_net * 10 / 100)):  # mean_1 < mean_2
                best_net = net_2
                chosen_histogram = histogram_pair[1]
                break
        else:  # mean_1 = mean_2
            if net_1 > (best_net + abs(best_net * 10 / 100)):
                lower_mean = result_df.loc[result_df["his"].isin(list_smacd_lower)][
                    net_name
                ].mean()
                upper_mean = result_df.loc[result_df["his"].isin(list_smacd_upper)][
                    net_name
                ].mean()
                if lower_mean > upper_mean:
                    best_net = net_1
                    chosen_histogram = histogram_pair[0]
                    break
                else:  # lower_mean <= upper_mean
                    best_net = net_2
                    chosen_histogram = histogram_pair[1]
                    break
    # print("best SMACD", chosen_histogram, best_net)
    # get the gamma configurations of the SMACD
    histogram_gamma_list = get_histogram_gamma_list(chosen_histogram, gamma_range)
    # print(histogram_gamma_list)
    # find closest best cmacd based on that
    chosen_histogram, best_net = compare_histogram_checks(
        result_df, net_name, histogram_gamma_list, chosen_histogram, best_net
    )
    # print("returned cmacd", chosen_histogram, best_net, result_df[result_df["his"] == chosen_histogram].index[0])
    return result_df[result_df["his"] == chosen_histogram].index[0]


# @timeit  # timing run only (see Timing section)
def get_rule_2(
    result_df: pd.DataFrame,
    name_mode: str,
    net_name: str,
    gamma_range: list,
    list_smacd_lower: list,
    list_smacd_upper: list,
    concat_list: list,
    chosen_histogram: str,
    best_net: float,
) -> int:
    """
    get_rule_2 returns histogram chosen by rule 1

    Arguments:
        result_df -- dataframe that containts results and benchmarks for a set of histograms/macds
        name_mode -- base macd/histogram text
        net_name -- the column to check for profit for eg: "net"
        gamma_range -- gamma values
        list_smacd_lower -- list of base macds lower than macd a: 0
        list_smacd_upper -- list of base macds higher than macd a: 0
        concat_list -- concatenated list of the previous two
        chosen_histogram -- starting histogram so that if there is no histogram better nothing changes
        best_net -- net gain of chosen_histogram

    Returns:
        chosen_histogram
    """

    base_list = []
    base_list.extend(list_smacd_lower)
    base_list.extend(list_smacd_upper)
    # find closest best SMACD
    # print(chosen_histogram, best_net)
    chosen_histogram, best_net = compare_histogram_checks(
        result_df, net_name, base_list, chosen_histogram, best_net
    )
    # print(chosen_histogram, best_net)
    # get the gamma configurations of the SMACD
    histogram_gamma_list = get_histogram_gamma_list(chosen_histogram, gamma_range)

    # find closest best cmacd based on that
    chosen_histogram, best_net = compare_histogram_checks(
        result_df, net_name, histogram_gamma_list, chosen_histogram, best_net
    )
    # print(chosen_histogram, best_net)
    return result_df[result_df["his"] == chosen_histogram].index[0]


# @timeit  # timing run only (see Timing section)
def get_rule_3(
    result_df: pd.DataFrame,
    name_mode: str,
    net_name: str,
    gamma_range: list,
    list_smacd_lower: list,
    list_smacd_upper: list,
    concat_list: list,
    chosen_histogram: str,
    best_net: float,
) -> tuple[int, list]:
    """
    get_rule_3 returns histogram chosen by rule 1

    Arguments:
        result_df -- dataframe that containts results and benchmarks for a set of histograms/macds
        name_mode -- base macd/histogram text
        net_name -- the column to check for profit for eg: "net"
        gamma_range -- gamma values
        list_smacd_lower -- list of base macds lower than macd a: 0
        list_smacd_upper -- list of base macds higher than macd a: 0
        concat_list -- concatenated list of the previous two
        chosen_histogram -- starting histogram so that if there is no histogram better nothing changes
        best_net -- net gain of chosen_histogram

    Returns:
        chosen_histogram
    """
    # get a list of best performing rows (if there is a list)
    list_of_best = result_df[result_df[net_name] == result_df[net_name].max()].values
    check_index = result_df[net_name].idxmax()

    # if only one element return it
    if len(list_of_best) == 1:
        return check_index, list_of_best
    else:  # if more than one element, return the one with the lowest index, closest to the Histogram a: 0 as in the default configuration of MACD
        # get two means to make a better list
        lower_mean = result_df.loc[result_df["his"].isin(list_smacd_lower)][
            net_name
        ].mean()
        upper_mean = result_df.loc[result_df["his"].isin(list_smacd_upper)][
            net_name
        ].mean()

        # make a list made of SMACDs and cmacds
        rule_3_list = [chosen_histogram]
        # rule_3_list = []
        rule_3_list_cmacd = [f"{chosen_histogram} g: {g}" for g in gamma_range]
        if lower_mean > upper_mean:
            for histogram_pair in concat_list:
                rule_3_list.append(histogram_pair[0])
                rule_3_list.append(histogram_pair[1])

                histogram_pair_0 = histogram_pair[0]
                histogram_pair_1 = histogram_pair[1]
                for g in gamma_range:
                    rule_3_list_cmacd.append(f"{histogram_pair_0} g: {g}")
                for g in gamma_range:
                    rule_3_list_cmacd.append(f"{histogram_pair_1} g: {g}")
        else:
            for histogram_pair in concat_list:
                rule_3_list.append(histogram_pair[1])
                rule_3_list.append(histogram_pair[0])

                histogram_pair_0 = histogram_pair[0]
                histogram_pair_1 = histogram_pair[1]

                for g in gamma_range:
                    rule_3_list_cmacd.append(f"{histogram_pair_1} g: {g}")
                for g in gamma_range:
                    rule_3_list_cmacd.append(f"{histogram_pair_0} g: {g}")
        rule_3_list.extend(rule_3_list_cmacd)

        # set lowest index to a high number
        lowest_index = len(rule_3_list) + 1000
        # go thourgh the list and find the lowest index
        for row_data in list_of_best:
            # if row_data[-1] == "MACD" or row_data[-1] == "MACD a: 0":
            #     print("cols", list(result_df['his']))
            #     print(row_data)
            #     print("list", rule_3_list)
            index_of_row_data = rule_3_list.index(row_data[-1])
            if index_of_row_data < lowest_index:
                lowest_index = index_of_row_data
                chosen_histogram = row_data[-1]

        return result_df[result_df["his"] == chosen_histogram].index[0], list_of_best


def get_his_from_solution(alpha: float, gamma: float, base_text: str) -> str:
    """
    get_his_from_solution helper function for get_genetic which gives histogram/macd name from alpha and gamma values

    Arguments:
        alpha -- alpha value
        gamma -- gamma calue
        base_text -- macd or histogram

    Returns:
        histogram/macd name based on alpha and gamma values
    """
    if gamma == 0:  # gamma=0 --> MACD and SMACD
        # SMACD
        his_name = f"{base_text} a: {alpha}"

        if alpha == 0:  # alpha=0 --> MACD
            his_name = f"{base_text} a: 0"
        # print(his_name)
    else:  # neither alpha or gamma are zero --> cmacd
        if alpha == 0:
            his_name = f"{base_text} a: 0 g: {gamma}"
        else:
            his_name = f"{base_text} a: {alpha} g: {gamma}"
        # print(his_name)
    return his_name


# @timeit  # timing run only (see Timing section)
def get_genetic(
    result_df: pd.DataFrame,
    name_mode: str,
    net_name: str,
    alpha_range: list,
    gamma_range: list,
    list_smacd_lower: list,
    list_smacd_upper: list,
    concat_list: list,
    chosen_histogram: str,
    best_net: float,
) -> int:
    """
    get_genetic returns histogram chosen by rule 1

    Arguments:
        result_df -- dataframe that containts results and benchmarks for a set of histograms/macds
        name_mode -- base macd/histogram text
        net_name -- the column to check for profit for eg: "net"
        alpha_range -- alpha values
        gamma_range -- gamma values
        list_smacd_lower -- list of base macds lower than macd a: 0
        list_smacd_upper -- list of base macds higher than macd a: 0
        concat_list -- concatenated list of the previous two
        chosen_histogram -- starting histogram so that if there is no histogram better nothing changes
        best_net -- net gain of chosen_histogram

    Returns:
        chosen_histogram
    """

    def fitness_func(ga_instance, solution, solution_idx):
        his_name = get_his_from_solution(solution[0], solution[1], name_mode)
        fitness = result_df[result_df["his"] == his_name][net_name].iloc[0]
        return fitness

    # 115 > 475 configs
    num_generations = 25
    num_parents_mating = 2
    sol_per_pop = 15
    num_genes = 2
    gene_type = [float, int]
    parent_selection_type = "tournament"
    K_tournament = 3
    crossover_type = "single_point"
    crossover_probability = 0.7
    mutation_type = "random"
    mutation_probability = 0.1

    gamma_range_0 = [0]
    gamma_range_0.extend(gamma_range)
    gene_space = [alpha_range, gamma_range_0]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_type=gene_type,
        parent_selection_type=parent_selection_type,
        K_tournament=K_tournament,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        gene_space=gene_space,
        random_seed=42,  # deterministic GA (Rule 4 / "GA") -- reproducible across runs
        # save_best_solutions=True,
        # save_solutions=True
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    his_name = get_his_from_solution(solution[0], solution[1], name_mode)
    return result_df[result_df["his"] == his_name].index[0]


# @timeit  # timing run only (see Timing section)
def find_best_index(
    result_df: pd.DataFrame,
    rule_number: int,
    net_name: str,
    name_mode: str,
    alpha_range: list,
    gamma_range: list,
) -> int | tuple[int, list]:
    """
    find_best_index general function that calls different rule functions

    Arguments:
        result_df -- dataframe that containts results and benchmarks for a set of histograms/macds
        rule_number -- which rule to be called
        net_name -- the column to check for profit for eg: "net"
        name_mode -- base macd/histogram text
        alpha_range -- alpha values
        gamma_range -- gamma values

    Returns:
        result of different rule functions
    """
    chosen_histogram = f"{name_mode} a: 0"
    baseIndex = result_df[result_df["his"] == chosen_histogram].index[0]
    best_net = result_df.iloc[baseIndex][net_name]
    # print(baseIndex, check_index)

    if name_mode == "MACD":
        # concat_list = ["MACD a: 0"]
        list_smacd_lower = [
            "MACD a: -0.1",
            "MACD a: -0.2",
            "MACD a: -0.3",
            "MACD a: -0.4",
            "MACD a: -0.5",
            "MACD a: -0.6",
            "MACD a: -0.7",
            "MACD a: -0.8",
            "MACD a: -0.9",
        ]
        list_smacd_upper = [
            "MACD a: 0.1",
            "MACD a: 0.2",
            "MACD a: 0.3",
            "MACD a: 0.4",
            "MACD a: 0.5",
            "MACD a: 0.6",
            "MACD a: 0.7",
            "MACD a: 0.8",
            "MACD a: 0.9",
        ]
    else:  # Histogram
        # concat_list = ["Histogram a: 0"]
        list_smacd_lower = [
            "Histogram a: -0.1",
            "Histogram a: -0.2",
            "Histogram a: -0.3",
            "Histogram a: -0.4",
            "Histogram a: -0.5",
            "Histogram a: -0.6",
            "Histogram a: -0.7",
            "Histogram a: -0.8",
            "Histogram a: -0.9",
        ]
        list_smacd_upper = [
            "Histogram a: 0.1",
            "Histogram a: 0.2",
            "Histogram a: 0.3",
            "Histogram a: 0.4",
            "Histogram a: 0.5",
            "Histogram a: 0.6",
            "Histogram a: 0.7",
            "Histogram a: 0.8",
            "Histogram a: 0.9",
        ]
    # concat_list.extend([histogram_pair for histogram_pair in zip(list_smacd_lower, list_smacd_upper)])
    concat_list = [
        histogram_pair for histogram_pair in zip(list_smacd_lower, list_smacd_upper)
    ]
    # print(concat_list)
    if rule_number == 1:  # rule 1
        return get_rule_1(
            result_df,
            name_mode,
            net_name,
            gamma_range,
            list_smacd_lower,
            list_smacd_upper,
            concat_list,
            chosen_histogram,
            best_net,
        )

    elif rule_number == 2:  # rule 2
        return get_rule_2(
            result_df,
            name_mode,
            net_name,
            gamma_range,
            list_smacd_lower,
            list_smacd_upper,
            concat_list,
            chosen_histogram,
            best_net,
        )

    elif rule_number == 3:  # rule 3
        return get_rule_3(
            result_df,
            name_mode,
            net_name,
            gamma_range,
            list_smacd_lower,
            list_smacd_upper,
            concat_list,
            chosen_histogram,
            best_net,
        )

    else:  # Genetic algorithm
        return get_genetic(
            result_df,
            name_mode,
            net_name,
            alpha_range,
            gamma_range,
            list_smacd_lower,
            list_smacd_upper,
            concat_list,
            chosen_histogram,
            best_net,
        )


# remove later
def get_standard_time(date_str: str) -> str:
    """
    get_standard_time changes the format and pads month and days

    Arguments:
        date_str -- date string

    Returns:
        padded string
    """
    year, month, day = date_str.split("-")
    if len(month) != 2:
        month = "0" + month
    if len(day) != 2:
        day = "0" + day
    return f"{year}-{month}-{day}"


def get_time_out(date_string: str) -> str:
    """
    get_time_out applies get_standard_time on two dates and joins them

    Arguments:
        date_string -- date string

    Returns:
        joined dates
    """
    date1, date2 = date_string.split(":")
    return get_standard_time(date1) + ":" + get_standard_time(date2)


def naked_crossover(
    target_df: pd.DataFrame,
    histogram_list: list,
    signal_line_list: list,
    trade_cost: bool,
    balance: float,
    risk_per_trade: float,
    signal_line_if: bool = False,
    signal_line_if_target: bool = False,
    last_close: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    naked_crossover runs crossover on dataframe and returns the results

    Arguments:
        target_df -- dataframe containing histograms and macds
        histogram_list -- list of histograms/macds to check
        signal_line_list -- list signal lines
        trade_cost -- False or True
        balance -- starting ballance or zero for one position scenario
        risk_per_trade -- how much capital to risk on each trade

    Keyword Arguments:
        signal_line_if -- system 2 (default: {False})
        signal_line_if_target -- system 3 (default: {False})
        last_close -- whether to close or roll back the last bet (default: {True})

    Returns:
        returns data frame and bets for every histogram
    """

    l = []
    all_his_bets_dict = dict()
    for his, sig in zip(histogram_list, signal_line_list):
        # print(his, sig, "in macd_crossover")
        (
            net,
            netbet,
            netpercent,
            bets,
            wins,
            longs,
            long_wins,
            long_profit,
            shorts,
            short_wins,
            short_profit,
            bets_list,
        ) = macd_crossover(
            target_df,
            "close",
            his,
            sig,
            balance=balance,
            risk_per_trade=risk_per_trade,
            trade_cost=trade_cost,
            signal_line_if=signal_line_if,
            signal_line_if_target=signal_line_if_target,
            last_close=last_close,
        )
        # print(his, sig, "in only_long_macd_crossover")
        # that same histogram but this time, only long positions are taken
        (
            long_only_net,
            long_only_netbet,
            long_only_netpercent,
            long_only_bets,
            long_only_wins,
            long_only_longs,
            long_only_long_wins,
            long_only_long_profit,
            long_only_shorts,
            long_only_short_wins,
            long_only_short_profit,
            long_only_bets_list,
        ) = only_long_macd_crossover(
            target_df,
            "close",
            his,
            sig,
            balance=balance,
            risk_per_trade=risk_per_trade,
            trade_cost=trade_cost,
            signal_line_if=signal_line_if,
            signal_line_if_target=signal_line_if_target,
            last_close=last_close,
        )

        l.append(
            [
                net,
                netbet,
                netpercent,
                bets,
                wins,
                longs,
                long_wins,
                long_profit,
                shorts,
                short_wins,
                short_profit,
                long_only_net,
                long_only_bets,
                long_only_wins,
                his,
            ]
        )

        all_his_bets_dict[f"{his}"] = bets_list
        all_his_bets_dict[f"{his} long"] = long_only_bets_list

    dff = pd.DataFrame(
        data=l,
        columns=[
            "net",
            "netbet",
            "netpercent",
            "bets",
            "wins",
            "longs",
            "long_wins",
            "long_profit",
            "shorts",
            "short_wins",
            "short_profit",
            "long_only_net",
            "long_only_bets",
            "long_only_wins",
            "his",
        ],
    )
    # dff.drop([0], inplace=True)
    dff.reset_index(inplace=True)
    dff.drop(columns=["index"], inplace=True)
    # print(dff)
    return dff, all_his_bets_dict


def make_obs_dict(target_keys: dict) -> dict:
    """
    make_obs_dict helper function to make results dictionary

    Arguments:
        targetKeys -- starting keys

    Returns:
        results dictionary
    """
    return {key: dict() for key in target_keys}


def add_to_inner_dict(target_dict: dict, label: str, tick_names: list, obs: list):
    """
    add_to_inner_dict helper function to congregate results

    Arguments:
        target_dict -- results dictionary
        label -- timespan label
        tick_names -- a list: [ticker, total]
        obs -- current observation
    """
    for tickName in tick_names:
        target_dict[tickName][label] = target_dict[tickName].get(label, list())
        target_dict[tickName][label].append(obs)


def check_list_maker(
    dff: pd.DataFrame, check_index: int, check_index_long: int
) -> list:
    """
    check_list_maker returns a list from two indexes

    Arguments:
        dff -- results dataframe
        check_index -- best index
        check_index_long -- best long index

    Returns:
        concatenated list of best index and best long index
    """
    check_index_list = list(dff.iloc[check_index])
    current_list_his_mode = check_index_list.pop()
    check_index_list.extend(list(dff.iloc[check_index_long][11:14]))
    check_index_list.append(current_list_his_mode)
    return check_index_list


# remove later
def what_was_chosen(histogram_name: str, name_mode: str) -> tuple[str, float, float]:
    """
    what_was_chosen returns info such as alpha and gamma from histogram for example:
        histogram_name --> Histogram a: 0 / MACD a: 0 --> MACD
        histogram_name --> Histogram a: 0.5 / MACD a: 0.5 --> SMACD
        histogram_name --> Histogram a: 0.5 g: 10 / MACD a: 0 g: 10 --> cmacd
        histogram_name --> Histogram a: -0.5 g: 1000 / MACD a: -0.1 g: 10 --> cmacd

    Arguments:
        histogram_name -- histogram or macd
        name_mode -- "Histogram" or "MACD"

    Returns:
        info about histogram
    """
    g_index = histogram_name.find("g", 7)
    if histogram_name == f"{name_mode} a: 0":
        return "MACD", "0", "0"
    elif g_index == -1:
        return (
            "SMACD",
            f"{'-' if histogram_name[-4] == '-' else ''}0.{histogram_name[-1]}",
            "0",
        )
    else:
        if histogram_name[g_index - 2] == "0":
            alpha = 0
        else:
            alpha = f"{'-' if histogram_name[g_index-5] == '-' else ''}0.{histogram_name[g_index-2]}"
        gamma = histogram_name[g_index + 3 :]
        return "cmacd", alpha, gamma


# remove later
def add_to_what_was_chosen(
    chosen_list_for_dict: dict, strategy: str, chosen_list: list
):
    """
    add_to_what_was_chosen aggregates results

    Arguments:
        chosen_list_for_dict -- general dictionary
        strategy -- crossover strategy
        chosen_list -- current list
    """
    # {'MACDCROSS': {'rule 1': [[0, 0, 0], []], 'rule 2': [[0, 0, 0], []], 'rule 3': [[0, 0, 0], []], 'Genetic': [[0, 0, 0], []]},
    # 'ZEROCROSS': {'rule 1': [[0, 0, 0], []], 'rule 2': [[0, 0, 0], []], 'rule 3': [[0, 0, 0], []], 'Genetic': [[0, 0, 0], []]}}
    # [what_was_chosen(rule1_name, name_mode), what_was_chosen(rule2_name, name_mode), what_was_chosen(rule3_name, name_mode), what_was_chosen(genetic_name, name_mode)]
    for i, element in enumerate(["rule 1", "rule 2", "rule 3", "Genetic"]):
        if chosen_list[i][0] == "MACD":
            index = 0
        elif chosen_list[i][0] == "SMACD":
            index = 1
        else:  # cmacd
            index = 2
        chosen_list_for_dict[strategy][element][0][index] += 1
        chosen_list_for_dict[strategy][element][1].append(
            [chosen_list[i][1], chosen_list[i][2]]
        )


def get_base_his(his_string):
    # print(his_string)
    if his_string[0] == "H":
        his_start = "Histogram"
        a_index = his_string.find("a", 9)
    else:
        his_start = "MACD"
        a_index = his_string.find("a", 5)
    first_check = his_string[a_index + 3]
    if first_check == "-":
        his_base = f"{his_start} a: {his_string[a_index+3:a_index+7]}"
    else:
        try:
            if his_string[a_index + 4] == ".":
                his_base = f"{his_start} a: {his_string[a_index+3:a_index+6]}"
            else:
                his_base = f"{his_start} a: {his_string[a_index+3]}"
        except:
            his_base = his_string
    return his_base

# delete old parts later
def get_result_from_df(
    df_test: pd.DataFrame,
    df_valid: pd.DataFrame,
    strategy: str,
    histogram_list: list,
    signal_line_list: list,
    target_dict: dict,
    label: str,
    tick: str,
    balance: float,
    alpha_range: list,
    gamma_range: list,
    risk_per_trade: float,
    signal_line_if: bool = False,
    signal_line_if_target: bool = False,
    bets_target_dict: dict = None,
    nets_target_dict: dict = None,
) -> tuple[list, list]:
    """
    get_result_from_df _summary_

    Arguments:
        df_test -- test part of dataframe
        df_valid -- validation part of dataframe
        strategy -- zero crossover or signal line crossover
        histogram_list -- list of histogram to check
        signal_line_list -- list of signal lines matching with histograms
        target_dict -- results dictionary / ex: {'total': {}, 'msft.us.txt': {}, 'axp.us.txt': {}, 'csco.us.txt': {}, 'amgn.us.txt': {}, 'nflx.us.txt': {}, 'hon.us.txt': {}, 'crm.us.txt': {},  'jpm.us.txt': {}, 'mcd.us.txt': {}, 'jnj.us.txt': {}, 'ba.us.txt': {}, 'mmm.us.txt': {}, 'googl.us.txt': {}, 'amzn.us.txt': {}, 'shw.us.txt': {}, 'dis.us.txt': {}, 'mrk.us.txt': {}, 'trv.us.txt': {}, 'vz.us.txt': {}, 'hd.us.txt': {}, 'ibm.us.txt': {}, 'wmt.us.txt': {}, 'nvda.us.txt': {}, 'gs.us.txt': {}, 'aapl.us.txt': {}, 'pg.us.txt': {}, 'cat.us.txt': {},  'unh.us.txt': {}, 'nke.us.txt': {}, 'cvx.us.txt': {}, 'v.us.txt': {}, 'ko.us.txt': {}}
        label -- timespan
        tick -- tickers
        balance -- starting balance
        alpha_range -- alpha values
        gamma_range -- alpha values
        risk_per_trade -- how much to risk on each trade

    Keyword Arguments:
        signal_line_if -- system 2 (default: {False})
        signal_line_if_target -- system 3 (default: {False})
        bets_target_dict -- dict to save the chosen configs' trimmed bets into for costs/drawdown; skipped if None (default: {None})
        nets_target_dict -- dict to save every config's validation net into for the data-snooping/SPA test; skipped if None (default: {None})

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        results
    """
    trade_cost = False
    name_mode = get_name_mode(strategy)
    current_bet_list = dict()

    dff, current_bet_list_optim = naked_crossover(
        df_test,
        histogram_list,
        signal_line_list,
        trade_cost,
        balance,
        risk_per_trade=risk_per_trade,
        signal_line_if=signal_line_if,
        signal_line_if_target=signal_line_if_target,
    )
    current_bet_list["optim"] = current_bet_list_optim
    # dff -> 'net', 'netbet', 'netpercent', 'bets', 'wins', 'longs', 'long_wins', 'long_profit',
    # 'shorts', 'short_wins', 'short_profit', "long_only_net", "long_only_bets", "long_only_wins", 'his'

    check_index_first = find_best_index(
        dff, 1, "net", name_mode, alpha_range, gamma_range
    )
    check_index_second = find_best_index(
        dff, 2, "net", name_mode, alpha_range, gamma_range
    )
    check_index_third, best_list = find_best_index(
        dff, 3, "net", name_mode, alpha_range, gamma_range
    )
    check_index_fourth = find_best_index(
        dff, 4, "net", name_mode, alpha_range, gamma_range
    )  # Genetic algorithm

    check_index_first_long = find_best_index(
        dff, 1, "long_only_net", name_mode, alpha_range, gamma_range
    )
    check_index_second_long = find_best_index(
        dff, 2, "long_only_net", name_mode, alpha_range, gamma_range
    )
    check_index_third_long, best_list_long = find_best_index(
        dff, 3, "long_only_net", name_mode, alpha_range, gamma_range
    )
    check_index_fourth_long = find_best_index(
        dff, 4, "long_only_net", name_mode, alpha_range, gamma_range
    )  # Genetic algorithm

    # NOTE: a second, older implementation of this selection (findBestIndex) used to run
    # here six more times and its results went into an `old_rules` list that nothing ever
    # read, beside a `new_rules` list that nothing read either. Both lists and the function
    # are deleted. The live selection is the find_best_index calls above; the individual
    # check_index_* values are what check_list_maker consumes.

    dff, current_bet_list_valid = naked_crossover(
        df_valid,
        histogram_list,
        signal_line_list,
        trade_cost,
        balance,
        risk_per_trade=risk_per_trade,
        signal_line_if=signal_line_if,
        signal_line_if_target=signal_line_if_target,
    )
    current_bet_list["valid"] = current_bet_list_valid

    check_index_first_list = check_list_maker(
        dff, check_index_first, check_index_first_long
    )
    check_index_second_list = check_list_maker(
        dff, check_index_second, check_index_second_long
    )
    check_index_third_list = check_list_maker(
        dff, check_index_third, check_index_third_long
    )
    check_index_fourth_list = check_list_maker(
        dff, check_index_fourth, check_index_fourth_long
    )

    intersetingMACD_first, rule1_name = make_output_list(check_index_first_list)
    intersetingMACD_Second, rule2_name = make_output_list(check_index_second_list)
    intersetingMACD_third, rule3_name = make_output_list(check_index_third_list)
    intersetingMACD_fourth, genetic_name = make_output_list(check_index_fourth_list)

    this_index = dff[dff["his"] == f"{name_mode} a: 0"].index[0]

    MACDDATA = list(dff.iloc[this_index])
    current_list_his_mode = MACDDATA.pop()
    last_three_elements = MACDDATA[-3:]
    MACDDATA.extend(last_three_elements)
    MACDDATA.append(current_list_his_mode)

    MACDDATA, macd_name = make_output_list(MACDDATA)

    currentObs = [
        intersetingMACD_first,
        intersetingMACD_Second,
        intersetingMACD_third,
        intersetingMACD_fourth,
        MACDDATA,
    ]
    # print(currentObs)

    # target_dict = observations_MACDCROSS_naked
    # label = f"{test_period_end.year}-{test_period_end.month}:{validation_period_end.year}-{validation_period_end.month}"
    # tick = aapl.ux.txt for eg
    add_to_inner_dict(target_dict, label, ["total", tick], currentObs)

    # save the chosen configs' trimmed bets (for post-hoc costs + drawdown) and every config's net
    # (for the post-hoc data-snooping / SPA test). mirrors the observations save above.
    #
    # BOTH dicts hold TWO entries per (ticker, window), in this order, because add_to_inner_dict
    # APPENDS:  [0] = the long/short arm,  [1] = the long-only arm.
    # The long-only arm is a SEPARATE SELECTION over the same config grid: check_list_maker builds
    # the reported row's columns 14-16 from check_index_*_long, which is its own find_best_index pass
    # on "long_only_net". Note it is NOT an argmax -- rules 1-3 are heuristics anchored on the MACD
    # baseline and only switch on a sufficient improvement, so the picked long-only config equalled
    # the grid's long-only maximum on just 2 of 20 rows measured. It is still a selection over 475
    # candidates, so it carries its own multiplicity, which is why it
    # needs its own SPA ([1] of the nets dict) -- and it is the only like-for-like arm against Wiles,
    # Kang and Aguirre, none of whom short. Its bets are saved too ([1] of the bets dict) so its
    # after-cost and risk columns can be produced post-hoc without re-running the sweep.
    # Readers that only want the headline arm read [0] and are unaffected by the second entry.
    if bets_target_dict is not None:
        chosen_indices = [
            check_index_first,
            check_index_second,
            check_index_third,
            check_index_fourth,
            this_index,
        ]
        chosen_bets = {
            name: [
                trim_bet(b)
                for b in current_bet_list_valid.get(dff.iloc[idx]["his"], [])
            ]
            for name, idx in zip(["Rule 1", "Rule 2", "Rule 3", "GA", "MACD"], chosen_indices)
        }
        add_to_inner_dict(bets_target_dict, label, [tick], chosen_bets)
        # [1] the long-only arm: the configs the LONG-ONLY rules picked, and their long-only bets, which
        # naked_crossover already stored under the "{his} long" key -- nothing is recomputed here. MACD
        # is the same config in both arms (it is the fixed benchmark, not a selection), so this_index
        # repeats; only the bet list differs, being its long-only leg.
        chosen_indices_long = [
            check_index_first_long,
            check_index_second_long,
            check_index_third_long,
            check_index_fourth_long,
            this_index,
        ]
        chosen_bets_long = {
            name: [
                trim_bet(b)
                for b in current_bet_list_valid.get(f'{dff.iloc[idx]["his"]} long', [])
            ]
            for name, idx in zip(["Rule 1", "Rule 2", "Rule 3", "GA", "MACD"], chosen_indices_long)
        }
        add_to_inner_dict(bets_target_dict, label, [tick], chosen_bets_long)
    if nets_target_dict is not None:
        all_nets = dict(zip(dff["his"].tolist(), dff["net"].tolist()))
        add_to_inner_dict(nets_target_dict, label, [tick], all_nets)
        # [1] every config's long-only net, so the long-only arm gets its own SPA over the same grid
        all_nets_long = dict(zip(dff["his"].tolist(), dff["long_only_net"].tolist()))
        add_to_inner_dict(nets_target_dict, label, [tick], all_nets_long)

    # remove later
    return current_bet_list, [
        what_was_chosen(rule1_name, name_mode),
        what_was_chosen(rule2_name, name_mode),
        what_was_chosen(rule3_name, name_mode),
        what_was_chosen(genetic_name, name_mode),
    ]


def run_experiment(
    time_format: str,
    balance: float,
    time_value: int = None,
    ticker: str = None,
    risk_per_trade: float = 25 / 100,
    signal_line_if: bool = False,
    signal_line_if_target: bool = False,
    macd_params: tuple = (12, 26, 9),
    session: str = "full",
):
    """
    run_experiment main function of the experiment

    Arguments:
        time_format -- time frame
        balance -- starting balance

    Keyword Arguments:
        time_value -- timespan duration value (default: {None})
        ticker -- ticker (default: {None})
        risk_per_trade -- how much to risk on each trade (default: {25/100})
        signal_line_if -- system 2 (default: {False})
        signal_line_if_target -- system 3 (default: {False})
        macd_params -- short, long, signal periods (default: {(12,26,9)})
    """
    starting_time = time.time()

    if balance:
        working_directory = "simulation"
    else:
        working_directory = "one_pos"
    # output_path = f'./results_{os.path.basename(__file__)[:-3]}_{os.environ["CONDA_DEFAULT_ENV"]}/{working_directory}'
    params_directory = "_".join(str(val) for val in macd_params)
    signal_lines_ifs_joined = "_".join(
        str(val) for val in [signal_line_if, signal_line_if_target]
    )
    # output_timeframe_for appends "_regular" for the guardrail run, so it cannot write
    # over the main results -- both of these lines used to be templated on time_format alone
    output_timeframe = output_timeframe_for(time_format, session)
    output_path = f"./results/{params_directory}/{output_timeframe}/{working_directory}/{signal_lines_ifs_joined}/"

    alpha_range = [
        -0.9,
        -0.8,
        -0.7,
        -0.6,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    gamma_range = [
        1,
        2,
        4,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        10000,
        100000,
    ]
    histogram_list, macd_list, signal_line_list = return_macd_lists(
        alpha_range, gamma_range
    )

    print("made macd_list and histogram_list")

    # the daily run READS 5-minute bars while staying labelled "1_day"; see data_prefix_for
    prefix = data_prefix_for(time_format)
    isMonth = "month"
    if time_format == "1_minute" or time_format == "5_minute":  # Minutely
        isMonth = "week"

    # datafiles = ["aapl.us.txt", "amzn.us.txt", "fb.us.txt", "googl.us.txt", "msft.us.txt", "nflx.us.txt"]
    datafiles = list(TICKER_FILES)

    upperKeys = ["total"]
    upperKeys.extend(datafiles)

    obs_list = [
        "observations_MACDCROSS_naked",
        "observations_ZEROCROSS_naked",
    ]
    # valuesDict = {v: None for v in [time_value]}
    # bets_list_for_ticks = {x: {obs: [] for obs in obs_list} for x in upperKeys}
    # chosen_list_for_dict = {
    #     x: {obs: [[0, 0, 0], []] for obs in ["rule 1", "rule 2", "rule 3", "Genetic"]}
    #     for x in ["MACDCROSS", "ZEROCROSS"]
    # }

    print(f"current value: {time_value}")

    # observations_MACDCROSS_naked = make_obs_dict(upperKeys)
    # observations_ZEROCROSS_naked = make_obs_dict(upperKeys)

    for tick in datafiles:
        # one rule for every timeframe: a named ticker runs alone, no ticker runs them all.
        # This used to be gated on time_format == "1_hour", so the daily run accepted a
        # ticker argument and silently ignored it, which made daily unshardable.
        if ticker is not None and tick != ticker:
            continue

        # {6: None}
        valuesDict = {v: None for v in [time_value]}

        # {'total': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'msft.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'axp.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'csco.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'amgn.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'nflx.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'hon.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'crm.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []},  'observations_ZEROCROSS_naked': []}, 'jpm.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'mcd.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'jnj.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'ba.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'mmm.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'googl.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'amzn.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'shw.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'dis.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'mrk.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'trv.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'vz.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'hd.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'ibm.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'wmt.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'nvda.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'gs.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'aapl.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'pg.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'cat.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []},  'observations_ZEROCROSS_naked': []}, 'unh.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'nke.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'cvx.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'v.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}, 'ko.us.txt': {'observations_MACDCROSS_naked': [], 'observations_ZEROCROSS_naked': []}}
        # bets_list_for_ticks = {x: {obs: [] for obs in obs_list} for x in upperKeys}

        # {'MACDCROSS': {'rule 1': [[0, 0, 0], []], 'rule 2': [[0, 0, 0], []], 'rule 3': [[0, 0, 0], []], 'Genetic': [[0, 0, 0], []]}, 'ZEROCROSS': {'rule 1': [[0, 0, 0], []], 'rule 2': [[0, 0, 0], []], 'rule 3': [[0, 0, 0], []], 'Genetic': [[0, 0, 0], []]}}
        chosen_list_for_dict = {
            x: {
                obs: [[0, 0, 0], []]
                for obs in ["rule 1", "rule 2", "rule 3", "Genetic"]
            }
            for x in ["MACDCROSS", "ZEROCROSS"]
        }
        # {'total': {}, 'msft.us.txt': {}, 'axp.us.txt': {}, 'csco.us.txt': {}, 'amgn.us.txt': {}, 'nflx.us.txt': {}, 'hon.us.txt': {}, 'crm.us.txt': {},  'jpm.us.txt': {}, 'mcd.us.txt': {}, 'jnj.us.txt': {}, 'ba.us.txt': {}, 'mmm.us.txt': {}, 'googl.us.txt': {}, 'amzn.us.txt': {}, 'shw.us.txt': {}, 'dis.us.txt': {}, 'mrk.us.txt': {}, 'trv.us.txt': {}, 'vz.us.txt': {}, 'hd.us.txt': {}, 'ibm.us.txt': {}, 'wmt.us.txt': {}, 'nvda.us.txt': {}, 'gs.us.txt': {}, 'aapl.us.txt': {}, 'pg.us.txt': {}, 'cat.us.txt': {},  'unh.us.txt': {}, 'nke.us.txt': {}, 'cvx.us.txt': {}, 'v.us.txt': {}, 'ko.us.txt': {}}
        observations_MACDCROSS_naked = make_obs_dict(upperKeys)
        observations_ZEROCROSS_naked = make_obs_dict(upperKeys)
        # parallel containers for the chosen-config bets (costs/drawdown) and all-config nets (SPA)
        bets_MACDCROSS_naked = make_obs_dict(upperKeys)
        bets_ZEROCROSS_naked = make_obs_dict(upperKeys)
        nets_MACDCROSS_naked = make_obs_dict(upperKeys)
        nets_ZEROCROSS_naked = make_obs_dict(upperKeys)

        print(f"current tick: {tick}")

        df = make_dataframe(
            prefix, tick, time_format, macd_params, alpha_range, gamma_range,
            session=session,
        )

        start = df.iloc[0].name
        end = df.iloc[-1].name
        test_period_start = start
        test_period_end = make_new_time(test_period_start, time_value, isMonth)
        validation_period_end = make_new_time(test_period_end, time_value, isMonth)

        print(
            f"before loop start: {start}, end: {end}, test_period_start: {test_period_start}, test_period_end: {test_period_end}, validation_period_end: {validation_period_end}"
        )

        while validation_period_end <= end:

            df_test = df[test_period_start:test_period_end]
            df_valid = df[test_period_end:validation_period_end]
            label = get_time_out(
                f"{test_period_end.year}-{test_period_end.month}-{test_period_end.day}:{validation_period_end.year}-{validation_period_end.month}-{validation_period_end.day}"
            )

            # MACD CROSS # NAKED
            strategy = "MACDCROSS"
            crossoverMode = "naked"
            print(f"start of {strategy}:{crossoverMode}")
            b_list, chosen_list_macdcross = get_result_from_df(
                df_test,
                df_valid,
                strategy,
                histogram_list,
                signal_line_list,
                observations_MACDCROSS_naked,
                label,
                tick,
                balance,
                alpha_range,
                gamma_range,
                risk_per_trade=risk_per_trade,
                signal_line_if=signal_line_if,
                signal_line_if_target=signal_line_if_target,
                bets_target_dict=bets_MACDCROSS_naked,
                nets_target_dict=nets_MACDCROSS_naked,
            )
            # print(bets_list_for_ticks)
            # bets_list_for_ticks[tick][obs_list[0]].append([label, b_list])
            # bets_list_for_ticks["total"][obs_list[0]].append([label, b_list])
            # print(chosen_list_for_dict)
            add_to_what_was_chosen(
                chosen_list_for_dict, strategy, chosen_list_macdcross
            )

            # ZERO CROSS # NAKED
            strategy = "ZEROCROSS"
            crossoverMode = "naked"
            print(f"start of {strategy}:{crossoverMode}")
            b_list, chosen_list_zerocross = get_result_from_df(
                df_test,
                df_valid,
                strategy,
                macd_list,
                signal_line_list,
                observations_ZEROCROSS_naked,
                label,
                tick,
                balance,
                alpha_range,
                gamma_range,
                risk_per_trade=risk_per_trade,
                signal_line_if=signal_line_if,
                signal_line_if_target=signal_line_if_target,
                bets_target_dict=bets_ZEROCROSS_naked,
                nets_target_dict=nets_ZEROCROSS_naked,
            )
            # print(bets_list_for_ticks)
            # bets_list_for_ticks[tick][obs_list[1]].append([label, b_list])
            # bets_list_for_ticks["total"][obs_list[1]].append([label, b_list])
            add_to_what_was_chosen(
                chosen_list_for_dict, strategy, chosen_list_zerocross
            )

            print(
                f"macd params: {macd_params}, working_directory: {working_directory}, signal_lines_ifs_joined: {signal_lines_ifs_joined}, tick: {tick}, tick index: {datafiles.index(tick) + 1}, time_format: {time_format}"
            )
            print("--- %s seconds ---" % (time.time() - starting_time))
            # same for all time formats

            if time_format == "1_day":
                test_period_start = make_new_time(test_period_start, 3, isMonth)
                test_period_end = make_new_time(test_period_end, 3, isMonth)
                validation_period_end = make_new_time(validation_period_end, 3, isMonth)
            elif time_format == "1_hour":
                test_period_start = make_new_time(test_period_start, 2, "week")
                test_period_end = make_new_time(test_period_end, 2, "week")
                validation_period_end = make_new_time(validation_period_end, 2, "week")
            else:  # minutely
                test_period_start = make_new_time(test_period_start, 1, isMonth)
                test_period_end = make_new_time(test_period_end, 1, isMonth)
                validation_period_end = make_new_time(validation_period_end, 1, isMonth)

            print(
                f"end of loop start: {start}, end: {end}, test_period_start: {test_period_start}, test_period_end: {test_period_end}, validation_period_end: {validation_period_end}"
            )

        print(
            f"out of loop start: {start}, end: {end}, test_period_start: {test_period_start}, test_period_end: {test_period_end}, validation_period_end: {validation_period_end}"
        )

        print("writing to dict")
        # print(valuesDict)
        valuesDict[time_value] = {
            "observations_MACDCROSS_naked": observations_MACDCROSS_naked,
            "observations_ZEROCROSS_naked": observations_ZEROCROSS_naked,
        }
        write_json_to_file(
            output_path,
            f"{tick}",
            valuesDict,
        )
        # write_json_to_file(
        #     output_path,
        #     f"{tick}_betslist",
        #     bets_list_for_ticks,
        # )
        write_json_to_file(
            output_path,
            f"{tick}_chosen_rules",
            chosen_list_for_dict,
        )
        # chosen-config trimmed bets (for post-hoc costs/drawdown) and all-config nets (for the SPA test)
        write_json_to_file(
            output_path,
            f"{tick}_bets",
            {
                time_value: {
                    "observations_MACDCROSS_naked": bets_MACDCROSS_naked,
                    "observations_ZEROCROSS_naked": bets_ZEROCROSS_naked,
                }
            },
        )
        write_json_to_file(
            output_path,
            f"{tick}_nets",
            {
                time_value: {
                    "observations_MACDCROSS_naked": nets_MACDCROSS_naked,
                    "observations_ZEROCROSS_naked": nets_ZEROCROSS_naked,
                }
            },
        )

    # print("writing to dict")
    # print(valuesDict)
    # if time_format == "1_day":
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}",
    #         valuesDict,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:betslist",
    #         bets_list_for_ticks,
    #     )
    # elif time_format == "1_hour":
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}",
    #         valuesDict,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:betslist",
    #         bets_list_for_ticks,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:chosen_rules",
    #         chosen_list_for_dict,
    #     )
    # else:
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:{ticker}",
    #         valuesDict,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:{ticker}:betslist",
    #         bets_list_for_ticks,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:{ticker}:chosen_rules",
    #         chosen_list_for_dict,
    #     )


def get_index(index_list: list, time_stamp: pd.Timestamp) -> int:
    """
    get_index takes a list and a timestamp and return the index of the closest timestamp in the list to the input timestamp

    Arguments:
        index_list -- a list of (sorted) timestamps
        time_stamp -- input timestamp object

    Returns:
        index of closest timestamp in the list
    """
    close_time_stamp = min(index_list, key=lambda d: abs(d - time_stamp))
    return index_list.index(close_time_stamp)


def run_experiment_minute(
    time_format: str,
    balance: float,
    time_value: int = None,
    ticker: str = None,
    risk_per_trade: float = 25 / 100,
    signal_line_if: bool = False,
    signal_line_if_target: bool = False,
    macd_params: tuple = (12, 26, 9),
    session: str = "full",
):
    """
    run_experiment_minute main function of the experiment for minutely time format

    Arguments:
        time_format -- time frame
        balance -- starting balance

    Keyword Arguments:
        time_value -- timespan duration value (default: {None})
        ticker -- ticker (default: {None})
        risk_per_trade -- how much to risk on each trade (default: {25/100})
        signal_line_if -- system 2 (default: {False})
        signal_line_if_target -- system 3 (default: {False})
        macd_params -- short, long, signal periods (default: {(12,26,9)})
    """
    starting_time = time.time()

    if balance:
        working_directory = "simulation"
    else:
        working_directory = "one_pos"
    # output_path = f'./results_{os.path.basename(__file__)[:-3]}_{os.environ["CONDA_DEFAULT_ENV"]}/{working_directory}'
    params_directory = "_".join(str(val) for val in macd_params)
    signal_lines_ifs_joined = "_".join(
        str(val) for val in [signal_line_if, signal_line_if_target]
    )
    # output_timeframe_for appends "_regular" for the guardrail run, so it cannot write
    # over the main results -- both of these lines used to be templated on time_format alone
    output_timeframe = output_timeframe_for(time_format, session)
    output_path = f"./results/{params_directory}/{output_timeframe}/{working_directory}/{signal_lines_ifs_joined}/"

    alpha_range = [
        -0.9,
        -0.8,
        -0.7,
        -0.6,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    gamma_range = [
        1,
        2,
        4,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        10000,
        100000,
    ]
    histogram_list, macd_list, signal_line_list = return_macd_lists(
        alpha_range, gamma_range
    )

    print("made macd_list and histogram_list")

    # the daily run READS 5-minute bars while staying labelled "1_day"; see data_prefix_for
    prefix = data_prefix_for(time_format)
    isMonth = "month"
    if time_format == "1_minute" or time_format == "5_minute":  # Minutely
        isMonth = "week"

    # datafiles = ["aapl.us.txt", "amzn.us.txt", "fb.us.txt", "googl.us.txt", "msft.us.txt", "nflx.us.txt"]
    datafiles = list(TICKER_FILES)

    upperKeys = ["total"]
    upperKeys.extend(datafiles)

    obs_list = [
        "observations_MACDCROSS_naked",
        "observations_ZEROCROSS_naked",
    ]
    # valuesDict = {v: None for v in [time_value]}
    # bets_list_for_ticks = {x: {obs: [] for obs in obs_list} for x in upperKeys}
    # chosen_list_for_dict = {
    #     x: {obs: [[0, 0, 0], []] for obs in ["rule 1", "rule 2", "rule 3", "Genetic"]}
    #     for x in ["MACDCROSS", "ZEROCROSS"]
    # }

    print(f"current value: {time_value}")

    # observations_MACDCROSS_naked = make_obs_dict(upperKeys)
    # observations_ZEROCROSS_naked = make_obs_dict(upperKeys)

    for tick in datafiles:

        # same rule as run_experiment. The filter used to be unconditional, so calling this
        # with ticker=None matched nothing and the driver silently did no work at all.
        if ticker is not None and tick != ticker:
            continue

        valuesDict = {v: None for v in [time_value]}
        # bets_list_for_ticks = {x: {obs: [] for obs in obs_list} for x in upperKeys}
        chosen_list_for_dict = {
            x: {
                obs: [[0, 0, 0], []]
                for obs in ["rule 1", "rule 2", "rule 3", "Genetic"]
            }
            for x in ["MACDCROSS", "ZEROCROSS"]
        }
        observations_MACDCROSS_naked = make_obs_dict(upperKeys)
        observations_ZEROCROSS_naked = make_obs_dict(upperKeys)
        # parallel containers for the chosen-config bets (costs/drawdown) and all-config nets (SPA)
        bets_MACDCROSS_naked = make_obs_dict(upperKeys)
        bets_ZEROCROSS_naked = make_obs_dict(upperKeys)
        nets_MACDCROSS_naked = make_obs_dict(upperKeys)
        nets_ZEROCROSS_naked = make_obs_dict(upperKeys)

        print(f"current tick: {tick}")

        df_read = read_and_change_data(prefix, tick, session=session)
        index_list = list(df_read.index)

        start = df_read.iloc[0].name
        end = df_read.iloc[-1].name

        nan_indexes = (max(macd_params[0], macd_params[1]) + macd_params[2]) - 2

        test_period_start = df_read.iloc[nan_indexes].name
        test_period_end = make_new_time(test_period_start, time_value, isMonth)
        validation_period_end = make_new_time(test_period_end, time_value, isMonth)

        print(
            f"before loop start: {start}, end: {end}, test_period_start: {test_period_start}, test_period_end: {test_period_end}, validation_period_end: {validation_period_end}"
        )

        while validation_period_end <= end:

            # test_period_start_index = index_list.index(test_period_start)
            test_period_start_index = get_index(index_list, test_period_start)

            calculate_period_start = df_read.iloc[
                max(test_period_start_index - nan_indexes - WARMUP_BARS, 0)
            ].name

            calculate_period_end = validation_period_end

            df = calculate_indicators(
                df_read[calculate_period_start:calculate_period_end],
                col="close",
                ema_mode="normal",
                short_period=macd_params[0],
                long_period=macd_params[1],
                signal_period=macd_params[2],
                alpha_range=alpha_range,
                gamma_range=gamma_range,
            )
            df_test = df[test_period_start:test_period_end]
            df_valid = df[test_period_end:validation_period_end]
            label = get_time_out(
                f"{test_period_end.year}-{test_period_end.month}-{test_period_end.day}:{validation_period_end.year}-{validation_period_end.month}-{validation_period_end.day}"
            )

            # MACD CROSS # NAKED
            strategy = "MACDCROSS"
            crossoverMode = "naked"
            print(f"start of {strategy}:{crossoverMode}")
            b_list, chosen_list_macdcross = get_result_from_df(
                df_test,
                df_valid,
                strategy,
                histogram_list,
                signal_line_list,
                observations_MACDCROSS_naked,
                label,
                tick,
                balance,
                alpha_range,
                gamma_range,
                risk_per_trade=risk_per_trade,
                signal_line_if=signal_line_if,
                signal_line_if_target=signal_line_if_target,
                bets_target_dict=bets_MACDCROSS_naked,
                nets_target_dict=nets_MACDCROSS_naked,
            )
            # print(bets_list_for_ticks)
            # bets_list_for_ticks[tick][obs_list[0]].append([label, b_list])
            # bets_list_for_ticks["total"][obs_list[0]].append([label, b_list])
            # print(chosen_list_for_dict)
            add_to_what_was_chosen(
                chosen_list_for_dict, strategy, chosen_list_macdcross
            )

            # ZERO CROSS # NAKED
            strategy = "ZEROCROSS"
            crossoverMode = "naked"
            print(f"start of {strategy}:{crossoverMode}")
            b_list, chosen_list_zerocross = get_result_from_df(
                df_test,
                df_valid,
                strategy,
                macd_list,
                signal_line_list,
                observations_ZEROCROSS_naked,
                label,
                tick,
                balance,
                alpha_range,
                gamma_range,
                risk_per_trade=risk_per_trade,
                signal_line_if=signal_line_if,
                signal_line_if_target=signal_line_if_target,
                bets_target_dict=bets_ZEROCROSS_naked,
                nets_target_dict=nets_ZEROCROSS_naked,
            )
            # print(bets_list_for_ticks)
            # bets_list_for_ticks[tick][obs_list[1]].append([label, b_list])
            # bets_list_for_ticks["total"][obs_list[1]].append([label, b_list])
            add_to_what_was_chosen(
                chosen_list_for_dict, strategy, chosen_list_zerocross
            )

            print(
                f"macd params: {macd_params}, working_directory: {working_directory}, signal_lines_ifs_joined: {signal_lines_ifs_joined}, tick: {tick}, tick index: {datafiles.index(tick) + 1}, time_format: {time_format}"
            )
            print("--- %s seconds ---" % (time.time() - starting_time))
            # same for all time formats

            if time_format == "1_day":
                test_period_start = make_new_time(test_period_start, 3, isMonth)
                test_period_end = make_new_time(test_period_end, 3, isMonth)
                validation_period_end = make_new_time(validation_period_end, 3, isMonth)
            elif time_format == "1_hour":
                test_period_start = make_new_time(test_period_start, 2, "week")
                test_period_end = make_new_time(test_period_end, 2, "week")
                validation_period_end = make_new_time(validation_period_end, 2, "week")
            else:  # minutely
                test_period_start = make_new_time(test_period_start, 1, isMonth)
                test_period_end = make_new_time(test_period_end, 1, isMonth)
                validation_period_end = make_new_time(validation_period_end, 1, isMonth)

            print(
                f"end of loop start: {start}, end: {end}, test_period_start: {test_period_start}, test_period_end: {test_period_end}, validation_period_end: {validation_period_end}"
            )

        print(
            f"out of loop start: {start}, end: {end}, test_period_start: {test_period_start}, test_period_end: {test_period_end}, validation_period_end: {validation_period_end}"
        )

        print("writing to dict")
        # print(valuesDict)
        valuesDict[time_value] = {
            "observations_MACDCROSS_naked": observations_MACDCROSS_naked,
            "observations_ZEROCROSS_naked": observations_ZEROCROSS_naked,
        }
        write_json_to_file(
            output_path,
            f"{tick}",
            valuesDict,
        )
        # write_json_to_file(
        #     output_path,
        #     f"{tick}_betslist",
        #     bets_list_for_ticks,
        # )
        write_json_to_file(
            output_path,
            f"{tick}_chosen_rules",
            chosen_list_for_dict,
        )
        # chosen-config trimmed bets (for post-hoc costs/drawdown) and all-config nets (for the SPA test)
        write_json_to_file(
            output_path,
            f"{tick}_bets",
            {
                time_value: {
                    "observations_MACDCROSS_naked": bets_MACDCROSS_naked,
                    "observations_ZEROCROSS_naked": bets_ZEROCROSS_naked,
                }
            },
        )
        write_json_to_file(
            output_path,
            f"{tick}_nets",
            {
                time_value: {
                    "observations_MACDCROSS_naked": nets_MACDCROSS_naked,
                    "observations_ZEROCROSS_naked": nets_ZEROCROSS_naked,
                }
            },
        )

    # print("writing to dict")
    # print(valuesDict)
    # if time_format == "1_day":
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}",
    #         valuesDict,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:betslist",
    #         bets_list_for_ticks,
    #     )
    # elif time_format == "1_hour":
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}",
    #         valuesDict,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:betslist",
    #         bets_list_for_ticks,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:chosen_rules",
    #         chosen_list_for_dict,
    #     )
    # else:
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:{ticker}",
    #         valuesDict,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:{ticker}:betslist",
    #         bets_list_for_ticks,
    #     )
    #     write_json_to_file(
    #         output_path,
    #         f"{signal_line_if}:{time_format}:{time_value}:{ticker}:chosen_rules",
    #         chosen_list_for_dict,
    #     )


# the 11 MACD parameter sets the paper sweeps, at module level so run_sweep.py can
# import them rather than keeping its own copy that could drift
MACD_PARAMS_LIST = [
    (12, 26, 9),
    (9, 19, 7),
    (14, 30, 9),
    (10, 22, 7),
    (6, 13, 4),
    (19, 39, 9),
    (8, 24, 9),
    (8, 17, 9),
    (5, 15, 9),
    (3, 10, 16),
    (5, 13, 3),
]


SIG_SYSTEMS = {
    1: (False, False),  # System 1 -- plain crossover
    2: (True, False),   # System 2 -- signal_line_if
    3: (False, True),   # System 3 -- signal_line_if_target
}


def run_scope_from_command_line() -> "argparse.Namespace":
    """
    run_scope_from_command_line reads which shard of the experiment to run from argv

    ONE RULE, applied to every shard flag: name a value, or omit the flag to run EVERY value
    of that dimension. So `--timeframe 1_day` runs all of daily in one process, while
    `--timeframe 5_minute --balance 20000 --sig 1 --ticker-index 7` runs a single shard.
    There are no interactive prompts: this is launched on a compute box by run_sweep.py,
    where a prompt reads EOF and kills the process at startup. `--timeframe` is required
    because there is no sensible "every timeframe".

    Unknown arguments are a hard error (`parse_args`, not `parse_known_args`). A mistyped
    flag used to be silently discarded, which left the run unsharded and looking fine.

    Usage:
        python experiment_main.py --timeframe 1_day
        python experiment_main.py --timeframe 1_hour --balance 0
        python experiment_main.py --timeframe 5_minute --balance 20000 --sig 1 --ticker-index 7

    Returns:
        the parsed namespace: timeframe, balance, sig, ticker_index (the last three None when omitted)
    """
    import argparse

    parser = argparse.ArgumentParser(
        add_help=True,
        description=(
            "Run one shard of the MACD experiment. Omit a shard flag to run every value "
            "of that dimension."
        ),
    )
    # 1_minute is deliberately NOT offered: the user removed it from the paper's scope,
    # and it is by far the most expensive timeframe to run
    parser.add_argument(
        "--timeframe",
        choices=["1_day", "1_hour", "5_minute"],
        required=True,
        help="which timeframe to run (required)",
    )
    parser.add_argument(
        "--balance",
        type=int,
        choices=[0, 20000],
        default=None,
        help="starting balance; 20000 is the compounding 'simulation' scenario and 0 the "
        "fixed-size 'one_pos' scenario. Omit to run both.",
    )
    parser.add_argument(
        "--sig",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="which signal-line system: 1 plain, 2 signal_line_if, 3 signal_line_if_target. "
        "Omit to run all three.",
    )
    parser.add_argument(
        "--ticker-index",
        type=int,
        choices=range(1, len(TICKER_FILES) + 1),
        metavar="1..%d" % len(TICKER_FILES),
        default=None,
        help="1-based index into the ticker universe. Omit to run every ticker.",
    )
    parser.add_argument(
        "--session",
        choices=["full", "regular"],
        default="full",
        help="'full' keeps extended hours and is the main runs' design. 'regular' keeps only "
        "09:30-16:00 exchange local time, for the regular-hours guardrail run, and writes to "
        "results/<params>/<timeframe>_regular/ so it cannot overwrite the main results. "
        "Not valid with --timeframe 1_day.",
    )
    scope = parser.parse_args()
    # Daily is the near-close build, which applies its OWN regular-session filter and takes the
    # second-to-last session bar, so there is no extended-hours exposure at daily to guard
    # against and filtering twice would shift the decision bar. read_and_change_data raises on
    # the combination anyway; catching it here fails in the first second of a shard instead of
    # part way through, which on a compute box is the difference between a typo and a lost run.
    if scope.timeframe == "1_day" and scope.session == "regular":
        parser.error(
            "--session regular is not valid with --timeframe 1_day: the daily near-close "
            "build already uses only regular-session bars (it reads the second-to-last "
            "session bar and fills at the official close), so there is nothing to filter."
        )
    return scope


def balances_to_run(scope) -> list:
    """
    balances_to_run gives the balances this process covers: the one named, or both

    Arguments:
        scope -- the namespace from run_scope_from_command_line

    Returns:
        a list of starting balances
    """
    return [scope.balance] if scope.balance is not None else [20000, 0]


def sigs_to_run(scope) -> list:
    """
    sigs_to_run gives the (signal_line_if, signal_line_if_target) pairs this process covers

    Arguments:
        scope -- the namespace from run_scope_from_command_line

    Returns:
        a list of (signal_line_if, signal_line_if_target) pairs
    """
    if scope.sig is not None:
        return [SIG_SYSTEMS[scope.sig]]
    return [SIG_SYSTEMS[key] for key in sorted(SIG_SYSTEMS)]


def tickers_to_run(scope) -> list:
    """
    tickers_to_run gives the ticker FILE NAMES this process covers: the one named, or all of them

    Arguments:
        scope -- the namespace from run_scope_from_command_line

    Returns:
        a list of ticker file names, e.g. ["aapl.us.txt"]
    """
    if scope.ticker_index is not None:
        return [TICKER_FILES[scope.ticker_index - 1]]
    return list(TICKER_FILES)


def single_ticker(scope):
    """
    single_ticker gives the one ticker file name this process covers, or None meaning every ticker

    The daily driver loops the ticker universe itself and shares its setup across tickers, so
    for daily it is cheaper to hand it None once than to call it 32 times. The hourly and
    minutely drivers are already called per ticker, so those branches use tickers_to_run.

    Arguments:
        scope -- the namespace from run_scope_from_command_line

    Returns:
        a ticker file name, or None for "every ticker"
    """
    return TICKER_FILES[scope.ticker_index - 1] if scope.ticker_index is not None else None


if __name__ == "__main__":

    # balance = 20000
    # balance = 0

    # 1_minute was dropped from the paper's scope (user, Oct 2026), so it is no longer
    # selectable here; the data stays on disk and the engine still understands the label.
    scope = run_scope_from_command_line()
    time_format = scope.timeframe

    # signal_line_if = bool(int(input("False(0) True(1): ")))
    # signal_line_if = False
    # signal_line_if_target = bool(int(input("False(0) True(1): ")))
    # signal_line_if_target = False

    risk_per_trade = 25 / 100

    macd_params_list = list(MACD_PARAMS_LIST)
    # macd_params = macd_params_list[0]

    if time_format == "1_day":  # all daily
        # split by balance
        # time_value = 6
        time_value = 12

        # ticker=None means every ticker, which run_experiment handles internally; daily is
        # cheap enough to run whole, and sharing its setup across tickers beats 32 calls
        for balance in balances_to_run(scope):
            for macd_params in macd_params_list:
                for signal_line_if, signal_line_if_target in sigs_to_run(scope):
                    run_experiment(
                        time_format=time_format,
                        balance=balance,
                        time_value=time_value,
                        ticker=single_ticker(scope),
                        risk_per_trade=risk_per_trade,
                        signal_line_if=signal_line_if,
                        signal_line_if_target=signal_line_if_target,
                        macd_params=macd_params,
                        session=scope.session,
                    )

    elif time_format == "1_hour":  # all hourly
        # split by balance and system and ticker part
        time_value = 3

        # the 2-group / 6-group ticker split is gone: a shard is one ticker, and omitting
        # --ticker-index runs them all. Same shape as the daily and minutely branches.
        for balance in balances_to_run(scope):
            for signal_line_if, signal_line_if_target in sigs_to_run(scope):
                for macd_params in macd_params_list:
                    for ticker in tickers_to_run(scope):
                        run_experiment(
                            time_format=time_format,
                            balance=balance,
                            time_value=time_value,
                            ticker=ticker,
                            risk_per_trade=risk_per_trade,
                            signal_line_if=signal_line_if,
                            signal_line_if_target=signal_line_if_target,
                            macd_params=macd_params,
                            session=scope.session,
                        )

    else:  # Minutely (1m and 5m)
        # split by balance and system and ticker part
        time_value = 8
        # # FAANG
        # ticker_list = ["msft", "nflx", "googl", "amzn", "aapl", "fb"]
        # ticker_dic = {key + 1: value for key, value in zip(range(6), ticker_list)}
        # ticker = ticker_dic[
        #     int(
        #         input(
        #             f"'msft - 1', 'nflx - 2', 'googl - 3', 'amzn - 4', 'aapl - 5', 'fb - 6': "
        #         )
        #     )
        # ]
        # run_experiment(
        #     time_format=time_format,
        #     balance=balance,
        #     time_value=time_value,
        #     ticker=ticker + ".us.txt",
        #     risk_per_trade=risk_per_trade,
        #     signal_line_if=signal_line_if,
        #     signal_line_if_target=signal_line_if_target,
        #     macd_params=macd_params,
        # )

        # DOW + FAANG
        # ticker = input(f"ticker ({TICKER_FILES}): ")
        # ticker = ticker + ".us.txt"


        # same shape as the other two branches; the 6-group ticker split is gone
        for balance in balances_to_run(scope):
            for signal_line_if, signal_line_if_target in sigs_to_run(scope):
                for macd_params in macd_params_list:
                    for ticker in tickers_to_run(scope):
                        run_experiment_minute(
                            time_format=time_format,
                            balance=balance,
                            time_value=time_value,
                            ticker=ticker,
                            risk_per_trade=risk_per_trade,
                            signal_line_if=signal_line_if,
                            signal_line_if_target=signal_line_if_target,
                            macd_params=macd_params,
                            session=scope.session,
                        )

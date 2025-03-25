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


def time_maked(t: datetime) -> str:
    """
    time_maked pads datetime for further work

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
    tick["<TIME>"] = tick["<TIME>"].apply(time_maked)
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


def read_and_change_data(prefix: str, file_name: str) -> pd.DataFrame:
    """
    read_and_change_data reads data and changes its columns and make the dataframe ready for calculating indicators

    Arguments:
        prefix -- data location
        file_name -- ticker name

    Returns:
        dataframe
    """
    df = read_data(prefix, file_name)
    change_raw_dataframe(df)
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
    start = pd.Timestamp("2009-07-01")
    return_dataframe = return_dataframe[start:]

    return return_dataframe


def make_dataframe(
    prefix: str,
    file_name: str,
    time_format: str,
    macd_params: tuple[int, int, int],
    alpha_range: list = np.arange(-0.90, 1, 0.1),
    gamma_range: list = range(0, 110, 10),
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

    Returns:
        dataframe
    """
    df = read_and_change_data(prefix, file_name)

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


def commission_cost(balance: float) -> float:
    """
    commission_cost if there are comissions for opening positions removes 4$ from balance

    Arguments:
        balance -- balance

    Returns:
        new balance after taking comission
    """

    return balance - 4


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


def open_long_position(
    candle_timestamp: str,
    candle: float,
    balance: float,
    trade_cost: bool,
    risk_per_trade: float,
    bets: int,
    longs: int,
    bets_list: list,
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

    Returns:
        tuple of changed variables
    """
    if balance:
        amount = (balance * risk_per_trade) // candle
        buy = candle * amount
        balance_before = balance
        balance -= buy
        if trade_cost:
            balance = commission_cost(balance)
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

    Returns:
        tuple of changed variables
    """
    if balance:
        amount = (balance * risk_per_trade) // candle
        buy = candle * amount
        balance_before = balance
        balance += buy
        if trade_cost:
            balance = commission_cost(balance)
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

    Returns:
        tuple of changed variables
    """
    if balance:
        if last_close:
            if trade_cost:
                balance += 4
            balance += last_candle * amount
            bets_list.pop()
            return 0, balance, wins, long_wins, long_profit

        outcome = "Loss"
        if last_candle < candle:
            wins += 1
            long_wins += 1
            outcome = "Win"
        if trade_cost:
            balance = commission_cost(balance)
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
            if trade_cost:
                balance = commission_cost(balance)
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

    Returns:
        tuple of changed variables
    """

    if balance:
        if last_close:
            if trade_cost:
                balance += 4
            balance -= last_candle * amount
            bets_list.pop()
            return 0, balance, wins, short_wins, short_profit

        outcome = "Loss"
        if last_candle > candle:
            wins += 1
            short_wins += 1
            outcome = "Win"
        if trade_cost:
            balance = commission_cost(balance)
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
            if trade_cost:
                balance = commission_cost(balance)
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
    signal_line_data = data["SignalLine"].to_list()
    target_signal_line_data = data[signal_line_target].to_list()

    signal, i = get_first_signal(target_his_data[0])
    candle = price_data[i]

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
                candle = price_data[i]
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
                candle = price_data[i]
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
                candle = price_data[i]
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
                candle = price_data[i]
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
            candle = price_data[i]
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
            candle = price_data[i]
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
    signal_line_data = data["SignalLine"].to_list()
    target_signal_line_data = data[signal_line_target].to_list()

    signal, i = get_first_signal(target_his_data[0])
    candle = price_data[i]

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
                candle = price_data[i]
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
                candle = price_data[i]
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
                candle = price_data[i]
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
                candle = price_data[i]
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
            candle = price_data[i]
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
    if not os.path.exists(output_path):
        os.makedirs(output_path)
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
        # save_best_solutions=True,
        # save_solutions=True
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    his_name = get_his_from_solution(solution[0], solution[1], name_mode)
    return result_df[result_df["his"] == his_name].index[0]


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
def findBestIndex(result_df, mode, net_name, name_mode, gamma_range):

    baseIndex = result_df[result_df["his"] == f"{name_mode} a: 0"].index[0]
    check_index = baseIndex
    # print(baseIndex, check_index)

    if mode == 1:  # rule 1
        # print(check_index, baseIndex, baseIndex + 1, baseIndex + baseIndex + 1)
        best_net = result_df.iloc[baseIndex][net_name]
        mean_lower = result_df.iloc[:baseIndex, 0].mean()
        mean_upper = result_df.iloc[baseIndex + 1 : baseIndex + baseIndex + 1, 0].mean()
        direction = 1 if mean_upper > mean_lower else -1
        # if net_name == "long_only_net":
        #     print(net_name,  baseIndex, check_index, direction, best_net)
        for i in range(1, baseIndex + 1):
            upper_jump = result_df.iloc[baseIndex + (direction * i)][net_name]
            lower_jump = result_df.iloc[baseIndex - (direction * i)][net_name]
            if upper_jump > lower_jump:
                if upper_jump > (best_net + abs(best_net * 10 / 100)):
                    check_index = baseIndex + (direction * i)
                    best_net = upper_jump
                    # print("first", check_index, upper_jump, best_net)
                    break
            elif lower_jump > upper_jump:
                if lower_jump > (best_net + abs(best_net * 10 / 100)):
                    check_index = baseIndex - (direction * i)
                    best_net = lower_jump
                    # print("second", check_index)
                    break
            else:  # upper_jump = lower_jump
                if upper_jump > (best_net + abs(best_net * 10 / 100)):
                    if mean_lower > mean_upper:
                        check_index = baseIndex - (direction * i)
                        best_net = lower_jump
                        # print("second", check_index)
                        break
                    else:  # mean_lower <= mean_upper
                        check_index = baseIndex + (direction * i)
                        best_net = upper_jump
                        # print("first", check_index, upper_jump, best_net)
                        break
        # if net_name == "long_only_net":
        #     print(net_name,  baseIndex, check_index, direction, best_net)
    elif mode == 2:  # rule 2
        check_index = result_df.iloc[: baseIndex + baseIndex + 1, :][net_name].idxmax()
        best_net = result_df.iloc[check_index][net_name]
    else:  # rule 3 return best out of all - over.
        check_index = result_df[net_name].idxmax()
        return check_index

    # continue for rule 1/2
    checkHis = result_df.iloc[check_index]["his"]
    # print(checkHis, name_mode)

    for number_his in gamma_range:
        current_check_his = checkHis + f" g: {number_his}"
        # print(current_check_his)
        # print(result_df[result_df["his"] == current_check_his][net_name])
        if (
            result_df[result_df["his"] == current_check_his][net_name].iloc[0]
        ) > best_net:
            check_index = result_df[result_df["his"] == current_check_his].iloc[0].name
            best_net = result_df[result_df["his"] == current_check_his][net_name].iloc[
                0
            ]
    return check_index


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

    check_index_first_old = findBestIndex(dff, 1, "net", name_mode, gamma_range)
    check_index_second_old = findBestIndex(dff, 2, "net", name_mode, gamma_range)
    check_index_third_old = findBestIndex(dff, 3, "net", name_mode, gamma_range)

    check_index_first_long_old = findBestIndex(
        dff, 1, "long_only_net", name_mode, gamma_range
    )
    check_index_second_long_old = findBestIndex(
        dff, 2, "long_only_net", name_mode, gamma_range
    )
    check_index_third_long_old = findBestIndex(
        dff, 3, "long_only_net", name_mode, gamma_range
    )
    old_rules = [
        check_index_first_old,
        check_index_second_old,
        check_index_third_old,
        check_index_first_long_old,
        check_index_second_long_old,
        check_index_third_long_old,
    ]
    new_rules = [
        check_index_first,
        check_index_second,
        check_index_third,
        check_index_first_long,
        check_index_second_long,
        check_index_third_long,
    ]

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
    output_path = f"./results/{params_directory}/{time_format}/{working_directory}/{signal_lines_ifs_joined}/"

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

    prefix = f"./data/stocks/{time_format}/"
    isMonth = "month"
    if time_format == "1_minute" or time_format == "5_minute":  # Minutely
        isMonth = "week"

    # datafiles = ["aapl.us.txt", "amzn.us.txt", "fb.us.txt", "googl.us.txt", "msft.us.txt", "nflx.us.txt"]
    datafiles = [
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
        if time_format == "1_hour":
            if tick != ticker:
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

        print(f"current tick: {tick}")

        df = make_dataframe(
            prefix, tick, time_format, macd_params, alpha_range, gamma_range
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
    output_path = f"./results/{params_directory}/{time_format}/{working_directory}/{signal_lines_ifs_joined}/"

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

    prefix = f"./data/stocks/{time_format}/"
    isMonth = "month"
    if time_format == "1_minute" or time_format == "5_minute":  # Minutely
        isMonth = "week"

    # datafiles = ["aapl.us.txt", "amzn.us.txt", "fb.us.txt", "googl.us.txt", "msft.us.txt", "nflx.us.txt"]
    datafiles = [
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

        if tick != ticker:
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

        print(f"current tick: {tick}")

        df_read = read_and_change_data(prefix, tick)
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
                max(test_period_start_index - nan_indexes - 500, 0)
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


if __name__ == "__main__":

    # balance = 20000
    # balance = 0

    time_format = int(
        input("time_format (1 Day (1), 1 Hour (2), 5 Minute (3), 1 Minute (4)): ")
    )

    if time_format == 1:
        time_format = "1_day"
    elif time_format == 2:
        time_format = "1_hour"
    elif time_format == 3:
        time_format = "5_minute"
    else:
        time_format = "1_minute"
    # time_format = "1_day"
    # time_format = "5_minute"

    # signal_line_if = bool(int(input("False(0) True(1): ")))
    # signal_line_if = False
    # signal_line_if_target = bool(int(input("False(0) True(1): ")))
    # signal_line_if_target = False

    risk_per_trade = 25 / 100

    macd_params_list = [
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
    # macd_params = macd_params_list[0]

    if time_format == "1_day":  # all daily
        # split by balance
        # time_value = 6
        time_value = 12

        balance = int(input("20000 (1), 0 (2): "))
        if balance == 1:
            balance = 20000
        else:
            balance = 0
        # balance = 20000

        for macd_params in macd_params_list:
            for signal_line_if, signal_line_if_target in [
                (False, False),
                (True, False),
                (False, True),
            ]:
                run_experiment(
                    time_format=time_format,
                    balance=balance,
                    time_value=time_value,
                    risk_per_trade=risk_per_trade,
                    signal_line_if=signal_line_if,
                    signal_line_if_target=signal_line_if_target,
                    macd_params=macd_params,
                )

    elif time_format == "1_hour":  # all hourly
        # split by balance and system and ticker part
        time_value = 3
        ticker_list = [
            "aapl",
            "amgn",
            "amzn",
            "axp",
            "ba",
            "cat",
            "crm",
            "csco",
            "cvx",
            "dis",
            "googl",
            "gs",
            "hd",
            "hon",
            "ibm",
            "jnj",
            "jpm",
            "ko",
            "mcd",
            "mmm",
            "mrk",
            "msft",
            "nflx",
            "nke",
            "nvda",
            "pg",
            "shw",
            "trv",
            "unh",
            "v",
            "vz",
            "wmt",
        ]

        balance = int(input("20000 (1), 0 (2): "))
        if balance == 1:
            balance = 20000
        else:
            balance = 0
        # balance = 20000
        ticker_index = int(input(f"1 (1-18), 2 (19-32): "))
        # ticker_index = 1
        if ticker_index == 1:
            lower = 0
            upper = 18
        else:
            lower = 18
            upper = 32
        ticker_part = ticker_list[lower:upper]

        sig_input = int(input(f"1 (False, False), 2 (True, False), 3 (False, True): "))
        # sig_input = 1
        if sig_input == 1:
            signal_line_if, signal_line_if_target = False, False
        elif sig_input == 2:
            signal_line_if, signal_line_if_target = True, False
        else:
            signal_line_if, signal_line_if_target = False, True

        # for signal_line_if, signal_line_if_target in [(False, False), (True, False), (False, True)]:
        for macd_params in macd_params_list:
            for ticker in ticker_part:
                run_experiment(
                    time_format=time_format,
                    balance=balance,
                    time_value=time_value,
                    ticker=ticker + ".us.txt",
                    risk_per_trade=risk_per_trade,
                    signal_line_if=signal_line_if,
                    signal_line_if_target=signal_line_if_target,
                    macd_params=macd_params,
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
        # ticker = input(f"ticker ({ticker_list}): ")
        # ticker = ticker + ".us.txt"

        ticker_list = [
            "aapl",
            "amgn",
            "amzn",
            "axp",
            "ba",
            "cat",
            "crm",
            "csco",
            "cvx",
            "dis",
            "googl",
            "gs",
            "hd",
            "hon",
            "ibm",
            "jnj",
            "jpm",
            "ko",
            "mcd",
            "mmm",
            "mrk",
            "msft",
            "nflx",
            "nke",
            "nvda",
            "pg",
            "shw",
            "trv",
            "unh",
            "v",
            "vz",
            "wmt",
        ]

        balance = int(input("20000 (1), 0 (2): "))
        if balance == 1:
            balance = 20000
        else:
            balance = 0
        # balance = 20000

        sig_input = int(input(f"1 (False, False), 2 (True, False), 3 (False, True): "))
        # sig_input = 1
        if sig_input == 1:
            signal_line_if, signal_line_if_target = False, False
        elif sig_input == 2:
            signal_line_if, signal_line_if_target = True, False
        else:
            signal_line_if, signal_line_if_target = False, True

        ticker_index = int(
            input(f"1 (1-6), 2 (7-12), 3 (13-18), 4 (19-24), 5 (25-30), 6 (31-32): ")
        )
        # ticker_index = 1
        lower = (ticker_index - 1) * 6 + 1  #
        upper = ticker_index * 6 if ticker_index != 6 else 32
        ticker_part = ticker_list[lower - 1 : upper]

        for macd_params in macd_params_list:
            for ticker in ticker_part:
                run_experiment_minute(
                    time_format=time_format,
                    balance=balance,
                    time_value=time_value,
                    ticker=ticker + ".us.txt",
                    risk_per_trade=risk_per_trade,
                    signal_line_if=signal_line_if,
                    signal_line_if_target=signal_line_if_target,
                    macd_params=macd_params,
                )

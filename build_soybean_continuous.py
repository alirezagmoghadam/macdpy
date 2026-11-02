# =====================================================================================
# build_soybean_continuous.py
#
# Builds a BACK-ADJUSTED CONTINUOUS soybean series out of the per-contract CME files in
# data/soybean/new_soy_archive (161 files = 7 contract months x 2000-2022).
#
# Why: a futures contract expires, so a long history has to be stitched from successive
# contracts. A naive splice puts a PRICE JUMP at every handover (the expiring contract and
# the incoming one trade at different prices), and a backtest books that jump as profit or
# loss that no trader ever experienced. Measured on the old files: ~-270 cents of such
# jumps inside Wiles's 160-day window, which is where our soybean "profit" came from.
#
# Design decisions (locked with the user):
#   ROLL RULE  -- roll to the next contract on the first shared date where its VOLUME or
#                 OPEN INTEREST overtakes the front contract's (what traders actually do).
#                 A contract that never overtakes is simply skipped, which naturally drops
#                 the thin Aug/Sep months without hard-coding anything.
#   ADJUSTMENT -- DIFFERENCE (subtract the gap), not ratio. Keeps the series in cents per
#                 bushel, which is exactly the unit Wiles reports.
#   MONTHS     -- all seven (F/H/K/N/Q/U/X). Aug (Q) and Sep (U) are thin; the roll rule
#                 skips them when they never take over.
#
# Side effect of difference adjustment: absolute price LEVELS become synthetic (old prices
# are shifted, and far enough back can go negative). Here that means 1356 bars (23.4%) have
# a non-positive adjusted close, all of them before 2009-03-02. Why that is safe, measured
# rather than assumed:
#   - The pipeline is EXACTLY invariant to a constant price shift. Lifting the whole series
#     by +664 leaves all 15 indicator columns identical to 0.0 -- including the CMACD
#     correntropy kernel -- and every strategy net identical to the cent. So the level is
#     not an input to any signal.
#   - P&L is a price DIFFERENCE, and differences are exact by construction (proved: on all
#     5802 day-pairs the series moves exactly as the contract actually held).
# Two things DO read the level, so they are the places to be careful:
#   - experiment_main's `amount = (balance * risk_per_trade) // candle` sizing. Soybean runs
#     with balance=0 (the one_pos scenario, amount=1), so it never takes that path. Running
#     this series with balance>0 would produce negative or exploding position sizes.
#   - Proportional bps costs. A bps cost is a fraction OF THE PRICE, so it must be priced
#     off `raw_close` (the real traded price, kept in the output for exactly this reason),
#     never off the adjusted close. On the adjusted close, 23.4% of bars would yield a
#     NEGATIVE cost, i.e. a credit.
# If the 2009-07-01 clip in calculate_indicators is lifted to use the full 23 years, decide
# then whether to add a constant lift to the series; it is free (see invariance above) but
# it is a change to the agreed anchor, so it is not applied here.
#
# Provenance: every price in the archive, and in this series, is on the CME quarter-cent
# tick grid (100.0% of 91,142 raw prices). soy_data.csv and soybeans_future.csv are only
# ~44% on grid, which is how we established they are not raw CME settlement data.
# =====================================================================================

import os
import re
import pandas as pd

# Everything that differs between one futures series and another lives here, so the same
# stitching algorithm serves more than one market. The Nikkei entry is ready for the
# front-month Osaka contract once per-contract files exist: Kang used JPX front-month
# futures and our nkx_d.csv is the cash index, which is a disclosed gap in that comparison.
# Nikkei 225 futures are quarterly (March/June/September/December), where soybean has seven
# contract months, and that is the only substantive difference.
markets = {
    "soybean": {
        "archive": "./data/soybean/new_soy_archive",
        "pattern": r"CME_S([FHKNQUX])(\d{4})\.csv",
        "prefix": "S",
        "months": {"F": 1, "H": 3, "K": 5, "N": 7, "Q": 8, "U": 9, "X": 11},
        "series": "./data/soybean/soybean_continuous_backadjusted.csv",
        "rolls": "./data/soybean/soybean_roll_log.csv",
    },
    "nikkei": {
        # per-contract files are NOT on disk yet; see VERIFY_THESE.md for sourcing. The
        # pattern below assumes Barchart's naming for the Osaka yen contract (root NO,
        # e.g. NOH11.csv = March 2011); adjust it to whatever the files are actually called.
        "archive": "./data/nikkei/nikkei_futures_archive",
        "pattern": r"NO([HMUZ])(\d{4})\.csv",
        "prefix": "NO",
        "months": {"H": 3, "M": 6, "U": 9, "Z": 12},
        "series": "./data/nikkei/nikkei_continuous_backadjusted.csv",
        "rolls": "./data/nikkei/nikkei_roll_log.csv",
    },
}

# soybean stays the default so existing callers and the gate are unaffected
ARCHIVE_DIR = markets["soybean"]["archive"]
OUT_SERIES = markets["soybean"]["series"]
OUT_ROLLS = markets["soybean"]["rolls"]

month_codes = markets["soybean"]["months"]
price_columns = ["Open", "High", "Low", "Close"]


def load_contracts(archive_dir: str, pattern: str = None, prefix: str = "S") -> dict:
    """
    load_contracts reads every per-contract csv in the archive into a date-indexed frame

    Arguments:
        archive_dir -- folder holding the per-contract files

    Keyword Arguments:
        pattern -- regex with two groups, the month code and the year, matching the file names. defaults to the soybean archive's CME_S<month><year>.csv (default: {None})
        prefix -- what to put in front of the month code and year when naming the contract, so a nikkei contract reads NOH2011 rather than SH2011 (default: {"S"})

    Returns:
        dict {contract code -> dataframe}, e.g. {"SQ2014": df}
    """
    if pattern is None:
        pattern = markets["soybean"]["pattern"]
    contracts = {}
    for name in sorted(os.listdir(archive_dir)):
        match = re.fullmatch(pattern, name)
        if not match:
            continue
        df = pd.read_csv(os.path.join(archive_dir, name), parse_dates=["Date"])
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        # one date must map to one row or the roll-gap lookup below returns a Series
        df = df[~df.index.duplicated(keep="last")]
        # a few early rows are settlement-only prints with no trading at all
        if len(df) == 0:
            continue
        contracts[f"{prefix}{match.group(1)}{match.group(2)}"] = df
    return contracts


def contract_sort_key(code: str, months: dict = None) -> tuple:
    """
    contract_sort_key orders contracts by their delivery date so "front" means nearest expiry

    Arguments:
        code -- contract code such as "SQ2014"

    Keyword Arguments:
        months -- mapping of month code letter to calendar month, defaulting to soybean's seven (default: {None})

    Returns:
        (year, month) tuple to sort on
    """
    if months is None:
        months = month_codes
    # the month code is the character before the 4-digit year, whatever the prefix length
    return int(code[-4:]), months[code[-5]]


def delivery_month(code: str, months: dict = None) -> pd.Timestamp:
    """
    delivery_month the first day of the calendar month a contract delivers in

    Arguments:
        code -- contract code such as "SQ2014"

    Keyword Arguments:
        months -- mapping of month code letter to calendar month (default: {None})

    Returns:
        timestamp of the first day of that contract's delivery month
    """
    year, month = contract_sort_key(code, months)
    return pd.Timestamp(year=year, month=month, day=1)


def collapse_rolls(rolls: list) -> list:
    """
    collapse_rolls merges handovers that land on the same date into a single log row, because a contract that is overtaken and then overtaken again within one day is never actually held and only clutters the audit trail

    Arguments:
        rolls -- the per-handover records built by back_adjust

    Returns:
        list of records with same-date chains merged and their gaps summed
    """
    merged = []
    for record in rolls:
        if merged and merged[-1]["roll_date"] == record["roll_date"]:
            # chain straight through to where we actually end up, and add the gaps so the
            # logged gap is the whole move we stepped over on that date
            merged[-1]["to_contract"] = record["to_contract"]
            merged[-1]["to_close"] = record["to_close"]
            merged[-1]["gap"] = merged[-1]["gap"] + record["gap"]
        else:
            merged.append(dict(record))
    return merged


def find_roll_date(front: pd.DataFrame, nxt: pd.DataFrame, not_before: pd.Timestamp) -> pd.Timestamp:
    """
    find_roll_date the first date on or after not_before, shared by both contracts, where the next contract's volume or open interest overtakes the front contract's

    Arguments:
        front -- the contract currently being held
        nxt -- the candidate contract to roll into
        not_before -- do not roll earlier than this date

    Returns:
        the roll date, or None if the next contract never overtakes while both trade
    """
    shared = front.index.intersection(nxt.index)
    shared = shared[shared >= not_before]
    for date in shared:
        front_row, next_row = front.loc[date], nxt.loc[date]
        overtakes_volume = next_row["Volume"] > front_row["Volume"]
        overtakes_interest = next_row["Open Interest"] > front_row["Open Interest"]
        if overtakes_volume or overtakes_interest:
            return date
    return None


def build_segments(contracts: dict, months: dict = None) -> list:
    """
    build_segments walks the contracts in delivery order and decides which one is "front" over each stretch of dates, skipping any contract that never takes over

    Arguments:
        contracts -- dict {contract code -> dataframe}

    Returns:
        list of (code, start_date, end_date, roll_date_into_next) with end_date exclusive
    """
    ordered = sorted(contracts, key=lambda code: contract_sort_key(code, months))
    segments = []
    current = ordered[0]
    # the archive's oldest contract starts PRINTING more than a year before it delivers,
    # and over that stretch there is no proper front month in the archive at all (the
    # contracts that really were front back then are simply not in the file). starting
    # there made the oldest contract "front" in 1998 and then forced a 357-day segment.
    # so floor the very first start at two months before its own delivery month, which is
    # about when it genuinely takes over, and drop the unusable history before that.
    floor = delivery_month(current, months) - pd.Timedelta(days=60)
    usable = contracts[current].index[contracts[current].index >= floor]
    start = usable.min() if len(usable) else contracts[current].index.min()
    index = 0
    while index < len(ordered) - 1:
        # look forward until some later contract actually overtakes the one we hold
        roll_date, next_code = None, None
        for candidate in ordered[index + 1 :]:
            roll_date = find_roll_date(contracts[current], contracts[candidate], start)
            if roll_date is not None:
                next_code = candidate
                break
        if roll_date is None:
            # nothing overtook it: hold to the end of its data and stop
            segments.append((current, start, contracts[current].index.max(), None))
            return segments
        segments.append((current, start, roll_date, roll_date))
        current, start = next_code, roll_date
        index = ordered.index(next_code)
    segments.append((current, start, contracts[current].index.max(), None))
    return segments


def back_adjust(contracts: dict, segments: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    back_adjust stitches the segments into one series and shifts every earlier segment by the cumulative roll gap so the series has no artificial jumps

    Arguments:
        contracts -- dict {contract code -> dataframe}
        segments -- output of build_segments

    Returns:
        (continuous dataframe, roll log dataframe)
    """
    # the gap at each handover, measured on the shared roll date
    gaps, rolls = [], []
    for position, (code, _, _, roll_date) in enumerate(segments[:-1]):
        next_code = segments[position + 1][0]
        old_close = contracts[code].loc[roll_date, "Close"]
        new_close = contracts[next_code].loc[roll_date, "Close"]
        gap = float(new_close - old_close)
        gaps.append(gap)
        rolls.append(
            {
                "roll_date": roll_date,
                "from_contract": code,
                "to_contract": next_code,
                "from_close": float(old_close),
                "to_close": float(new_close),
                "gap": gap,
            }
        )
    # cumulative offset: the newest segment is the anchor and is never shifted, and every
    # earlier segment is lifted by the sum of all gaps that happen at or after its own roll.
    # walking backwards from the anchor keeps every handover continuous at once.
    offsets = [0.0] * len(segments)
    running = 0.0
    for position in range(len(segments) - 2, -1, -1):
        running += gaps[position]
        offsets[position] = running

    frames = []
    for position, (code, start, end, roll_date) in enumerate(segments):
        df = contracts[code]
        # a segment owns [start, end): the roll date itself belongs to the NEXT segment
        piece = df.loc[(df.index >= start) & (df.index < end)].copy()
        if position == len(segments) - 1:
            piece = df.loc[df.index >= start].copy()
        if len(piece) == 0:
            continue
        piece["raw_close"] = piece["Close"]
        for column in price_columns:
            piece[column] = piece[column] + offsets[position]
        piece["contract"] = code
        frames.append(piece)

    continuous = pd.concat(frames).sort_index()
    continuous = continuous[~continuous.index.duplicated(keep="last")]
    return continuous, pd.DataFrame(collapse_rolls(rolls))


def run(market: str = "soybean"):
    """
    run builds the back-adjusted continuous series for one market and writes it plus an auditable roll log

    Keyword Arguments:
        market -- key into `markets`; "soybean" is the only one with files on disk so far (default: {"soybean"})
    """
    config = markets[market]
    contracts = load_contracts(config["archive"], config["pattern"], config["prefix"])
    print(f"loaded {len(contracts)} contracts from {config['archive']}")
    segments = build_segments(contracts, config["months"])
    continuous, rolls = back_adjust(contracts, segments)
    continuous.to_csv(config["series"])
    rolls.to_csv(config["rolls"], index=False)
    print(f"segments used: {len(segments)}   rolls: {len(rolls)}")
    print(f"series rows: {len(continuous)}   {continuous.index.min().date()} -> {continuous.index.max().date()}")
    print(f"wrote {config['series']}")
    print(f"wrote {config['rolls']}")


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "soybean")

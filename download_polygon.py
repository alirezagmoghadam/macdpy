"""
download_polygon.py downloads the real vendor data the paper's results are computed from,
using the reader's OWN Polygon.io API key.

This is polygon_test.ipynb turned into a script. That matters: the notebook is what
actually produced the data behind the published numbers, so this script keeps its request
parameters, its column layout and its file naming EXACTLY, rather than being a cleaner
reimplementation that might differ in some detail nobody notices. The output of this
script is byte-comparable with the files in data/stocks/ apart from whatever the vendor has
revised since.

FOUR THINGS THE NOTEBOOK DID NOT DO, each of which stops a fresh clone:

  1. ITS TICKER LIST WAS SHORT. The notebook downloads 30 symbols. The engine's
     TICKER_FILES has 32 -- the Dow 30 of the sample period plus NFLX and GOOGL, which were
     fetched in a separate pass with a different list still visible commented out in the
     notebook. A reader running the notebook as-is would get 30 files and the engine would
     fail on nflx.us.txt and googl.us.txt. This script reads TICKER_FILES from the engine,
     so the two cannot drift apart again.

  2. IT DID NOT CREATE ITS OUTPUT FOLDERS. `to_csv` into ./data/stocks/1_day/ raises on a
     fresh clone, after the download has already been paid for in API calls.

  3. ITS API KEY WAS A BARE NAME. The notebook has `# API_KEY = ""` commented out and then
     uses API_KEY, so it raises NameError unless you happen to have defined it in another
     cell. A key must never be committed, so this takes it from --api-key or the
     POLYGON_API_KEY environment variable and refuses to run without one.

  4. NO RESUME AND NO RATE LIMITING. 32 tickers x 3 timeframes is 96 paginated downloads.
     Polygon's free tier allows 5 calls a minute, and a 429 part-way through lost
     everything. This skips files already on disk, so an interrupted run continues where it
     stopped, and it backs off and retries on a rate limit.

WHAT YOU WILL AND WILL NOT GET. Polygon serves history according to your subscription: the
free and Starter tiers cap it at 2 years, which is why the files in this repository begin
in February 2020 rather than the 2019-01-01 the request asks for. If your plan has a
shorter window than the paper's sample you will get a shorter sample, and the walk-forward
window counts will not match the paper's. That is a subscription limit, not a bug, and the
script prints the span it actually received for every file so the difference is visible
rather than silent.

The 1-minute timeframe is NOT downloaded by default. It was dropped from the paper's scope
and is roughly five times the size of everything else combined. Pass
--timeframes 1_minute if you want it.

Usage:

    cd Code_1_working
    pip install polygon-api-client            # not in requirements.txt, see the note there
    export POLYGON_API_KEY=your_key_here      # or pass --api-key
    python download_polygon.py                # all 32 tickers, 3 timeframes

    python download_polygon.py --tickers aapl msft --timeframes 1_day
    python download_polygon.py --dry-run      # show the plan, download nothing

Cost check before you start: --dry-run prints the number of requests and files.
"""
import argparse
import datetime
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# The vendor column layout, exactly as polygon_test.ipynb wrote it. The engine's reader
# depends on every one of these being present -- `change_raw_dataframe` drops five of them
# BY NAME, so a missing column is a KeyError rather than a silently narrower frame.
COLUMNS = [
    "<TICKER>", "<PER>", "<DATE>", "<TIME>", "<OPEN>", "<HIGH>",
    "<LOW>", "<CLOSE>", "<VOL>", "<OPENINT>", "<VWAP>", "<TRANSACTIONS>",
]

# (multiplier, timespan) per folder name, matching the notebook's config loop. The folder
# name is what data_prefix_for builds, so these strings are not cosmetic.
TIMEFRAMES = {
    "1_day": (1, "day"),
    "1_hour": (1, "hour"),
    "5_minute": (5, "minute"),
    "1_minute": (1, "minute"),
}

# The paper's sample window, as requested by polygon_test.ipynb. What you actually receive
# depends on your subscription's history limit.
DEFAULT_FROM = "2019-01-01"
DEFAULT_TO = "2025-02-06"

FALLBACK_TICKERS = [
    "aapl", "amgn", "amzn", "axp", "ba", "cat", "crm", "csco", "cvx", "dis",
    "googl", "gs", "hd", "hon", "ibm", "jnj", "jpm", "ko", "mcd", "mmm",
    "mrk", "msft", "nflx", "nke", "nvda", "pg", "shw", "trv", "unh", "v",
    "vz", "wmt",
]


def ticker_universe():
    """
    ticker_universe returns the engine's ticker list, read from experiment_main so this script cannot download a different universe from the one the engine expects

    This is the fix for defect 1 above: the notebook's hardcoded list of 30 was two short of
    the engine's 32, and nothing connected the two.

    Returns:
        list of bare ticker symbols, lowercase, in the engine's order
    """
    try:
        sys.path.insert(0, HERE)
        from experiment_main import TICKER_FILES

        return [name.split(".")[0] for name in TICKER_FILES]
    except Exception as why:
        print(f"[polygon] NOTE could not import experiment_main ({why}); using the "
              "built-in list of 32. Check it against TICKER_FILES if the engine later "
              "cannot find a file.")
        return list(FALLBACK_TICKERS)


def period_value(multiplier: int, timespan: str) -> int:
    """
    period_value gives the <PER> column's value for a timeframe, reproducing the notebook's rule exactly

    The notebook wrote `60 if timespan == "hour" else multiplier`, so daily bars carry PER=1
    and so do 1-minute bars. That collision is harmless -- `change_raw_dataframe` drops
    <PER> before anything reads it -- but it is reproduced rather than corrected, because
    the files already on disk have it and a reader diffing their download against ours
    should see no difference.

    Arguments:
        multiplier -- the aggregate multiplier
        timespan -- "day", "hour" or "minute"

    Returns:
        the <PER> value
    """
    return 60 if timespan == "hour" else multiplier


def fetch_one(client, ticker, multiplier, timespan, from_date, to_date, retries=5):
    """
    fetch_one downloads every aggregate bar for one ticker at one timeframe, retrying on a rate limit

    Arguments:
        client -- a polygon RESTClient
        ticker -- bare symbol, uppercased for the request
        multiplier -- aggregate multiplier
        timespan -- "day", "hour" or "minute"
        from_date -- ISO start date
        to_date -- ISO end date

    Keyword Arguments:
        retries -- attempts before giving up, with an increasing wait between them (default: {5})

    Returns:
        DataFrame in the vendor layout, possibly empty if the vendor returned no bars
    """
    rows = []
    attempt = 0
    while True:
        try:
            rows = list(client.list_aggs(
                ticker=ticker.upper(),
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                adjusted=True,
                limit=50000,
            ))
            break
        except Exception as why:
            attempt += 1
            message = str(why)
            # a rate limit or a transient gateway error is worth waiting out; a bad key or
            # an unknown symbol is not, and retrying it five times just wastes the reader's
            # quota on a request that will never succeed
            transient = any(
                marker in message
                for marker in ("429", "too many", "Too Many", "502", "503", "504", "timeout")
            )
            if attempt >= retries or not transient:
                raise
            wait = 15 * attempt
            print(f"      rate limited or transient error, waiting {wait}s "
                  f"(attempt {attempt}/{retries})")
            time.sleep(wait)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    # the notebook's own timestamp handling: Polygon returns epoch milliseconds, read as
    # UTC and split into a zero-padded YYYYMMDD and HHMMSS. The zero padding is load
    # bearing -- make_index concatenates <DATE> and <TIME> as strings, and an unpadded
    # 50000 gives "20200207 50000", which does not parse as a time
    stamps = [
        datetime.datetime.fromtimestamp(row.timestamp / 1000.0, tz=datetime.timezone.utc)
        for row in rows
    ]
    count = len(rows)
    frame = pd.DataFrame({
        "<TICKER>": [f"{ticker.upper()}.US"] * count,
        "<PER>": [period_value(multiplier, timespan)] * count,
        "<DATE>": [f"{s.year}{s.month:02d}{s.day:02d}" for s in stamps],
        "<TIME>": [f"{s.hour:02d}{s.minute:02d}{s.second:02d}" for s in stamps],
        "<OPEN>": [row.open for row in rows],
        "<HIGH>": [row.high for row in rows],
        "<LOW>": [row.low for row in rows],
        "<CLOSE>": [row.close for row in rows],
        "<VOL>": [row.volume for row in rows],
        "<OPENINT>": [0] * count,
        "<VWAP>": [row.vwap for row in rows],
        "<TRANSACTIONS>": [row.transactions for row in rows],
    })[COLUMNS]
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the paper's vendor data from Polygon.io using your own API "
                    "key. Reproduces polygon_test.ipynb, which is what produced the data "
                    "behind the published results.",
    )
    parser.add_argument("--api-key", default=os.environ.get("POLYGON_API_KEY"),
                        help="your Polygon.io key. Defaults to $POLYGON_API_KEY. Never "
                             "commit it.")
    parser.add_argument("--tickers", nargs="+", metavar="SYM",
                        help="bare symbols to fetch. Default: the engine's 32.")
    parser.add_argument("--timeframes", nargs="+", default=["1_day", "1_hour", "5_minute"],
                        choices=sorted(TIMEFRAMES),
                        help="which timeframes. 1_minute is out of the paper's scope and "
                             "very large, so it is not a default.")
    parser.add_argument("--from-date", default=DEFAULT_FROM,
                        help=f"start of the request window (default: {DEFAULT_FROM}). What "
                             "you receive depends on your subscription's history limit.")
    parser.add_argument("--to-date", default=DEFAULT_TO,
                        help=f"end of the request window (default: {DEFAULT_TO})")
    parser.add_argument("--out", default=os.path.join(HERE, "data", "stocks"),
                        help="where the timeframe folders are written")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist. Without it, existing "
                             "files are skipped so an interrupted run resumes.")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to wait between requests. Set to 13 on the free "
                             "tier, which allows 5 calls a minute (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without calling the API")
    args = parser.parse_args(argv)

    tickers = args.tickers if args.tickers else ticker_universe()
    tickers = [symbol.lower().replace(".us", "").replace(".txt", "") for symbol in tickers]

    plan = []
    for timeframe in args.timeframes:
        for ticker in tickers:
            target = os.path.join(args.out, timeframe, f"{ticker}.us.txt")
            exists = os.path.exists(target)
            if exists and not args.force:
                continue
            plan.append((timeframe, ticker, target))

    skipped = len(args.timeframes) * len(tickers) - len(plan)
    print(f"[polygon] {len(tickers)} tickers x {len(args.timeframes)} timeframes")
    print(f"[polygon] window {args.from_date} .. {args.to_date}")
    print(f"[polygon] {len(plan)} files to download, {skipped} already on disk")
    if args.sleep:
        estimate = len(plan) * args.sleep / 60.0
        print(f"[polygon] at --sleep {args.sleep}s that is at least "
              f"{estimate:.0f} minutes of waiting")

    if args.dry_run:
        print("[polygon] --dry-run: nothing downloaded")
        for timeframe, ticker, target in plan[:10]:
            print(f"             would write {target}")
        if len(plan) > 10:
            print(f"             ... and {len(plan) - 10} more")
        return 0

    if not plan:
        print("[polygon] nothing to do. Pass --force to re-download.")
        return 0

    if not args.api_key:
        print("[polygon] NO API KEY. Pass --api-key or set POLYGON_API_KEY.\n"
              "          Get one at https://polygon.io. The paper's data needs a plan "
              "with at least 5 years of\n"
              "          intraday history; the free tier caps history at 2 years, which "
              "gives a shorter sample\n"
              "          than the paper's and therefore different window counts.")
        return 2

    try:
        from polygon import RESTClient
    except ImportError:
        print("[polygon] polygon-api-client is not installed. It is deliberately not in "
              "requirements.txt\n"
              "          because the engine does not need it -- only this script does:\n"
              "              pip install polygon-api-client")
        return 2

    client = RESTClient(api_key=args.api_key)
    for timeframe in args.timeframes:
        os.makedirs(os.path.join(args.out, timeframe), exist_ok=True)

    failures = []
    for position, (timeframe, ticker, target) in enumerate(plan, start=1):
        multiplier, timespan = TIMEFRAMES[timeframe]
        print(f"[{position:>3}/{len(plan)}] {timeframe:10} {ticker:6} ", end="", flush=True)
        try:
            frame = fetch_one(
                client, ticker, multiplier, timespan, args.from_date, args.to_date,
            )
        except Exception as why:
            print(f"FAILED: {why}")
            failures.append((timeframe, ticker, str(why)))
            continue

        if frame.empty:
            print("NO BARS RETURNED -- not written")
            failures.append((timeframe, ticker, "vendor returned no bars"))
            continue

        frame.to_csv(target, index=False)
        # print the span actually received, so a subscription history limit is visible on
        # every line instead of being discovered later as a short sample
        print(f"{len(frame):>8,} bars  {frame['<DATE>'].iloc[0]} .. "
              f"{frame['<DATE>'].iloc[-1]}")
        if args.sleep:
            time.sleep(args.sleep)

    print()
    print(f"[polygon] {len(plan) - len(failures)} of {len(plan)} files written")
    if failures:
        print(f"[polygon] {len(failures)} FAILED -- rerun to retry just these:")
        for timeframe, ticker, why in failures:
            print(f"             {timeframe:10} {ticker:6} {why[:70]}")
        return 1

    requested_start = args.from_date.replace("-", "")
    print()
    print("[polygon] CHECK THE SPANS PRINTED ABOVE. If they start later than "
          f"{requested_start},\n"
          "          your subscription's history limit is shorter than the request and "
          "your sample is\n"
          "          shorter than the paper's. The walk-forward window counts will differ "
          "accordingly.")
    print()
    print("  Next:")
    print("      python preflight_results.py --root ./results   # after a run")
    print("      python experiment_main.py --timeframe 1_day --balance 20000 --sig 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""run_sweep.py -- launch the sharded MACD experiment across every core on the compute box.

This is the script to run on the Xeon. `experiment_main.py` runs exactly ONE shard; this
enumerates the shards, runs them across a worker pool, logs each one, and reports at the end
whether every expected result file actually exists.

    python run_sweep.py --timeframe 5_minute                  # every shard, all cores
    python run_sweep.py --timeframe 5_minute --jobs 128
    python run_sweep.py --timeframe 5_minute --resume          # only the missing work
    python run_sweep.py --timeframe 1_day --dry-run            # list the shards, run nothing
    python run_sweep.py --timeframe 1_hour --balance 0 --sig 1 # a slice of the sweep

A shard is one (balance, sig, ticker); the 11 MACD parameter sets are looped inside the
shard. So a full timeframe is 2 x 3 x 32 = 192 shards.

WHY `--resume` MATTERS. All 32 ticker shards for a given (balance, sig) write into the SAME
directory, with the ticker in the FILE name. So a shard that dies leaves no marker -- the
directory looks populated and you would only notice by counting files. `--resume` skips
shards whose files are all present and re-runs the rest, so re-issuing the same command after
a crash, a reboot or a full disk picks up exactly where it stopped.

WHY THE DIRECTORIES ARE MADE UP FRONT. `write_json_to_file` creates its own directory, and
with 32 processes arriving at the same new directory at once a bare check-then-create races.
Making all of them here, single-threaded, before any shard starts, means that never happens.
(The engine also passes `exist_ok=True`, so a hand-run shard is safe without this script.)

Nothing here duplicates the experiment's configuration: the ticker universe, the signal-line
systems and the MACD parameter sets are imported from experiment_main, and the output paths
are rebuilt with the same rule the engine uses.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from experiment_main import (  # noqa: E402
    MACD_PARAMS_LIST,
    SIG_SYSTEMS,
    TICKER_FILES,
    output_timeframe_for,
)

# the four files the engine writes per (ticker, macd config); see write_json_to_file callers
RESULT_SUFFIXES = ("", "_chosen_rules", "_bets", "_nets")
TIMEFRAMES = ("1_day", "1_hour", "5_minute")
BALANCES = (20000, 0)


def output_dir(macd_params, timeframe, balance, sig, session="full"):
    """
    output_dir rebuilds the directory the engine writes into, by the same rule the engine uses

    Arguments:
        macd_params -- (short, long, signal)
        timeframe -- 1_day/1_hour/5_minute
        balance -- 20000 (the compounding 'simulation' scenario) or 0 ('one_pos')
        sig -- 1, 2 or 3, keying SIG_SYSTEMS

    Returns:
        a path relative to this folder, e.g. results/12_26_9/5_minute/simulation/False_False
    """
    params_directory = "_".join(str(val) for val in macd_params)
    working_directory = "simulation" if balance else "one_pos"
    signal_lines_ifs_joined = "_".join(str(val) for val in SIG_SYSTEMS[sig])
    # the SAME helper the engine uses, so the launcher cannot disagree with it about where a
    # guardrail run lands -- results/<params>/<timeframe>_regular/...
    return os.path.join(
        "results", params_directory, output_timeframe_for(timeframe, session),
        working_directory, signal_lines_ifs_joined
    )


def expected_files(timeframe, balance, sig, ticker, session="full"):
    """
    expected_files lists every result file one shard should leave behind

    Arguments:
        timeframe -- 1_day/1_hour/5_minute
        balance -- 20000 or 0
        sig -- 1, 2 or 3
        ticker -- a ticker file name, e.g. "aapl.us.txt"

    Returns:
        a list of paths, 4 per MACD parameter set
    """
    paths = []
    for macd_params in MACD_PARAMS_LIST:
        directory = output_dir(macd_params, timeframe, balance, sig, session)
        paths.extend(os.path.join(directory, ticker + suffix) for suffix in RESULT_SUFFIXES)
    return paths


def shard_complete(timeframe, balance, sig, ticker, root=None, session="full"):
    """
    shard_complete says whether every file a shard should have written is present

    This is what --resume tests. It has to look at the FILES rather than at anything the
    shard left behind, because all 32 ticker shards share one output directory and a shard
    that died leaves no marker of its own.

    Arguments:
        timeframe -- 1_day/1_hour/5_minute
        balance -- 20000 or 0
        sig -- 1, 2 or 3
        ticker -- a ticker file name

    Keyword Arguments:
        root -- the folder the relative result paths hang off (default: this folder)

    Returns:
        True when all 4 files exist for all 11 MACD parameter sets
    """
    base = HERE if root is None else root
    return all(
        os.path.exists(os.path.join(base, path))
        for path in expected_files(timeframe, balance, sig, ticker, session)
    )


def shard_name(timeframe, balance, sig, ticker, session="full"):
    """
    shard_name gives a short filename-safe label for a shard, used for its log file

    Returns:
        e.g. "5_minute-b20000-s1-aapl"
    """
    # the session goes in the name too, or a guardrail shard's log would overwrite the main
    # run's log for the same (timeframe, balance, sig, ticker)
    suffix = "" if session == "full" else "-" + session
    return "%s%s-b%d-s%d-%s" % (timeframe, suffix, balance, sig, ticker.split(".")[0])


def run_shard(timeframe, balance, sig, ticker_index, log_dir, session="full"):
    """
    run_shard runs one shard as a subprocess, with its output in its own log file

    Arguments:
        timeframe -- 1_day/1_hour/5_minute
        balance -- 20000 or 0
        sig -- 1, 2 or 3
        ticker_index -- 1-based index into TICKER_FILES
        log_dir -- where to write the shard's log

    Returns:
        (name, returncode, seconds, log_path)
    """
    ticker = TICKER_FILES[ticker_index - 1]
    name = shard_name(timeframe, balance, sig, ticker, session)
    command = [
        sys.executable,
        "experiment_main.py",
        "--timeframe", timeframe,
        "--balance", str(balance),
        "--sig", str(sig),
        "--ticker-index", str(ticker_index),
        "--session", session,
    ]
    log_path = os.path.join(log_dir, name + ".log")
    started = time.time()
    with open(log_path, "w") as handle:
        handle.write("$ %s\n\n" % " ".join(command))
        handle.flush()
        # cwd=HERE because the engine reads ./data and writes ./results relative to itself
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, cwd=HERE)
    return name, code, time.time() - started, log_path


def main():
    parser = argparse.ArgumentParser(
        description="Launch the sharded MACD experiment and report completeness."
    )
    parser.add_argument("--timeframe", choices=TIMEFRAMES, required=True)
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count(),
        help="shards to run at once (default: every core, %d here)" % (os.cpu_count() or 1),
    )
    parser.add_argument(
        "--balance", type=int, choices=list(BALANCES), default=None,
        help="restrict the sweep to one balance (default: both)",
    )
    parser.add_argument(
        "--sig", type=int, choices=sorted(SIG_SYSTEMS), default=None,
        help="restrict the sweep to one signal-line system (default: all three)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip shards whose result files are all already present",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list the shards and exit without running any"
    )
    parser.add_argument(
        "--session", choices=["full", "regular"], default="full",
        help="'full' is the main run. 'regular' is the regular-hours guardrail run: it keeps "
        "only 09:30-16:00 bars and writes to results/<params>/<timeframe>_regular/, so it "
        "never touches the main results. Not valid with --timeframe 1_day.",
    )
    parser.add_argument(
        "--log-dir", default=os.path.join("logs", "sweep"),
        help="where per-shard logs go (default: logs/sweep)",
    )
    args = parser.parse_args()
    # the engine rejects this too; refusing here means the whole sweep fails in the first
    # second rather than every one of its shards failing separately
    if args.timeframe == "1_day" and args.session == "regular":
        parser.error(
            "--session regular is not valid with --timeframe 1_day: the daily near-close "
            "build already uses only regular-session bars."
        )

    balances = [args.balance] if args.balance is not None else list(BALANCES)
    sigs = [args.sig] if args.sig is not None else sorted(SIG_SYSTEMS)
    indices = range(1, len(TICKER_FILES) + 1)

    # every directory, made once, single-threaded, before anything is launched
    directories = {
        output_dir(macd_params, args.timeframe, balance, sig, args.session)
        for macd_params, balance, sig in itertools.product(MACD_PARAMS_LIST, balances, sigs)
    }
    if not args.dry_run:
        for directory in sorted(directories):
            os.makedirs(os.path.join(HERE, directory), exist_ok=True)
    print("output directories: %d (%s)" % (
        len(directories), "would create" if args.dry_run else "created"))

    all_shards = [(b, s, i) for b, s, i in itertools.product(balances, sigs, indices)]
    todo, skipped = [], []
    for balance, sig, index in all_shards:
        ticker = TICKER_FILES[index - 1]
        if args.resume and shard_complete(
                args.timeframe, balance, sig, ticker, session=args.session):
            skipped.append(shard_name(args.timeframe, balance, sig, ticker, args.session))
        else:
            todo.append((balance, sig, index))

    print("shards: %d total, %d to run, %d already complete" % (
        len(all_shards), len(todo), len(skipped)))
    if args.dry_run:
        for balance, sig, index in todo:
            print("  would run %s" % shard_name(
                args.timeframe, balance, sig, TICKER_FILES[index - 1], args.session))
        return 0
    if not todo:
        print("nothing to do")
        return 0

    os.makedirs(os.path.join(HERE, args.log_dir), exist_ok=True)
    log_dir = os.path.join(HERE, args.log_dir)
    jobs = max(1, min(args.jobs, len(todo)))
    print("running %d shards, %d at a time; logs in %s" % (len(todo), jobs, args.log_dir))

    started = time.time()
    failures, done = [], 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_shard, args.timeframe, balance, sig, index, log_dir,
                        args.session): (balance, sig, index)
            for balance, sig, index in todo
        }
        for future in as_completed(futures):
            name, code, seconds, log_path = future.result()
            done += 1
            status = "ok" if code == 0 else "FAILED (exit %d)" % code
            print("  [%d/%d] %-34s %-18s %6.1fs" % (done, len(todo), name, status, seconds))
            if code != 0:
                failures.append((name, code, log_path))

    # the completeness check: a shard can exit 0 and still not have written everything
    missing = []
    for balance, sig, index in all_shards:
        ticker = TICKER_FILES[index - 1]
        for path in expected_files(args.timeframe, balance, sig, ticker, args.session):
            if not os.path.exists(os.path.join(HERE, path)):
                missing.append(path)
    expected_total = len(all_shards) * len(MACD_PARAMS_LIST) * len(RESULT_SUFFIXES)

    print()
    print("=" * 72)
    print("elapsed %.1f min" % ((time.time() - started) / 60.0))
    print("shards run    : %d, %d failed" % (len(todo), len(failures)))
    print("result files  : expected %d, found %d, missing %d" % (
        expected_total, expected_total - len(missing), len(missing)))
    for name, code, log_path in failures:
        print("  FAILED %s (exit %d) -> %s" % (name, code, log_path))
    for path in missing[:20]:
        print("  MISSING %s" % path)
    if len(missing) > 20:
        print("  ... and %d more" % (len(missing) - 20))
    if failures or missing:
        print("re-run the same command with --resume to fill the gaps")
    else:
        print("complete")
    print("=" * 72)
    return 1 if (failures or missing) else 0


if __name__ == "__main__":
    sys.exit(main())

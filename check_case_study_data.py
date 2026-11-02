# =====================================================================================
# check_case_study_data.py
#
# Acceptance gate for the three published case-study comparisons (Wiles/soybean,
# Kang/nikkei, Aguirre/nasdaq). Run this BEFORE trusting any comparison number, and
# re-run it whenever new data is obtained: it is the check that was never run.
#
# There are TWO kinds of check here and the difference matters:
#   (A) PAPER-DEPENDENT  -- does our data reproduce a number the paper reports?
#       These assume the paper is truthful. If a paper misreported, (A) can fail even
#       though our data is fine, so an (A) failure alone is not proof we are wrong.
#   (B) SELF-CONTAINED   -- is our series internally the instrument we claim it is?
#       Roll gaps, trading past contract expiry, whole-market open interest, bars on
#       days the exchange was shut. These are evidence about OUR data only and hold
#       REGARDLESS of whether any paper lied. A (B) failure is ours to fix.
# =====================================================================================

import pandas as pd

# -------------------------------------------------------------------------------------
# what each paper reports (the expectations we test against)
# -------------------------------------------------------------------------------------
# Wiles & Enke (2015) p.88-89: CME August 2014 soybean contract (SQ2014); 160 test
# points 2013-12-26..2014-08-14; Table 3 buy-and-hold over that window = -25.3 cents.
wiles_test_start, wiles_test_end = "2013-12-26", "2014-08-14"
wiles_test_rows = 160
wiles_buy_and_hold = -25.3

# Kang (2021) JRFM 14(1) 37, p.6/p.9: daily closes of Nikkei 225 Futures near maturity
# from JPX Data Cloud, 2011-01-04..2019-12-30, covering 2204 trading days.
kang_start, kang_end = "2011-01-04", "2019-12-31"
kang_trading_days = 2204
# Japanese national holidays on which the Osaka exchange is CLOSED -- a JPX-sourced
# series cannot have price bars on these dates.
japan_holidays_2011 = [
    "2011-01-10", "2011-02-11", "2011-03-21", "2011-04-29", "2011-05-03",
    "2011-05-04", "2011-05-05", "2011-11-03", "2011-11-23", "2011-12-23",
]

# US market holidays, used to tell an OSAKA-listed contract from a CME-listed one. This
# pair of calendars identifies the exchange on its own: nkf.csv trades through 20/20
# Japanese holidays but only 8/15 US ones, which makes it the CME Nikkei 225 contract
# (NIY/NKD), not the Osaka contract Kang used. Its daily changes also correlate only 0.52
# with the index, because the CME day settles at 16:00 CT, about 06:00 JST the NEXT day --
# a different 24-hour window from the Japanese cash session. Any replacement futures file
# must pass the OSE side of this test before it is used.
us_holidays = [
    "2011-01-17", "2011-05-30", "2011-07-04", "2011-11-24", "2011-12-26",
    "2015-01-19", "2015-05-25", "2015-07-03", "2015-11-26", "2015-12-25",
    "2018-01-15", "2018-05-28", "2018-07-04", "2018-11-22", "2018-12-25",
]

# CME soybean futures trade in quarter-cent ticks, so every genuine settlement price is a
# multiple of 0.25. a series that is not on this grid is not raw CME data, whatever it is
# labelled -- this one check retired two of our candidate files.
soybean_tick = 0.25

# Agudelo Aguirre et al. (2021), JEFAS 26(52): the investment runs 2015-08-11..2017-10-27
# (808 days) starting from USD 5,036,790 for 1,000 units, and buy-and-hold ends at USD
# 6,701,263. Those two anchors are index levels to three decimals, so they identify his
# series exactly -- and ours reproduces both to the cent.
aguirre_anchors = {"2015-08-11": 5036.790, "2017-10-27": 6701.263}

# Agudelo Aguirre et al. (2020), IMFI 17(4), Table 2. These are TRANSACTION prices -- the
# column is "asset quotation at the time of the sell or the buy completion" -- not closes.
# Testing them against our close was the wrong comparison and made good data look wrong;
# the right test is whether each falls inside our day's trading range.
aguirre_quoted_closes = {
    "2015-08-21": 4735.28, "2015-09-24": 4684.09, "2015-10-02": 4673.35,
    "2015-12-02": 5165.48, "2016-09-09": 5177.17, "2018-02-14": 7083.65,
    "2019-02-25": 7581.98,
}


def load_series(path: str) -> pd.DataFrame:
    """
    load_series reads one case-study price file into a date-indexed frame, coping with the two formats on disk (semicolon with "31 Dec 2014" dates, or comma with ISO dates)

    Arguments:
        path -- the data file to read

    Returns:
        dataframe indexed by date and sorted, with lowercase column names
    """
    # four formats live on disk: scraped vendor (semicolon, day-first dates), plain ISO
    # csv with a "date"/"dates" column, metastock (<DATE> as YYYYMMDD), and an
    # investing.com export (quoted numbers with thousands separators, "Price" = close)
    for sep, dayfirst in ((";", True), (",", False)):
        try:
            df = pd.read_csv(path, sep=sep)
        except Exception:
            continue
        df.columns = [str(c).strip().strip('"').lstrip("﻿").lower() for c in df.columns]
        # locate the date column under any of its spellings
        date_col = next((c for c in df.columns if c in ("date", "dates", "<date>")), None)
        if date_col is None:
            continue
        if date_col == "<date>":
            df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
        else:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        # normalise the close column name
        if "close" not in df.columns:
            for alias in ("<close>", "price", "nearby_close"):
                if alias in df.columns:
                    df["close"] = df[alias]
                    break
        if "close" not in df.columns:
            continue
        # investing.com quotes numbers as "1,019.25" strings
        df["close"] = pd.to_numeric(
            df["close"].astype(str).str.replace(",", "").str.replace('"', ""),
            errors="coerce",
        )
        df = df.dropna(subset=["close"])
        # a single-contract file carries its own open interest under a different name
        if "open interest" not in df.columns and "openint" in df.columns:
            df["open interest"] = df["openint"]
        if len(df) > 100:
            return df
    raise ValueError("could not parse " + path)


def report(kind: str, label: str, passed: bool, detail: str):
    # one line per check; kind is "A" (paper-dependent) or "B" (self-contained)
    status = "PASS" if passed else "FAIL"
    print(f"  [{kind}] {status}  {label:38} {detail}")


def check_soybean():
    """
    check_soybean tests every candidate soybean series against Wiles's reported test window and buy-and-hold figure, plus the self-contained signs of a continuous (rolled) series
    """
    print("\n=== SOYBEAN / Wiles (2015) -- expects CME August-2014 contract (SQ2014) ===")
    candidates = [
        # the single August-2014 contract Wiles actually traded -- this is the ONE to use
        # for his comparison: it reproduces his -25.3 buy-and-hold to within rounding
        "./data/soybean/CME_SQ2014.csv",
        # RETIRED: naively-spliced vendor nearby series. carries ~196 cents of phantom roll
        # P&L inside Wiles's window (B&H -221 against his -25.3). kept here only so the
        # gate keeps printing the evidence for why we stopped using it.
        "./data/soybean/soy_data.csv",
        "./data/soybean/soybean_JUL14.txt",
        "./data/soybean/soybean_nearby.txt",
        "./data/soybean/soybeans_future.csv",
    ]
    for path in candidates:
        try:
            df = load_series(path)
        except Exception as exc:
            print(f"\n  {path}: UNREADABLE ({exc})")
            continue
        window = df.loc[wiles_test_start:wiles_test_end]
        print(f"\n  {path}  (span {df.index.min().date()} -> {df.index.max().date()})")
        if len(window) < 2:
            print("    window empty -- file does not cover Wiles's test period")
            continue
        # (A) his reported row count and buy-and-hold
        # his "160 test points" are one-day-ahead forecasts, and 161 closes are what yield
        # 160 of them, so either count is consistent with the paper
        report("A", "test window rows == 160 or 161",
               len(window) in (wiles_test_rows, wiles_test_rows + 1),
               f"got {len(window)} (161 closes -> 160 one-day-ahead points)")
        bh = window["close"].iloc[-1] - window["close"].iloc[0]
        report("A", "buy-and-hold == -25.3 cents", abs(bh - wiles_buy_and_hold) < 5.0,
               f"got {bh:+.2f} (expected {wiles_buy_and_hold})")
        # (B) a single expiring contract cannot trade past expiry and does not carry
        #     whole-market open interest.
        #     NOTE: there used to be a "no one-day move >40c" check here as a proxy for
        #     roll detection. it was dropped because it is a false positive: SQ2014 has two
        #     such moves (2014-05-01 and the 2014-06-30 USDA acreage report) and both are
        #     genuine market-wide events -- every 2014 contract moved together on those
        #     days, and neither falls on a roll date. roll soundness is now tested directly
        #     against the roll log in check_soybean_continuous below, which does not need a
        #     magnitude guess.
        past_expiry = df.index.max() > pd.Timestamp("2014-09-30")
        report("B", "stops trading at Aug-2014 expiry", not past_expiry,
               f"last bar {df.index.max().date()}")
        if "open interest" in df.columns:
            oi = pd.to_numeric(window["open interest"].astype(str).str.replace(",", ""),
                               errors="coerce").median()
            report("B", "open interest is single-contract", bool(oi < 200000),
                   f"median {oi:,.0f}")
        # (B) the tick grid. this is the cheapest and sharpest provenance test we have: the
        #     CME archive, our continuous series and soybean_nearby.txt are all 100% on the
        #     quarter-cent grid, while soy_data.csv (44.3%) and soybeans_future.csv (44.8%)
        #     are not, with remainders of an eighth of a cent. off-grid prices cannot be CME
        #     settlements, so those two files are some other processed product.
        on_grid = (((df["close"] * 100).round() % int(soybean_tick * 100)) == 0).mean()
        report("B", "prices on the CME 0.25c tick grid", on_grid > 0.999,
               f"{100 * on_grid:.1f}% on grid")


def check_soybean_continuous():
    """
    check_soybean_continuous tests our own back-adjusted soybean series for roll soundness directly against its roll log and the per-contract archive it was built from, which needs no magnitude guess about what counts as a roll

    These are all (B) checks: they say whether OUR series is the instrument we claim it is,
    and hold regardless of what any paper reported.
    """
    print("\n=== SOYBEAN continuous / ours -- back-adjusted off the per-contract archive ===")
    try:
        import build_soybean_continuous as builder
        contracts = builder.load_contracts(builder.ARCHIVE_DIR)
        series = pd.read_csv(builder.OUT_SERIES, parse_dates=["Date"]).set_index("Date")
        rolls = pd.read_csv(builder.OUT_ROLLS, parse_dates=["roll_date"])
    except Exception as exc:
        print(f"  not built yet ({exc}) -- run build_soybean_continuous.py first")
        return
    print(f"\n  {builder.OUT_SERIES}  (span {series.index.min().date()} -> {series.index.max().date()}, "
          f"{len(series)} rows, {len(rolls)} rolls)")

    # the central invariant: on every day the series must move exactly as much as the
    # contract we were actually HOLDING coming into that day. any roll gap left in the
    # series shows up here as a day where the two disagree, so this catches phantom P&L
    # without having to guess a threshold.
    violations = 0
    dates = series.index
    for position in range(1, len(dates)):
        today, yesterday = dates[position], dates[position - 1]
        held = series.loc[yesterday, "contract"]
        frame = contracts.get(held)
        if frame is None or today not in frame.index or yesterday not in frame.index:
            violations += 1
            continue
        own_move = float(frame.loc[today, "Close"] - frame.loc[yesterday, "Close"])
        series_move = float(series.loc[today, "Close"] - series.loc[yesterday, "Close"])
        if abs(series_move - own_move) > 1e-9:
            violations += 1
    report("B", "no phantom roll P&L on any day", violations == 0,
           f"{len(dates) - 1 - violations}/{len(dates) - 1} days match the held contract")

    # difference adjustment shifts old prices down, and far enough back they can go
    # negative. that is harmless for how we use the series -- P&L is a price DIFFERENCE, and
    # the whole pipeline is exactly invariant to a constant price shift (measured: all 15
    # indicator columns identical to 0.0, including the CMACD correntropy kernel, and every
    # strategy net identical to the cent). but two things DO read the price level: the
    # engine's `amount = (balance * risk_per_trade) // candle` sizing when balance is
    # non-zero, and any proportional bps cost. soybean runs with balance=0 so it never hits
    # the sizing path, and bps costs must be priced off raw_close rather than the adjusted
    # close anyway. flagging it here so the hazard is visible if either ever changes.
    negative = int((series["Low"] <= 0).sum())
    report("B", "no non-positive adjusted levels", negative == 0,
           f"{negative} bars ({100 * negative / len(series):.1f}%), min low "
           f"{series['Low'].min():.2f} -- harmless while balance=0 and costs use raw_close")

    # the adjusted close must stay on the tick grid: we only ever subtract real gaps, which
    # are themselves differences of on-grid prices
    on_grid = (((series["Close"] * 100).round() % 25) == 0).mean()
    report("B", "adjusted close still on the tick grid", on_grid > 0.999,
           f"{100 * on_grid:.1f}% on grid")

    # every handover must happen while the contract is still alive, i.e. before the month
    # it delivers in -- rolling at or after delivery means we traded a contract in delivery
    late = int((rolls["from_contract"].map(builder.delivery_month) <= rolls["roll_date"]).sum())
    report("B", "every roll precedes its delivery month", late == 0, f"{late} late rolls")

    # NOTE: a "no >40c move lands on a roll date" check was tried here and dropped. it is a
    # false positive for the same reason the magnitude check above was: a roll date can also
    # just be a day the held contract genuinely moved a lot. the two hits it flagged
    # (2005-06-27 and 2022-07-26) both match the held contract's own move exactly, with
    # their gaps (+3.75 and -149.00) fully absorbed, and 2 hits is below the 3.8 that chance
    # predicts from 137 rolls at the series' big-move rate. the exact invariant above
    # already proves what that check was groping at, so it added only noise.

    print(f"        total roll gap removed: {rolls['gap'].sum():+.2f} cents "
          f"(largest single {rolls['gap'].abs().max():.2f})")


def check_nikkei():
    """
    check_nikkei tests the nikkei series against Kang's trading-day count and against the self-contained test that a JPX series cannot trade on Japanese public holidays
    """
    print("\n=== NIKKEI / Kang (2021) -- expects JPX front-month, 2204 trading days ===")
    for path in ["./data/nikkei/nkf.csv", "./data/nikkei/nkx_d.csv"]:
        try:
            df = load_series(path)
        except Exception as exc:
            print(f"\n  {path}: UNREADABLE ({exc})")
            continue
        window = df.loc[kang_start:kang_end]
        print(f"\n  {path}  (span {df.index.min().date()} -> {df.index.max().date()})")
        # (A) his stated trading-day count
        report("A", "trading days == 2204", len(window) == kang_trading_days,
               f"got {len(window)} ({len(window) - kang_trading_days:+d})")
        # (B) bars on days the Osaka exchange is shut prove it is not a JPX series
        present = [h for h in japan_holidays_2011 if pd.Timestamp(h) in window.index]
        report("B", "no bars on Japanese holidays", len(present) == 0,
               f"{len(present)}/{len(japan_holidays_2011)} holidays have bars")

        # (B) the companion test: a series that is shut on Japanese holidays should be OPEN
        #     on US ones. Being open on Japanese holidays and shut on US ones is the
        #     signature of the CME contract rather than the Osaka one.
        us_present = [h for h in us_holidays if pd.Timestamp(h) in df.index]
        report("B", "trades on US holidays (so not CME-listed)",
               len(us_present) > 0.8 * len(us_holidays),
               f"{len(us_present)}/{len(us_holidays)} US holidays have bars")

        # (B) is this a futures file at all, or the cash index? Kang used front-month
        #     futures, so a replacement should carry volume and open interest.
        has_oi = "open interest" in df.columns
        report("B", "carries futures fields (volume / open interest)", has_oi,
               "futures-style file" if has_oi else "index-style file, no open interest")


def check_nasdaq():
    """
    check_nasdaq tests whether our nasdaq series reproduces the closing values Aguirre quotes
    """
    print("\n=== NASDAQ / Aguirre (2020) -- expects the NASDAQ index series he used ===")
    for path in ["./data/nasdaq/ndq.csv", "./data/nasdaq/ndq.txt"]:
        try:
            df = load_series(path)
        except Exception as exc:
            print(f"\n  {path}: UNREADABLE ({exc})")
            continue
        print(f"\n  {path}  (span {df.index.min().date()} -> {df.index.max().date()})")
        # (A) the 2021 study's two anchors are index levels and must match outright
        hits, details = 0, []
        for date, expected in aguirre_anchors.items():
            stamp = pd.Timestamp(date)
            if stamp not in df.index:
                details.append(f"{date}:missing")
                continue
            got = float(df.loc[stamp, "close"])
            if abs(got - expected) < 0.05:
                hits += 1
            else:
                details.append(f"{date}:{got:.2f}vs{expected}")
        report("A", "2021 anchors reproduced exactly", hits == len(aguirre_anchors),
               f"{hits}/{len(aguirre_anchors)} match within 0.05 point")
        if details:
            print(f"        mismatches: {', '.join(details)}")

        # (A) the 2020 study's Table 2 figures are intraday TRANSACTION prices, so the test
        # is range containment, not equality with our close
        if {"high", "low"} <= set(df.columns):
            inside, outside = 0, []
            for date, price in aguirre_quoted_closes.items():
                stamp = pd.Timestamp(date)
                if stamp not in df.index:
                    outside.append(f"{date}:missing")
                    continue
                low, high = float(df.loc[stamp, "low"]), float(df.loc[stamp, "high"])
                if low <= price <= high:
                    inside += 1
                else:
                    outside.append(f"{date}:{price}not in[{low:.0f},{high:.0f}]")
            report("A", "2020 transaction prices within our range",
                   inside == len(aguirre_quoted_closes),
                   f"{inside}/{len(aguirre_quoted_closes)} inside the day's high-low")
            if outside:
                print(f"        outside: {', '.join(outside[:4])}")


def run():
    """
    run executes every case-study data check and prints a PASS/FAIL line per check
    """
    print("ACCEPTANCE GATE for the published case-study comparisons")
    print("  [A] = paper-dependent (assumes the paper is truthful)")
    print("  [B] = self-contained (valid regardless of whether the paper is truthful)")
    check_soybean()
    check_soybean_continuous()
    check_nikkei()
    check_nasdaq()
    print("\nAny [B] failure means OUR data is the wrong instrument -- ours to fix.")
    print("An [A] failure with all [B] passing may mean the paper misreported.")


if __name__ == "__main__":
    run()

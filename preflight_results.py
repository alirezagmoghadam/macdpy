"""PREFLIGHT GATE for an engine results tree.

Run this against ANY results/ tree -- the small review tree, or Monday's full 32-ticker output --
BEFORE trusting a single number out of it.

    python preflight_results.py                      # ./results, every cell it finds
    python preflight_results.py --root ./results --time-value 12 --timeframe 1_day

WHY THIS EXISTS. run_sweep.py's completeness report is EXISTENCE ONLY (os.path.exists), so a
shard that wrote the full skeleton with an empty own-slice counts toward "complete". And the
checks that would catch the two worst silent failures are not in the engine or the table layer:

  * a WINDOW-ORDER permutation between {tick} and {tick}_bets is invisible to everything
    downstream. collect_cell_bets throws the label away and appends in dict order, while
    cell_full_rows addresses windows BY POSITION against the mother file's key list. The
    alignment guard compares LENGTHS, which a permutation preserves.
  * saved bet lists can be silently EMPTY: the engine builds them with
    `current_bet_list_valid.get(key, [])`, a .get with a [] default, so a key that does not
    resolve yields an empty list per window and nothing raises.

THE REPORTED ROW IS 18 WIDE (check_list_maker):
     0-10  the NET winner's long/short metrics  (col 3 = Transactions)
    11-13  the NET winner's long_only_*          ("Long 1";  col 12 = its transactions)
    14-16  the LONG-ONLY winner's long_only_*    ("Long 2";  col 15 = its transactions)
      17   the NET winner's config name
So arm [0] of _bets must account for col 3, and arm [1] for col 15.
"""
import argparse
import io
import json
import os
import sys

OBS_MODES = ["observations_MACDCROSS_naked", "observations_ZEROCROSS_naked"]
ENTRIES = ["Rule 1", "Rule 2", "Rule 3", "GA", "MACD"]
COL_TRANSACTIONS = 3        # long/short arm
COL_LONG2_TRANSACTIONS = 15  # the long-only winner's transactions

fails = []
passes = 0


def check(name, ok, detail=""):
    global passes
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if ok:
        passes += 1
    else:
        fails.append(name)


def load(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_cells(root):
    """every directory holding per-ticker result files"""
    out = []
    for dp, _dirs, files in os.walk(root):
        ticks = sorted(f for f in files if f.endswith(".txt") and not f.endswith("_bets")
                       and not f.endswith("_nets") and not f.endswith("_chosen_rules"))
        if ticks:
            out.append((dp.replace("\\", "/"), ticks))
    return out


BH_CONFIG = "12_26_9"       # the config bh.ipynb hardcodes when it builds its window grid
BH_BALANCE_KEY = "20000"    # regime_balance_key in make_tables_notebooks


def check_bh_grids(root, cells, bh_root=None):
    """
    check_bh_grids that bh.ipynb's window grid lines up with the engine's, per ticker

    The per-regime tables pair bh's windows to the engine's BY POSITION, because a window
    label is derived from each series' own post-dropna first bar and so differs per config and
    per ticker. Position pairing needs two things that nothing else checks:

      * EQUAL COUNTS per ticker. cell_full_rows raises on this, but only once the table build
        is already running; here it is caught before the sweep's output is trusted. Measured on
        the saved hourly run, 31 tickers had 118 windows and one had 117.
      * THE SAME DATA VINTAGE. Equal counts with different dates is the one failure position
        pairing cannot detect, and it is exactly what a stale bh_result looks like. So for
        12_26_9 -- the config bh itself ran -- the labels must be IDENTICAL, not merely equally
        many. Measured on the saved tree, the engine's daily grid started 2021-02-07 while
        bh_result's started 2021-03-26: two vintages, same count of 12.
    """
    bh_path = os.path.join(bh_root or root, "buy_and_hold", "bh_result")
    if not os.path.exists(bh_path):
        # NOT counted as a pass. A skipped vintage check is exactly the state in which stale
        # regime tables get believed, so it has to read as "not checked", not as "fine".
        print("  SKIP  bh grid alignment -- no %s yet. Run bh.ipynb, or point --bh-root at it;"
              % bh_path)
        print("        until then the per-regime tables are UNVERIFIED.")
        return
    try:
        bh = load(bh_path)
    except (IOError, OSError, ValueError) as exc:
        check("bh grid alignment", False, "bh_result unreadable: %s" % exc)
        return

    count_bad, vintage_bad, compared = [], [], 0
    for cell, ticks in cells:
        parts = [p for p in cell.split("/") if p]
        timeframe = next((p for p in parts if p in ("1_day", "1_hour", "5_minute")), None)
        params = parts[parts.index(timeframe) - 1] if timeframe in parts else None
        node = bh.get(BH_BALANCE_KEY, {}).get(timeframe, {})
        if not node:
            continue
        for tick in ticks:
            windows = node.get(tick)
            if not isinstance(windows, dict):
                continue
            bh_labels = sorted(w for w in windows if ":" in w)
            try:
                mother = load(os.path.join(cell, tick))
            except (IOError, OSError, ValueError):
                continue
            tv = list(mother)[0]
            mode = next((m for m in OBS_MODES if m in mother[tv]), None)
            if mode is None:
                continue
            eng_labels = sorted(w for w in mother[tv][mode].get(tick, {}) if ":" in w)
            if not eng_labels or not bh_labels:
                continue
            compared += 1
            if len(eng_labels) != len(bh_labels):
                count_bad.append((cell, tick, len(eng_labels), len(bh_labels)))
            elif params == BH_CONFIG and eng_labels != bh_labels:
                vintage_bad.append((cell, tick, eng_labels[0], bh_labels[0]))

    check("bh grid alignment: equal window counts per ticker", not count_bad,
          "%d of %d (ticker, cell) pairs compared" % (compared - len(count_bad), compared))
    check("bh grid alignment: %s labels identical (same data vintage)" % BH_CONFIG,
          not vintage_bad,
          "no %s cells compared" % BH_CONFIG if compared == 0 else "")
    for cell, tick, a, b in count_bad[:4]:
        print("        COUNT  %s %s: engine %d windows, bh %d" % (cell, tick, a, b))
    for cell, tick, a, b in vintage_bad[:4]:
        print("        VINTAGE %s %s: engine starts %s, bh starts %s -- re-run bh.ipynb"
              % (cell, tick, a, b))


def main():
    ap = argparse.ArgumentParser(description="Preflight gate for an engine results tree.")
    ap.add_argument("--root", default="./results")
    ap.add_argument("--timeframe", default=None, help="only cells under this timeframe")
    ap.add_argument("--bh-root", default=None,
                    help="where buy_and_hold/bh_result lives, if not under --root")
    ap.add_argument("--quiet-cells", action="store_true", help="suppress the per-cell table")
    args = ap.parse_args()

    cells = find_cells(args.root)
    if args.timeframe:
        cells = [c for c in cells if "/%s/" % args.timeframe in c[0] + "/"]
    if not cells:
        print("no result cells found under %s" % args.root)
        return 1

    print("=" * 78)
    print("PREFLIGHT -- %d cells under %s" % (len(cells), args.root))
    print("=" * 78)

    empty_slices = []
    label_mismatch = []
    order_mismatch = []
    tx_arm0_bad = []
    tx_arm1_bad = []
    nets_keyset_bad = []
    arm1_identical = []
    arm1_nonfinite = []
    empty_bet_windows = []
    per_cell = []
    total_rows = 0

    for cell, ticks in cells:
        cell_rows = 0
        for tick in ticks:
            try:
                mother = load(os.path.join(cell, tick))
                bets = load(os.path.join(cell, tick + "_bets"))
                nets = load(os.path.join(cell, tick + "_nets"))
            except (IOError, OSError, ValueError) as exc:
                label_mismatch.append((cell, tick, "unreadable: %s" % exc))
                continue
            tv = list(mother)[0]
            for mode in OBS_MODES:
                m_slice = mother[tv][mode].get(tick, {})
                if not m_slice:
                    empty_slices.append((cell, tick, mode))
                    continue
                b_slice = bets[list(bets)[0]][mode].get(tick, {})
                n_slice = nets[list(nets)[0]][mode].get(tick, {})

                m_labels, b_labels, n_labels = list(m_slice), list(b_slice), list(n_slice)
                if not (set(m_labels) == set(b_labels) == set(n_labels)):
                    label_mismatch.append((cell, tick, mode))
                    continue
                # ORDER, not just membership -- a permutation is what nothing else catches
                if not (m_labels == b_labels == n_labels):
                    order_mismatch.append((cell, tick, mode))

                nets_keysets = set()
                for w in m_labels:
                    rows = m_slice[w][0]
                    cell_rows += len(rows)
                    arms_b = b_slice[w]
                    arms_n = n_slice[w]
                    if len(arms_b) != 2 or len(arms_n) != 2:
                        label_mismatch.append((cell, tick, mode, w, "arm count"))
                        continue
                    nets_keysets.add(frozenset(arms_n[0]))
                    nets_keysets.add(frozenset(arms_n[1]))
                    # arm [1] must not be a copy of arm [0], and must be finite
                    same = sum(1 for k in arms_n[0] if arms_n[0][k] == arms_n[1].get(k))
                    if arms_n[0] and same == len(arms_n[0]):
                        arm1_identical.append((cell, tick, mode, w))
                    for k, v in arms_n[1].items():
                        if not isinstance(v, (int, float)) or v != v or abs(v) == float("inf"):
                            arm1_nonfinite.append((cell, tick, mode, w, k))
                            break
                    for arm_i, arm in enumerate(arms_b):
                        for e in ENTRIES:
                            if e in arm and len(arm[e]) == 0:
                                empty_bet_windows.append((cell, tick, mode, w, arm_i, e))
                # transaction identity, summed over windows, per entry
                for e_i, e in enumerate(ENTRIES):
                    tx0 = sum(len(b_slice[w][0].get(e, [])) for w in m_labels)
                    tx1 = sum(len(b_slice[w][1].get(e, [])) for w in m_labels)
                    row0 = sum(m_slice[w][0][e_i][COL_TRANSACTIONS] for w in m_labels)
                    row1 = sum(m_slice[w][0][e_i][COL_LONG2_TRANSACTIONS] for w in m_labels)
                    if tx0 != row0:
                        tx_arm0_bad.append((cell, tick, mode, e, tx0, row0))
                    if tx1 != row1:
                        tx_arm1_bad.append((cell, tick, mode, e, tx1, row1))
                if len(nets_keysets) > 1:
                    nets_keyset_bad.append((cell, tick, mode, len(nets_keysets)))
        total_rows += cell_rows
        per_cell.append((cell, len(ticks), cell_rows))

    if not args.quiet_cells:
        print()
        for cell, nt, rows in per_cell:
            print("  %-62s %d tickers  %5d rows" % (cell[-62:], nt, rows))

    print()
    print("=" * 78)
    print("RESULTS  (%d reported rows across %d cells)" % (total_rows, len(cells)))
    print("=" * 78)
    check("every shard's OWN ticker slice is populated", not empty_slices,
          "%d empty" % len(empty_slices))
    check("{tick}, _bets and _nets agree on the window label SET", not label_mismatch,
          "%d mismatches" % len(label_mismatch))
    check("...and on the window label ORDER (a permutation is otherwise invisible)",
          not order_mismatch, "%d permuted" % len(order_mismatch))
    check("arm [0] bet counts == mother col 3 (Transactions)", not tx_arm0_bad,
          "%d mismatches" % len(tx_arm0_bad))
    check("arm [1] bet counts == mother col 15 (Long 2 Transactions)", not tx_arm1_bad,
          "%d mismatches" % len(tx_arm1_bad))
    check("_nets key set identical across windows AND arms", not nets_keyset_bad,
          "%d cells differ" % len(nets_keyset_bad))
    check("arm [1] is not a copy of arm [0]", not arm1_identical,
          "%d windows identical" % len(arm1_identical))
    check("arm [1] values are all finite numbers", not arm1_nonfinite,
          "%d bad" % len(arm1_nonfinite))

    # empty bet lists are a WARNING, not a failure: a rule can legitimately not trade
    print()
    if empty_bet_windows:
        by_arm = {0: 0, 1: 0}
        for e in empty_bet_windows:
            by_arm[e[4]] += 1
        print("  NOTE  %d (window, entry) bet lists are EMPTY  [arm0 %d, arm1 %d]"
              % (len(empty_bet_windows), by_arm[0], by_arm[1]))
        print("        Legitimate when a rule took no trade in a window, but it is also what a")
        print("        failed .get(key, []) looks like. Spot-check a few before trusting costs.")
        for e in empty_bet_windows[:4]:
            print("          %s %s arm[%d] %s %s" % (e[1], e[3], e[4], e[5], e[2][13:22]))
    else:
        print("  NOTE  no empty bet lists anywhere.")

    for name, items in [("label/arm mismatches", label_mismatch),
                        ("order permutations", order_mismatch),
                        ("arm0 transaction mismatches", tx_arm0_bad),
                        ("arm1 transaction mismatches", tx_arm1_bad)]:
        for it in items[:4]:
            print("    %s: %s" % (name, it))

    print()
    check_bh_grids(args.root, cells, args.bh_root)

    print()
    print("=" * 78)
    print("PREFLIGHT: %d passed, %d failed" % (passes, len(fails)))
    for f in fails:
        print("   FAILED: %s" % f)
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

"""
ccxt_to_tslab.py

Fetches OHLCV candles from a crypto exchange via CCXT and writes them to a
TSLab-compatible offline CSV data file.

TSLab "CSV file" offline data provider format (per doc.tslab.pro):

    Column order : DATE;TIME;OPEN;HIGH;LOW;CLOSE;VOL;
    Row format   : MM/dd/yyyy;HH:mm;<open>;<high>;<low>;<close>;<volume>;

  - NO header row in the file
  - NO empty lines (TSLab stops reading at the first empty line)
  - ';' column delimiter, '.' decimal separator, no thousands separators
  - trailing ';' at the end of every line
  - time in the file is interpreted by TSLab as UTC
  - minimum supported bar size is 1 minute

Usage (run from VS Code terminal):

    pip install ccxt
    python ccxt_to_tslab.py --symbol BTC/USDT --timeframe 1m \
        --start 2022-01-01 --end 2024-01-01

    # Incremental update: the file already exists, so only candles after its
    # last row are fetched and appended (--start is not needed, --end
    # defaults to "now"):
    python ccxt_to_tslab.py --symbol BTC/USDT --timeframe 1m

    # Several symbols and timeframes in one run (each combination goes into
    # its own auto-named file):
    python ccxt_to_tslab.py --symbol BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1m,1h

    # Only report gaps in already downloaded files (no exchange requests):
    python ccxt_to_tslab.py --symbol BTC/USDT,ETH/USDT --check-gaps

Notes:
  - With several symbols/timeframes, a failure in one (unknown market, network
    error, mismatched file) is reported and the run continues with the next.
    A summary is printed at the end; the exit code is 1 if anything failed.
    All of them share the same --end, so files updated together end on the
    same bar. --out can't be combined with several symbols/timeframes.
  - Without --out the file goes to <data-dir>/<exchange>_<symbol>_<timeframe>.csv,
    e.g. data/binance_btc_usdt_1m.csv. <data-dir> defaults to the 'data'
    folder next to this script, regardless of the current directory.
  - --start/--end are interpreted as UTC. --end is exclusive and defaults to
    the current time (i.e. up to the last closed candle).
  - If --out already exists, the script resumes from the bar after the file's
    last row and appends (--start is ignored). A partially written last line
    (e.g. from a crash) is cut off first. Use --full to re-download from
    --start and overwrite the file instead.
  - A full (non-incremental) download is written to '<out>.tmp' and renamed
    over --out only on success, so a failed run never destroys an existing file.
  - The currently still-forming (unclosed) candle is always dropped, even if
    the exchange returns it -- including it would be a data-integrity bug
    equivalent to look-ahead bias.
  - ccxt OHLCV volume for spot exchanges is base-asset volume (BTC here),
    not quote-asset (USDT) volume.
  - Gaps (missing bars, usually exchange downtime) are listed with their exact
    UTC range, both for newly downloaded data and, with --check-gaps, for
    whole existing files. Gap detection is skipped for the monthly (1M)
    timeframe, whose bar length varies.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone

import ccxt

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_GAPS_PRINTED = 50


def parse_args():
    p = argparse.ArgumentParser(description="CCXT -> TSLab offline CSV OHLCV exporter")
    p.add_argument("--exchange", default="binance", help="ccxt exchange id (default: binance)")
    p.add_argument("--symbol", default="BTC/USDT",
                   help="trading pair or comma-separated list, e.g. BTC/USDT,ETH/USDT (default: BTC/USDT)")
    p.add_argument("--timeframe", default="1m",
                   help="candle timeframe or comma-separated list, e.g. 1m or 1m,1h (default: 1m)")
    p.add_argument("--start", help="UTC start, 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' "
                                   "(required for a new file or with --full)")
    p.add_argument("--end", help="UTC end (exclusive), same format as --start (default: now)")
    p.add_argument("--out", help="output file path (default: <data-dir>/<exchange>_<symbol>_<timeframe>.csv)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="folder for auto-named output files (default: 'data' next to this script)")
    p.add_argument("--full", action="store_true",
                   help="ignore an existing --out file and re-download from --start")
    p.add_argument("--check-gaps", action="store_true",
                   help="don't download anything, only scan existing files and list gaps")
    p.add_argument("--limit", type=int, default=1000, help="candles per request (default: 1000)")
    p.add_argument("--max-retries", type=int, default=5, help="retries per failed request (default: 5)")
    return p.parse_args()


def split_list(value):
    """'a, b,a' -> ['a', 'b'] (order kept, duplicates and blanks dropped)."""
    return list(dict.fromkeys(s.strip() for s in value.split(",") if s.strip()))


def default_out_path(data_dir, exchange_id, symbol, timeframe):
    """<data_dir>/<exchange>_<symbol>_<timeframe>.csv, e.g. binance_btc_usdt_1m.csv.
    Futures symbols like 'BTC/USDT:USDT' become 'btc_usdt_usdt'."""
    sym = re.sub(r"[^a-z0-9]+", "_", symbol.lower()).strip("_")
    # Windows file names are case-insensitive: keep 1M (month) apart from 1m.
    tf = timeframe[:-1] + "mo" if timeframe.endswith("M") else timeframe
    return os.path.join(data_dir, f"{exchange_id}_{sym}_{tf}.csv")


def parse_utc(date_str):
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{date_str}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM")


def timeframe_to_ms(tf, exchange):
    return int(exchange.parse_timeframe(tf) * 1000)


def fmt_num(x):
    """Fixed-point formatting: never scientific notation, no redundant trailing zeros."""
    s = f"{float(x):.8f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def safe_fetch(exchange, symbol, timeframe, since_ms, limit, max_retries):
    for attempt in range(1, max_retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            if attempt == max_retries:
                raise
            wait = min(60, 2 ** attempt)
            sys.stderr.write(
                f"\n[retry {attempt}/{max_retries}] {type(e).__name__}: {e}. Sleeping {wait}s\n"
            )
            time.sleep(wait)


def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit, max_retries):
    """Yield fully-closed candles [ts, o, h, l, c, v] in [since_ms, until_ms)."""
    tf_ms = timeframe_to_ms(timeframe, exchange)
    cursor = since_ms
    last_ts = None
    total = 0

    while cursor < until_ms:
        now_ms = exchange.milliseconds()
        batch = safe_fetch(exchange, symbol, timeframe, cursor, limit, max_retries)
        if not batch:
            break

        for candle in batch:
            ts = candle[0]
            if ts < cursor or ts >= until_ms:
                continue
            if last_ts is not None and ts <= last_ts:
                continue  # duplicate/overlap from the exchange
            if ts + tf_ms > now_ms:
                continue  # candle not closed yet -- never export the forming bar
            last_ts = ts
            total += 1
            yield candle

        next_cursor = batch[-1][0] + tf_ms
        if next_cursor <= cursor:
            sys.stderr.write("\nNo progress from exchange response, stopping early.\n")
            break
        cursor = next_cursor

        sys.stderr.write(f"\r  fetched {total} closed candles, cursor={iso(cursor)}   ")
        sys.stderr.flush()

    sys.stderr.write(f"\nTotal fetched: {total} closed candles.\n")


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")


class GapTracker:
    """Records bars that are missing between consecutive timestamps.
    tf_ms=None disables detection (variable-length bars, e.g. 1M)."""

    def __init__(self, tf_ms, prev_ts=None):
        self.tf_ms = tf_ms
        self.prev = prev_ts
        self.first = None
        self.gaps = []  # (first missing bar ts, last missing bar ts, bars missing)

    def add(self, ts):
        if self.first is None:
            self.first = ts
        if self.tf_ms and self.prev is not None and ts - self.prev > self.tf_ms:
            self.gaps.append((self.prev + self.tf_ms, ts - self.tf_ms, (ts - self.prev) // self.tf_ms - 1))
        self.prev = ts

    def track(self, candles):
        for candle in candles:
            self.add(candle[0])
            yield candle

    @property
    def missing(self):
        return sum(g[2] for g in self.gaps)


def print_gaps(gaps):
    for start, end, n in gaps[:MAX_GAPS_PRINTED]:
        span = iso(start) if start == end else f"{iso(start)} -> {iso(end)}"
        sys.stderr.write(f"    {span:<36} {n:>7} bars missing\n")
    if len(gaps) > MAX_GAPS_PRINTED:
        rest = gaps[MAX_GAPS_PRINTED:]
        sys.stderr.write(f"    ... and {len(rest)} more gaps ({sum(g[2] for g in rest)} bars)\n")


def repair_tail(path, chunk=4096):
    """Cut off a partial last line (no trailing newline). Returns bytes removed."""
    with open(path, "rb+") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return 0
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return 0
        pos = size
        while pos > 0:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            idx = f.read(step).rfind(b"\n")
            if idx != -1:
                cut = pos + idx + 1
                f.truncate(cut)
                return size - cut
        f.truncate(0)
        return size


def read_last_lines(path, n, chunk=4096):
    """Return up to n last lines of a file without reading the whole file."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        data = b""
        while pos > 0 and data.count(b"\n") <= n:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return [line.decode("ascii").rstrip("\r") for line in lines[-n:]]


def parse_row_ts(line):
    """Timestamp (ms, UTC) of a TSLab CSV row 'MM/dd/yyyy;HH:mm;...'."""
    parts = line.split(";")
    try:
        dt = datetime.strptime(f"{parts[0]} {parts[1]}", "%m/%d/%Y %H:%M")
    except (IndexError, ValueError):
        raise ValueError(f"Not a TSLab CSV row: '{line}'")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


class SymbolError(Exception):
    """A problem with one symbol/file; other symbols in the run can still proceed."""


def resume_point(path, timeframe, tf_ms):
    """Timestamp (ms) of the bar right after the last row of an existing file,
    or None if the file is empty. Raises SymbolError if the file's bar spacing
    doesn't match the requested timeframe (e.g. appending 1h bars to a 1m file)."""
    removed = repair_tail(path)
    if removed:
        sys.stderr.write(f"Removed partial last line ({removed} bytes) from {path}\n")

    lines = read_last_lines(path, 10)
    if not lines:
        return None
    if any(not line.strip() for line in lines):
        raise SymbolError(f"{path} has empty lines at the end; TSLab stops reading there. Fix the file first.")

    try:
        stamps = [parse_row_ts(line) for line in lines]
    except ValueError as e:
        raise SymbolError(f"{path}: {e}")
    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    if any(d <= 0 for d in deltas):
        raise SymbolError(f"{path}: last rows are not in ascending time order.")
    # Variable-length months can't be checked this way.
    if deltas and not timeframe.endswith("M") and min(deltas) != tf_ms:
        raise SymbolError(
            f"{path}: bar spacing in the file ({min(deltas) // 60_000} min) doesn't match "
            f"--timeframe {timeframe}. Wrong file or timeframe?"
        )
    return stamps[-1] + tf_ms


def write_tslab_csv(candles, out_path, mode="w"):
    written = 0
    with open(out_path, mode, newline="") as f:
        for ts, o, h, l, c, v in candles:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            f.write(
                f"{dt.strftime('%m/%d/%Y')};{dt.strftime('%H:%M')};"
                f"{fmt_num(o)};{fmt_num(h)};{fmt_num(l)};{fmt_num(c)};{fmt_num(v)};\n"
            )
            written += 1
    return written


def new_result(symbol, timeframe, out_path, status="ok", note=""):
    return {"symbol": symbol, "timeframe": timeframe, "out": out_path,
            "rows": 0, "status": status, "note": note}


def process_symbol(exchange, symbol, timeframe, out_path, args, until_ms):
    """Download/append one symbol+timeframe. Returns a result dict for the summary."""
    result = new_result(symbol, timeframe, out_path)
    tf_ms = timeframe_to_ms(timeframe, exchange)

    since_ms = None
    append = os.path.isfile(out_path) and not args.full
    if append:
        since_ms = resume_point(out_path, timeframe, tf_ms)
        if since_ms is None:
            append = False  # existing but empty file -- treat as new
        elif args.start:
            sys.stderr.write(f"{out_path} exists: resuming from its last row, --start ignored.\n")

    if since_ms is None:
        if not args.start:
            raise SymbolError("file doesn't exist yet (or --full): --start is required")
        since_ms = int(parse_utc(args.start).timestamp() * 1000)
        if until_ms <= since_ms:
            raise SymbolError("--end must be after --start")

    sys.stderr.write(
        f"Exchange={args.exchange}  Symbol={symbol}  Timeframe={timeframe}\n"
        f"Range (UTC, end exclusive): {iso(since_ms)} -> {iso(until_ms)}\n"
        f"Output: {out_path} ({'append' if append else 'overwrite'})\n\n"
    )

    if until_ms - since_ms < tf_ms:
        sys.stderr.write("Already up to date, nothing to fetch.\n")
        result["status"] = "up to date"
        return result

    # When appending, the file's last bar is the reference point, so a hole
    # right at the seam is reported as a gap too.
    tracker = GapTracker(None if timeframe.endswith("M") else tf_ms,
                         prev_ts=since_ms - tf_ms if append else None)
    candles = tracker.track(fetch_ohlcv_range(
        exchange, symbol, timeframe, since_ms, until_ms, args.limit, args.max_retries
    ))
    if append:
        # Rows already appended before a failure are valid; the next run resumes after them.
        written = write_tslab_csv(candles, out_path, mode="a")
    else:
        tmp_path = out_path + ".tmp"
        try:
            written = write_tslab_csv(candles, tmp_path)
            if written:  # don't create an empty file / wipe an existing one with --full
                os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    result["rows"] = written
    sys.stderr.write(f"\nDone. {written} rows {'appended to' if append else 'written to'} {out_path}\n")

    notes = []
    now_ms = exchange.milliseconds()
    if written == 0:
        result["status"] = "warning"
        notes.append("exchange returned no closed candles for this range")
    else:
        if not append and tracker.first > since_ms:
            # Usually just the listing date of the pair -- informational only.
            notes.append(f"data starts at {iso(tracker.first)}")
        if tracker.gaps:
            result["status"] = "warning"
            notes.append(f"{len(tracker.gaps)} gaps, {tracker.missing} bars missing")
            sys.stderr.write(f"WARNING: {len(tracker.gaps)} gaps in the downloaded data (UTC):\n")
            print_gaps(tracker.gaps)
        # The bar after the last one should have existed (inside the range and already closed).
        last = tracker.prev
        if last + tf_ms < until_ms and last + 2 * tf_ms <= now_ms:
            result["status"] = "warning"
            notes.append(f"no data after {iso(last)}")
    for note in notes:
        sys.stderr.write(f"Note: {note}\n")
    result["note"] = "; ".join(notes)
    return result


def scan_file(path, tf_ms):
    """Read a whole TSLab CSV and return (rows, GapTracker). Raises SymbolError
    on malformed rows, empty lines or rows out of time order."""
    tracker = GapTracker(tf_ms)
    day_ms = {}  # 'MM/dd/yyyy' -> midnight UTC in ms; strptime per row is too slow for millions of rows
    rows = 0
    with open(path, "r", newline="") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                raise SymbolError(f"line {n} is empty; TSLab stops reading there")
            d = line[:10]
            try:
                base = day_ms.get(d)
                if base is None:
                    base = day_ms[d] = int(datetime(int(d[6:10]), int(d[0:2]), int(d[3:5]),
                                                    tzinfo=timezone.utc).timestamp() * 1000)
                if line[10] != ";" or line[13] != ":" or line[16] != ";":
                    raise ValueError
                ts = base + int(line[11:13]) * 3_600_000 + int(line[14:16]) * 60_000
            except (ValueError, IndexError):
                raise SymbolError(f"line {n} is not a TSLab CSV row: '{line.rstrip()}'")
            if tracker.prev is not None and ts <= tracker.prev:
                raise SymbolError(f"line {n} ({iso(ts)}) is not after the previous row ({iso(tracker.prev)})")
            tracker.add(ts)
            rows += 1
    return rows, tracker


def check_file(symbol, timeframe, out_path, exchange):
    """--check-gaps mode: report gaps in an existing file without downloading anything."""
    result = new_result(symbol, timeframe, out_path)
    if not os.path.isfile(out_path):
        raise SymbolError("file not found")
    tf_ms = None if timeframe.endswith("M") else timeframe_to_ms(timeframe, exchange)

    sys.stderr.write(f"Checking {out_path} ...\n")
    rows, tracker = scan_file(out_path, tf_ms)
    if rows == 0:
        result["status"] = "warning"
        result["note"] = "file is empty"
        sys.stderr.write("  file is empty\n")
        return result

    sys.stderr.write(f"  {rows} rows, {iso(tracker.first)} -> {iso(tracker.prev)} (UTC)\n")
    if tf_ms is None:
        result["note"] = "gap check not supported for monthly bars"
    elif tracker.gaps:
        result["status"] = "gaps"
        result["note"] = f"{len(tracker.gaps)} gaps, {tracker.missing} bars missing"
        sys.stderr.write(f"  {len(tracker.gaps)} gaps, {tracker.missing} bars missing:\n")
        print_gaps(tracker.gaps)
    else:
        result["note"] = "no gaps"
        sys.stderr.write("  no gaps\n")
    return result


def print_summary(results):
    ws = max(len(r["symbol"]) for r in results)
    wt = max(len(r["timeframe"]) for r in results)
    sys.stderr.write("\n" + "=" * 60 + "\nSummary:\n")
    for r in results:
        rows = f"+{r['rows']}" if r["rows"] else "-"
        note = f"  {r['note']}" if r["note"] else ""
        sys.stderr.write(
            f"  {r['symbol']:<{ws}}  {r['timeframe']:<{wt}}  {r['status']:<10}  {rows:>9}  "
            f"{os.path.basename(r['out'])}{note}\n"
        )


def main():
    args = parse_args()

    symbols = split_list(args.symbol)
    timeframes = split_list(args.timeframe)
    if not symbols or not timeframes:
        sys.exit("--symbol and --timeframe must not be empty")
    jobs = [(s, tf) for s in symbols for tf in timeframes]
    if args.out and len(jobs) > 1:
        sys.exit("--out can only be used with a single --symbol and --timeframe; "
                 "omit it to auto-name files in --data-dir")

    exchange_class = getattr(ccxt, args.exchange, None)
    if exchange_class is None:
        sys.exit(f"Unknown ccxt exchange id: {args.exchange}")
    exchange = exchange_class({"enableRateLimit": True})

    if exchange.timeframes:
        unsupported = [tf for tf in timeframes if tf not in exchange.timeframes]
        if unsupported:
            sys.exit(
                f"{args.exchange} does not support timeframe(s) {', '.join(unsupported)}. "
                f"Available: {sorted(exchange.timeframes.keys())}"
            )

    if not args.check_gaps:
        try:
            exchange.load_markets()
        except ccxt.BaseError as e:
            sys.exit(f"Cannot load markets from {args.exchange}: {type(e).__name__}: {e}")
        if not args.out:
            os.makedirs(args.data_dir, exist_ok=True)

    # One shared end for all jobs, so files updated together end on the same bar.
    until_ms = int(parse_utc(args.end).timestamp() * 1000) if args.end else exchange.milliseconds()

    results = []
    for i, (symbol, timeframe) in enumerate(jobs, 1):
        if len(jobs) > 1:
            sys.stderr.write(f"\n[{i}/{len(jobs)}] {symbol} {timeframe}\n")
        out_path = args.out or default_out_path(args.data_dir, args.exchange, symbol, timeframe)
        try:
            if args.check_gaps:
                results.append(check_file(symbol, timeframe, out_path, exchange))
            elif symbol not in exchange.markets:
                raise SymbolError(f"no such market on {args.exchange}")
            else:
                results.append(process_symbol(exchange, symbol, timeframe, out_path, args, until_ms))
        except (SymbolError, ccxt.BaseError, OSError) as e:
            note = str(e) if isinstance(e, SymbolError) else f"{type(e).__name__}: {e}"
            results.append(new_result(symbol, timeframe, out_path, "FAILED", note))
            sys.stderr.write(f"\nERROR: {symbol} {timeframe}: {note}\n")

    if len(jobs) > 1:
        print_summary(results)
    if any(r["status"] == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

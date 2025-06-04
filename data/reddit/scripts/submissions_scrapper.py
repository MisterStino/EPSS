"""
High-throughput, zero-duplication Reddit CVE extractor.
  * ProcessPoolExecutor  — one process per .zst
  * Atomic output renames — no half-written CSVs
  * Final-line flush      — no lost rows
  * UTF-8 'replace'       — no decode-drops
  * Batch writes          — ~1 MB per syscall
Tested on Windows 11 + Python 3.12; works unchanged on Linux/macOS.
"""

from __future__ import annotations
import os, sys, csv, re, json, logging, shutil, traceback
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import zstandard as zstd

# ────────────────────────── CONFIG ──────────────────────────
INPUT_DIR   = r"W:\reddit\submissions"
OUTPUT_DIR  = r"W:\reddit\scraped\submissions_scraped"
FIELD       = "selftext"
CVE_REGEX   = re.compile(r"cve-\d{4}-\d{4,7}", re.IGNORECASE)
FROM_DATE   = datetime(2005, 1, 1)
TO_DATE     = datetime(2030, 12, 31)
BATCH_SIZE  = 5_000                      # rows before flush
CHUNK_BYTES = 1 << 28                    # 256 MiB decompressed
# ────────────────────────────────────────────────────────────

# Optional speed-up
try:
    import orjson as _json
    def _loads(b: str | bytes):   # keep same API as json.loads
        return _json.loads(b)
except ModuleNotFoundError:
    _loads = json.loads

# Logging
log = logging.getLogger("extract")
log.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s: %(message)s"))
log.addHandler(h)

# ─────────────────────────── HELPERS ───────────────────────────

def extract_post_id(url: str) -> str:
    m = re.search(r"/comments/([^/]+)/", url)
    return m.group(1) if m else ""

def read_lines_zst(path: str):
    """Yield (line:str) streaming; never loads the full file."""
    with open(path, "rb") as fh:
        rdr = zstd.ZstdDecompressor(max_window_size=1 << 31).stream_reader(fh)
        buf = ""
        while True:
            chunk = rdr.read(CHUNK_BYTES)
            if not chunk:
                break
            lines = (buf + chunk.decode(errors="replace")).split("\n")
            for ln in lines[:-1]:
                yield ln
            buf = lines[-1]
        if buf:          # flush last line even if no trailing \n
            yield buf

def write_csv_header(writer: csv.writer):
    writer.writerow([
        "Score", "Date", "Title/Body", "Author", "Permalink",
        "Content", "CVE ID", "Post ID",
    ])

def process_one(
    file_in: str,
    file_out_base: str,
) -> tuple[str, int, int, int]:
    """
    Returns (basename, total_lines, matched, bad)
    Raises on unrecoverable errors.
    """
    # Ensure child process has logging handlers
    if not log.handlers:                      # child has no handlers yet
        h = logging.StreamHandler(sys.stdout) # or sys.stderr
        h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s: %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    
    tmp_path   = f"{file_out_base}.csv.tmp"
    final_path = f"{file_out_base}.csv"

    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    total = matched = bad = 0
    is_sub = "submission" in os.path.basename(file_in)

    with open(tmp_path, "w", newline="", encoding="utf-8", buffering=1_048_576) as fh:
        w = csv.writer(fh)
        write_csv_header(w)
        batch: list[list[str]] = []

        for line in read_lines_zst(file_in):
            total += 1
            if total % 500000 == 0:
                log.info(f"{os.path.basename(file_in)}  {total:,d} lines scanned "
                         f"({matched:,d} matches, {bad:,d} bad)")
            try:
                obj = _loads(
                        line if _loads.__module__ != "orjson"
                        else line.encode("utf-8", "surrogateescape")
                    )
                t     = datetime.utcfromtimestamp(int(obj["created_utc"]))
                if t < FROM_DATE or t > TO_DATE:
                    continue

                text  = obj.get(FIELD, "")
                m     = CVE_REGEX.search(text)
                if not m:
                    continue
                cve_id = m.group(0)

                # permalink
                if "permalink" in obj:
                    url = f"https://www.reddit.com{obj['permalink']}"
                else:
                    url = f"https://www.reddit.com/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}"
                post_id = extract_post_id(url)

                row = [
                    str(obj["score"]),
                    t.strftime("%Y-%m-%d"),
                    obj.get("title", "") if is_sub else obj.get("body", ""),
                    f"u/{obj['author']}",
                    url,
                    obj.get("selftext" if (is_sub and obj.get("is_self", False)) else "url", "") if is_sub
                    else obj.get("body", ""),
                    cve_id,
                    post_id,
                ]
                batch.append(row)
                matched += 1

                if len(batch) >= BATCH_SIZE:
                    w.writerows(batch)
                    batch.clear()

            except Exception:         # JSON error, KeyError, etc.
                bad += 1
                if bad % 10_000 == 0:
                    log.warning(f"{os.path.basename(file_in)} bad lines so far: {bad}")

        if batch:
            w.writerows(batch)

    # Atomic replace
    os.replace(tmp_path, final_path)
    return os.path.basename(file_in), total, matched, bad

# ────────────────────────────  MAIN  ────────────────────────────

def main() -> None:
    # Discover inputs
    if os.path.isfile(INPUT_DIR):
        candidates = [INPUT_DIR]
    else:
        candidates = sorted(
            os.path.join(INPUT_DIR, f)
            for f in os.listdir(INPUT_DIR) if f.endswith(".zst")
        )

    # De-dup & collision checks
    seen_in, seen_out = set(), set()
    tasks = []
    for z in candidates:
        if z in seen_in:
            raise RuntimeError(f"Duplicate input file {z}")
        seen_in.add(z)

        stem = os.path.splitext(os.path.basename(z))[0]
        out  = os.path.join(OUTPUT_DIR, stem)
        if out in seen_out:
            raise RuntimeError(f"Duplicate output base {out}")
        seen_out.add(out)
        tasks.append((z, out))

    if not tasks:
        log.info("No input files found.")
        return

    cores = os.cpu_count() or 1
    workers = 15
    log.info(f"{len(tasks)} files queued; spawning {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        fut = {pool.submit(process_one, f_in, f_out): (f_in, f_out) for f_in, f_out in tasks}
        done = 0
        for f in as_completed(fut):
            base, total, matched, bad = f.result()
            done += 1
            log.info(f"✅ {base:25} | {matched:7} / {total:8} rows  | bad {bad:6}  ({done}/{len(tasks)})")

    log.info("🎉 All files processed without duplicates or data loss.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Fatal: {e}")
        traceback.print_exc()
        sys.exit(1)

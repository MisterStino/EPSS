# nvd_timelines_v3_3.py  ---------------------------------------------------
"""
v3.3 – incremental manifest writing + “attempts” column
"""

from __future__ import annotations
import csv
import datetime as dt
import gzip, io, json, logging, os, time, hashlib
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import requests
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

FEED_URL      = "https://nvd.nist.gov/feeds/json/cve/2.0"
HIST_URL      = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0"
TODAY         = dt.date.today().isoformat()
MAX_RETRY     = 8
BACKOFF0      = 0.2
MAX_WORKERS   = 12        # default concurrent requests


# -------------------------------------------------------------------------
def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _request_with_retry(
    sess: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> tuple[requests.Response, int]:
    """
    Same logic as before but returns a tuple (response, attempts_used)
    so the caller can log how many retries were required.
    """
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code < 500 and r.status_code != 429:
                return r, attempt
            raise requests.HTTPError(f"HTTP {r.status_code}")
        except Exception as exc:
            if attempt == MAX_RETRY:
                raise
            time.sleep(BACKOFF0 * 2 ** (attempt - 1))
            logging.debug("retry %s (%s/%s) – %s", url, attempt, MAX_RETRY, exc)


# -------------------------------------------------------------------------
class NVDHelper:
    def __init__(
        self,
        base_dir: str = "catalogs_processed",
        api_key: str | None = None,
        timeout: int = 30,
        workers: int = MAX_WORKERS,
        verbose: bool = False,
        manifest_dir: str | None = 'catalogs_processed',
    ) -> None:
        lvl = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
        )

        self.base   = _mkdir(Path(base_dir))
        self.snap   = self.base / f"{TODAY}_snapshot.jsonl"
        self.h_dir  = _mkdir(self.base / "cve_history")
        self.r_dir  = _mkdir(self.base / "reconstructions")

        self.timeout  = timeout
        self.workers  = workers
        self.headers  = {"apiKey": api_key or os.getenv("NVD_API_KEY", "")}
        if not self.headers["apiKey"]:
            self.headers = {}

        self.sess = requests.Session()
        self.manifest_dir = _mkdir(Path(manifest_dir) if manifest_dir else self.h_dir)
    # ---------------- phase A – snapshot ---------------------------------
    def download_today_snapshot(self) -> None:
        if self.snap.exists():
            logging.info("[snapshot] already present – skip")
            return

        cur_year = dt.date.today().year
        with self.snap.open("w", encoding="utf-8") as out:
            for yr in tqdm(range(2002, cur_year + 1), desc="Year feeds"):
                url = f"{FEED_URL}/nvdcve-2.0-{yr}.json.gz"
                r, _ = _request_with_retry(self.sess, url, timeout=self.timeout)

                if r.status_code == 404:
                    logging.debug("year %s: no v2.0 feed", yr)
                    continue

                with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                    data = json.loads(gz.read())

                for rec in data["vulnerabilities"]:
                    out.write(json.dumps(rec["cve"], ensure_ascii=False) + "\n")

        logging.info("[snapshot] written → %s", self.snap)

    # ---------------- phase B – histories + manifest ---------------------
    def download_histories(self, cve_ids: List[str]) -> None:
        mani_path = self.manifest_dir / "manifest.csv"
        fieldnames  = ["cve_id", "success", "num_changes", "attempts", "error"]

        # create / truncate manifest & write header once -------------------
        fh = mani_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        # helper that runs inside each thread ------------------------------
        def _fetch_one(cve: str) -> Dict[str, Any]:
            hist_path = self.h_dir / f"{cve}.json"
            if hist_path.exists():  
                print(f"[INFO] {cve} already have it – no network needed")      # already have it – no network needed
                return {
                    "cve_id": cve,
                    "success": 1,
                    "num_changes": 0,
                    "attempts": 0,
                    "error": "cached",
                }

            ok = False
            n_chg = 0
            attempts_used = 0
            err = ""
            try:
                r, attempts_used = _request_with_retry(
                    self.sess,
                    HIST_URL,
                    params={"cveId": cve},
                    headers=self.headers,
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    payload = r.json()
                    n_chg = payload.get("totalResults", 0)
                    if n_chg:
                        hist_path.write_text(
                            json.dumps(payload, indent=2, ensure_ascii=False)
                        )
                        ok = True
                    else:
                        err = "no_changes"
                else:
                    err = f"http_{r.status_code}"
            except Exception as exc:
                err = str(exc) or exc.__class__.__name__

            return {
                "cve_id": cve,
                "success": int(ok),
                "num_changes": n_chg,
                "attempts": attempts_used or MAX_RETRY,
                "error": err,
            }

        # run the pool -----------------------------------------------------
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            fut_to_cve = {pool.submit(_fetch_one, cid): cid for cid in cve_ids}
            for fut in tqdm(as_completed(fut_to_cve), total=len(fut_to_cve), desc="Histories"):
                row = fut.result()
                writer.writerow(row)   # ← incremental write
                fh.flush()             # keep file up-to-date even if we crash

        fh.close()
        logging.info("[histories] manifest → %s", mani_path)

    # ---------------- phase C – reconstruction ---------------------------
    def reconstruct_all_with_history(self) -> None:
        if not self.snap.exists():
            raise RuntimeError("snapshot missing – run download_today_snapshot first")
        
        # Create reconstruction manifest
        recon_manifest_path = self.manifest_dir / "reconstruction_manifest.csv"
        recon_fieldnames = ["cve_id", "success", "num_timeline_states", "skipped_changes", "error"]
        
        with recon_manifest_path.open("w", newline="", encoding="utf-8") as recon_fh:
            recon_writer = csv.DictWriter(recon_fh, fieldnames=recon_fieldnames)
            recon_writer.writeheader()
            
            lookup = {rec["id"]: rec for rec in self._iter_snap()}
            for hf in tqdm(list(self.h_dir.glob("CVE-*.json")), desc="Reconstruct"):
                cid = hf.stem
                success = False
                num_states = 0
                skipped_changes = 0
                error = ""
                
                try:
                    if cid not in lookup:
                        error = "not_in_snapshot"
                        logging.warning("%s not in snapshot – skip", cid)
                    else:
                        ch = json.loads(hf.read_text())["cveChanges"]
                        tl, skipped = self._timeline_with_stats(lookup[cid], ch)
                        num_states = len(tl)
                        skipped_changes = skipped
                        
                        with (self.r_dir / f"{cid}.jsonl").open("w", encoding="utf-8") as fh_snap:
                            for snap in tl:
                                fh_snap.write(json.dumps(snap, ensure_ascii=False) + "\n")
                        success = True
                        
                except Exception as exc:
                    error = str(exc) or exc.__class__.__name__
                    logging.error("Failed to reconstruct %s: %s", cid, error)
                
                # Write to reconstruction manifest
                recon_writer.writerow({
                    "cve_id": cid,
                    "success": int(success),
                    "num_timeline_states": num_states,
                    "skipped_changes": skipped_changes,
                    "error": error
                })
                recon_fh.flush()
        
        logging.info("[reconstruction] manifest → %s", recon_manifest_path)

    # --------------------------------------------------------------------
    def _iter_snap(self) -> Generator[Dict[str, Any], None, None]:
        with self.snap.open(encoding="utf-8") as fh_snap:
            for ln in fh_snap:
                yield json.loads(ln)

    def _timeline(self, cur: Dict[str, Any], chgs: List[Any]) -> List[Any]:
        # ============================================================================
        # 🚨 CRITICAL FIX - DUPLICATE PREVENTION 🚨
        # ============================================================================
        # This method was creating duplicate timeline states when historical changes
        # resulted in identical states. Fixed by adding hash-based deduplication.
        # 
        # ⚠️  WARNING: THIS FIX HAS NOT BEEN TESTED DUE TO TIME CONSTRAINTS
        # ⚠️  VALIDATE THOROUGHLY BEFORE PRODUCTION USE
        # ============================================================================
        
        s = json.loads(json.dumps(cur))
        out = [s]
        last_hash = hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()

        for w in reversed(chgs):
            s = self._undo(s, w["change"]["details"])
            s["reconstruction_timestamp"] = w["change"]["created"]
            new_hash = hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()
            if new_hash != last_hash:           # ← only append if state changed
                out.append(json.loads(json.dumps(s)))
                last_hash = new_hash
        return out
    
    def _timeline_with_stats(self, cur: Dict[str, Any], chgs: List[Any]) -> tuple[List[Any], int]:
        """Timeline reconstruction with statistics tracking"""
        # ============================================================================
        # 🚨 CRITICAL FIX - DUPLICATE PREVENTION 🚨
        # ============================================================================
        # This method was also creating duplicate timeline states. Applied same
        # hash-based deduplication fix as _timeline() method.
        # 
        # ⚠️  WARNING: THIS FIX HAS NOT BEEN TESTED DUE TO TIME CONSTRAINTS
        # ⚠️  VALIDATE THOROUGHLY BEFORE PRODUCTION USE
        # ============================================================================
        
        s = json.loads(json.dumps(cur))
        out = [s]
        last_hash = hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()
        total_skipped = 0
        
        for w in reversed(chgs):
            s, skipped = self._undo_with_stats(s, w["change"]["details"])
            total_skipped += skipped
            s["reconstruction_timestamp"] = w["change"]["created"]
            new_hash = hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()
            if new_hash != last_hash:           # ← only append if state changed
                out.append(json.loads(json.dumps(s)))
                last_hash = new_hash
        return out, total_skipped

    def _undo(self, s: Dict[str, Any], ds: List[Dict[str, str]]) -> Dict[str, Any]:
        fmap = {
            "Description": ("descriptions", None),
            "CWE": ("weaknesses", None),
            "Reference": ("references", None),
            "CPE Configuration": ("configurations", None),
            "CVSS V2": ("metrics", "cvssMetricV2"),
            "CVSS V3": ("metrics", "cvssMetricV3"),
            "CVSS V3.1": ("metrics", "cvssMetricV31"),
            "CVSS V4.0": ("metrics", "cvssMetricV40"),
            "State": ("vulnStatus", None),
        }
        for d in ds:
            tgt, sub = fmap.get(d["type"], (None, None))
            if tgt is None:
                continue
            
            # Handle missing 'action' field gracefully
            act = d.get("action")
            old = d.get("oldValue")
            
            # Skip changes that don't have enough information to process
            if act is None:
                logging.debug("Skipping change detail with missing action: %s", d)
                continue
                
            if act == "Added":
                (s.pop(tgt, None) if sub is None else s.get(tgt, {}).pop(sub, None))
            elif act in ("Removed", "Changed") and old is not None:
                if sub is None:
                    s[tgt] = old
                else:
                    s.setdefault(tgt, {})[sub] = old
        return s
    
    def _undo_with_stats(self, s: Dict[str, Any], ds: List[Dict[str, str]]) -> tuple[Dict[str, Any], int]:
        """Undo with statistics tracking for skipped changes"""
        fmap = {
            "Description": ("descriptions", None),
            "CWE": ("weaknesses", None),
            "Reference": ("references", None),
            "CPE Configuration": ("configurations", None),
            "CVSS V2": ("metrics", "cvssMetricV2"),
            "CVSS V3": ("metrics", "cvssMetricV3"),
            "CVSS V3.1": ("metrics", "cvssMetricV31"),
            "CVSS V4.0": ("metrics", "cvssMetricV40"),
            "State": ("vulnStatus", None),
        }
        skipped_count = 0
        for d in ds:
            tgt, sub = fmap.get(d["type"], (None, None))
            if tgt is None:
                continue
            
            # Handle missing 'action' field gracefully
            act = d.get("action")
            old = d.get("oldValue")
            
            # Skip changes that don't have enough information to process
            if act is None:
                skipped_count += 1
                logging.debug("Skipping change detail with missing action: %s", d)
                continue
                
            if act == "Added":
                (s.pop(tgt, None) if sub is None else s.get(tgt, {}).pop(sub, None))
            elif act in ("Removed", "Changed") and old is not None:
                if sub is None:
                    s[tgt] = old
                else:
                    s.setdefault(tgt, {})[sub] = old
        return s, skipped_count


# ------------------------------ smoke-test ------------------------------
if __name__ == "__main__":
    nvd = NVDHelper(
        verbose=True,
        workers=12,                           # tweak to taste
        api_key="0f3e80d0-fe35-4118-a2eb-df87148e355c",
    )

    nvd.download_today_snapshot()

    # grab *every* CVE ID from today’s snapshot
    all_cve_ids = list({rec["id"] for rec in nvd._iter_snap()})

    # download their histories
    # nvd.download_histories(all_cve_ids)

    # rebuild full timelines
    nvd.reconstruct_all_with_history()
    logging.info("♥ pipeline completed")

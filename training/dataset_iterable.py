# SPDX-License-Identifier: MIT
"""
Streaming IterableDataset + per-batch pad/mask for the EPSS LSTM.

Reads `work/epss_stage1.arrow` (written by 00_build_arrow.py)
and yields one variable-length CVE sequence at a time.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator, List, Tuple
import re, json, joblib

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence

import pyarrow as pa
import pyarrow.ipc as ipc


# ───────────────────────────── helper ──────────────────────────────
def _build_y_and_mh(eps: torch.Tensor, horizon: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicates legacy rolling-window target construction."""
    T = len(eps)
    y  = torch.zeros(T, horizon, dtype=torch.float32)
    mh = torch.zeros_like(y)
    for t in range(T):
        k = min(horizon, T - t - 1)
        if k:
            y[t, :k]  = eps[t + 1 : t + 1 + k]
            mh[t, :k] = 1
    return y, mh


# ───────────────────────────── dataset ─────────────────────────────
class CVEIterableDataset(IterableDataset):
    """
    Streams CVE sequences from the Arrow file with **no padding**.
    """

    BOOL_REGEX = re.compile(r'^(has_|is_)')

    def __init__(self,
                 arrow_path: str | Path,
                 horizon: int = 30):
        super().__init__()
        self.arrow_path = Path(arrow_path)
        self.horizon    = horizon

        self._derive_column_groups()  # sets self.num_cols, etc.

        # Position → name mapping for fast row access
        schema = self._schema
        self.col2idx = {field.name: i for i, field in enumerate(schema)}

    # -----------------------------------------------------------------
    def _derive_column_groups(self) -> None:
        """Read schema + vocab to assemble the four feature groups."""

        self._schema = ipc.open_file(pa.memory_map(str(self.arrow_path), 'r')).schema
        names        = [f.name for f in self._schema]

        # categorical list is truth-source → vocab.json
        vocab = json.load(open(self.arrow_path.parent / "vocab.json"))
        self.cat_cols: List[str] = list(vocab.keys())

        # boolean: has_* / is_*  but not categoricals
        self.bool_cols = [n for n in names
                          if self.BOOL_REGEX.match(n) and n not in self.cat_cols]

        # flags are fixed names
        self.flag_cols = ["flag_train", "flag_val", "flag_test"]

        # numeric  = every numeric column not in the other three sets
        reserved = set(self.cat_cols + self.bool_cols + self.flag_cols
                       + ["cve", "date"])
        num_cols = []
        for f in self._schema:
            if f.name in reserved:        # skip reserved
                continue
            if pa.types.is_integer(f.type) or pa.types.is_floating(f.type):
                num_cols.append(f.name)
        self.num_cols = num_cols            # ← includes the *_missing columns

        # quick sanity
        assert len(self.num_cols) == 32,   "Expected 32 numeric (16+16) columns"
        assert len(self.bool_cols) == 2,   "Expected 2 boolean event columns"
        assert len(self.cat_cols) == 6,    "Expected 6 categorical columns"

    # -----------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        worker = get_worker_info()
        w_id, w_n = (worker.id, worker.num_workers) if worker else (0, 1)

        reader: ipc.RecordBatchFileReader = ipc.open_file(
            pa.memory_map(str(self.arrow_path), 'r'))

        cur_cve = None
        seq_idx = -1

        buf_num, buf_bool, buf_cat = [], [], []
        buf_eps, buf_flag = [], []

        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            cols  = batch.columns

            for r in range(batch.num_rows):
                cve = cols[self.col2idx['cve']][r].as_py()

                # ── sequence boundary ──────────────────────────────
                if cur_cve is None:
                    cur_cve = cve
                if cve != cur_cve:
                    seq_idx += 1
                    if seq_idx % w_n == w_id:
                        yield self._flush(buf_num, buf_bool, buf_cat,
                                          buf_eps, buf_flag)
                    cur_cve = cve
                    buf_num.clear();  buf_bool.clear(); buf_cat.clear()
                    buf_eps.clear();  buf_flag.clear()

                # ── append current row ─────────────────────────────
                buf_num .append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.num_cols])
                buf_bool.append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.bool_cols])
                buf_cat .append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.cat_cols])
                buf_eps .append(cols[self.col2idx['epss']][r].as_py())
                buf_flag.append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.flag_cols])

        # flush last CVE
        if buf_num:
            seq_idx += 1
            if seq_idx % w_n == w_id:
                yield self._flush(buf_num, buf_bool, buf_cat,
                                  buf_eps, buf_flag)

    # -----------------------------------------------------------------
    def _flush(self, num_l, bool_l, cat_l, eps_l, flag_l):
        """Convert accumulated python lists → CPU tensors."""
        num_t  = torch.tensor(num_l,  dtype=torch.float32)  # [T,32]
        boo_t  = torch.tensor(bool_l, dtype=torch.uint8)    # [T,2]
        cat_t  = torch.tensor(cat_l,  dtype=torch.int64)    # [T,6]
        eps_t  = torch.tensor(eps_l,  dtype=torch.float32)  # [T]
        flag_t = torch.tensor(flag_l, dtype=torch.uint8)    # [T,3]
        return num_t, boo_t, cat_t, eps_t, flag_t


# ─────────────────────────── collate fn ────────────────────────────
def pad_and_mask(batch,
                 *,
                 flag_kind: str,       # "train" | "val" | "test"
                 horizon:   int = 30):
    """Pads variable-length samples and rebuilds the three masks."""

    kidx = {"train": 0, "val": 1, "test": 2}[flag_kind]

    nums, boos, cats, epss, flags = zip(*batch)
    lengths = [t.size(0) for t in nums]
    pad = lambda seqs, val: pad_sequence(seqs, batch_first=True,
                                         padding_value=val)

    num_pad = pad(nums, 0.0)        # [B,L,32]
    boo_pad = pad(boos, 0)          # [B,L, 2]
    cat_pad = pad(cats, 0)          # [B,L, 6]

    # build Y & m_h per sequence, then pad
    Ys, mHs = [], []
    for eps in epss:
        y, mh = _build_y_and_mh(eps, horizon)
        Ys.append(y);  mHs.append(mh)
    Y_pad  = pad(Ys, 0.0)           # [B,L,H]
    mH_pad = pad(mHs, 0)            # [B,L,H]

    # masks
    mT_pad = pad([torch.ones(l) for l in lengths], 0.0)          # [B,L]
    mE_pad = pad([f[:, kidx].float() for f in flags], 0.0)       # [B,L]

    return num_pad, boo_pad, cat_pad, Y_pad, mT_pad, mH_pad, mE_pad


# ───────────────────────── smoke-test ─────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from functools import partial

    ds = CVEIterableDataset("work/epss_stage1.arrow", horizon=30)
    ld = DataLoader(ds, batch_size=4,
                    collate_fn=partial(pad_and_mask,
                                       flag_kind="train", horizon=30),
                    num_workers=0)

    batch = next(iter(ld))
    n, b, c, y, mt, mh, me = batch
    print("✓ loader OK  shapes:",
          n.shape, b.shape, c.shape, y.shape, mt.shape, mh.shape, me.shape) 
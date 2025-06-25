#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
cve_iter_dataset.py – streaming IterableDataset + pad/mask collate
for the Arrow file written by 00_build_arrow.py.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple, Sequence, List
import json, re

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.types as patypes


# ─────────────────────────── helpers ────────────────────────────
def _build_y_and_mh(eps: torch.Tensor, horizon: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For every timestep t, set
       y[t, k] = eps[t + 1 + k]   for k < horizon  and value exists.
    mh is the horizon-mask (1 where y is valid).
    """
    T = eps.size(0)
    y  = torch.zeros(T, horizon, dtype=torch.float32)
    mh = torch.zeros_like(y)
    for t in range(T):
        k = min(horizon, T - t - 1)
        if k:
            y[t, :k]  = eps[t + 1 : t + 1 + k]
            mh[t, :k] = 1
    return y, mh


# ─────────────────────────── dataset ────────────────────────────
class CVEIterableDataset(IterableDataset):
    """
    Memory-maps the Arrow IPC file and yields *unpadded* sequences,
    one per CVE.  Padding happens in the collate_fn.
    """

    BOOL_RE = re.compile(r"^(has_|is_)")

    def __init__(self,
                 arrow_path: str | Path,
                 horizon: int = 30) -> None:
        super().__init__()
        self.arrow_path = Path(arrow_path)
        self.horizon    = horizon

        # ── derive column groups from schema + vocab ──────────────
        file = ipc.open_file(pa.memory_map(str(self.arrow_path), "r"))
        self._schema = file.schema

        with (self.arrow_path.parent / "vocab.json").open() as fh:
            vocab = json.load(fh)
        self.cat_cols: List[str] = list(vocab.keys())

        names = [f.name for f in self._schema]
        self.bool_cols = [n for n in names
                          if self.BOOL_RE.match(n) and n not in self.cat_cols]

        self.flag_cols = ["flag_train", "flag_val", "flag_test"]

        # FIXED: Remove 'epss' from reserved set so it becomes an input feature
        # EPSS will be both an input feature AND the target variable
        reserved = set(self.cat_cols + self.bool_cols +
                       self.flag_cols + ["cve", "date"])
        self.num_cols = [
            f.name for f in self._schema
            if f.name not in reserved and
               (patypes.is_integer(f.type) or patypes.is_floating(f.type))
        ]
        assert self.num_cols, "No numeric columns detected – check schema."

        self.col2idx = {f.name: i for i, f in enumerate(self._schema)}
        self.collected_cve_ids: List[str] = []

        print(f"[CVEDataset] {len(self.num_cols)} numeric  |  "
              f"{len(self.bool_cols)} boolean  |  {len(self.cat_cols)} categorical")

    # ─────────────────────────── iterator ─────────────────────────
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        reader = ipc.open_file(pa.memory_map(str(self.arrow_path), "r"))

        worker = get_worker_info()
        wid, wnum = (worker.id, worker.num_workers) if worker else (0, 1)

        cur_cve: str | None = None
        buf_num: list[list]  = []
        buf_bool: list[list] = []
        buf_cat: list[list]  = []
        buf_eps: list[float] = []
        buf_flag: list[list] = []
        buf_date: list[int]  = []

        def flush():
            "Yield current CVE (if any) and clear buffers."
            if not buf_num:
                return
            # simple hash-based sharding so every CVE goes to one worker
            if hash(cur_cve) % wnum != wid:
                buf_num.clear(); buf_bool.clear(); buf_cat.clear()
                buf_eps.clear(); buf_flag.clear(); buf_date.clear()
                return
            self.collected_cve_ids.append(cur_cve)
            yield (
                torch.tensor(buf_num,  dtype=torch.float32),
                torch.tensor(buf_bool, dtype=torch.float32),
                torch.tensor(buf_cat,  dtype=torch.int64),
                torch.tensor(buf_eps,  dtype=torch.float32),
                torch.tensor(buf_flag, dtype=torch.uint8),
                torch.tensor(buf_date, dtype=torch.int64),
            )
            buf_num.clear(); buf_bool.clear(); buf_cat.clear()
            buf_eps.clear(); buf_flag.clear(); buf_date.clear()

        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            cols  = batch.columns

            for r in range(batch.num_rows):
                cve = cols[self.col2idx["cve"]][r].as_py()

                if cur_cve is None:
                    cur_cve = cve
                if cve != cur_cve:          # sequence boundary
                    yield from flush()
                    cur_cve = cve

                buf_num .append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.num_cols])
                buf_bool.append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.bool_cols])
                buf_cat .append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.cat_cols])
                buf_eps .append(cols[self.col2idx["epss"]][r].as_py())
                buf_flag.append([cols[self.col2idx[c]][r].as_py()
                                 for c in self.flag_cols])
                # Arrow scalar: .value is int64 nanoseconds
                buf_date.append(cols[self.col2idx["date"]][r].value)

        yield from flush()                  # final CVE


# ─────────────────────────── collate fn ──────────────────────────
def pad_and_mask(batch: Sequence[Tuple[torch.Tensor, ...]],
                 *,
                 flag_kind: str,             # "train" | "val" | "test"
                 horizon: int = 30):
    """Pads ragged sequences and rebuilds Y, mT, mH, mE."""

    split_idx = {"train": 0, "val": 1, "test": 2}[flag_kind]

    nums, boos, cats, epss, flags, dates = zip(*batch)
    lengths = [t.size(0) for t in nums]

    pad = lambda seqs, val: pad_sequence(seqs, batch_first=True,
                                         padding_value=val)

    num_pad = pad(nums, 0.0)
    boo_pad = pad(boos, 0.0)
    cat_pad = pad(cats, 0)

    Ys, mHs = [], []
    for e in epss:
        y, mh = _build_y_and_mh(e, horizon)
        Ys.append(y);  mHs.append(mh)
    Y_pad  = pad(Ys, 0.0)
    mH_pad = pad(mHs, 0)

    mT_pad = pad([torch.ones(l, dtype=torch.float32) for l in lengths], 0.0)
    mE_pad = pad([f[:, split_idx].float() for f in flags], 0.0)

    date_pad = pad(dates, 0)

    return num_pad, boo_pad, cat_pad, Y_pad, mT_pad, mH_pad, mE_pad, date_pad


# ─────────────────────────── smoke test ─────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from functools import partial

    ARROW = Path(__file__).resolve().parent.parent / "work" / "epss_stage1.arrow"
    ds = CVEIterableDataset(ARROW, horizon=30)

    dl = DataLoader(
        ds,
        batch_size=4,
        collate_fn=partial(pad_and_mask, flag_kind="train", horizon=30),
        num_workers=0,           # change to >0 if you like
        pin_memory=False
    )

    batch = next(iter(dl))
    n, b, c, y, mt, mh, me, dt = batch
    print("✓ loader OK  shapes:",
          n.shape, b.shape, c.shape, y.shape, mt.shape, mh.shape, me.shape, dt.shape)
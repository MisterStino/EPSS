#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
SUS Dataset: Stochastic Under-Sampling for EPSS Time Series
===========================================================

Implements Stochastic Under-Sampling (SUS) from Silvestrin et al.'s 
"A Framework for Imbalanced Time-series Forecasting" to address within-CVE
imbalance by probabilistically sampling windows based on future change magnitude.

Key differences from CVEIterableDatasetFixed:
- Yields fixed-length windows (30 days) instead of full sequences
- Applies probabilistic sampling based on weight function g_t = |y(t+Δ) - y(t)|
- Sampling probability: p_t = (g_t^β) / Z
- No padding needed since all windows are same length
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple, List
import json, re, hashlib, random

import torch
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.types as patypes
import numpy as np


class CVEIterableDatasetSUS(IterableDataset):
    """
    SUS Dataset: Probabilistically samples windows based on future EPSS changes
    
    This addresses the "flat-hugging" problem by showing the network far fewer
    "boring" windows and many more "interesting" ones without duplicating data.
    """

    BOOL_RE = re.compile(r"^(has_|is_)")

    def __init__(self,
                 arrow_path: str | Path,
                 horizon: int = 30,
                 look_ahead: int = 5,
                 beta: float = 3.0,
                 z_norm: float | None = None,
                 seed: int = 42) -> None:
        """
        Args:
            arrow_path: Path to Arrow file
            horizon: Window length (30 days)
            look_ahead: Days ahead to compute future change (Δ in paper)
            beta: Exponent for probability function (higher = more selective)
            z_norm: Normalization constant Z (must be ≥ max weight^β to ensure p≤1)
            seed: Random seed for reproducible sampling
        """
        super().__init__()
        self.arrow_path = Path(arrow_path)
        self.horizon = horizon
        self.look_ahead = look_ahead
        self.beta = beta
        self.Z = z_norm
        
        # Set random seed for reproducible sampling
        random.seed(seed)
        torch.manual_seed(seed)

        # ── derive column groups from schema + vocab ──────────────
        with ipc.open_file(pa.memory_map(str(self.arrow_path), "r")) as tmp_reader:
            self._schema = tmp_reader.schema

        with (self.arrow_path.parent / "vocab.json").open() as fh:
            vocab = json.load(fh)
        self.cat_cols: List[str] = list(vocab.keys())

        names = [f.name for f in self._schema]
        self.bool_cols = [n for n in names
                          if self.BOOL_RE.match(n) and n not in self.cat_cols]

        self.flag_cols = ["flag_train", "flag_val", "flag_test"]

        # Exclude epss_target from input features (it's used only for targets)
        reserved = set(self.cat_cols + self.bool_cols +
                       self.flag_cols + ["cve", "date", "epss_target"])
        self.num_cols = [
            f.name for f in self._schema
            if f.name not in reserved and
               (patypes.is_integer(f.type) or patypes.is_floating(f.type))
        ]
        assert self.num_cols, "No numeric columns detected – check schema."

        self.col2idx = {f.name: i for i, f in enumerate(self._schema)}
        self.collected_cve_ids: List[str] = []

        # ── Check date column format ──────────────────────────────
        date_field = None
        for field in self._schema:
            if field.name == "date":
                date_field = field
                break
        
        if date_field is None:
            raise ValueError("No 'date' column found in Arrow schema")
        
        # Determine date conversion strategy
        if str(date_field.type) == "date32[day]":
            self.date_conversion = "days_since_epoch"
        elif str(date_field.type).startswith("timestamp"):
            self.date_conversion = "timestamp"
        else:
            self.date_conversion = "timestamp"

        print(f"[CVEDatasetSUS] {len(self.num_cols)} numeric  |  "
              f"{len(self.bool_cols)} boolean  |  {len(self.cat_cols)} categorical")
        print(f"[CVEDatasetSUS] SUS params: horizon={horizon}, look_ahead={look_ahead}, "
              f"beta={beta}, Z={z_norm}")

    def _convert_date_value(self, arrow_scalar) -> int:
        """Convert Arrow date scalar to nanoseconds since epoch"""
        if self.date_conversion == "days_since_epoch":
            days_since_epoch = arrow_scalar.value
            nanoseconds = days_since_epoch * 86400 * 1_000_000_000
            return nanoseconds
        else:
            type_str = str(arrow_scalar.type)
            if "[" in type_str and "]" in type_str:
                ts_unit = type_str.split("[")[-1].rstrip("]")
                factor = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}.get(ts_unit, 1.0)
                return int(arrow_scalar.value * factor)
            else:
                return arrow_scalar.value

    @staticmethod
    def _compute_weight(eps_window: np.ndarray, look_ahead: int) -> float:
        """
        Compute weight function g_t = |y(t+Δ) - y(t)| from paper
        
        Args:
            eps_window: EPSS values for window [t-horizon+1, ..., t]
            look_ahead: Δ (days ahead to look for change)
            
        Returns:
            Weight (future change magnitude)
        """
        if len(eps_window) < look_ahead + 1:
            return 0.0
        
        current = eps_window[-look_ahead - 1]  # y(t-Δ) to avoid future leak
        future_window = eps_window[-look_ahead:]  # next Δ days
        future_max = np.max(future_window)  # peak in next Δ days
        
        return abs(float(future_max - current))

    def _emit_windows(self, buf_num, buf_bool, buf_cat, buf_eps, buf_flag, buf_date):
        """
        Emit windows from current CVE using SUS sampling
        
        This is the core SUS implementation:
        1. Slide window of length horizon
        2. Compute weight g_t for each window
        3. Sample with probability p_t = (g_t^β) / Z
        """
        T = len(buf_eps)
        if T < self.horizon:
            return  # Sequence too short
        
        eps_np = np.array(buf_eps, dtype=np.float32)
        
        # Slide window through sequence
        for t in range(self.horizon - 1, T):
            s = t - self.horizon + 1  # Window start
            
            # Check if we have enough future data for weight computation
            if t + self.look_ahead >= T:
                continue
                
            # Extract window [s:t+1] = [t-horizon+1, ..., t]
            window_eps = eps_np[s:t+1+self.look_ahead]  # Include future for weight
            
            # Compute weight using first horizon values for prediction,
            # but include look_ahead for weight computation
            g = self._compute_weight(window_eps, self.look_ahead)
            
            # Skip if Z not set (should be pre-computed)
            if self.Z is None:
                continue
                
            # Compute sampling probability: p = (g^β) / Z
            p = min(1.0, (g ** self.beta) / self.Z)
            
            # Sample: keep window with probability p
            if p >= 1.0 or random.random() < p:
                yield (
                    torch.tensor(buf_num[s:t+1],  dtype=torch.float32),
                    torch.tensor(buf_bool[s:t+1], dtype=torch.float32),
                    torch.tensor(buf_cat[s:t+1],  dtype=torch.int64),
                    torch.tensor(buf_eps[s:t+1],  dtype=torch.float32),
                    torch.tensor(buf_flag[s:t+1], dtype=torch.uint8),
                    torch.tensor(buf_date[s:t+1], dtype=torch.int64),
                )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Iterate through Arrow file, applying SUS to each CVE"""
        if self.Z is None:
            raise ValueError("Z normalization constant must be set before iteration. "
                           "Use compute_weight_quantile.py to pre-compute Z.")
        
        with ipc.open_file(pa.memory_map(str(self.arrow_path), "r")) as reader:
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
                """Apply SUS to current CVE and yield windows"""
                if not buf_num:
                    return
                # Worker sharding (same as original)
                digest = int(hashlib.md5(cur_cve.encode("utf-8")).hexdigest(), 16)
                if digest % wnum != wid:
                    buf_num.clear(); buf_bool.clear(); buf_cat.clear()
                    buf_eps.clear(); buf_flag.clear(); buf_date.clear()
                    return
                
                self.collected_cve_ids.append(cur_cve)
                
                # Emit windows using SUS sampling
                yield from self._emit_windows(buf_num, buf_bool, buf_cat, 
                                            buf_eps, buf_flag, buf_date)
                
                buf_num.clear(); buf_bool.clear(); buf_cat.clear()
                buf_eps.clear(); buf_flag.clear(); buf_date.clear()

            for b in range(reader.num_record_batches):
                batch = reader.get_batch(b)
                cols  = batch.columns

                for r in range(batch.num_rows):
                    cve = cols[self.col2idx["cve"]][r].as_py()

                    if cur_cve is None:
                        cur_cve = cve
                    if cve != cur_cve:  # sequence boundary
                        yield from flush()
                        cur_cve = cve

                    buf_num .append([cols[self.col2idx[c]][r].as_py()
                                     for c in self.num_cols])
                    buf_bool.append([cols[self.col2idx[c]][r].as_py()
                                     for c in self.bool_cols])
                    buf_cat .append([cols[self.col2idx[c]][r].as_py()
                                     for c in self.cat_cols])
                    buf_eps .append(cols[self.col2idx["epss_target"]][r].as_py())
                    buf_flag.append([cols[self.col2idx[c]][r].as_py()
                                     for c in self.flag_cols])
                    
                    date_scalar = cols[self.col2idx["date"]][r]
                    date_ns = self._convert_date_value(date_scalar)
                    buf_date.append(date_ns)

            yield from flush()  # final CVE


def collate_fixed_win(batch: List[Tuple[torch.Tensor, ...]]):
    """
    Simple collate function for fixed-length windows
    
    Since all windows are exactly horizon length, we can just stack tensors
    without padding. This is much simpler than pad_and_mask_fixed.
    """
    if not batch:
        return tuple()
        
    return tuple(torch.stack(tensors, 0) for tensors in zip(*batch))


def collate_sus_train(batch: List[Tuple[torch.Tensor, ...]], 
                      horizon: int = 30,
                      flag_kind: str = "train"):
    """
    Collate function for SUS dataset that builds same output structure as pad_and_mask_fixed
    
    This converts SUS windows into the format expected by the training loop:
    num, boo, cat, Y, mt, mh, me, date_pad, lengths
    """
    from .dataset_iterable_fixed import _build_y_and_mh
    
    if not batch:
        return tuple()
    
    # Stack the 6 tensors from SUS dataset
    nums, boos, cats, epss, flags, dates = tuple(torch.stack(tensors, 0) for tensors in zip(*batch))
    
    # Build targets and masks (same as pad_and_mask_fixed)
    split_idx = {"train": 0, "val": 1, "test": 2}[flag_kind]
    
    B, L = nums.shape[:2]  # Batch size, sequence length (should be horizon)
    
    # Build Y and mH using vectorized function
    Ys, mHs = [], []
    for i in range(B):
        y, mh = _build_y_and_mh(epss[i], horizon)
        Ys.append(y)
        mHs.append(mh)
    
    Y = torch.stack(Ys)    # [B, L, H]
    mH = torch.stack(mHs)  # [B, L, H]
    
    # Time mask: all ones since windows are fixed length
    mT = torch.ones(B, L, dtype=torch.float32)
    
    # Eval mask: extract appropriate flag column
    mE = flags[:, :, split_idx].float()  # [B, L]
    
    # Lengths: all sequences are same length
    lengths = torch.full((B,), L, dtype=torch.long)
    
    return nums, boos, cats, Y, mT, mH, mE, dates, lengths


# ─────────────────────────── smoke test ─────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    ARROW = Path(__file__).resolve().parent.parent / "work" / "epss_stage1.arrow"
    
    print("Testing SUS dataset loader...")
    print("NOTE: This test requires Z to be pre-computed")
    
    # Test with dummy Z (in practice, use compute_weight_quantile.py)
    ds = CVEIterableDatasetSUS(ARROW, horizon=30, look_ahead=5, 
                              beta=3.0, z_norm=0.5)  # Dummy Z

    # Test DataLoader
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fixed_win, num_workers=0)

    try:
        batch = next(iter(dl))
        print(f"✓ Batch shapes: {[t.shape for t in batch]}")
        print(f"✓ All tensors have fixed length: {batch[0].shape[1]} == horizon")
        print("✓ SUS dataset test passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Make sure to pre-compute Z using compute_weight_quantile.py") 
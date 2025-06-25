# SPDX-License-Identifier: MIT
"""
hf_dataset.py – zero-copy HuggingFace loader for EPSS Arrow file.
"""
from pathlib import Path
import numpy as np, torch, datasets, pyarrow as pa
from torch.nn.utils.rnn import pad_sequence
import json

HORIZON = 30          # keep in sync with trainer

class EPSSDatasetHF(torch.utils.data.IterableDataset):
    def __init__(self, arrow_path: str | Path, split: str):
        super().__init__()
        self.arrow_path = str(arrow_path)
        assert split in {"train", "val", "test"}
        self.split = split

        # 1️⃣  open the mmap’d Arrow file once
        self.raw = datasets.Dataset.from_file(self.arrow_path, memory_map=True)

        # 2️⃣  mask by flag column **inside Arrow compute layer**
        self.raw = self.raw.filter(
            lambda f, split=split: f[f"flag_{split}"], 
            num_proc=0,  # run in C++ kernel, no Python loop
        )

        # 3️⃣  sort ensures contiguous rows per CVE (already true but cheap)
        self.raw = self.raw.sort("cve")

        # 4️⃣  group by cve → each yield is *one* sequence (zero-copy slice)
        self.grouped = self.raw.groupby("cve", batch_size=-1)

        # 5️⃣  tell HF we want numpy back → torch later, still zero-copy
        self.grouped.set_format(type="numpy")

        # 6️⃣  pre-compute column lists (schema is tiny)
        schema = pa.ipc.open_file(pa.memory_map(self.arrow_path, "r")).schema
        names  = [f.name for f in schema]
        self.num_cols  = [n for n in names if n.endswith("_delta") or n.startswith("z_")]
        self.bool_cols = [n for n in names if n.startswith(("has_", "is_"))]
        self.cat_cols  = json.load(open(Path(self.arrow_path).with_suffix("").parent/"vocab.json")).keys()

    # --------------------------------------------------------------------- #
    def _build_future_targets(self, eps):
        """
        Vectorised horizon target builder (no Python loop).
        eps : (T,) float32 → y,(T,H)  mh,(T,H)
        """
        T = len(eps)
        idx = np.arange(T)[:, None]
        fut = idx + np.arange(1, HORIZON + 1)
        valid = fut < T
        y  = np.where(valid, eps[fut], 0.).astype("float32")
        mh = valid.astype("uint8")
        return y, mh

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        it = self.grouped.shard(worker.num_workers, worker.id) if worker else self.grouped

        for batch in it:
            # batch is still column-major – slice once, no Python-per-row
            num  = np.vstack([batch[c] for c in self.num_cols]).T
            boo  = np.vstack([batch[c] for c in self.bool_cols]).T
            cat  = np.vstack([batch[c] for c in self.cat_cols]).T
            eps  = np.asarray(batch["epss"], dtype="float32")
            date = np.asarray(batch["date"], dtype="int64")

            y, mh = self._build_future_targets(eps)

            yield (
                torch.from_numpy(num),
                torch.from_numpy(boo),
                torch.from_numpy(cat),
                torch.from_numpy(y),
                torch.ones(len(eps), dtype=torch.float32),   # m_t (all real rows)
                torch.from_numpy(mh),
                torch.ones(len(eps), dtype=torch.float32),   # m_eval (all rows in split)
                torch.from_numpy(date),
            )

# ------------------------------------------------------------------------- #
def pad_and_mask_hf(batch):
    pad = lambda seqs, val: pad_sequence(seqs, batch_first=True,
                                         padding_value=val)
    num, boo, cat, y, mt, mh, me, dt = zip(*batch)
    return (
        pad(num, 0.), pad(boo, 0.), pad(cat, 0),
        pad(y, 0.),   pad(mt, 0.),  pad(mh, 0), pad(me, 0.), pad(dt, 0)
    )

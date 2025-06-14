# %%
#!/usr/bin/env python
# lstm_epss_fullsequence_cuda_v3.py
"""
Feature-rich, leakage-proof EPSS forecaster

• loads the full Spark-generated Parquet (≈68 cols, one row per (cve, date))
• converts all timestamps to non-leaking "age" deltas
• builds numerics, booleans, categoricals with train-only statistics
• keeps full history in every split (warm-up) but masks the loss accordingly
• trains a mixed-type Seq-to-Seq LSTM that predicts the next 30-day EPSS path
"""

# ───────────────────────────── CELL 1: IMPORTS & SETUP ──────────────────────
import json, numpy as np, pandas as pd, torch, torch.nn as nn, duckdb
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ──────────────────────────── helpers ───────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[INFO] GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("[WARN] CUDA unavailable → CPU")
    return torch.device("cpu")

def transform_epss(arr: np.ndarray,
                   mode: str = "logit",
                   eps : float = 1e-6) -> np.ndarray:
    """
    Stabilised transforms to map [0,1] → ℝ (helps optimisation).
    Choose one of: "inverted_log" | "cloglog" | "logit"
    """
    p = np.clip(arr.astype("float64"), eps, 1.0 - eps)
    if   mode == "inverted_log": out = -np.log(p)
    elif mode == "cloglog":      out = np.log(-np.log(1.0 - p))
    elif mode == "logit":        out = np.log(p / (1.0 - p))
    else: raise ValueError(mode)
    return out.astype("float32")

# ───────────────────────── column bookkeeping ───────────────────────────────
DROP_COLS = [                     # large strings / provenance we ignore
    "cve_date_key", "original_date",
    "reconstruction_timestamp", "reconstruction_timestamp_raw",
    "details_combined", "details_longest", "event_data_merged",
    "description_all", "description_en",
    "primary_cvss_vec", "cve_tags", "reference_count",
]

TS_SAFE   = ["published_date"]                     # never in the future
TS_LEAKY  = ["last_modified_date", "snapshot_date"]# may exceed row date

BOOL_COLS = (lambda df: [c for c in df.columns
                         if c.startswith(("has_", "is_"))
                         or c in ("same_day_multi_source",
                                  "has_v2","has_v30","has_v31","has_v40")])

CAT_COLS  = [
    "primary_cvss_ver", "primary_cvss_sev", "dominant_event_type",
    "prev_event_type", "primary_source", "cwe_id",
    "vuln_status", "source_identifier",
]

EMB_DIM   = 8          # per-categorical embedding size
SENTINEL  = -10.0      # ≥ |10σ| after standardisation

# %%
# ───────────────────────────── CELL 2: DATA LOADING ─────────────────────────
from pathlib import Path

# Use simple path relative to current working directory (works on Paperspace)
PARQUET_DIR = Path("data/minimal_v1_timeseries.parquet")

if not PARQUET_DIR.exists():
    raise FileNotFoundError(f"Parquet path not found: {PARQUET_DIR}")

# DuckDB needs parquet_scan() for multi-file datasets
parquet_glob = str(PARQUET_DIR / "*.parquet") if PARQUET_DIR.is_dir() else str(PARQUET_DIR)

# Probe schema using LIMIT 0 instead of DESCRIBE
schema_df = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}') LIMIT 0").df()
probe_cols = schema_df.columns.tolist()

# Build EXCLUDE clause
drop_real = [c for c in DROP_COLS if c in probe_cols]
quote_cols = ", ".join(f'"{c}"' for c in drop_real)
exclude = f" EXCLUDE ({quote_cols})" if drop_real else ""

BIG = duckdb.sql(f"SELECT *{exclude} FROM parquet_scan('{parquet_glob}')").df()
BIG["date"] = pd.to_datetime(BIG["date"])
print(f"[INFO] loaded {len(BIG):,} rows × {BIG.shape[1]} columns (DuckDB)")

# ───────────────────────────── 1.1  dynamic column filtering ────────────────
# Filter hardcoded lists to only include columns that actually exist
TS_SAFE   = [c for c in TS_SAFE  if c in BIG.columns]
TS_LEAKY  = [c for c in TS_LEAKY if c in BIG.columns]
CAT_COLS  = [c for c in CAT_COLS if c in BIG.columns]

print(f"[INFO] Available TS_SAFE: {TS_SAFE}")
print(f"[INFO] Available TS_LEAKY: {TS_LEAKY}")
print(f"[INFO] Available CAT_COLS: {CAT_COLS}")

# %%
# ───────────────────────────── CELL 3: DATA PREPROCESSING ───────────────────
for col in TS_SAFE:
    BIG[f"{col}_delta"] = (BIG["date"]
                           - pd.to_datetime(BIG[col])).dt.days.astype("float32")

for col in TS_LEAKY:
    d = (BIG["date"] - pd.to_datetime(BIG[col])).dt.days.astype("float32")
    d[d < 0] = np.nan                      # clip future leakage → NaN
    BIG[f"{col}_delta"] = d

BIG.drop(columns=TS_SAFE + TS_LEAKY, inplace=True)
if TS_SAFE + TS_LEAKY:  # Only check if we have timestamp columns
    assert (BIG.filter(regex="_delta$") < 0).stack().sum() == 0

# ───────────────────────────── 3  calendar split ────────────────────────────
days     = np.sort(BIG["date"].unique())
VAL_CUT  = pd.to_datetime(days[int(0.64 * len(days))])
TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])

BIG["flag_train"] = (BIG["date"] <  VAL_CUT).astype("uint8")
BIG["flag_val"]   = ((BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT)).astype("uint8")
BIG["flag_test"]  = (BIG["date"] >= TEST_CUT).astype("uint8")

print(f"[INFO] VAL from {VAL_CUT.date()} | TEST from {TEST_CUT.date()}")

# %%
# ───────────────────────────── CELL 4: FEATURE ENGINEERING ──────────────────
BIG["epss"] = transform_epss(BIG["epss"].values, mode="logit", eps=1e-6)

# booleans ───────────────────────────────────────────────────────────────────
for col in BOOL_COLS(BIG):
    if pd.api.types.is_bool_dtype(BIG[col]):
        BIG[col] = BIG[col].fillna(False).astype("uint8")
    else:
        BIG[col] = BIG[col].fillna(0).astype("uint8")

# categoricals (vocab from *true* training rows only) ────────────────────────
train_mask = BIG["flag_train"] == 1           # warm-up excluded
VOCAB = {}
for col in CAT_COLS:
    cats = BIG.loc[train_mask, col].dropna().unique()
    VOCAB[col] = {"UNK": 0,
                  **{c: i + 1 for i, c in enumerate(sorted(cats))}}
    BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int16")
json.dump(VOCAB, open("vocab.json", "w"))

# numerics (everything that is number-typed *and* not bool flag) ─────────────
NUM_COLS = [c for c, t in BIG.dtypes.items()
            if np.issubdtype(t, np.number)
            and c not in BOOL_COLS(BIG) + ["flag_train", "flag_val", "flag_test"]]

scaler = StandardScaler().fit(BIG.loc[train_mask, NUM_COLS])
BIG[NUM_COLS] = scaler.transform(BIG[NUM_COLS]).astype('float32')

for col in NUM_COLS:
    miss = BIG[col].isna()
    BIG[f"{col}_missing"] = miss.astype("uint8")
    BIG.loc[miss, col] = SENTINEL

# %%
# ───────────────────────────── CELL 5: DATASET PREPARATION ──────────────────
class CVEDataset(Dataset):
    """One item = a zero-padded full history for a CVE + all masks."""
    def __init__(self, df: pd.DataFrame, L_max: int,
                 horizon: int, flag_col: str, cat_cols: list):

        self.num, self.boo, self.cat = [], [], []
        self.Y, self.m_t, self.m_h, self.m_eval = [], [], [], []
        self.cat_cols = cat_cols

        num = df[NUM_COLS].to_numpy("float32")
        boo = df[BOOL_COLS(df)].to_numpy("uint8")
        # Handle case where no categorical columns exist
        cat = (
            df[cat_cols].to_numpy("int64").astype("int64")  # torch.long later
            if cat_cols else np.empty((len(df), 0), dtype="int64")
        )
        epss= df["epss"].to_numpy("float32").reshape(-1, 1)
        flag= df[flag_col].to_numpy("float32")

        for _, idx in df.groupby("cve", sort=False).groups.items():
            idx = np.asarray(idx)
            T   = len(idx)
            pad = L_max - T

            self.num.append(torch.from_numpy(np.pad(num[idx], ((0, pad), (0, 0)))))
            self.boo.append(torch.from_numpy(np.pad(boo[idx], ((0, pad), (0, 0)))))
            # Handle empty categorical case
            if cat_cols:
                self.cat.append(torch.from_numpy(np.pad(cat[idx], ((0, pad), (0, 0)))))
            else:
                self.cat.append(torch.zeros((L_max, 0), dtype=torch.int64))

            self.m_t   .append(torch.from_numpy(np.r_[np.ones(T), np.zeros(pad)]))
            self.m_eval.append(torch.from_numpy(np.r_[flag[idx], np.zeros(pad)]))

            y  = np.zeros((L_max, horizon), "float32")
            mh = np.zeros_like(y)
            for t in range(T):
                k = min(horizon, T - t - 1)
                if k:
                    y[t, :k]  = epss[idx][t + 1:t + 1 + k, 0]
                    mh[t, :k] = 1
            self.Y .append(torch.from_numpy(y))
            self.m_h.append(torch.from_numpy(mh))

    def __len__(self): return len(self.num)

    def __getitem__(self, i):
        return (self.num[i], self.boo[i], self.cat[i],
                self.Y[i],  self.m_t[i], self.m_h[i], self.m_eval[i])

def collate(batch):  # stacks list-of-tuples → tuple-of-tensors
    return tuple(torch.stack(x, 0) for x in zip(*batch))

# ───────────────────────────── 6  model ──────────────────────────────────────
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_num: int, n_bool: int, cat_sizes,
                 horizon: int = 30, hidden: int = 512,
                 layers: int = 3, emb_dim: int = EMB_DIM,
                 dropout: float = 0.3):
        super().__init__()
        # Handle case where no categorical columns exist
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim) for s in cat_sizes])
        in_dim   = n_num + n_bool + emb_dim * len(cat_sizes)

        self.lstm = nn.LSTM(in_dim, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, num, boo, cat):
        # Handle case where no categorical embeddings exist
        if len(self.emb):                           # tiny micro-perf
            cat = cat.long()                        # ⚠ NEW: ensure correct dtype
            e = torch.cat([emb(cat[..., i]) for i, emb in enumerate(self.emb)], dim=-1)
            x = torch.cat([num, boo.float(), e], dim=-1)
        else:
            x = torch.cat([num, boo.float()], dim=-1)
        h, _ = self.lstm(x)
        return self.head(h)

# ───────────────────────────── 7  masked loss ───────────────────────────────
def masked_mse(pred, true, m_t, m_h, m_eval):
    mask = m_t * m_eval                        # [B,L]
    err  = (pred - true) ** 2                  # [B,L,H]
    return (err * mask.unsqueeze(-1) * m_h).sum() / m_h.sum()

# %%
# ───────────────────────────── CELL 6: MODEL TRAINING & EVALUATION ──────────
# Main training routine - optimized for notebook execution
import time

print("🚀 STARTING MODEL TRAINING PIPELINE")
print("=" * 50)

torch.manual_seed(0)
dev = get_device()

HORIZON, BATCH, EPOCHS, LR = 30, 512, 12, 1e-3

# ──────────────────────── STEP 1: Dataset Preparation ───────────────────────
print(f"\n[STEP 1/5] Computing sequence lengths...")
start_time = time.time()
L_max = BIG.groupby("cve", observed=True).size().max()
elapsed = time.time() - start_time
print(f"✓ Max sequence length: {L_max} (computed in {elapsed:.1f}s)")

print(f"\n[STEP 2/5] Creating datasets (this may take several minutes)...")
start_time = time.time()

print("  → Training dataset...")
tr_ds = CVEDataset(BIG, L_max, HORIZON, "flag_train", CAT_COLS)
print(f"    ✓ {len(tr_ds):,} training sequences")

print("  → Validation dataset...")
va_ds = CVEDataset(BIG, L_max, HORIZON, "flag_val", CAT_COLS)
print(f"    ✓ {len(va_ds):,} validation sequences")

print("  → Test dataset...")
te_ds = CVEDataset(BIG, L_max, HORIZON, "flag_test", CAT_COLS)
print(f"    ✓ {len(te_ds):,} test sequences")

elapsed = time.time() - start_time
print(f"✓ All datasets created in {elapsed:.1f}s")

# ──────────────────────── STEP 3: DataLoader Creation ───────────────────────
print(f"\n[STEP 3/5] Creating data loaders...")
start_time = time.time()

tr_ld = DataLoader(tr_ds, BATCH, True,  collate_fn=collate, pin_memory=True)
va_ld = DataLoader(va_ds, BATCH, False, collate_fn=collate, pin_memory=True)
te_ld = DataLoader(te_ds, BATCH, False, collate_fn=collate, pin_memory=True)

elapsed = time.time() - start_time
print(f"✓ Data loaders ready: {len(tr_ld)} train, {len(va_ld)} val, {len(te_ld)} test batches ({elapsed:.1f}s)")

# ──────────────────────── STEP 4: Model Setup ───────────────────────────────
print(f"\n[STEP 4/5] Setting up model...")
start_time = time.time()

model = Seq2SeqLSTM(
            n_num   = len(NUM_COLS),
            n_bool  = len(BOOL_COLS(BIG)),
            cat_sizes=[len(VOCAB[c]) for c in CAT_COLS],
            horizon = HORIZON).to(dev)

if hasattr(torch, "compile") and dev.type == "cuda":
    print("  → Compiling model for GPU optimization...")
    model = torch.compile(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
elapsed = time.time() - start_time
print(f"✓ Model ready: {total_params:,} total params, {trainable_params:,} trainable ({elapsed:.1f}s)")

opt = torch.optim.Adam(model.parameters(), lr=LR)

# ──────────────────────── STEP 5: Training ──────────────────────────────────
print(f"\n[STEP 5/5] Training for {EPOCHS} epochs...")
print("=" * 50)
for ep in range(1, EPOCHS + 1):
    model.train(); tr_loss = 0.0
    for num, boo, cat, Y, mt, mh, me in tqdm(tr_ld, desc=f"train {ep}/{EPOCHS}"):
        num, boo, cat, Y, mt, mh, me = (z.to(dev, non_blocking=True)
                                        for z in (num, boo, cat, Y, mt, mh, me))
        opt.zero_grad()
        loss = masked_mse(model(num, boo, cat), Y, mt, mh, me)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item()
    tr_loss /= len(tr_ld)

    model.eval(); va_loss = 0.0
    with torch.no_grad():
        for num, boo, cat, Y, mt, mh, me in va_ld:
            num, boo, cat, Y, mt, mh, me = (z.to(dev, non_blocking=True)
                                            for z in (num, boo, cat, Y, mt, mh, me))
            va_loss += masked_mse(model(num, boo, cat), Y, mt, mh, me).item()
    va_loss /= len(va_ld)
    print(f"epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")

print("\n" + "=" * 50)
print("🎯 FINAL EVALUATION")
print("=" * 50)
model.eval(); tot_mse = tot_mae = tot_n = 0.0
with torch.no_grad():
    for num, boo, cat, Y, mt, mh, me in te_ld:
        num, boo, cat, Y, mt, mh, me = (z.to(dev, non_blocking=True)
                                        for z in (num, boo, cat, Y, mt, mh, me))
        P = model(num, boo, cat)
        m = mh * me.unsqueeze(-1)
        err = P - Y
        tot_mse += (err.pow(2) * m).sum().item()
        tot_mae += (err.abs() * m).sum().item()
        tot_n   += m.sum().item()

print(f"[RESULT] test MSE {tot_mse / tot_n:.4f} | MAE {tot_mae / tot_n:.4f}")

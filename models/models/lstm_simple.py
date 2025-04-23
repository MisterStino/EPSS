# lstm_epss_fullhistory_cuda.py
"""
Train an LSTM that, for every prefix of a CVE history, predicts the next
10 daily EPSS-invlog values.  Optimised for an NVIDIA GPU (RTX 3050 Ti).

Input folders (already inverse-log scaled & calendar-split):
    TRAIN_DIR / VAL_DIR / TEST_DIR

Each Parquet row:
    cve, date, epss, age_epss_pub
"""

import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ───────────────────────── device helper ────────────────────────────
def get_device() -> torch.device:
    """Prefer NVIDIA CUDA, fall back to CPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using NVIDIA GPU: {name}")
        # cuDNN autotuner → fastest kernels for this GPU + shapes
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    print("[WARN] CUDA unavailable – using CPU.")
    return torch.device("cpu")


# ───────────────────────── Pandas typing helper ─────────────────────
def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"]           = pd.to_datetime(df["date"], errors="raise")
    df["cve"]            = df["cve"].astype("category")
    df["epss"]           = df["epss"].astype("float32")
    df["age_epss_pub"]   = df["age_epss_pub"].astype("int32")
    return df.sort_values(["cve", "date"]).reset_index(drop=True)


# ───────────────────────── Dataset (all prefixes) ────────────────────
class CVEPrefixDataset(Dataset):
    """
    One sample = (prefix [T,2], next-10 target [10])
    """
    def __init__(self, df: pd.DataFrame, horizon: int = 10):
        self.X, self.y, self.len = [], [], []
        for _, grp in df.groupby("cve", observed=True):
            vals = grp[["epss", "age_epss_pub"]].values.astype("float32")
            T    = len(vals)
            if T <= horizon:
                continue
            for t in range(T - horizon):
                self.X.append(torch.from_numpy(vals[: t + 1]))           # (L,2)
                self.y.append(torch.from_numpy(vals[t + 1:t + 1 + horizon, 0]))
                self.len.append(t + 1)
        print(f"[INFO] Dataset windows: {len(self):,}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.len[idx]


def collate_pad(batch):
    x_list, y_list, len_list = zip(*batch)
    lengths  = torch.tensor(len_list, dtype=torch.int64)
    x_padded = pad_sequence(x_list, batch_first=True)        # right-pad zeros
    y_target = torch.stack(y_list)                           # (B, 10)
    return x_padded, lengths, y_target


# ───────────────────────── model ─────────────────────────────────────
class LSTMForecast(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128,
                 num_layers=2, output_dim=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_padded, lengths):
        packed = pack_padded_sequence(x_padded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)     # h_n: (layers, B, hidden)
        return self.head(h_n[-1])           # (B, 10)


# ───────────────────────── main ─────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    device = get_device()

    # ─── constants ──────────────────────────────────────────────────
    TRAIN_DIR    = "data/full_db/ml-sets-sampled/logit-scaled/train"
    VAL_DIR      = "data/full_db/ml-sets-sampled/logit-scaled/val"
    TEST_DIR     = "data/full_db/ml-sets-sampled/logit-scaled/test"
    HORIZON      = 10
    BATCH        = 200           # fits into 4 GiB – raise if memory allows
    EPOCHS       = 12
    LR           = 1e-3
    NUM_WORKERS  = 0             # stays 0 for Windows → no spawn/pickle overhead
    PIN_MEMORY   = True          # fine even with 0 workers

    # ─── 1) load splits ─────────────────────────────────────────────
    df_train = cast_types(pd.read_parquet(TRAIN_DIR, engine="pyarrow"))
    df_val   = cast_types(pd.read_parquet(VAL_DIR,   engine="pyarrow"))
    df_test  = cast_types(pd.read_parquet(TEST_DIR,  engine="pyarrow"))
    print(f"[INFO] rows – train {len(df_train):,} | val {len(df_val):,} | test {len(df_test):,}")

    # ─── 2) scaling ─────────────────────────────────────────────────
    scaler = StandardScaler().fit(df_train[["epss", "age_epss_pub"]])
    for _df in (df_train, df_val, df_test):
        _df[["epss", "age_epss_pub"]] = scaler.transform(_df[["epss", "age_epss_pub"]])

    # ─── 3) datasets & loaders ──────────────────────────────────────
    train_ds = CVEPrefixDataset(df_train, HORIZON)
    val_ds   = CVEPrefixDataset(df_val,   HORIZON)
    test_ds  = CVEPrefixDataset(df_test,  HORIZON)

    # **IMPORTANT**: persistent_workers & prefetch_factor only if NUM_WORKERS > 0
    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )

    # ─── 4) model / optim ───────────────────────────────────────────
    model  = LSTMForecast().to(device)
    if hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    loss_fn = nn.MSELoss()
    optim_  = optim.Adam(model.parameters(), lr=LR)

    # ─── 5) training loop ───────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        print(f"[INFO] Starting: Epoch {epoch:02d}/{EPOCHS} [train]")
        model.train()
        tr_loss, t0 = 0.0, time.perf_counter()
        for xb, lens, yb in tqdm(train_ld, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]"):
            xb, lens, yb = (x.to(device, non_blocking=True) for x in (xb, lens, yb))
            optim_.zero_grad()
            pred = model(xb, lens)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim_.step()
            tr_loss += loss.item()
        tr_loss /= len(train_ld)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, lens, yb in val_ld:
                xb, lens, yb = (x.to(device, non_blocking=True) for x in (xb, lens, yb))
                val_loss += loss_fn(model(xb, lens), yb).item()
        val_loss /= len(val_ld)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train {tr_loss:.4f} | val {val_loss:.4f}")

    # ─── 6) test evaluation ─────────────────────────────────────────
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, lens, yb in test_ld:
            xb, lens = xb.to(device, non_blocking=True), lens.to(device, non_blocking=True)
            preds.append(model(xb, lens).cpu())
            trues.append(yb)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    print(f"[RESULT] test MSE {mse:.4f} | MAE {mae:.4f}")

    # ─── 7) save csv ────────────────────────────────────────────────
    pd.DataFrame(
        np.hstack([preds, trues]),
        columns=[f"pred_t+{i+1}" for i in range(HORIZON)] +
                [f"true_t+{i+1}" for i in range(HORIZON)]
    ).to_csv("predictions_lstm_fullhistory.csv", index=False)
    print("[INFO] predictions saved → predictions_lstm_fullhistory.csv")

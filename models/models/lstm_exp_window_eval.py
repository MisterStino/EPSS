#!/usr/bin/env python
# lstm_epss_fullsequence_cuda.py
"""
One‑file runnable:
• reads the *single* Parquet  ➜ NumPy logit transform
• creates train / val / test sets with a per‑row loss flag
• trains the LSTM with full‑history context and leakage‑free metrics
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─────────────────────────────────── helpers ──────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[INFO] GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("[WARN] CUDA unavailable → CPU")
    return torch.device("cpu")


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"]         = pd.to_datetime(df["date"], errors="raise")
    df["cve"]          = df["cve"].astype("category")
    df["epss"]         = df["epss"].astype("float32")
    df["age_epss_pub"] = df["age_epss_pub"].astype("int32")
    return df.sort_values(["cve", "date"]).reset_index(drop=True)

# ──────────────────────────────── transforms ─────────────────────────────────

def transform_epss(
    arr: np.ndarray,
    mode: str = "inverted_log",
    eps: float = 1e-6
) -> np.ndarray:
    """
    arr: 1D array of probabilities in [0,1].
    mode: "inverted_log" | "cloglog" | "logit"
    eps:  clipping epsilon to avoid log(0) or division by zero.
    Returns a float32 array of same shape.
    """
    # 1) clip into (eps, 1-eps)
    p = np.clip(arr, eps, 1.0 - eps)

    # 2) apply transform
    if mode == "inverted_log":
        # y = -log(p)
        out = -np.log(p)
    elif mode == "cloglog":
        # y = log(-log(1-p))
        out = np.log(-np.log(1.0 - p))
    elif mode == "logit":
        # y = log(p / (1-p))
        out = np.log(p / (1.0 - p))
    else:
        raise ValueError(f"Unknown transform: {mode!r}")

    return out.astype(np.float32)


# ────────────────────────────── ❶ load & transform ───────────────────────────
FILE = "data/full_db/sampled/final_full_data_sampled.parquet"
BIG  = cast_types(pd.read_parquet(FILE, engine="pyarrow"))

# transform to: "inverted_log", "cloglog", "logit"
MODE = "logit"

# apply the numpy transform
BIG["epss"] = transform_epss(BIG["epss"].values, mode=MODE, eps=1e-6)
print(f"[INFO] loaded full table: {BIG.shape}  (epss transformed via {MODE})")

# ───────────────────────────── ❷ choose calendar cuts ────────────────────────
days     = np.sort(BIG["date"].unique())
n_days   = len(days)
TEST_CUT = days[int(0.80 * n_days)]
VAL_CUT  = days[int(0.64 * n_days)]
VAL_CUT = pd.to_datetime(VAL_CUT)
TEST_CUT = pd.to_datetime(TEST_CUT)
print(f"[INFO] VAL from {VAL_CUT.date()} | TEST from {TEST_CUT.date()}")

# ───────────────────────────── ❸ mark use_for_loss ───────────────────────────
def flag(df, cond):
    out = np.zeros(len(df), np.float32)
    out[cond] = 1.0
    return out

train_flag = flag(BIG, BIG["date"] <  VAL_CUT)
val_flag   = flag(BIG, (BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT))
test_flag  = flag(BIG, BIG["date"] >= TEST_CUT)

df_tr  = BIG.copy(); df_tr ["use_for_loss"] = train_flag
df_val = BIG.copy(); df_val["use_for_loss"] = val_flag
df_te  = BIG.copy(); df_te ["use_for_loss"] = test_flag

for name, df in zip(("train","val","test"), (df_tr, df_val, df_te)):
    print(f"[INFO] {name:<5} rows {len(df):,}  unique CVE {df['cve'].nunique():,}")

# ──────────────────────────── feature standardisation ────────────────────────
scaler = StandardScaler().fit(df_tr[["age_epss_pub"]])
for df in (df_tr, df_val, df_te):
    df["age_epss_pub"] = scaler.transform(df[["age_epss_pub"]])

# ─────────────────────────────── dataset class ───────────────────────────────
class CVEFullDataset(Dataset):
    """
    Returns tuple (X, Y, m_t, m_h, m_eval)
      X      : [L_max, 2]
      Y      : [L_max, H]
      m_t    : [L_max]          real rows vs padding
      m_h    : [L_max, H]       future availability
      m_eval : [L_max]          1 if this row counts toward the loss
    """
    def __init__(self, df, L_max: int, horizon: int = 10):
        self.X, self.Y, self.m_t, self.m_h, self.m_eval = [], [], [], [], []
        for _, g in df.groupby("cve", observed=True):
            vals  = g[["epss", "age_epss_pub"]].to_numpy("float32")
            evalm = g["use_for_loss"].to_numpy("float32")
            T     = len(vals); pad = L_max - T

            self.X.append(torch.from_numpy(np.pad(vals,  ((0,pad),(0,0)), "constant")))
            self.m_t.append(torch.from_numpy(np.r_[np.ones(T), np.zeros(pad)].astype("float32")))
            self.m_eval.append(torch.from_numpy(np.r_[evalm,     np.zeros(pad)].astype("float32")))

            Y  = np.zeros((L_max, horizon), np.float32)
            mh = np.zeros_like(Y)
            for t in range(T):
                k = min(horizon, T - t - 1)
                if k:
                    Y[t, :k]  = vals[t+1:t+1+k, 0]
                    mh[t, :k] = 1
            self.Y.append(torch.from_numpy(Y))
            self.m_h.append(torch.from_numpy(mh))
        print(f"[INFO] CVEs: {len(self)}  L_max={L_max}")

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.Y[i], self.m_t[i], self.m_h[i], self.m_eval[i])


def collate(batch):
    return tuple(torch.stack(x, 0) for x in zip(*batch))

# ───────────────────────────────── model ─────────────────────────────────────
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, layers=2,
                 horizon=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h)

# ─────────────────────── loss that honours m_eval ────────────────────────────
def masked_mse(pred, true, m_t, m_h, m_eval):
    mask = m_t * m_eval                        # 0/1 floats
    err  = (pred - true) ** 2
    err  = err * mask.unsqueeze(-1) * m_h
    return err.sum() / m_h.sum()

# ──────────────────────────── training script ───────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = get_device()

    HORIZON = 10; BATCH = 64; EPOCHS = 12; LR = 1e-3

    # ---------- dataset & loaders
    L_max = BIG.groupby("cve", observed=True).size().max()
    tr_ds = CVEFullDataset(df_tr,  L_max, HORIZON)
    va_ds = CVEFullDataset(df_val, L_max, HORIZON)
    te_ds = CVEFullDataset(df_te,  L_max, HORIZON)

    tr_ld = DataLoader(tr_ds, BATCH, shuffle=True,  collate_fn=collate,
                       num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, BATCH, shuffle=False, collate_fn=collate,
                       num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, BATCH, shuffle=False, collate_fn=collate,
                       num_workers=0, pin_memory=True)

    model = Seq2SeqLSTM(horizon=HORIZON).to(dev)
    if hasattr(torch, "compile") and dev.type == "cuda":
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------- train / val
    for ep in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0
        for X, Y, mt, mh, me in tqdm(tr_ld, desc=f"train {ep}/{EPOCHS}"):
            X, Y, mt, mh, me = (z.to(dev, non_blocking=True) for z in (X, Y, mt, mh, me))
            opt.zero_grad()
            loss = masked_mse(model(X), Y, mt, mh, me)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        model.eval(); va_loss = 0
        with torch.no_grad():
            for X, Y, mt, mh, me in va_ld:
                X, Y, mt, mh, me = (z.to(dev, non_blocking=True) for z in (X, Y, mt, mh, me))
                va_loss += masked_mse(model(X), Y, mt, mh, me).item()
        va_loss /= len(va_ld)
        print(f"epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")

    # ---------- test metric
    model.eval(); pred_, true_, mask_, eval_ = [], [], [], []
    with torch.no_grad():
        for X, Y, mt, mh, me in te_ld:
            pred_.append(model(X.to(dev)).cpu())
            true_.append(Y)
            mask_.append(mh)
            eval_.append(me)
    P = torch.cat(pred_); T = torch.cat(true_); MH = torch.cat(mask_); ME = torch.cat(eval_)
    
    # DEBUG: Let's measure the actual tensor sizes
    print(f"\n=== DEBUGGING TENSOR SIZES ===")
    print(f"P shape: {P.shape}, size: {P.numel() * 4 / 1024 / 1024:.1f} MB")
    print(f"T shape: {T.shape}, size: {T.numel() * 4 / 1024 / 1024:.1f} MB") 
    print(f"MH shape: {MH.shape}, size: {MH.numel() * 4 / 1024 / 1024:.1f} MB")
    print(f"ME shape: {ME.shape}, size: {ME.numel() * 4 / 1024 / 1024:.1f} MB")
    
    print(f"\nME.unsqueeze(-1) would have shape: {ME.unsqueeze(-1).shape}")
    print(f"ME.unsqueeze(-1) size: {ME.unsqueeze(-1).numel() * 4 / 1024 / 1024:.1f} MB")
    
    print(f"\nBroadcasted multiplication result would be:")
    print(f"Shape: {torch.broadcast_shapes(MH.shape, ME.unsqueeze(-1).shape)}")
    result_numel = torch.broadcast_shapes(MH.shape, ME.unsqueeze(-1).shape)
    result_size = result_numel[0] * result_numel[1] * result_numel[2] * 4 / 1024 / 1024
    print(f"Size: {result_size:.1f} MB")
    
    print(f"\nTotal memory needed for operation: {MH.numel() * 4 / 1024 / 1024 + ME.unsqueeze(-1).numel() * 4 / 1024 / 1024 + result_size:.1f} MB")
    print(f"================================\n")
    
    # combine masks into a float mask of shape (N, L_max, H)
    m_comb = MH * ME.unsqueeze(-1)                # zeros out any position not in test
    err    = (P - T) ** 2

    mse = (err * m_comb).sum() / m_comb.sum()

    # similarly for MAE
    abs_err = (P - T).abs()
    mae     = (abs_err * m_comb).sum() / m_comb.sum()
    print(f"[RESULT] test MSE {mse:.4f} | MAE {mae:.4f}")

    # get cve ids order for correct mapping in results: we use the same order as in the dataset constructor
    # (i.e. the order of the groupby keys)
    ordered_cves = list(df_te.groupby("cve", observed=True).groups.keys())

    np.savez("predictions_fullseq.npz",
        cves      = np.array(ordered_cves, dtype=object),  # CVE → i mapping
        pred      = P.numpy(),
        true      = T.numpy(),
        mask_h    = MH.numpy(),
        eval_mask = ME.numpy()
    )
    print("[INFO] saved → predictions_fullseq.npz")

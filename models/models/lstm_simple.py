#!/usr/bin/env python
# lstm_epss_fullsequence_cuda.py


import time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ───── device helper ────────────────────────────────────────────────
def get_device():
    # Checks if CUDA is available. Uses GPU if present, else CPU.
    # When a GPU is used, it enables benchmark mode for cudnn optimizations.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[INFO] GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("[WARN] CUDA unavailable → CPU")
    return torch.device("cpu")

# ───── cast helper ──────────────────────────────────────────────────
def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each column has proper type:
      - 'date' is converted to datetime
      - 'cve' is treated as a category
      - 'epss' is float32
      - 'age_epss_pub' is int32
    Then sorts by ['cve', 'date'] to maintain temporal order.
    """
    df = df.copy()
    df["date"]         = pd.to_datetime(df["date"], errors="raise")
    df["cve"]          = df["cve"].astype("category")
    df["epss"]         = df["epss"].astype("float32")
    df["age_epss_pub"] = df["age_epss_pub"].astype("int32")
    return df.sort_values(["cve", "date"]).reset_index(drop=True)

# ───── dataset ─────────────────────────────────────────────────────
class CVEFullDataset(Dataset):
    """
    For each CVE produce a padded sequence:
      X  : [L_max, 2]  -> (epss_t, age_t)
      Y  : [L_max, H]  -> (epss_{t+1 … t+H})
      m_t: [L_max]     -> valid timesteps
      m_h: [L_max, H]  -> valid future horizons
    """
    def __init__(self, df, L_max: int, horizon: int = 10):
        self.X, self.Y, self.m_t, self.m_h = [], [], [], []
        for _, grp in df.groupby("cve", observed=True):
            vals = grp[["epss","age_epss_pub"]].values.astype("float32")
            T    = len(vals)
            # pad inputs
            pad  = L_max - T
            self.X.append(torch.from_numpy(np.pad(vals, ((0,pad),(0,0)), "constant")))
            # timestep mask
            mt = np.zeros(L_max, np.float32); mt[:T] = 1
            self.m_t.append(torch.from_numpy(mt))
            # build future targets
            Y  = np.zeros((L_max, horizon), np.float32)
            mh = np.zeros_like(Y)
            for t in range(T):
                start = t + 1
                end   = min(start + horizon, T)
                k     = end - start          # number of real future days
                if k > 0:
                    Y[t, :k]  = vals[start:end, 0]   # epss_{t+1 …}
                    mh[t, :k] = 1
            self.Y.append(torch.from_numpy(Y))
            self.m_h.append(torch.from_numpy(mh))
        print(f"[INFO] CVEs: {len(self)}  L_max={L_max}")
    def __len__(self):  return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.Y[i], self.m_t[i], self.m_h[i]

def collate(batch):
    """
    Stacks each part of the tuple across batch dimension:
      - X:  (B, L_max, 2) 
      - Y:  (B, L_max, 10)
      - m_t: (B, L_max)
      - m_h: (B, L_max, 10)
    """
    return tuple(torch.stack(e, 0) for e in zip(*batch))

# ───── sanity check ────────────────────────────────────────────────
def sanity_check(pred, Y, m_t, m_h):
    B, L, H = pred.shape

    # ❶ Shapes
    assert Y.shape == pred.shape
    assert m_h.shape == (B, L, H)

    # ❷ No horizon mask outside real rows
    assert ((m_t == 0).unsqueeze(-1) & (m_h == 1)).sum() == 0

    # ❸ horizon‑0 mask == “a real next‑day exists”
    # shift m_t one step to the *left* and compare
    next_day_exists = torch.zeros_like(m_t)
    next_day_exists[:, :-1] = m_t[:, 1:]
    assert (m_h[:, :, 0] == next_day_exists).all(), "horizon‑0 mask wrong"

    # ❹ targets that are declared valid are non‑zero
    assert (Y[m_h.bool()] != 0).all()

    # ❺ finite loss
    loss = ((pred - Y)**2 * m_t.unsqueeze(-1) * m_h).sum() / m_h.sum()
    assert torch.isfinite(loss)

    print("[INFO] sanity‑check passed")



# ───── model ────────────────────────────────────────────────────────
class Seq2SeqLSTM(nn.Module):
    """
    Simple LSTM-based Seq2Seq:
      - Input dimension = 2 (epss + age_epss_pub)
      - Hidden dimension = 128
      - Output dimension = horizon (10)
      - dropout in LSTM layers
    """
    def __init__(self, input_dim=2, hidden_dim=128, layers=2,
                 horizon=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x shape: (B, L, input_dim)
        # Output shape from LSTM is (B, L, hidden_dim)
        h, _ = self.lstm(x)
        # Final linear transforms hidden states to predictions (B, L, output_dim)
        return self.head(h)


# ───── masked loss ─────────────────────────────────────────────────
def masked_mse(pred, true, m_t, m_h):
    """
    Calculates MSE only where masks are 1.
      - pred & true shapes: (B, L, horizon)
      - m_t shape: (B, L)
      - m_h shape: (B, L, horizon)
    The final MSE is scaled by the sum of valid entries.
    """
    err = (pred - true) ** 2
    err = err * m_t.unsqueeze(-1) * m_h
    return err.sum() / m_h.sum()

# ───── main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set a random seed for reproducibility
    torch.manual_seed(0)
    dev = get_device()

    # Paths to training, validation, test sets
    TRAIN = "data/full_db/ml-sets-sampled/logit-scaled/train"
    VAL   = "data/full_db/ml-sets-sampled/logit-scaled/val"
    TEST  = "data/full_db/ml-sets-sampled/logit-scaled/test"

    # Model hyperparameters
    HORIZON = 10
    BATCH   = 32
    EPOCHS  = 12
    LR      = 1e-3

    # ----- load data -----
    # 1) Read raw data from parquet
    # 2) Convert columns to appropriate types and sort
    df_tr = cast_types(pd.read_parquet(TRAIN, engine="pyarrow"))
    df_va = cast_types(pd.read_parquet(VAL,   engine="pyarrow"))
    df_te = cast_types(pd.read_parquet(TEST,  engine="pyarrow"))
    print(f"[INFO] rows  train {len(df_tr):,} | val {len(df_va):,} | test {len(df_te):,}")

    # sanity check: how many unique CVEs in each set?
    print(f"[INFO] unique CVEs train {df_tr['cve'].nunique():,} | val {df_va['cve'].nunique():,} | test {df_te['cve'].nunique():,}")    

    # Standard scaling for 'age_epss_pub' column
    # (Assumes it's numeric and can be standardized.)
    scaler = StandardScaler().fit(df_tr[["age_epss_pub"]])
    for df in (df_tr, df_va, df_te):
        df["age_epss_pub"] = scaler.transform(df[["age_epss_pub"]])

    # L_max = longest sequence among all CVE grouping, and future proof it: we could decide to prepend the training sequences to the cve's validation and test sequences: realistic data availability. But then they are longer than the training sequences -> need to decide on all sets what the max length is. Also, maybe real world deployment validity of model might go up: it is trained on shorter sequences than it is used on. So we observe deployment behavior in test and validation sets.
    max_tr = df_tr.groupby("cve", observed=True).size().max()
    max_va = df_va.groupby("cve", observed=True).size().max()
    max_te = df_te.groupby("cve", observed=True).size().max()
    L_max = max(max_tr, max_va, max_te)
    print("[INFO] longest time series across sets:", L_max)

    # Set up datasets and their corresponding Dataloaders
    tr_ds = CVEFullDataset(df_tr, L_max, HORIZON)
    va_ds = CVEFullDataset(df_va, L_max, HORIZON)
    te_ds = CVEFullDataset(df_te, L_max, HORIZON)
    tr_ld = DataLoader(tr_ds, BATCH, shuffle=True,  collate_fn=collate, num_workers=0,  pin_memory=True)
    va_ld = DataLoader(va_ds, BATCH, shuffle=False, collate_fn=collate, num_workers=0,  pin_memory=True)
    te_ld = DataLoader(te_ds, BATCH, shuffle=False, collate_fn=collate, num_workers=0,  pin_memory=True)

    print(f"[INFO] train batches {len(tr_ld):,} | val batches {len(va_ld):,} | test batches {len(te_ld):,}")
    print(f"[INFO] train shape {tr_ld.batch_size} | val batch size {va_ld.batch_size} | test batch size {te_ld.batch_size}")
    print(f"[INFO] train shape X  for first batch {tr_ld.dataset.X[0].shape} | val batch shape X {va_ld.dataset.X[0].shape} | test batch shape {te_ld.dataset.X[0].shape}")
    print(f"[INFO] train shape Y for first batch {tr_ld.dataset.Y[0].shape} | val batch shape Y {va_ld.dataset.Y[0].shape} | test batch shape {te_ld.dataset.Y[0].shape}")
    print(f"[INFO] train shape m_t for first batch {tr_ld.dataset.m_t[0].shape} | val batch shape m_t {va_ld.dataset.m_t[0].shape} | test batch shape {te_ld.dataset.m_t[0].shape}")
    print(f"[INFO] train shape m_h for first batch {tr_ld.dataset.m_h[0].shape} | val batch shape m_h {va_ld.dataset.m_h[0].shape} | test batch shape {te_ld.dataset.m_h[0].shape}")    
    # Create model instance, possibly compile for speed if CUDA and torch.compile are available
    model = Seq2SeqLSTM(horizon=HORIZON).to(dev)
    if hasattr(torch, "compile") and dev.type == "cuda":
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ----- training -----
    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        # keep track of batch count for sanity check
        batch_count = 0
        # Train loop with mask-aware MSE
        for X, Y, mt, mh in tqdm(tr_ld, desc=f"train {ep}/{EPOCHS}"):
            # Move data to device (GPU/CPU)
            X, Y, mt, mh = (z.to(dev, non_blocking=True) for z in (X, Y, mt, mh))
            opt.zero_grad()

            # Forward pass
            pred = model(X)
            # masked MSE loss
            loss = masked_mse(pred, Y, mt, mh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # sanity check: pred shape
            if ep==1 and batch_count==0:                 # run sanity check for shapes once for first epoch, first batch
                sanity_check(pred, Y, mt, mh)
            tr_loss += loss.item()
            batch_count += 1
        tr_loss /= len(tr_ld)
        
        
        # Validation loop, same masked MSE metric
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for X, Y, mt, mh in va_ld:
                X, Y, mt, mh = (z.to(dev, non_blocking=True) for z in (X, Y, mt, mh))
                va_loss += masked_mse(model(X), Y, mt, mh).item()
        va_loss /= len(va_ld)
        print(f"epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")

    # ----- test metrics -----
    # Collect predictions and compare to true values under the mask
    model.eval()
    pred_, true_, mask_ = [], [], []
    with torch.no_grad():
        for X, Y, mt, mh in te_ld:
            pred_.append(model(X.to(dev)).cpu())
            true_.append(Y)
            mask_.append(mh)
    P = torch.cat(pred_)
    T = torch.cat(true_)
    M = torch.cat(mask_)
    mse = ((P - T) ** 2)[M.bool()].mean().item()
    mae = (P - T).abs()[M.bool()].mean().item()
    print(f"[RESULT] test MSE {mse:.4f} | MAE {mae:.4f}")

    # ----- save csv -----
    # Save predictions, true values, and mask
    np.savez("predictions_fullseq.npz", pred=P.numpy(), true=T.numpy(), mask=M.numpy())
    print("[INFO] saved → predictions_fullseq.npz")

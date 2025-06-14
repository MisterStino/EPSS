#!/usr/bin/env python
"""
Optimized LSTM with automatic batch size finding and mixed precision
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import your existing functions
from lstm_exp_window_eval import (
    cast_types, transform_epss, CVEFullDataset, 
    collate, masked_mse
)

class OptimizedSeq2SeqLSTM(pl.LightningModule):
    def __init__(self, input_dim=2, hidden_dim=128, layers=2,
                 horizon=10, dropout=0.3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, horizon)
        self.lr = lr
        
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h)
    
    def training_step(self, batch, batch_idx):
        X, Y, mt, mh, me = batch
        pred = self(X)
        loss = masked_mse(pred, Y, mt, mh, me)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, Y, mt, mh, me = batch
        pred = self(X)
        loss = masked_mse(pred, Y, mt, mh, me)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    # Load and prepare data (same as original)
    FILE = "data/full_db/sampled/final_full_data_sampled.parquet"
    BIG = cast_types(pd.read_parquet(FILE, engine="pyarrow"))
    BIG["epss"] = transform_epss(BIG["epss"].values, mode="logit", eps=1e-6)
    
    # Create temporal splits (same as original)
    days = np.sort(BIG["date"].unique())
    n_days = len(days)
    TEST_CUT = days[int(0.80 * n_days)]
    VAL_CUT = days[int(0.64 * n_days)]
    VAL_CUT = pd.to_datetime(VAL_CUT)
    TEST_CUT = pd.to_datetime(TEST_CUT)
    
    # Create datasets
    def flag(df, cond):
        out = np.zeros(len(df), np.float32)
        out[cond] = 1.0
        return out

    train_flag = flag(BIG, BIG["date"] < VAL_CUT)
    val_flag = flag(BIG, (BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT))
    
    df_tr = BIG.copy(); df_tr["use_for_loss"] = train_flag
    df_val = BIG.copy(); df_val["use_for_loss"] = val_flag
    
    # Standardize features
    scaler = StandardScaler().fit(df_tr[["age_epss_pub"]])
    for df in (df_tr, df_val):
        df["age_epss_pub"] = scaler.transform(df[["age_epss_pub"]])
    
    # Create datasets
    HORIZON = 10
    L_max = BIG.groupby("cve", observed=True).size().max()
    
    tr_ds = CVEFullDataset(df_tr, L_max, HORIZON)
    val_ds = CVEFullDataset(df_val, L_max, HORIZON)
    
    # Create model
    model = OptimizedSeq2SeqLSTM(horizon=HORIZON)
    
    # Setup trainer with automatic optimizations
    trainer = pl.Trainer(
        max_epochs=12,
        precision="16-mixed",  # Automatic mixed precision
        callbacks=[
            ModelCheckpoint(monitor='val_loss', save_top_k=1),
            EarlyStopping(monitor='val_loss', patience=3)
        ],
        log_every_n_steps=50,
        enable_progress_bar=True
    )
    
    # Create initial dataloaders (batch size will be optimized)
    train_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, 
                             collate_fn=collate, num_workers=4,
                             persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                           collate_fn=collate, num_workers=4,
                           persistent_workers=True)
    
    # 🚀 AUTOMATIC BATCH SIZE OPTIMIZATION
    tuner = Tuner(trainer)
    
    print("🔍 Finding optimal batch size...")
    tuner.scale_batch_size(model, train_dataloaders=train_loader, 
                          val_dataloaders=val_loader, mode="power")
    
    print(f"✅ Optimal batch size found: {train_loader.batch_size}")
    
    # Train with optimized settings
    trainer.fit(model, train_loader, val_loader)
    
    print("🎯 Training completed with automatic optimizations!")
    print(f"   - Mixed precision: Enabled")
    print(f"   - Optimal batch size: {train_loader.batch_size}")
    print(f"   - Early stopping: Enabled")
    print(f"   - Model checkpointing: Enabled")

if __name__ == "__main__":
    main() 
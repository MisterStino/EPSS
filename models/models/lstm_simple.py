import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def get_device():
    """
    Returns the best available device:
      1) CUDA if available,
      2) then Intel GPU (torch.xpu) if available,
      3) otherwise CPU.
    Also prints info about CPU, NVIDIA GPU count, Intel GPU availability.
    """
    num_cpus = os.cpu_count()
    print(f"[INFO] Number of CPU cores available: {num_cpus}")
    
    if torch.cuda.is_available():
        num_nvidia = torch.cuda.device_count()
        print(f"[INFO] NVIDIA GPU(s) available: {num_nvidia}")
    else:
        print(f"[INFO] No NVIDIA GPU available (torch.cuda.is_available() == False).")
    
    has_xpu = False
    if getattr(torch, "xpu", None) is not None:
        # Check if xpu is truly available
        has_xpu = torch.xpu.is_available()
        print(f"[INFO] Intel GPU availability (torch.xpu.is_available()): {has_xpu}")
    else:
        print("[INFO] No Intel GPU extension (torch.xpu) found in this PyTorch version.")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif has_xpu:
        return torch.device("xpu")
    else:
        return torch.device("cpu")


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MyLSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=10, dropout=0.2):
        super(MyLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)  # out: (batch_size, seq_len, hidden_dim)
        last_output = out[:, -1, :]  # (batch_size, hidden_dim)
        y_hat = self.fc(last_output) # (batch_size, 10)
        return y_hat


def create_sequences(dataframe, lookback=30, horizon=10):
    X_list = []
    Y_list = []
    for cve, group in dataframe.groupby("cve"):
        group = group.sort_values("date")
        arr = group[["epss", "age_epss_pub"]].values  # shape: [time, 2]
        
        for i in range(len(arr) - lookback - horizon + 1):
            x_window = arr[i : i + lookback, :]  
            y_future = arr[i + lookback : i + lookback + horizon, 0]  # epss is col 0
            X_list.append(x_window)
            Y_list.append(y_future)
    return np.array(X_list), np.array(Y_list)

def cast_datatypes(df):
    """
    Casts the datatypes of the DataFrame columns to appropriate types.
    Raises early if any of the casts fail.
    """
    df = df.copy()
    
    # 1) datetime (errors='raise' will blow up if there are bad values)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    
    # 2) string identifiers → category
    df["cve"] = df["cve"].astype("category")
    
    # 3) numeric columns
    #    If you plan to Standard‐scale them anyway, float32 is fine.
    df["epss"] = pd.to_numeric(df["epss"], errors="raise").astype("float32")
    
    #    If you really want integer "days since publication", use an integer dtype:
    # df["age_epss_pub"] = pd.to_numeric(df["age_epss_pub"], errors="raise").astype("int32")
    #
    #    Or, if you prefer to treat age as a continuous feature and save half the
    #    memory vs float64, float32 is OK:
    df["age_epss_pub"] = pd.to_numeric(df["age_epss_pub"], errors="raise").astype('int32')
    
    return df

if __name__ == "__main__":
    #####################################
    # 1) Load and Sort Data
    #####################################
    df = pd.read_parquet("data/full_db/sampled/final_full_data_sampled.parquet")
    df = cast_datatypes(df)
    df = df.sort_values(by=["date", "cve"]).reset_index(drop=True)
    print("[INFO] Data loaded and sorted by date, cve.")
    print(df.head())

    #####################################
    # 2) Time-based Split
    #    Enough days for 30-day lookback + 10-day horizon
    #####################################
    train_end_date = pd.to_datetime("2023-12-30")
    val_end_date   = pd.to_datetime("2024-03-30")

    df_train = df[df["date"] <= train_end_date]
    df_val   = df[(df["date"] > train_end_date) & (df["date"] <= val_end_date)]
    df_test  = df[df["date"] > val_end_date]

    print(f"[INFO] Train size (rows): {len(df_train)}")
    print(f"[INFO] Val size   (rows): {len(df_val)}")
    print(f"[INFO] Test size  (rows): {len(df_test)}")

    #####################################
    # 3) Scaling: Fit on train, apply to val/test
    #####################################
    scaler = StandardScaler()
    scaler.fit(df_train[["epss", "age_epss_pub"]])

    # Use .loc to avoid SettingWithCopyWarning
    df_train.loc[:, ["epss", "age_epss_pub"]] = scaler.transform(df_train[["epss", "age_epss_pub"]])
    df_val.loc[:, ["epss", "age_epss_pub"]]   = scaler.transform(df_val[["epss", "age_epss_pub"]])
    df_test.loc[:, ["epss", "age_epss_pub"]]  = scaler.transform(df_test[["epss", "age_epss_pub"]])
    print("[INFO] Scaling done (fitted on train, applied to val/test).")

    #####################################
    # 4) Window creation
    #####################################
    lookback = 30
    horizon  = 10

    X_train, Y_train = create_sequences(df_train, lookback, horizon)
    X_val,   Y_val   = create_sequences(df_val,   lookback, horizon)
    X_test,  Y_test  = create_sequences(df_test,  lookback, horizon)

    print(f"[INFO] X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"[INFO] X_val:   {X_val.shape},   Y_val:   {Y_val.shape}")
    print(f"[INFO] X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")

    #####################################
    # 5) Dataset and DataLoader
    #####################################
    train_dataset = TimeSeriesDataset(X_train, Y_train)
    val_dataset   = TimeSeriesDataset(X_val,   Y_val)
    test_dataset  = TimeSeriesDataset(X_test,  Y_test)

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"[INFO] Batches -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    #####################################
    # 6) Model, Device, Loss, Optimizer
    #####################################
    device = get_device()
    print(f"[INFO] Device chosen: {device}")

    model = MyLSTMModel(input_dim=2, hidden_dim=64, num_layers=2, output_dim=10, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if hasattr(torch, "compile") and (device.type in ["cuda", "xpu"]):
        model = torch.compile(model)
        print("[INFO] Model compiled with torch.compile().")

    #####################################
    # 7) Training and Validation
    #####################################
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
        train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_valb, Y_valb in val_loader:
                X_valb = X_valb.to(device)
                Y_valb = Y_valb.to(device)
                val_pred = model(X_valb)
                val_loss_sum += criterion(val_pred, Y_valb).item()
        val_loss = val_loss_sum / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    #####################################
    # 8) Test Evaluation and Saving
    #####################################
    model.eval()
    test_preds_list = []
    test_tgts_list  = []

    with torch.no_grad():
        for X_testb, Y_testb in test_loader:
            X_testb = X_testb.to(device)
            Y_testb = Y_testb.to(device)
            Y_test_pred = model(X_testb)  # (batch_size, horizon)
            test_preds_list.append(Y_test_pred.cpu().numpy())
            test_tgts_list.append(Y_testb.cpu().numpy())

    test_preds = np.concatenate(test_preds_list, axis=0)
    test_tgts  = np.concatenate(test_tgts_list,  axis=0)

    test_mse = np.mean((test_preds - test_tgts)**2)
    test_mae = np.mean(np.abs(test_preds - test_tgts))
    print(f"[RESULT] Test MSE: {test_mse:.4f}")
    print(f"[RESULT] Test MAE: {test_mae:.4f}")

    df_preds = pd.DataFrame({f"Pred_Step_{i+1}": test_preds[:, i] for i in range(horizon)})
    df_tgts  = pd.DataFrame({f"True_Step_{i+1}": test_tgts[:, i] for i in range(horizon)})
    df_out   = pd.concat([df_preds, df_tgts], axis=1)
    df_out.to_csv("predictions_lstm.csv", index=False)
    print("[INFO] Predictions saved to 'predictions_lstm.csv'.")


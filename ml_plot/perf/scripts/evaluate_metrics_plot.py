import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xarray as xr

ds = xr.open_dataset("ml_plot\perf\files\predictions_stream.nc")

pred = ds["pred"].values          # [N, T, H]
true = ds["true"].values
mask_h = ds["mask_h"].values      # [N, T, H]
eval_mask = ds["eval_mask"].values  # [N, T]

print(f"Prediction shape: {pred.shape}")
print(f"Mask shape: {mask_h.shape}")
print(f"Eval mask shape: {eval_mask.shape}")

for h in range(pred.shape[2]):  # loop over horizons
    all_preds, all_trues = [], []

    for i in range(pred.shape[0]):
        valid = (mask_h[i, :, h] > 0) & (eval_mask[i, :] > 0)
        if np.any(valid):
            all_preds.append(pred[i, valid, h])
            all_trues.append(true[i, valid, h])

    if all_preds:
        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)

        mae = mean_absolute_error(trues, preds)
        rmse = mean_squared_error(trues, preds) ** 0.5
        mape = np.mean(np.abs((trues - preds) / (trues + 1e-8))) * 100
        smape = 100 * np.mean(2 * np.abs(trues - preds) / (np.abs(preds) + np.abs(trues) + 1e-8))
        r2 = r2_score(trues, preds)

        print(f"Horizon {h+1:2d}: MAE={mae:.4f}, RMSE={rmse:.4f}, "
              f"MAPE={mape:.2f}%, SMAPE={smape:.2f}%, R²={r2:.4f}")


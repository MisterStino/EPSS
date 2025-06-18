#!/usr/bin/env python3
"""Debug the date conversion issue in LSTM predictions"""

import torch
import numpy as np
import pandas as pd

# Simulate the problematic date conversion from LSTM script
print("=== DEBUGGING DATE CONVERSION ISSUE ===")

# This simulates what happens in the LSTM script
print("\n1. Original date tensor (int64 nanoseconds):")
# Example date tensor with some real dates and padding
date_tensor = torch.tensor([
    19436000000000,  # Some nanosecond timestamp
    19437000000000,  # Another nanosecond timestamp  
    19438000000000,  # Another nanosecond timestamp
    0,               # Padding value
    0,               # Padding value
], dtype=torch.int64)
print(f"Date tensor: {date_tensor}")

print("\n2. Converting to numpy:")
DT_numpy = date_tensor.numpy()
print(f"DT_numpy: {DT_numpy}")

print("\n3. Replace padding (0) with NaT:")
DT_numpy[DT_numpy == 0] = np.datetime64("NaT").view("int64")
print(f"DT_numpy after NaT replacement: {DT_numpy}")

print("\n4. View as datetime64[ns]:")
dates_converted = DT_numpy.view("datetime64[ns]")
print(f"Converted dates: {dates_converted}")

print("\n=== THE PROBLEM ===")
print("The issue is that the date values are being stored as small integers")
print("(like 19436, 19437) which when interpreted as nanoseconds since 1970")
print("give dates very close to 1970-01-01!")

print("\n=== WHAT SHOULD HAPPEN ===")
print("Dates should be proper timestamps like:")
proper_timestamp = pd.Timestamp("2023-01-01").value  # nanoseconds since 1970
print(f"Proper 2023-01-01 timestamp: {proper_timestamp}")
print(f"As datetime64: {np.datetime64(proper_timestamp, 'ns')}")

print("\n=== CHECKING ACTUAL DATA ===")
# Let's see what's in the Arrow file
import pyarrow as pa
import pyarrow.ipc as ipc

try:
    reader = ipc.open_file(pa.memory_map("work/epss_stage1.arrow", "r"))
    schema = reader.schema
    print(f"\nArrow schema:")
    for field in schema:
        if 'date' in field.name.lower():
            print(f"  {field.name}: {field.type}")
    
    # Get a sample batch
    batch = reader.get_batch(0)
    date_col_idx = None
    for i, field in enumerate(schema):
        if field.name == 'date':
            date_col_idx = i
            break
    
    if date_col_idx is not None:
        date_column = batch.columns[date_col_idx]
        print(f"\nFirst 5 date values from Arrow:")
        for i in range(min(5, len(date_column))):
            raw_value = date_column[i].value  # This is the .value that gets stored
            as_datetime = pd.to_datetime(raw_value, unit='ns')
            print(f"  Raw: {raw_value} → DateTime: {as_datetime}")
            
except Exception as e:
    print(f"Error reading Arrow file: {e}")

print("\n=== SOLUTION ===")
print("The problem is in the dataset_iterable.py line:")
print("buf_date.append(cols[self.col2idx['date']][r].value)")
print("\nThe .value gives nanoseconds, but they seem to be wrong values.")
print("We need to check how dates are being processed in the Arrow creation step.") 
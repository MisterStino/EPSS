import torch
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Iterator, Tuple, List, Optional
import json
import time


class OptimizedCVEDataset(IterableDataset):
    def __init__(self, arrow_path, horizon=30, max_sequence_length=200):
        super().__init__()
        self.arrow_path = Path(arrow_path)
        self.horizon = horizon
        self.max_sequence_length = max_sequence_length
        self._setup_columns()
        print(f"[OptimizedDataset] Max sequence length: {max_sequence_length}")
    
    def _setup_columns(self):
        memory_map = pa.memory_map(str(self.arrow_path), "r")
        file_reader = ipc.open_file(memory_map)
        schema = file_reader.schema
        
        with (self.arrow_path.parent / "vocab.json").open() as f:
            vocab = json.load(f)
        
        self.cat_cols = list(vocab.keys())
        field_names = [f.name for f in schema]
        
        self.bool_cols = [name for name in field_names 
                         if (name.startswith(('has_', 'is_')) and name not in self.cat_cols)]
        
        self.flag_cols = ["flag_train", "flag_val", "flag_test"]
        
        reserved = set(self.cat_cols + self.bool_cols + self.flag_cols + 
                      ["cve", "date", "epss"])
        
        self.num_cols = [f.name for f in schema 
                        if (f.name not in reserved and 
                            str(f.type) in ['double', 'float', 'int64', 'int32'])]
        
        memory_map.close()
    
    def __iter__(self):
        memory_map = pa.memory_map(str(self.arrow_path), "r")
        file_reader = ipc.open_file(memory_map)
        
        table = file_reader.read_all()
        cve_column = table["cve"]
        unique_cves = pc.unique(cve_column).to_pylist()
        
        print(f"[OptimizedDataset] Processing {len(unique_cves)} unique CVEs")
        
        for cve_id in unique_cves:
            mask = pc.equal(cve_column, cve_id)
            cve_data = table.filter(mask)
            cve_data = cve_data.sort_by("date")
            
            if len(cve_data) == 0:
                continue
                
            if len(cve_data) > self.max_sequence_length:
                cve_data = cve_data.slice(len(cve_data) - self.max_sequence_length)
            
            try:
                # Vectorized conversion
                num_arrays = []
                for col in self.num_cols:
                    if col in cve_data.column_names:
                        col_array = cve_data[col].to_numpy(zero_copy_only=False)
                        col_array = np.nan_to_num(col_array, nan=-100.0)
                        num_arrays.append(col_array)
                
                if not num_arrays:
                    continue
                    
                numeric_data = np.column_stack(num_arrays).astype(np.float32)
                
                # Boolean features
                bool_arrays = []
                for col in self.bool_cols:
                    if col in cve_data.column_names:
                        col_array = cve_data[col].to_numpy(zero_copy_only=False)
                        bool_arrays.append(col_array.astype(np.float32))
                
                boolean_data = np.column_stack(bool_arrays) if bool_arrays else np.zeros((len(cve_data), 0), dtype=np.float32)
                
                # Categorical features  
                cat_arrays = []
                for col in self.cat_cols:
                    if col in cve_data.column_names:
                        col_array = cve_data[col].to_numpy(zero_copy_only=False)
                        cat_arrays.append(col_array)
                
                categorical_data = np.column_stack(cat_arrays).astype(np.int64) if cat_arrays else np.zeros((len(cve_data), 0), dtype=np.int64)
                
                # EPSS values
                epss_array = cve_data["epss"].to_numpy(zero_copy_only=False).astype(np.float32)
                
                # Flags
                flag_arrays = []
                for col in self.flag_cols:
                    if col in cve_data.column_names:
                        col_array = cve_data[col].to_numpy(zero_copy_only=False)
                        flag_arrays.append(col_array)
                
                flags_data = np.column_stack(flag_arrays).astype(np.uint8) if flag_arrays else np.zeros((len(cve_data), len(self.flag_cols)), dtype=np.uint8)
                
                # Dates  
                date_array = cve_data["date"].to_numpy(zero_copy_only=False).astype(np.int64)
                
                yield (
                    torch.from_numpy(numeric_data),
                    torch.from_numpy(boolean_data),
                    torch.from_numpy(categorical_data),
                    torch.from_numpy(epss_array),
                    torch.from_numpy(flags_data),
                    torch.from_numpy(date_array),
                )
                
            except Exception as e:
                print(f"[WARNING] Failed to process CVE {cve_id}: {e}")
                continue
        
        memory_map.close()


def optimized_collate_fn(batch, flag_kind="train", horizon=30):
    split_idx = {"train": 0, "val": 1, "test": 2}[flag_kind]
    
    (nums, bools, cats, epss_vals, flags, dates) = zip(*batch)
    
    # Build targets and masks
    targets_list = []
    horizon_masks_list = []
    
    for epss in epss_vals:
        T = len(epss)
        targets = torch.zeros(T, horizon, dtype=torch.float32)
        masks = torch.zeros(T, horizon, dtype=torch.uint8)
        
        for t in range(T):
            available = min(horizon, T - t - 1)
            if available > 0:
                targets[t, :available] = epss[t+1:t+1+available]
                masks[t, :available] = 1
        
        targets_list.append(targets)
        horizon_masks_list.append(masks)
    
    # Pad sequences
    num_pad = pad_sequence(nums, batch_first=True, padding_value=0.0)
    bool_pad = pad_sequence(bools, batch_first=True, padding_value=0.0)
    cat_pad = pad_sequence(cats, batch_first=True, padding_value=0)
    targets_pad = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
    horizon_masks_pad = pad_sequence(horizon_masks_list, batch_first=True, padding_value=0)
    
    # Time masks
    lengths = [len(seq) for seq in nums]
    max_len = max(lengths)
    time_masks = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, length in enumerate(lengths):
        time_masks[i, :length] = 1.0
    
    # Eval masks
    eval_masks = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, flag_seq in enumerate(flags):
        eval_masks[i, :len(flag_seq)] = flag_seq[:, split_idx].float()
    
    date_pad = pad_sequence(dates, batch_first=True, padding_value=0)
    
    return (num_pad, bool_pad, cat_pad, targets_pad, 
            time_masks, horizon_masks_pad, eval_masks, date_pad)


def create_optimized_dataloader(arrow_path, batch_size=32, flag_kind="train", max_sequence_length=200):
    dataset = OptimizedCVEDataset(arrow_path, max_sequence_length=max_sequence_length)
    
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(4, max(1, cpu_count // 2))
    
    from functools import partial
    collate_fn = partial(optimized_collate_fn, flag_kind=flag_kind)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=optimal_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=optimal_workers > 0,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    arrow_path = Path("ml_pipeline/work/epss_stage1.arrow")
    if arrow_path.exists():
        print("Testing optimized data loader...")
        loader = create_optimized_dataloader(arrow_path, batch_size=16, max_sequence_length=150)
        
        start_time = time.time()
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            print(f"Batch {i+1}: {[t.shape if hasattr(t, 'shape') else len(t) for t in batch]}")
        
        elapsed = time.time() - start_time
        print(f"Time for 3 batches: {elapsed:.2f}s ({elapsed/3:.3f}s per batch)")
    else:
        print("Arrow file not found.") 
#!/usr/bin/env python3
"""
Data Loading Performance Profiler
=================================

This script profiles the data loading pipeline to identify performance bottlenecks.
It measures timing at each stage of the data loading process.

Usage: python -m ml_pipeline.profile_data_loading
"""

import time
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
import pyarrow as pa
import pyarrow.ipc as ipc

# Import your components
from ml_pipeline.training.dataset_iterable import CVEIterableDataset, pad_and_mask


class DataLoadingProfiler:
    """Profiles data loading performance at each stage"""
    
    def __init__(self):
        self.results = {}
        
    def log(self, message):
        print(f"[PROFILER] {message}")
        
    def time_operation(self, name, func, *args, **kwargs):
        """Time an operation and store results"""
        self.log(f"Starting: {name}")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        self.results[name] = elapsed
        self.log(f"Completed: {name} in {elapsed:.3f}s")
        return result
    
    def profile_arrow_file_access(self, arrow_path):
        """Profile Arrow file access patterns"""
        self.log("=" * 50)
        self.log("PROFILING ARROW FILE ACCESS")
        self.log("=" * 50)
        
        # File size
        file_size = Path(arrow_path).stat().st_size
        self.log(f"Arrow file size: {file_size / 1024 / 1024:.1f} MB")
        
        # Memory mapping
        def map_file():
            return pa.memory_map(str(arrow_path), "r")
        
        memory_map = self.time_operation("Arrow memory mapping", map_file)
        
        # Opening file
        def open_file():
            return ipc.open_file(memory_map)
        
        file_reader = self.time_operation("Arrow file opening", open_file)
        
        # Schema access
        def get_schema():
            return file_reader.schema
        
        schema = self.time_operation("Schema access", get_schema)
        self.log(f"Schema fields: {len(schema)}")
        
        # Batch info
        num_batches = file_reader.num_record_batches
        self.log(f"Number of record batches: {num_batches}")
        
        # First batch access
        def get_first_batch():
            return file_reader.get_batch(0)
        
        first_batch = self.time_operation("First batch access", get_first_batch)
        self.log(f"First batch rows: {first_batch.num_rows}")
        
        return file_reader, schema
    
    def profile_dataset_iteration(self, arrow_path, num_samples=5):
        """Profile dataset iteration speed"""
        self.log("=" * 50)
        self.log("PROFILING DATASET ITERATION")
        self.log("=" * 50)
        
        # Dataset creation
        def create_dataset():
            return CVEIterableDataset(arrow_path, horizon=30)
        
        dataset = self.time_operation("Dataset creation", create_dataset)
        
        # Manual iteration (no DataLoader)
        self.log(f"Testing manual iteration for {num_samples} samples...")
        
        def iterate_samples():
            count = 0
            for sample in dataset:
                count += 1
                if count >= num_samples:
                    break
            return count
        
        sample_count = self.time_operation(f"Manual iteration ({num_samples} samples)", iterate_samples)
        
        if sample_count > 0:
            avg_time = self.results[f"Manual iteration ({num_samples} samples)"] / sample_count
            self.log(f"Average time per sample: {avg_time:.3f}s")
        
        return dataset
    
    def profile_dataloader(self, dataset, batch_sizes=[1, 4, 16, 64], num_batches=3):
        """Profile DataLoader with different configurations"""
        self.log("=" * 50)
        self.log("PROFILING DATALOADER CONFIGURATIONS")
        self.log("=" * 50)
        
        configs = [
            {"num_workers": 0, "pin_memory": False},
            {"num_workers": 0, "pin_memory": True},
        ]
        
        if torch.cuda.is_available():
            self.log("CUDA available - testing with pin_memory")
        else:
            self.log("CUDA not available - CPU only")
        
        for batch_size in batch_sizes:
            self.log(f"\nTesting batch_size={batch_size}")
            
            for config in configs:
                config_name = f"bs{batch_size}_nw{config['num_workers']}_pm{config['pin_memory']}"
                
                def create_and_test_loader():
                    # Recreate dataset to reset iterator
                    test_dataset = CVEIterableDataset(dataset.arrow_path, horizon=30)
                    
                    loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=partial(pad_and_mask, flag_kind="train", horizon=30),
                        **config
                    )
                    
                    # Time loading a few batches
                    batch_times = []
                    for i, batch in enumerate(loader):
                        if i >= num_batches:
                            break
                        batch_start = time.time()
                        # Simulate processing (move to device if needed)
                        if torch.cuda.is_available() and config['pin_memory']:
                            batch = [t.cuda() if isinstance(t, torch.Tensor) else t for t in batch]
                        batch_time = time.time() - batch_start
                        batch_times.append(batch_time)
                    
                    return sum(batch_times) / len(batch_times) if batch_times else 0
                
                avg_batch_time = self.time_operation(config_name, create_and_test_loader)
                self.log(f"  {config_name}: {avg_batch_time:.3f}s avg per batch")
    
    def profile_collate_function(self, dataset, batch_sizes=[1, 4, 16, 64]):
        """Profile the collate function separately"""
        self.log("=" * 50)
        self.log("PROFILING COLLATE FUNCTION")
        self.log("=" * 50)
        
        # Collect some raw samples
        samples = []
        count = 0
        for sample in dataset:
            samples.append(sample)
            count += 1
            if count >= max(batch_sizes):
                break
        
        for batch_size in batch_sizes:
            batch_samples = samples[:batch_size]
            
            def time_collate():
                return pad_and_mask(batch_samples, flag_kind="train", horizon=30)
            
            result = self.time_operation(f"Collate batch_size={batch_size}", time_collate)
            
            # Analyze result shapes
            if result:
                shapes = [t.shape if hasattr(t, 'shape') else len(t) for t in result]
                self.log(f"  Result shapes: {shapes}")
    
    def analyze_arrow_file_structure(self, arrow_path):
        """Analyze the Arrow file structure for optimization opportunities"""
        self.log("=" * 50)
        self.log("ANALYZING ARROW FILE STRUCTURE")
        self.log("=" * 50)
        
        memory_map = pa.memory_map(str(arrow_path), "r")
        file_reader = ipc.open_file(memory_map)
        
        schema = file_reader.schema
        self.log(f"Total columns: {len(schema)}")
        
        # Analyze column types
        type_counts = {}
        for field in schema:
            type_name = str(field.type)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        self.log("Column types:")
        for type_name, count in sorted(type_counts.items()):
            self.log(f"  {type_name}: {count} columns")
        
        # Analyze batch sizes
        batch_sizes = []
        total_rows = 0
        for i in range(file_reader.num_record_batches):
            batch = file_reader.get_batch(i)
            batch_sizes.append(batch.num_rows)
            total_rows += batch.num_rows
        
        self.log(f"Total rows: {total_rows:,}")
        self.log(f"Number of batches: {len(batch_sizes)}")
        self.log(f"Average batch size: {sum(batch_sizes) / len(batch_sizes):.1f}")
        self.log(f"Min batch size: {min(batch_sizes)}")
        self.log(f"Max batch size: {max(batch_sizes)}")
        
        # Estimate CVE count (rough)
        sample_batch = file_reader.get_batch(0)
        sample_cves = set()
        cve_col_idx = None
        for i, field in enumerate(schema):
            if field.name == "cve":
                cve_col_idx = i
                break
        
        if cve_col_idx is not None:
            cve_column = sample_batch.columns[cve_col_idx]
            for j in range(min(1000, sample_batch.num_rows)):  # Sample first 1000 rows
                sample_cves.add(cve_column[j].as_py())
            
            self.log(f"Estimated CVEs (from first 1000 rows): {len(sample_cves)}")
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        self.log("=" * 50)
        self.log("PERFORMANCE ANALYSIS SUMMARY")
        self.log("=" * 50)
        
        # Sort results by time
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        
        self.log("Slowest operations:")
        for i, (name, time_taken) in enumerate(sorted_results[:10]):
            self.log(f"{i+1:2d}. {name}: {time_taken:.3f}s")
        
        # Look for specific bottlenecks
        bottlenecks = []
        
        # Check if Arrow operations are slow
        arrow_ops = [name for name in self.results.keys() if 'Arrow' in name or 'batch' in name]
        arrow_time = sum(self.results[name] for name in arrow_ops)
        if arrow_time > 1.0:
            bottlenecks.append(f"Arrow file operations slow: {arrow_time:.3f}s total")
        
        # Check if collate function is slow
        collate_ops = [name for name in self.results.keys() if 'Collate' in name]
        if collate_ops:
            avg_collate = sum(self.results[name] for name in collate_ops) / len(collate_ops)
            if avg_collate > 0.1:
                bottlenecks.append(f"Collate function slow: {avg_collate:.3f}s average")
        
        # Check if iteration is slow
        iter_ops = [name for name in self.results.keys() if 'iteration' in name]
        if iter_ops:
            iter_time = self.results[iter_ops[0]]
            if iter_time > 2.0:
                bottlenecks.append(f"Dataset iteration slow: {iter_time:.3f}s")
        
        if bottlenecks:
            self.log("\nIDENTIFIED BOTTLENECKS:")
            for bottleneck in bottlenecks:
                self.log(f"  ⚠️  {bottleneck}")
        else:
            self.log("\nNo major bottlenecks identified in profiled operations")
    
    def run_full_profile(self, arrow_path):
        """Run complete profiling suite"""
        self.log("🔍 STARTING COMPREHENSIVE DATA LOADING PROFILE")
        self.log(f"Arrow file: {arrow_path}")
        print()
        
        try:
            # Profile Arrow file access
            file_reader, schema = self.profile_arrow_file_access(arrow_path)
            
            # Analyze file structure
            self.analyze_arrow_file_structure(arrow_path)
            
            # Profile dataset iteration
            dataset = self.profile_dataset_iteration(arrow_path, num_samples=3)
            
            # Profile collate function
            self.profile_collate_function(dataset, batch_sizes=[1, 4, 16])
            
            # Profile DataLoader
            self.profile_dataloader(dataset, batch_sizes=[1, 4, 16], num_batches=2)
            
            # Generate report
            self.generate_performance_report()
            
        except Exception as e:
            self.log(f"Error during profiling: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main profiling function"""
    arrow_path = Path("work/epss_stage1.arrow")
    
    if not arrow_path.exists():
        print(f"Error: Arrow file not found at {arrow_path}")
        print("Please run the pipeline first to generate the Arrow file.")
        return
    
    profiler = DataLoadingProfiler()
    profiler.run_full_profile(arrow_path)


if __name__ == "__main__":
    main() 
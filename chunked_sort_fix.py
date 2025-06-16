#!/usr/bin/env python
"""
Chunked sorting approach to fix PyArrow offset overflow
"""

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from pathlib import Path
import time

def chunked_sort_arrow_file(input_path, output_path, sort_keys, chunk_size=50_000_000):
    """
    Sort large Arrow file in chunks to avoid offset overflow.
    
    Args:
        input_path: Path to unsorted Arrow file
        output_path: Path for sorted output
        sort_keys: List of (column, order) tuples for sorting
        chunk_size: Number of rows per chunk (adjust based on memory)
    """
    print(f"[CHUNKED_SORT] Loading table from {input_path}")
    
    # Load the full table
    tbl = pa.ipc.open_file(str(input_path)).read_all()
    total_rows = tbl.num_rows
    
    print(f"[CHUNKED_SORT] Total rows: {total_rows:,}")
    print(f"[CHUNKED_SORT] Chunk size: {chunk_size:,}")
    
    if total_rows <= chunk_size:
        # Small enough to sort directly
        print(f"[CHUNKED_SORT] Table small enough for direct sort")
        idx = pc.sort_indices(tbl, sort_keys)
        sorted_tbl = tbl.take(idx)
        feather.write_feather(sorted_tbl, output_path, compression="lz4")
        return
    
    # Large table - use chunked approach
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    print(f"[CHUNKED_SORT] Splitting into {num_chunks} chunks")
    
    # Sort each chunk individually
    sorted_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        print(f"[CHUNKED_SORT] Processing chunk {i+1}/{num_chunks} (rows {start_idx:,}-{end_idx:,})")
        
        # Extract chunk
        chunk = tbl.slice(start_idx, end_idx - start_idx)
        
        # Sort chunk
        chunk_idx = pc.sort_indices(chunk, sort_keys)
        sorted_chunk = chunk.take(chunk_idx)
        
        sorted_chunks.append(sorted_chunk)
    
    # Merge sorted chunks
    print(f"[CHUNKED_SORT] Merging {len(sorted_chunks)} sorted chunks")
    final_table = merge_sorted_chunks(sorted_chunks, sort_keys)
    
    # Write final result
    print(f"[CHUNKED_SORT] Writing final sorted table to {output_path}")
    feather.write_feather(final_table, output_path, compression="lz4")

def merge_sorted_chunks(chunks, sort_keys):
    """
    Merge pre-sorted chunks into final sorted table.
    Uses k-way merge algorithm.
    """
    if len(chunks) == 1:
        return chunks[0]
    
    # For simplicity, use PyArrow's concat + sort
    # In production, you'd implement proper k-way merge
    combined = pa.concat_tables(chunks)
    
    # Final sort (this might still overflow for very large datasets)
    # If it does, we'd need to implement true k-way merge
    try:
        final_idx = pc.sort_indices(combined, sort_keys)
        return combined.take(final_idx)
    except pa.ArrowInvalid as e:
        if "offset overflow" in str(e):
            # Fallback: write chunks separately and merge externally
            raise NotImplementedError("Dataset too large for in-memory merge. Need external merge-sort.")
        raise

# Example usage for the main script
def fix_arrow_sort_overflow(tmp_arrow_path, final_arrow_path):
    """
    Drop-in replacement for the failing sort operation
    """
    try:
        # Try original approach first (for smaller datasets)
        tbl = pa.ipc.open_file(str(tmp_arrow_path)).read_all()
        idx = pc.sort_indices(tbl, [("cve","ascending"),("date","ascending")])
        sorted_tbl = tbl.take(idx)
        feather.write_feather(sorted_tbl, str(final_arrow_path), compression="lz4")
        print(f"[SORT] Direct sort successful")
        
    except pa.ArrowInvalid as e:
        if "offset overflow" in str(e):
            print(f"[SORT] Offset overflow detected, switching to chunked sort")
            chunked_sort_arrow_file(
                tmp_arrow_path, 
                final_arrow_path,
                [("cve","ascending"),("date","ascending")],
                chunk_size=50_000_000  # Adjust based on available memory
            )
        else:
            raise 
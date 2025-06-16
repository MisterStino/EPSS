#!/usr/bin/env python
"""
External sorting approach using DuckDB to avoid PyArrow limitations
"""

import duckdb
import pyarrow.feather as feather
from pathlib import Path
import time

def external_sort_via_duckdb(tmp_arrow_path, final_arrow_path):
    """
    Use DuckDB's external sorting to handle arbitrarily large datasets.
    This completely bypasses PyArrow's offset limitations.
    """
    print(f"[EXTERNAL_SORT] Using DuckDB external sort")
    
    con = duckdb.connect(":memory:")
    
    # Configure DuckDB for large external sorts
    con.execute("SET memory_limit='8GB'")  # Adjust based on available RAM
    con.execute("SET threads=16")
    con.execute("SET temp_directory='/tmp'")  # Ensure temp space available
    
    t0 = time.time()
    
    # Read from Arrow file, sort externally, write to new Arrow file
    con.execute(f"""
        COPY (
            SELECT * FROM read_ipc('{tmp_arrow_path}')
            ORDER BY cve, date
        ) TO '{final_arrow_path}' (FORMAT 'arrow', COMPRESSION 'lz4')
    """)
    
    elapsed = time.time() - t0
    print(f"[EXTERNAL_SORT] Complete ({elapsed:.1f}s)")
    
    # Verify the result
    result_table = feather.read_table(str(final_arrow_path))
    print(f"[EXTERNAL_SORT] Sorted {result_table.num_rows:,} rows")
    
    return True

def hybrid_sort_approach(tmp_arrow_path, final_arrow_path, memory_threshold=100_000_000):
    """
    Hybrid approach: try PyArrow first, fallback to DuckDB external sort
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    
    # Check table size first
    tbl = pa.ipc.open_file(str(tmp_arrow_path)).read_all()
    num_rows = tbl.num_rows
    
    print(f"[HYBRID_SORT] Table has {num_rows:,} rows")
    
    if num_rows <= memory_threshold:
        # Small enough for PyArrow in-memory sort
        try:
            print(f"[HYBRID_SORT] Attempting PyArrow in-memory sort")
            idx = pc.sort_indices(tbl, [("cve","ascending"),("date","ascending")])
            sorted_tbl = tbl.take(idx)
            feather.write_feather(sorted_tbl, str(final_arrow_path), compression="lz4")
            print(f"[HYBRID_SORT] PyArrow sort successful")
            return True
            
        except pa.ArrowInvalid as e:
            if "offset overflow" in str(e):
                print(f"[HYBRID_SORT] PyArrow overflow, falling back to DuckDB")
            else:
                raise
    else:
        print(f"[HYBRID_SORT] Table too large for PyArrow, using DuckDB")
    
    # Fallback to DuckDB external sort
    return external_sort_via_duckdb(tmp_arrow_path, final_arrow_path) 
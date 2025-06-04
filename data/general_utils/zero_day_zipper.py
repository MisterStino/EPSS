"""
ZeroDayZipper: A robust feature merger for EPSS time-series data.

This class provides a clean, automated way to merge any feature dataset onto the 
EPSS time-series table while maintaining data integrity and following best practices.

Example usage:
    # Basic usage with file paths
    zipper = ZeroDayZipper(
        epss_path="data/epss/processed/epss_processed.parquet",
        feature_source="github",
        feature_file="github_commit_timestamps_9k.csv"
    )
    result_df = zipper.merge()
    
    # Advanced usage with custom DataFrames
    zipper = ZeroDayZipper(epss_df=my_epss_df)
    result_df = zipper.add_features(my_feature_df, source_name="custom")
"""

import os
import time
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from t3_spark.session import get_spark_session

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress indicator
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass


class ZeroDayZipper:
    """
    A robust feature merger for EPSS time-series data.
    
    This class handles the complete pipeline of merging feature data onto EPSS
    time-series while ensuring data integrity, proper normalization, and 
    comprehensive validation.
    """
    
    def __init__(
        self,
        epss_path: Optional[str] = None,
        epss_df: Optional[DataFrame] = None,
        feature_source: Optional[str] = None,
        feature_file: Optional[str] = None,
        feature_df: Optional[DataFrame] = None,
        spark_session: Optional[Any] = None,
        verbose: bool = True,
        fun_mode: bool = True,
        auto_fill_nulls: bool = False
    ):
        """
        Initialize the ZeroDayZipper.
        
        Args:
            epss_path: Path to EPSS parquet file (default: standard location)
            epss_df: Pre-loaded EPSS DataFrame (alternative to epss_path)
            feature_source: Source name (e.g., 'github', 'reddit') for file-based loading
            feature_file: Feature filename in data/<source>/raw/ directory
            feature_df: Pre-loaded feature DataFrame (alternative to file-based loading)
            spark_session: Custom Spark session (will create one if not provided)
            verbose: Whether to print progress information
            fun_mode: Whether to show battle cries and ASCII art
            auto_fill_nulls: Whether to automatically fill nulls (default: False, shows in red)
        """
        self.verbose = verbose
        self.fun_mode = fun_mode
        self.auto_fill_nulls = auto_fill_nulls
        self.start_time = time.time()
        self.spark = spark_session or get_spark_session("ZeroDayZipper")
        
        if self.fun_mode:
            self._show_banner()
        
        # Initialize EPSS data with persistence
        self.epss_df = self._load_epss_data(epss_path, epss_df).persist()
        self.epss_rows = self.epss_df.count()  # Cache the count
        
        # Initialize feature data if provided
        self.feature_source = feature_source
        self.feature_df = None
        if feature_df is not None:
            self.feature_df = self._prepare_feature_dataframe(feature_df, feature_source or "custom")
        elif feature_source and feature_file:
            self.feature_df = self._load_feature_data(feature_source, feature_file)
        
        # Track merged results
        self.merged_df: Optional[DataFrame] = None
        self.feature_columns: List[str] = []
        
        if self.verbose:
            self._log(f"✓ ZeroDayZipper initialized with EPSS data ({self.epss_rows:,} rows)")
            if self.feature_df:
                feature_rows = self.feature_df.count()
                self._log(f"✓ Feature data loaded: {self.feature_source} ({feature_rows:,} rows)")
    
    def _show_banner(self) -> None:
        """Show the ZeroDayZipper ASCII banner."""
        banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║  ⚡ ZERO-DAY ZIPPER ⚡  │  🔥 FEATURE FUSION ENGINE 🔥 ║
    ║                                                          ║
    ║     ████████╗██████╗ ██████╗                             ║
    ║        ███╔══██╗██╔══██╗██╔══██╗                         ║
    ║        ███║  ██║██║  ██║██████╔╝                         ║
    ║        ███║  ██║██║  ██║██╔══██╗                         ║
    ║        ███████╔╝██████╔╝██║  ██║                         ║
    ║        ╚══════╝ ╚═════╝ ╚═╝  ╚═╝                         ║
    ║                                                          ║
    ║  🎯 Target: EPSS Time-Series  │  🚀 Mission: Data Fusion║
    ╚══════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def _battle_cry(self, message: str) -> None:
        """Show a battle cry with some flair."""
        if self.fun_mode and self.verbose:
            print(f"\n🎮 {message}")
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ZeroDayZipper] {message}")
    
    def _progress_bar(self, description: str, total: int = 100):
        """Create a progress bar context manager."""
        if TQDM_AVAILABLE and self.fun_mode:
            return tqdm(total=total, desc=description, bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}')
        else:
            return tqdm()  # Fallback no-op
    
    def _load_epss_data(self, epss_path: Optional[str], epss_df: Optional[DataFrame]) -> DataFrame:
        """Load and prepare EPSS data."""
        self._battle_cry("🔌 Jacked-in! Siphoning bits from the matrix …")
        
        with self._progress_bar("Loading EPSS data") as pbar:
            if epss_df is not None:
                df = epss_df
                pbar.update(50)
            else:
                path = epss_path or "data/epss/processed/epss_processed.parquet"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"EPSS data not found at: {path}")
                df = self.spark.read.parquet(path)
                pbar.update(50)
            
            # Normalize EPSS data
            self._battle_cry("🧼 Scrubbing CVEs—now 99.9% germ-free!")
            normalized_df = (df
                           .withColumn("cve", F.trim(F.upper(F.col("cve"))))
                           .withColumn("date", F.to_date(F.col("date"))))
            pbar.update(50)
            
        return normalized_df
    
    def _load_feature_data(self, source: str, filename: str) -> DataFrame:
        """Load feature data from file."""
        self._battle_cry("🔌 Jacked-in! Siphoning bits from the matrix …")
        
        feature_path = f"data/{source}/raw/{filename}"
        
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature data not found at: {feature_path}")
        
        with self._progress_bar(f"Loading {source} data") as pbar:
            # Generic loader for CSV or Parquet
            if feature_path.lower().endswith(".csv"):
                df = (self.spark.read
                      .option("header", "true")
                      .option("inferSchema", "true")
                      .csv(feature_path))
                pbar.update(50)
            elif feature_path.lower().endswith(".parquet"):
                df = self.spark.read.parquet(feature_path)
                pbar.update(50)
            else:
                raise ValueError(f"Unsupported file format. Expected .csv or .parquet, got: {feature_path}")
            
            prepared_df = self._prepare_feature_dataframe(df, source)
            pbar.update(50)
        
        return prepared_df
    
    def _prepare_feature_dataframe(self, df: DataFrame, source_name: str) -> DataFrame:
        """Prepare and normalize feature DataFrame."""
        # Detect and harmonize column names
        cve_col = self._detect_cve_column(df.columns)
        date_col = self._detect_date_column(df.columns)
        
        # Rename and normalize
        self._battle_cry("🧼 Scrubbing CVEs—now 99.9% germ-free!")
        prepared_df = (df
                      .withColumnRenamed(cve_col, "cve")
                      .withColumnRenamed(date_col, "date")
                      .withColumn("cve", F.trim(F.upper(F.col("cve"))))
                      .withColumn("date", F.to_date(F.col("date"))))
        
        # Fast duplicate check - much more efficient than groupBy
        self._battle_cry("🕵️‍♂️ Clone patrol engaged… EXECUTING ORDER 66!")
        self._fast_duplicate_check(prepared_df, source_name)
        
        # Return with persistence and deduplication
        return prepared_df.dropDuplicates(["cve", "date"]).persist()
    
    def _fast_duplicate_check(self, df: DataFrame, source_name: str) -> None:
        """Fast duplicate check using count comparison instead of expensive groupBy."""
        original_count = df.count()
        deduped_count = df.dropDuplicates(["cve", "date"]).count()
        
        if deduped_count != original_count:
            duplicates_found = original_count - deduped_count
            self._battle_cry(f"⚠️ {duplicates_found} duplicate keys detected in {source_name} – will be dropped!")
            
            if self.verbose:
                self._log(f"Duplicate summary: {original_count:,} → {deduped_count:,} rows ({duplicates_found:,} duplicates removed)")
    
    def _detect_cve_column(self, columns: List[str]) -> str:
        """Detect CVE column name from available columns."""
        cve_candidates = [c for c in columns if c.lower() in {"cve", "cve_id"}]
        if not cve_candidates:
            raise ValueError(f"No CVE column found. Expected 'cve' or 'cve_id', got: {columns}")
        return cve_candidates[0]
    
    def _detect_date_column(self, columns: List[str]) -> str:
        """Detect date column name from available columns."""
        date_candidates = [c for c in columns if c.lower() == "date"]
        if not date_candidates:
            raise ValueError(f"No date column found. Expected 'date', got: {columns}")
        return date_candidates[0]
    
    def _show_null_summary(self, df: DataFrame, feature_cols: List[str]) -> None:
        """Show summary of null values in red."""
        if not feature_cols:
            return
            
        print("\n🔍 NULL VALUE SUMMARY:")
        print("=" * 50)
        
        for col in feature_cols:
            null_count = df.filter(F.col(col).isNull()).count()
            total_count = df.count()
            null_pct = (null_count / total_count * 100) if total_count > 0 else 0
            
            if null_count > 0:
                # Red color for null values
                print(f"\033[91m❌ {col}: {null_count:,} nulls ({null_pct:.2f}%)\033[0m")
            else:
                print(f"✅ {col}: No nulls")
        
        print("=" * 50)
        if not self.auto_fill_nulls:
            print("💡 Tip: Set auto_fill_nulls=True to automatically fill nulls with 0")
    
    def add_features(
        self, 
        feature_df: DataFrame, 
        source_name: str = "custom",
        fill_value: Union[int, float, str] = 0
    ) -> 'ZeroDayZipper':
        """
        Add features from a DataFrame to the current dataset.
        
        Args:
            feature_df: DataFrame containing features to merge
            source_name: Name for this feature source (for logging)
            fill_value: Value to fill missing features (only used if auto_fill_nulls=True)
            
        Returns:
            Self for method chaining
        """
        prepared_df = self._prepare_feature_dataframe(feature_df, source_name)
        
        if self.merged_df is None:
            # First merge - use EPSS as base
            base_df = self.epss_df
            expected_count = self.epss_rows  # Use cached count
        else:
            # Subsequent merge - use existing merged result
            base_df = self.merged_df
            expected_count = base_df.count()
        
        # Perform LEFT join with broadcast hint for small feature tables
        self._battle_cry("💥 Fusion core online. Stand back—merging universes!")
        with self._progress_bar("Merging features") as pbar:
            joined_df = base_df.join(
                F.broadcast(prepared_df),  # Broadcast hint for optimization
                on=["cve", "date"], 
                how="left"
            )
            pbar.update(50)
            
            # Get new feature columns (excluding cve, date)
            new_feature_cols = [c for c in prepared_df.columns if c not in ("cve", "date")]
            self.feature_columns.extend(new_feature_cols)
            
            # Handle nulls based on auto_fill_nulls setting
            if self.auto_fill_nulls:
                self._battle_cry("👻 Ghost-busting NULLs—no spooky gaps allowed.")
                self.merged_df = joined_df.fillna(fill_value, subset=new_feature_cols).persist()
            else:
                self._battle_cry("🔍 Preserving NULLs for inspection—check the red warnings!")
                self.merged_df = joined_df.persist()
                # Show null summary in red
                self._show_null_summary(self.merged_df, new_feature_cols)
            
            pbar.update(50)
        
        # Validation using cached count
        actual_count = self.merged_df.count()  # This hits the cache now
        
        if actual_count != expected_count:
            raise RuntimeError(
                f"Row count mismatch after merge! Expected {expected_count:,}, got {actual_count:,}. "
                "This indicates a problem with the LEFT join."
            )
        
        self._battle_cry("✅ Body count verified. No rows left behind.")
        self._log(f"✓ Added {len(new_feature_cols)} features from {source_name} "
                 f"({prepared_df.count():,} feature rows)")
        
        return self
    
    def merge(self, fill_value: Union[int, float, str] = 0) -> DataFrame:
        """
        Perform the merge operation and return the result.
        
        Args:
            fill_value: Value to fill missing features (only used if auto_fill_nulls=True)
            
        Returns:
            Merged DataFrame with all features
        """
        if self.feature_df is None:
            raise ValueError("No feature data available. Provide feature_df or feature_source/feature_file.")
        
        self.add_features(self.feature_df, self.feature_source or "unknown", fill_value)
        return self.merged_df
    
    def save(
        self, 
        output_path: Optional[str] = None, 
        mode: str = "overwrite"
    ) -> str:
        """
        Save the merged result to parquet.
        
        Args:
            output_path: Custom output path (default: data/<source>/processed/<source>_processed.parquet)
            mode: Write mode (default: "overwrite")
            
        Returns:
            Path where the file was saved
        """
        if self.merged_df is None:
            raise ValueError("No merged data available. Call merge() or add_features() first.")
        
        if output_path is None:
            if self.feature_source:
                output_path = f"data/{self.feature_source}/processed/{self.feature_source}_processed.parquet"
            else:
                output_path = "data/merged/merged_features.parquet"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._battle_cry("🚀 Payload dropped to disk. Mission accomplished!")
        with self._progress_bar("Saving data") as pbar:
            self.merged_df.write.mode(mode).parquet(output_path)  # Uses cached data
            pbar.update(100)
        
        # Show runtime summary
        runtime = time.time() - self.start_time
        self._battle_cry(f"🏁 ZeroDayZipper finished in {runtime:.2f}s. Coffee may now be consumed.")
        
        self._log(f"✓ Saved merged data to: {output_path}")
        return output_path
    
    def preview(self, n: int = 10) -> None:
        """Show a preview of the merged data."""
        if self.merged_df is None:
            self._log("No merged data available yet.")
            return
        
        self._battle_cry("✨ Flashbang! Eyes on these shiny rows →")
        self._log(f"Merged data preview ({self.merged_df.count():,} total rows):")  # Uses cached data
        self.merged_df.show(n, truncate=False)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of the current feature set."""
        if self.merged_df is None:
            return {"status": "No merged data available"}
        
        return {
            "total_rows": self.merged_df.count(),  # Uses cached data
            "total_columns": len(self.merged_df.columns),
            "feature_columns": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "base_columns": ["cve", "date", "epss"],
        }
    
    def validate_integrity(self) -> bool:
        """Validate data integrity of the merged result."""
        if self.merged_df is None:
            self._log("No merged data to validate.")
            return False
        
        # Check row count matches EPSS using cached count
        merged_count = self.merged_df.count()  # Uses cached data
        
        if self.epss_rows != merged_count:
            self._log(f"✗ Row count mismatch: EPSS={self.epss_rows:,}, Merged={merged_count:,}")
            return False
        
        # Check for required columns
        required_cols = {"cve", "date"}
        missing_cols = required_cols - set(self.merged_df.columns)
        if missing_cols:
            self._log(f"✗ Missing required columns: {missing_cols}")
            return False
        
        self._battle_cry("✅ Body count verified. No rows left behind.")
        self._log("✓ Data integrity validation passed")
        return True
    
    def __repr__(self) -> str:
        """String representation of the ZeroDayZipper."""
        status = "merged" if self.merged_df else "initialized"
        feature_info = f", {len(self.feature_columns)} features" if self.feature_columns else ""
        return f"ZeroDayZipper(status={status}, source={self.feature_source}{feature_info})"


# Convenience function for quick usage
def quick_merge(
    feature_source: str,
    feature_file: str,
    epss_path: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
    fun_mode: bool = True,
    auto_fill_nulls: bool = False
) -> str:
    """
    Quick one-liner to merge features and save result.
    
    Args:
        feature_source: Source name (e.g., 'github', 'reddit')
        feature_file: Feature filename
        epss_path: Custom EPSS path (optional)
        output_path: Custom output path (optional)
        verbose: Whether to show progress
        fun_mode: Whether to show battle cries and ASCII art
        auto_fill_nulls: Whether to automatically fill nulls
        
    Returns:
        Path where the merged file was saved
    """
    zipper = ZeroDayZipper(
        epss_path=epss_path,
        feature_source=feature_source,
        feature_file=feature_file,
        verbose=verbose,
        fun_mode=fun_mode,
        auto_fill_nulls=auto_fill_nulls
    )
    zipper.merge()
    return zipper.save(output_path) 
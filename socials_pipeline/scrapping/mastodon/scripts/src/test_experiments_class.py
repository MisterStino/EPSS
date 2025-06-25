#!/usr/bin/env python3
"""
Test: Can the experiments.py class replicate contained_classify.py workflow?
This test attempts to use the generic Experiment class for the same task.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import the experiments class
from .experiments import Experiment, classification_metrics

# Import the existing components from contained_classify
from .contained_classify import (
    jsonl_to_truth, toots_to_cve_date, 
    CLASS_ORDER, CVSS_PROXY,
    VulnerabilitySeverityClassifier
)

# Test configuration
NVD_JSONL_PATH = Path("catalogs_processed/2025-06-08_snapshot.jsonl")
MASTODON_CSV_PATH = Path("data/mastodon/mastodon_raw.csv")

class CVEModelWrapper:
    """Wrapper to make VulnerabilitySeverityClassifier compatible with experiments.py"""
    def __init__(self):
        self.clf = VulnerabilitySeverityClassifier(device="auto")
        if not self.clf.is_ready():
            raise RuntimeError("Model failed to load")
    
    def predict(self, texts: list[str]) -> list[str]:
        """Convert the tuple return to just severity predictions"""
        severity_list, _ = self.clf.predict(texts)
        return [s.upper() for s in severity_list]

def test_experiments_vs_contained_classify():
    """Test if we can replicate contained_classify.py using experiments.py"""
    
    print("🧪 Testing experiments.py vs contained_classify.py")
    print("=" * 60)
    
    # Step 1: Load and process data (same as contained_classify)
    print("📊 Loading NVD ground truth...")
    truth_df = jsonl_to_truth(NVD_JSONL_PATH)
    
    print("🐘 Loading Mastodon data...")
    social_df = toots_to_cve_date(MASTODON_CSV_PATH)
    
    # Step 2: Merge the data (this is where experiments.py breaks down)
    print("🔗 Merging social and truth data...")
    merged_df = social_df.merge(truth_df, on="cve", how="inner")
    merged_df = merged_df[merged_df["cvss_base_severity"].isin(CLASS_ORDER)]
    
    print(f"   Merged dataset: {len(merged_df)} rows")
    
    # Step 3: Attempt to use the Experiment class
    print("🤖 Testing with Experiment class...")
    
    try:
        # Create model wrapper
        model = CVEModelWrapper()
        
        # Define metrics function
        def combined_metrics(y_true, y_pred):
            """Try to replicate the dual metrics from contained_classify"""
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Classification metrics
            cls_metrics = classification_metrics(y_true, y_pred, CLASS_ORDER)
            
            # Regression metrics (convert severity to CVSS scores)
            y_true_scores = [CVSS_PROXY.get(sev, 0.0) for sev in y_true]
            y_pred_scores = [CVSS_PROXY.get(sev, 0.0) for sev in y_pred]
            
            mae = mean_absolute_error(y_true_scores, y_pred_scores)
            rmse = np.sqrt(mean_squared_error(y_true_scores, y_pred_scores))
            r2 = r2_score(y_true_scores, y_pred_scores)
            
            # Combine metrics
            cls_metrics["regression"] = {"mae": mae, "rmse": rmse, "r2": r2}
            return cls_metrics
        
        # Create experiment
        exp = Experiment.from_dataframe(
            merged_df,
            model=model,
            target_col="cvss_base_severity",
            text_col="text",
            metric_fn=combined_metrics,
            run_name="cve_severity_test",
            batch_size=32,
            out_dir=Path("data/mastodon/scripts/src/test_runs")
        )
        
        # Run experiment
        print("   Running experiment...")
        results = exp.run()
        
        # Print results
        print("\n🎯 EXPERIMENT RESULTS:")
        print(f"   Accuracy: {results['accuracy']:.3%}")
        print(f"   Regression R²: {results['regression']['r2']:.3f}")
        print("   ✅ SUCCESS: Experiment class worked!")
        
        return results
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return None

def analyze_limitations():
    """Analyze the fundamental limitations of the experiments.py approach"""
    
    print("\n🔍 ANALYSIS: Limitations of experiments.py for contained_classify workflow")
    print("=" * 80)
    
    limitations = [
        "1. **Single DataFrame Assumption**: experiments.py assumes text and targets are in same DataFrame",
        "   Reality: contained_classify loads separate NVD truth and Mastodon social data",
        "",
        "2. **Complex Preprocessing**: experiments.py has simple preprocess_fn hook",
        "   Reality: contained_classify does CVE extraction, account classification, date aggregation",
        "",
        "3. **Data Merging**: experiments.py doesn't handle merging multiple data sources",
        "   Reality: contained_classify merges social_df and truth_df on CVE ID after prediction",
        "",
        "4. **Dual Metrics**: experiments.py supports one metric_fn", 
        "   Reality: contained_classify computes both classification AND regression metrics",
        "",
        "5. **Model Interface**: experiments.py assumes simple predict() -> predictions",
        "   Reality: VulnerabilitySeverityClassifier returns (predictions, confidences) tuple",
        "",
        "6. **Date Filtering**: experiments.py has no built-in temporal filtering",
        "   Reality: contained_classify supports eval_dates filtering",
        "",
        "CONCLUSION: The experiments.py class is too simplistic for contained_classify's workflow"
    ]
    
    for limitation in limitations:
        print(limitation)

if __name__ == "__main__":
    # Run the test
    results = test_experiments_vs_contained_classify()
    
    # Analyze limitations regardless of test outcome
    analyze_limitations() 
#!/usr/bin/env python3
"""
DEMONSTRATION: Correct True Positive Definition for EPSS Early Warning

This script demonstrates the CORRECT evaluation logic for early warning systems
using synthetic data to avoid memory constraints. It shows exactly how True Positives
should be defined and counted for temporal forecasting problems.

Key Insights:
1. Event-specific evaluation (not CVE-level aggregation)
2. Exact horizon-to-date matching: T_anchor + h + 1 = T_event
3. Individual event counting for contingency tables
4. Correct lead time calculation: lead_time = h + 1

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def create_synthetic_example():
    """
    Create a synthetic example to demonstrate correct evaluation logic.
    """
    print("🔬 CREATING SYNTHETIC EXAMPLE")
    print("="*50)
    
    # Small synthetic dataset for demonstration
    C, T, H = 5, 20, 5  # 5 CVEs, 20 time points, 5 horizons
    threshold = 0.7
    
    # Create synthetic predictions (log-space)
    np.random.seed(42)
    pred_log = np.random.normal(-2.0, 0.5, (C, T, H))  # Most predictions below threshold
    
    # Create some high predictions for demonstration
    pred_log[0, 10, 0] = 0.5   # CVE 0, t=10, h=0 -> predicts t=11, prob ≈ 0.65
    pred_log[1, 8, 2] = 1.2    # CVE 1, t=8, h=2 -> predicts t=11, prob ≈ 0.77
    pred_log[2, 5, 4] = 0.9    # CVE 2, t=5, h=4 -> predicts t=10, prob ≈ 0.71
    pred_log[3, 15, 1] = 1.5   # CVE 3, t=15, h=1 -> predicts t=17, prob ≈ 0.82
    
    # Convert to probabilities
    pred_prob = np.clip(np.exp(pred_log) - 1e-6, 0.0, 1.0)
    
    # Create synthetic ground truth events
    events = [
        (0, 11, 0.85),  # CVE 0, event at t=11, EPSS=0.85
        (1, 11, 0.75),  # CVE 1, event at t=11, EPSS=0.75  
        (2, 10, 0.72),  # CVE 2, event at t=10, EPSS=0.72
        (3, 17, 0.90),  # CVE 3, event at t=17, EPSS=0.90
        (4, 8, 0.78),   # CVE 4, event at t=8, EPSS=0.78 (no prediction above threshold)
    ]
    
    # Create evaluation masks (all valid for simplicity)
    eval_mask = np.ones((C, T), dtype=bool)
    mask_h = np.ones((C, T, H), dtype=bool)
    
    return pred_prob, events, eval_mask, mask_h, C, T, H, threshold


def demonstrate_correct_tp_logic(pred_prob, events, eval_mask, mask_h, C, T, H, threshold):
    """
    Demonstrate the CORRECT True Positive evaluation logic.
    """
    print("\n🎯 DEMONSTRATING CORRECT TRUE POSITIVE LOGIC")
    print("="*50)
    
    tp_events = []
    fn_events = []
    
    print("\nAnalyzing each event individually:")
    print("-" * 40)
    
    for event_idx, (cve_idx, event_t, epss_value) in enumerate(events):
        print(f"\nEvent {event_idx + 1}: CVE {cve_idx}, t={event_t}, EPSS={epss_value:.2f}")
        print(f"  Checking all prediction horizons that could warn about this event...")
        
        warnings_found = []
        
        # Check all possible prediction horizons that could warn about this event
        for h in range(H):
            required_anchor = event_t - h - 1
            
            print(f"    h={h}: anchor t={required_anchor} -> predicts t={required_anchor + h + 1}", end="")
            
            # Validate anchor time and masks
            if (required_anchor >= 0 and required_anchor < T and 
                eval_mask[cve_idx, required_anchor] and
                mask_h[cve_idx, required_anchor, h]):
                
                pred_value = pred_prob[cve_idx, required_anchor, h]
                above_threshold = pred_value >= threshold
                lead_time = h + 1
                
                print(f" | pred={pred_value:.3f}, ≥{threshold}: {above_threshold}", end="")
                
                if above_threshold:
                    print(f" ✅ WARNING (lead: {lead_time} days)")
                    warnings_found.append({
                        'anchor_t': required_anchor,
                        'horizon': h,
                        'lead_time': lead_time,
                        'pred_prob': pred_value
                    })
                else:
                    print(f" ❌ No warning")
            else:
                print(f" | INVALID (anchor out of bounds or masked)")
        
        if warnings_found:
            # True Positive: event was warned about
            earliest_warning = min(warnings_found, key=lambda x: x['anchor_t'])
            tp_events.append({
                'cve_idx': cve_idx,
                'event_t': event_t,
                'epss_value': epss_value,
                'warning': earliest_warning,
                'total_warnings': len(warnings_found)
            })
            print(f"  ✅ TRUE POSITIVE: {len(warnings_found)} warning(s), earliest lead: {earliest_warning['lead_time']} days")
        else:
            # False Negative: event occurred but no warning
            fn_events.append({
                'cve_idx': cve_idx,
                'event_t': event_t,
                'epss_value': epss_value
            })
            print(f"  ❌ FALSE NEGATIVE: No warnings found")
    
    return tp_events, fn_events


def demonstrate_fp_logic(pred_prob, events, eval_mask, mask_h, C, T, H, threshold):
    """
    Demonstrate False Positive detection logic.
    """
    print(f"\n🚨 DEMONSTRATING FALSE POSITIVE DETECTION")
    print("="*50)
    
    # Create event lookup for fast checking
    event_lookup = {}
    for cve_idx, event_t, _ in events:
        if cve_idx not in event_lookup:
            event_lookup[cve_idx] = set()
        event_lookup[cve_idx].add(event_t)
    
    fp_warnings = []
    
    print("\nScanning all predictions ≥ threshold for False Positives:")
    print("-" * 50)
    
    for cve_idx in range(C):
        for t in range(T):
            if eval_mask[cve_idx, t]:
                for h in range(H):
                    if mask_h[cve_idx, t, h]:
                        pred_value = pred_prob[cve_idx, t, h]
                        
                        if pred_value >= threshold:
                            target_t = t + h + 1
                            
                            # Check if this prediction corresponds to an actual event
                            has_event = (cve_idx in event_lookup and 
                                       target_t in event_lookup[cve_idx])
                            
                            print(f"CVE {cve_idx}, t={t}, h={h} -> target t={target_t}: pred={pred_value:.3f}", end="")
                            
                            if has_event:
                                print(f" ✅ MATCHES EVENT (True Positive)")
                            else:
                                print(f" ❌ NO EVENT (False Positive)")
                                fp_warnings.append({
                                    'cve_idx': cve_idx,
                                    'anchor_t': t,
                                    'horizon': h,
                                    'target_t': target_t,
                                    'pred_prob': pred_value
                                })
    
    return fp_warnings


def compute_demonstration_metrics(tp_events, fn_events, fp_warnings):
    """
    Compute metrics from the demonstration.
    """
    print(f"\n📊 COMPUTING CORRECTED METRICS")
    print("="*50)
    
    TP = len(tp_events)
    FN = len(fn_events)
    FP = len(fp_warnings)
    
    # For demonstration, estimate TN as a reasonable number
    total_events = TP + FN
    total_predictions_above_threshold = TP + FP
    # Rough estimate: if we have few events and few high predictions, most predictions are TN
    TN = max(100, 10 * (TP + FN + FP))  # Conservative estimate
    
    print(f"Event-Level Contingency Table:")
    print(f"  True Positives (TP):  {TP}")
    print(f"  False Negatives (FN): {FN}")
    print(f"  False Positives (FP): {FP}")
    print(f"  True Negatives (TN):  {TN} (estimated)")
    
    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # TSS (True Skill Statistic)
    tss_num = TP * TN - FP * FN
    tss_den = (TP + FN) * (FP + TN)
    tss = tss_num / tss_den if tss_den > 0 else 0.0
    
    print(f"\nCorrected Skill Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  TSS:       {tss:.3f}")
    
    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "Precision": precision, "Recall": recall, "F1": f1, "TSS": tss
    }


def demonstrate_lead_time_analysis(tp_events):
    """
    Demonstrate lead-time analysis.
    """
    print(f"\n⏰ LEAD-TIME ANALYSIS")
    print("="*50)
    
    if not tp_events:
        print("No True Positive events found - no lead times to analyze.")
        return {}
    
    lead_times = [event['warning']['lead_time'] for event in tp_events]
    
    print(f"Lead times from {len(tp_events)} True Positive events:")
    for i, event in enumerate(tp_events):
        warning = event['warning']
        print(f"  Event {i+1}: CVE {event['cve_idx']}, lead time: {warning['lead_time']} days")
    
    lead_times = np.array(lead_times)
    pct_ge7 = 100 * np.mean(lead_times >= 7) if len(lead_times) > 0 else 0
    median_lt = np.median(lead_times) if len(lead_times) > 0 else 0
    mean_lt = np.mean(lead_times) if len(lead_times) > 0 else 0
    max_lt = np.max(lead_times) if len(lead_times) > 0 else 0
    
    print(f"\nLead-time Statistics:")
    print(f"  Events with warnings: {len(tp_events)}")
    print(f"  % warned ≥7 days:     {pct_ge7:.1f}%")
    print(f"  Median lead time:     {median_lt:.1f} days")
    print(f"  Mean lead time:       {mean_lt:.1f} days")
    print(f"  Maximum lead time:    {max_lt:.1f} days")
    
    return {
        "num_tp_with_leads": len(tp_events),
        "pct_warned_ge7d": pct_ge7,
        "median_lead": median_lt,
        "mean_lead": mean_lt,
        "max_lead": max_lt
    }


def create_comparison_with_wrong_method(pred_prob, events, threshold):
    """
    Show how the WRONG CVE-level aggregation method would evaluate this.
    """
    print(f"\n❌ COMPARISON: WRONG CVE-LEVEL AGGREGATION METHOD")
    print("="*50)
    
    C, T, H = pred_prob.shape
    
    # Wrong method: aggregate at CVE level
    cve_has_event = np.zeros(C, dtype=bool)
    cve_has_high_pred = np.zeros(C, dtype=bool)
    
    # Mark CVEs with events
    for cve_idx, event_t, epss_value in events:
        cve_has_event[cve_idx] = True
    
    # Mark CVEs with any prediction ≥ threshold
    for cve_idx in range(C):
        if np.any(pred_prob[cve_idx] >= threshold):
            cve_has_high_pred[cve_idx] = True
    
    # Wrong contingency table
    wrong_tp = np.sum(cve_has_event & cve_has_high_pred)
    wrong_fn = np.sum(cve_has_event & ~cve_has_high_pred)
    wrong_fp = np.sum(~cve_has_event & cve_has_high_pred)
    wrong_tn = np.sum(~cve_has_event & ~cve_has_high_pred)
    
    print(f"WRONG CVE-Level Contingency Table:")
    print(f"  True Positives (TP):  {wrong_tp}")
    print(f"  False Negatives (FN): {wrong_fn}")
    print(f"  False Positives (FP): {wrong_fp}")
    print(f"  True Negatives (TN):  {wrong_tn}")
    
    # Wrong metrics
    wrong_precision = wrong_tp / (wrong_tp + wrong_fp) if (wrong_tp + wrong_fp) > 0 else 0.0
    wrong_recall = wrong_tp / (wrong_tp + wrong_fn) if (wrong_tp + wrong_fn) > 0 else 0.0
    wrong_f1 = 2 * wrong_precision * wrong_recall / (wrong_precision + wrong_recall) if (wrong_precision + wrong_recall) > 0 else 0.0
    
    print(f"\nWRONG Metrics:")
    print(f"  Precision: {wrong_precision:.3f}")
    print(f"  Recall:    {wrong_recall:.3f}")
    print(f"  F1 Score:  {wrong_f1:.3f}")
    
    print(f"\n🔍 PROBLEMS WITH CVE-LEVEL AGGREGATION:")
    print(f"  1. Loses temporal information - can't compute lead times")
    print(f"  2. One high prediction anywhere 'covers' all events for that CVE")
    print(f"  3. Doesn't check if prediction horizon matches event date")
    print(f"  4. Can't distinguish between different events on same CVE")
    print(f"  5. Artificially inflates performance by ignoring temporal precision")


def main():
    """
    Main demonstration of correct evaluation logic.
    """
    print("🚀 DEMONSTRATION: CORRECT TRUE POSITIVE DEFINITION")
    print("="*60)
    print("This script demonstrates the CORRECT way to evaluate early warning")
    print("systems for temporal forecasting problems using synthetic data.")
    print("="*60)
    
    # Create synthetic example
    pred_prob, events, eval_mask, mask_h, C, T, H, threshold = create_synthetic_example()
    
    # Demonstrate correct TP logic
    tp_events, fn_events = demonstrate_correct_tp_logic(
        pred_prob, events, eval_mask, mask_h, C, T, H, threshold
    )
    
    # Demonstrate FP logic
    fp_warnings = demonstrate_fp_logic(
        pred_prob, events, eval_mask, mask_h, C, T, H, threshold
    )
    
    # Compute corrected metrics
    metrics = compute_demonstration_metrics(tp_events, fn_events, fp_warnings)
    
    # Lead-time analysis
    lead_stats = demonstrate_lead_time_analysis(tp_events)
    
    # Show comparison with wrong method
    create_comparison_with_wrong_method(pred_prob, events, threshold)
    
    print(f"\n✅ DEMONSTRATION COMPLETE")
    print("="*60)
    print("KEY TAKEAWAYS:")
    print("1. True Positives must be defined at the EVENT level, not CVE level")
    print("2. Prediction horizons must EXACTLY match event dates: T_anchor + h + 1 = T_event")
    print("3. Lead time = h + 1 (horizon h=0 means 1-day lead time)")
    print("4. Each event should be evaluated independently")
    print("5. CVE-level aggregation loses critical temporal information")
    print("\nThis corrected approach provides meaningful early warning evaluation")
    print("that properly captures the temporal dynamics of forecasting systems.")


if __name__ == '__main__':
    main() 
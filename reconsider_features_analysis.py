#!/usr/bin/env python
"""
Reconsidering Feature Analysis for EPSS Time Series Data

Key insight: 253M rows = ~82K CVEs × ~3K daily time steps
High "missingness" actually means "no events happening" - which is normal!
The sparse events (1-2%) are likely the KEY predictive signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("🔄 RECONSIDERING FEATURE ANALYSIS WITH TIME SERIES CONTEXT")
print("=" * 80)

# Key dataset statistics
total_rows = 253_839_731
unique_cves = 82_179
unique_dates = 1_160
avg_timesteps_per_cve = total_rows / unique_cves

print(f"📊 Dataset Structure Understanding:")
print(f"  • Total time steps: {total_rows:,}")
print(f"  • Unique CVEs: {unique_cves:,}")
print(f"  • Unique dates: {unique_dates:,}")
print(f"  • Avg time steps per CVE: {avg_timesteps_per_cve:,.0f}")
print(f"  • Expected dense coverage: {unique_cves * unique_dates:,}")
print(f"  • Coverage ratio: {total_rows / (unique_cves * unique_dates):.1%}")

def analyze_event_density(missing_pct, total_rows, unique_cves):
    """Analyze what high 'missingness' means in event context"""
    
    non_null_rows = total_rows * (1 - missing_pct/100)
    events_per_cve_avg = non_null_rows / unique_cves
    
    return {
        'non_null_observations': int(non_null_rows),
        'avg_events_per_cve': events_per_cve_avg,
        'interpretation': 'HIGH' if events_per_cve_avg > 100 else 'MEDIUM' if events_per_cve_avg > 10 else 'LOW'
    }

print(f"\n🎯 EVENT DENSITY REANALYSIS")
print("=" * 60)

# Reanalyze the "high missing" columns
event_columns = {
    'has_discovery': 98.4,
    'has_release': 98.4, 
    'has_threat': 98.4,
    'has_remediation': 98.4,
    'event_sequence': 98.4,
    'cumulative_source_count': 98.4,
    'total_events_so_far': 98.4,
    'prev_event_type': 98.4,
    'event_stage_num': 98.4,
    'max_stage_reached': 98.4,
    'dominant_event_type': 100.0,
    'primary_source': 100.0
}

print("Column Analysis (Sparse Events ≠ Useless!):")
print("-" * 60)

for col, missing_pct in event_columns.items():
    analysis = analyze_event_density(missing_pct, total_rows, unique_cves)
    
    print(f"\n📍 {col}")
    print(f"   Missing: {missing_pct}% (= 'no events happening')")
    print(f"   Active observations: {analysis['non_null_observations']:,}")
    print(f"   Events per CVE (avg): {analysis['avg_events_per_cve']:.1f}")
    
    if missing_pct == 100.0:
        recommendation = "🗑️  DROP - Truly empty"
        reason = "No data whatsoever"
    elif analysis['avg_events_per_cve'] < 1:
        recommendation = "🤔 CONSIDER DROP - Very rare events"  
        reason = f"<1 event per CVE on average"
    elif analysis['avg_events_per_cve'] < 10:
        recommendation = "✅ KEEP - Rare but meaningful events"
        reason = f"~{analysis['avg_events_per_cve']:.1f} events per CVE could be highly predictive"
    else:
        recommendation = "✅ KEEP - Frequent events"
        reason = f"~{analysis['avg_events_per_cve']:.1f} events per CVE"
    
    print(f"   {recommendation}")
    print(f"   Reasoning: {reason}")

print(f"\n🧠 KEY INSIGHTS:")
print("=" * 60)
print("1. HIGH MISSINGNESS ≠ BAD DATA in time series!")
print("   - 98% 'missing' = 98% of time nothing happens (normal!)")
print("   - 2% non-missing = discrete events that drive EPSS changes")
print()
print("2. SPARSE EVENTS ARE LIKELY THE MOST PREDICTIVE:")
print("   - Vulnerability discovery → EPSS spike")
print("   - Patch release → EPSS change") 
print("   - Threat assessment → EPSS adjustment")
print()
print("3. EVEN 1 EVENT PER CVE IS VALUABLE:")
print("   - In time series forecasting, rare events often drive the biggest changes")
print("   - Model needs to learn: 'when X event happens, EPSS changes by Y'")

print(f"\n📋 REVISED FEATURE RECOMMENDATIONS:")
print("=" * 60)

revised_recommendations = {
    'KEEP_EVENT_FEATURES': [
        'has_discovery',      # Discovery events → EPSS spikes
        'has_release',        # Release events → EPSS changes  
        'event_sequence',     # Event ordering within CVE lifecycle
        'cumulative_source_count',  # Multiple sources = higher attention
        'total_events_so_far',      # Event accumulation over time
        'prev_event_type',          # Event type transitions
        'event_stage_num',          # Stage in vulnerability lifecycle
        'max_stage_reached',        # Maturity of vulnerability process
    ],
    
    'DROP_TRULY_EMPTY': [
        'dominant_event_type',  # 100% missing - truly empty
        'primary_source',       # 100% missing - truly empty
        'has_threat',          # Always False when present
        'has_remediation',     # Always False when present
    ],
    
    'KEEP_WITH_COMMENT': [
        # Event features that capture discrete, predictive events
        'has_discovery',       # ~1.6% coverage = ~40K discovery events across all CVEs
        'has_release',         # ~1.6% coverage = ~40K release events  
        'event_sequence',      # Event ordering - critical for sequence modeling
        'cumulative_source_count', # Source accumulation over time
        'total_events_so_far', # Event count progression
        'prev_event_type',     # Event type transitions
        'event_stage_num',     # Stage progression
        'max_stage_reached',   # Maximum stage reached
    ]
}

print("\n✅ KEEP - Event Features (Sparse but Predictive):")
for feature in revised_recommendations['KEEP_EVENT_FEATURES']:
    events_per_cve = analyze_event_density(98.4, total_rows, unique_cves)['avg_events_per_cve']
    print(f"  • {feature}  # ~{events_per_cve:.1f} events/CVE, drives EPSS changes")

print("\n🗑️  DROP - Truly Empty:")
for feature in revised_recommendations['DROP_TRULY_EMPTY']:
    print(f"  • {feature}  # 100% missing or constant False")

print(f"\n💡 FEATURE ENGINEERING INSIGHTS:")
print("=" * 60)
print("1. Event features should be treated as categorical/binary signals")
print("2. Consider creating 'days_since_last_X_event' features")
print("3. Event sequences could be encoded as categorical transitions") 
print("4. Cumulative event counts are valuable temporal features")
print("5. The LSTM should learn event → EPSS change patterns")

print(f"\n📊 MEMORY IMPACT RECALCULATION:")
print("=" * 60)
original_drop_count = 47
revised_drop_count = len(revised_recommendations['DROP_TRULY_EMPTY']) + 30  # Keep more event features
print(f"Original recommendation: Drop {original_drop_count} columns")
print(f"Revised recommendation: Drop ~{revised_drop_count} columns") 
print(f"Event features to keep: {len(revised_recommendations['KEEP_EVENT_FEATURES'])}")
print(f"These event features are sparse (low memory) but high predictive value!") 
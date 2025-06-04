import pandas as pd
import numpy as np

# Load the historical NVD data
print("Loading historical NVD data...")
# df = pd.read_csv('catalogs_processed/cve_catalog.csv')

df = pd.read_csv('catalogs_processed/data/historical_nvd.csv')

# Display basic information about the dataset
print(f"\nDataset Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\n" + "="*80)
print("ANALYSIS: CVEs That Get CVSS Scores Over Time (2023+ Only)")
print("="*80)

# Convert date column to datetime for proper sorting

# convert date_published to datetime
df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce', utc=True).dt.tz_localize(None)

# filter on cve with cve_id that has a value of 2023 or later in the cve_id column
df = df[df['cve_id'].str.startswith('CVE-2023-') | df['cve_id'].str.startswith('CVE-2024-') | df['cve_id'].str.startswith('CVE-2025-')]


print(df.head())

# Extract only the columns we need: cve_id, date_published, cvss_score, cvss_v2_score, cvss_v3_score, cvss_v4_score, 
columns_needed = ['cve_id', 'date_published', 'cvss_score', 'cvss_v2_score', 'cvss_v3_score', 'cvss_v4_score', 'description']
df_filtered = df[columns_needed].copy()

print(f"\nAfter column selection:")
print(f"Shape: {df_filtered.shape}")
print(f"Columns: {list(df_filtered.columns)}")

# group by cve_id and select only the cve groups where there is at leas 1 cvss score for one of the columns in 1 of the rows and at least 1 description
cvss_columns = ['cvss_score', 'cvss_v2_score', 'cvss_v3_score', 'cvss_v4_score']

# Filter to keep only rows that have both a description AND at least one CVSS score
print(f"\nFiltering rows with both description and CVSS score...")
print(f"Original rows after column selection: {len(df_filtered):,}")

# Create boolean masks
has_description = df_filtered['description'].notna()
has_any_cvss = df_filtered[cvss_columns].notna().any(axis=1)

# Apply both conditions
df_final = df_filtered[has_description & has_any_cvss].copy()

print(f"Rows with description: {has_description.sum():,}")
print(f"Rows with any CVSS score: {has_any_cvss.sum():,}")
print(f"Rows with BOTH description AND CVSS score: {len(df_final):,}")
print(f"Final dataset shape: {df_final.shape}")
print(f"Final rows: {df_final.shape[0]:,}")
print(f"Final columns: {df_final.shape[1]}")
print(f"Unique CVEs in final dataset: {df_final['cve_id'].nunique():,}")

# Show some statistics about the filtered data
print(f"\nCVSS Score Statistics:")
for col in cvss_columns:
    non_null_count = df_final[col].notna().sum()
    percentage = (non_null_count / len(df_final)) * 100
    print(f"  {col}: {non_null_count:,} non-null values ({percentage:.1f}%)")

description_count = df_final['description'].notna().sum()
description_percentage = (description_count / len(df_final)) * 100
print(f"  description: {description_count:,} non-null values ({description_percentage:.1f}%)")

print(f"\nSample of filtered data:")
print(df_final.head())
df_final.to_csv('models/llm/scripts/filtered_cves.csv', index=False)

# throw away any rows that do no have both a description and a cvss score
df_final = df_final[df_final['description'].notna() & df_final[cvss_columns].notna().any(axis=1)]

# load the file again
df_final = pd.read_csv('models/llm/scripts/filtered_cves.csv')

#keep only columns cve_id, cvss_score, description
df_final = df_final[['cve_id', 'cvss_score', 'description']]
print(df_final.head())

# ============================================================================
# VULNERABILITY SEVERITY PREDICTION USING CIRCL MODELS
# ============================================================================

print(f"\n" + "="*80)
print("VULNERABILITY SEVERITY PREDICTION")
print("="*80)

# Install required packages if not already installed
import subprocess
import sys

# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load the CIRCL severity classification model
print(f"\nLoading CIRCL vulnerability severity classification model...")
model_name = "CIRCL/vulnerability-severity-classification-roberta-base"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Severity labels (in order expected by the model)
    SEVERITY_LABELS = ["low", "medium", "high", "critical"]
    
    # CVSS score mapping (standard ranges)
    SEVERITY_TO_CVSS = {
        "low": 3.0,      # 0.1-3.9
        "medium": 6.0,   # 4.0-6.9  
        "high": 8.0,     # 7.0-8.9
        "critical": 9.5  # 9.0-10.0
    }
    
    print(f"✅ Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Severity labels: {SEVERITY_LABELS}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Continuing without AI predictions...")
    model = None

# Function to predict severity and CVSS score
def predict_severity_and_cvss(descriptions):
    """Predict severity and map to CVSS scores"""
    if model is None:
        return ["unknown"] * len(descriptions), [0.0] * len(descriptions)
    
    # Tokenize descriptions
    inputs = tokenizer(
        descriptions, 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_indices = torch.argmax(probabilities, dim=-1)
    
    # Convert to severity labels and CVSS scores
    predicted_severities = [SEVERITY_LABELS[idx] for idx in predicted_indices.cpu().numpy()]
    predicted_cvss = [SEVERITY_TO_CVSS[severity] for severity in predicted_severities]
    
    return predicted_severities, predicted_cvss

# Test on first 10 rows
print(f"\n" + "-"*60)
print("TESTING ON FIRST 100 ROWS")
print("-"*60)

test_df = df_final.head(100).copy()
print(f"Testing on {len(test_df)} rows...")

# Get descriptions for prediction
descriptions = test_df['description'].fillna("").tolist()

# Predict severity and CVSS scores
predicted_severities, predicted_cvss = predict_severity_and_cvss(descriptions)

# Add predictions to test dataframe
test_df['ai_severity'] = predicted_severities
test_df['ai_cvss_score'] = predicted_cvss

# Calculate differences
test_df['cvss_difference'] = test_df['ai_cvss_score'] - test_df['cvss_score']
test_df['abs_cvss_difference'] = abs(test_df['cvss_difference'])

print(f"\nCVE PREDICTIONS (Original vs AI Predicted):")
print("="*80)
print(f"{'CVE ID':<20} {'Original':<10} {'Predicted':<10} {'Difference':<12} {'Severity':<10}")
print("-"*80)

for idx, row in test_df.iterrows():
    cve_id = row['cve_id']
    original = row['cvss_score']
    predicted = row['ai_cvss_score']
    difference = row['cvss_difference']
    severity = row['ai_severity']
    
    print(f"{cve_id:<20} {original:<10.1f} {predicted:<10.1f} {difference:+<12.1f} {severity:<10}")

# Summary statistics
print(f"\n" + "="*80)
print("SUMMARY STATISTICS (100 CVEs)")
print("="*80)
print(f"Mean Absolute Error: {test_df['abs_cvss_difference'].mean():.2f}")
print(f"Root Mean Square Error: {np.sqrt((test_df['cvss_difference']**2).mean()):.2f}")
print(f"Original CVSS range: {test_df['cvss_score'].min():.1f} - {test_df['cvss_score'].max():.1f}")
print(f"AI CVSS range: {test_df['ai_cvss_score'].min():.1f} - {test_df['ai_cvss_score'].max():.1f}")

# Show severity distribution
print(f"\nSeverity Distribution:")
severity_counts = test_df['ai_severity'].value_counts()
for severity, count in severity_counts.items():
    print(f"  {severity}: {count} CVEs ({count/len(test_df)*100:.1f}%)")

# Show accuracy within different thresholds
within_05 = (test_df['abs_cvss_difference'] <= 0.5).sum()
within_10 = (test_df['abs_cvss_difference'] <= 1.0).sum()
within_15 = (test_df['abs_cvss_difference'] <= 1.5).sum()

print(f"\nAccuracy within thresholds:")
print(f"  Within ±0.5: {within_05}/{len(test_df)} ({within_05/len(test_df)*100:.1f}%)")
print(f"  Within ±1.0: {within_10}/{len(test_df)} ({within_10/len(test_df)*100:.1f}%)")
print(f"  Within ±1.5: {within_15}/{len(test_df)} ({within_15/len(test_df)*100:.1f}%)")

print(f"\nTest completed! ✅")


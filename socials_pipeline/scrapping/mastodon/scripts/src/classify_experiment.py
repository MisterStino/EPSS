# first we load in: data/mastodon/mastodon_cve_centric_classifications.csv
import pandas as pd
import sys
import os


from models.llm.src.label import VulnerabilitySeverityClassifier

# Load the Mastodon CVE-centric dataset
mas_df = pd.read_csv('data/mastodon/mastodon_cve_centric_classifications.csv')

# Check the actual columns in the dataset
print("Dataset columns:", mas_df.columns.tolist())
print("Dataset shape:", mas_df.shape)

# The dataset already has 'created_at_dt' column and uses 'date' for the main date
# Convert the date column to datetime if not already
if mas_df['date'].dtype == 'object':
    mas_df['date'] = pd.to_datetime(mas_df['date'])

# Next we only keep dates after February 4 2022
# Fix timezone issue by making the comparison timestamp timezone-aware
cutoff_date = pd.Timestamp('2022-02-04', tz='UTC')
mas_df = mas_df[mas_df['date'] >= cutoff_date]

print(f"After date filtering: {len(mas_df)} rows")
print("Sample of filtered data:")
print(mas_df[['cve', 'acct', 'date', 'account_type']].head())

# Initialize the VulnerabilitySeverityClassifier
print("\n🤖 Initializing VulnerabilitySeverityClassifier...")
try:
    # Use CPU to avoid GPU memory issues with large batches
    classifier = VulnerabilitySeverityClassifier(device="auto")
    print("✅ Classifier initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize classifier: {e}")
    exit(1)

# Check if classifier is ready
if classifier.is_ready():
    print("✅ Classifier loaded successfully!")
    print("Model info:", classifier.get_model_info())
else:
    print("❌ Classifier failed to load!")
    exit(1)

# Apply predictions to the content column in batches
print(f"\n🔮 Applying severity predictions to {len(mas_df)} posts...")
print("Processing in batches to manage memory...")

# Get content for prediction (handle missing content)
content_texts = mas_df['content'].fillna('').astype(str).tolist()

# Process in batches of 100 to avoid memory issues
batch_size = 100
predicted_severities = []
predicted_cvss = []

for i in range(0, len(content_texts), batch_size):
    batch_end = min(i + batch_size, len(content_texts))
    batch_texts = content_texts[i:batch_end]
    
    print(f"Processing batch {i//batch_size + 1}/{(len(content_texts) + batch_size - 1)//batch_size} (rows {i+1}-{batch_end})")
    
    # Apply predictions to batch
    batch_severities, batch_cvss = classifier.predict(batch_texts)
    
    predicted_severities.extend(batch_severities)
    predicted_cvss.extend(batch_cvss)

# Add predictions as new columns
mas_df['severity_prediction'] = predicted_severities
mas_df['cvss_prediction'] = predicted_cvss

print("✅ Predictions completed!")

# Show sample results
print("\n📊 Sample predictions:")
sample_results = mas_df[['cve', 'content', 'severity_prediction', 'cvss_prediction']].head()
for _, row in sample_results.iterrows():
    print(f"\nCVE: {row['cve']}")
    print(f"Content: {row['content'][:100]}...")
    print(f"Predicted Severity: {row['severity_prediction']} (CVSS: {row['cvss_prediction']})")

# Summary statistics
print(f"\n📈 Prediction Summary:")
print("Severity distribution:")
print(mas_df['severity_prediction'].value_counts())
print(f"\nCVSS score statistics:")
print(mas_df['cvss_prediction'].describe())

# Save the results
output_file = 'data/mastodon/mas_df_pred.csv'
mas_df.to_csv(output_file, index=False)
print(f"\n💾 Results saved to: {output_file}")
print(f"Dataset shape: {mas_df.shape}")
print("New columns added: severity_prediction, cvss_prediction")

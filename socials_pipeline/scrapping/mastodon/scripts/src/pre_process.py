import pandas as pd
import numpy as np
from collections import Counter
import re

def extract_advanced_features(account_data):
    """Extract sophisticated behavioral features for bot detection"""
    
    # Basic metrics
    post_count = len(account_data)
    if post_count < 3:  # Too few posts to classify reliably
        return None
    
    time_span = (account_data['created_at_dt'].max() - account_data['created_at_dt'].min()).days
    posts_per_day = post_count / max(time_span, 1)
    
    # Temporal patterns
    hours = account_data['hour'].values
    hour_variance = np.var(hours)
    hour_entropy = -sum(p * np.log2(p + 1e-10) for p in np.bincount(hours, minlength=24) / len(hours) if p > 0)
    
    # Posting rhythm analysis
    timestamps = account_data['created_at_dt'].values
    time_diffs = np.diff(timestamps).astype('timedelta64[m]').astype(int)  # minutes between posts
    rhythm_variance = np.var(time_diffs) if len(time_diffs) > 1 else 0
    
    # Weekend behavior
    weekend_posts = account_data[account_data['day_of_week'].isin([5, 6])].shape[0]
    weekend_ratio = weekend_posts / post_count
    
    # Content analysis
    content_lengths = account_data['content'].str.len()
    content_variance = np.var(content_lengths)
    content_mean = np.mean(content_lengths)
    
    # URL pattern analysis
    urls_per_post = account_data['content'].str.count(r'https?://').mean()
    
    # Content diversity (unique words)
    all_text = ' '.join(account_data['content'].fillna('').astype(str))
    words = re.findall(r'\w+', all_text.lower())
    unique_word_ratio = len(set(words)) / max(len(words), 1) if words else 0
    
    # Hashtag patterns
    hashtag_count = account_data['content'].str.count(r'#\w+').sum()
    hashtags_per_post = hashtag_count / post_count
    
    # CVE pattern analysis
    cve_pattern_consistency = account_data['cve_ids'].str.match(r'CVE-\d{4}-\d+').mean()
    
    return {
        'post_count': post_count,
        'posts_per_day': posts_per_day,
        'hour_variance': hour_variance,
        'hour_entropy': hour_entropy,
        'rhythm_variance': rhythm_variance,
        'weekend_ratio': weekend_ratio,
        'content_variance': content_variance,
        'content_mean': content_mean,
        'urls_per_post': urls_per_post,
        'unique_word_ratio': unique_word_ratio,
        'hashtags_per_post': hashtags_per_post,
        'cve_pattern_consistency': cve_pattern_consistency,
        'time_span': time_span
    }

def classify_account_advanced(features, account_name):
    """Advanced classification using multiple behavioral indicators"""
    
    if features is None:
        return "INSUFFICIENT_DATA", []
    
    bot_score = 0
    confidence_factors = []
    
    # 1. Account name indicators (strong signal)
    name_lower = account_name.lower()
    if any(indicator in name_lower for indicator in ['rss', 'bot', '_feed', 'auto', 'scraper']):
        bot_score += 25
        confidence_factors.append("RSS/Bot name")
    
    # 2. Extreme posting frequency (very strong signal)
    if features['posts_per_day'] > 20:
        bot_score += 20
        confidence_factors.append(f"Extreme frequency ({features['posts_per_day']:.1f}/day)")
    elif features['posts_per_day'] > 10:
        bot_score += 10
        confidence_factors.append(f"High frequency ({features['posts_per_day']:.1f}/day)")
    
    # 3. Temporal consistency patterns
    if features['hour_variance'] < 5 and features['post_count'] > 50:
        bot_score += 15
        confidence_factors.append(f"Robotic timing (var={features['hour_variance']:.1f})")
    elif features['hour_variance'] < 10 and features['post_count'] > 100:
        bot_score += 10
        confidence_factors.append(f"Consistent timing (var={features['hour_variance']:.1f})")
    
    # 4. Low temporal entropy (posts at same hours)
    if features['hour_entropy'] < 2 and features['post_count'] > 30:
        bot_score += 12
        confidence_factors.append(f"Low time diversity (entropy={features['hour_entropy']:.1f})")
    
    # 5. Posting rhythm analysis
    if features['rhythm_variance'] < 100 and features['post_count'] > 20:
        bot_score += 8
        confidence_factors.append("Regular posting rhythm")
    
    # 6. Content uniformity
    if features['content_variance'] < 100 and features['post_count'] > 50:
        bot_score += 12
        confidence_factors.append(f"Uniform content (var={features['content_variance']:.0f})")
    
    # 7. High URL usage (feed behavior)
    if features['urls_per_post'] > 0.8:
        bot_score += 8
        confidence_factors.append(f"High URL usage ({features['urls_per_post']:.2f}/post)")
    
    # 8. Low content diversity
    if features['unique_word_ratio'] < 0.3 and features['post_count'] > 20:
        bot_score += 10
        confidence_factors.append(f"Low vocabulary diversity ({features['unique_word_ratio']:.2f})")
    
    # 9. Weekend posting (bots don't rest)
    if features['weekend_ratio'] > 0.3 and features['post_count'] > 50:
        bot_score += 6
        confidence_factors.append(f"High weekend activity ({features['weekend_ratio']:.2f})")
    
    # 10. Perfect CVE pattern matching (automated parsing)
    if features['cve_pattern_consistency'] > 0.95 and features['post_count'] > 20:
        bot_score += 8
        confidence_factors.append("Perfect CVE formatting")
    
    # Human indicators (negative scoring)
    if features['hour_entropy'] > 3.5:
        bot_score -= 5
        confidence_factors.append("Diverse posting times")
    
    if features['unique_word_ratio'] > 0.6:
        bot_score -= 8
        confidence_factors.append("High vocabulary diversity")
    
    if features['posts_per_day'] < 2 and features['content_variance'] > 200:
        bot_score -= 10
        confidence_factors.append("Low frequency + varied content")
    
    # Classification with confidence levels
    if bot_score >= 30:
        return "BOT_HIGH_CONFIDENCE", confidence_factors
    elif bot_score >= 20:
        return "BOT_MEDIUM_CONFIDENCE", confidence_factors  
    elif bot_score >= 10:
        return "LIKELY_BOT", confidence_factors
    elif bot_score >= 5:
        return "SUSPICIOUS", confidence_factors
    else:
        return "HUMAN", confidence_factors

def extract_cve_ids(cve_string):
    """Extract and clean individual CVE IDs from a string"""
    if pd.isna(cve_string) or not isinstance(cve_string, str):
        return []
    
    # Find all CVE patterns in the string
    cve_pattern = r'CVE-\d{4}-\d+'
    cve_matches = re.findall(cve_pattern, cve_string, re.IGNORECASE)
    
    # Standardize format (uppercase)
    standardized_cves = [cve.upper() for cve in cve_matches]
    
    # Return unique CVEs only (deduplicate within same post)
    return list(set(standardized_cves))

def explode_cve_rows(df):
    """Transform post-centric data to CVE-centric data"""
    
    print("🔄 TRANSFORMING TO CVE-CENTRIC STRUCTURE")
    print("-" * 50)
    
    # First, examine current CVE ID patterns
    print("Examining current CVE ID patterns...")
    sample_cves = df['cve_ids'].dropna().head(10)
    for i, cve_str in enumerate(sample_cves, 1):
        extracted = extract_cve_ids(cve_str)
        print(f"  {i}. Original: {cve_str[:100]}...")
        print(f"     Extracted: {extracted}")
    
    # Extract all CVE IDs from each row
    df['extracted_cves'] = df['cve_ids'].apply(extract_cve_ids)
    
    # Filter out rows with no valid CVEs
    df_with_cves = df[df['extracted_cves'].str.len() > 0].copy()
    
    print(f"\nRows with valid CVEs: {len(df_with_cves):,}")
    print(f"Total CVE mentions to explode: {df_with_cves['extracted_cves'].str.len().sum():,}")
    
    # Explode the list of CVEs into separate rows
    df_exploded = df_with_cves.explode('extracted_cves').copy()
    
    # Rename and clean up
    df_exploded['cve_id'] = df_exploded['extracted_cves']
    df_exploded = df_exploded.drop(columns=['cve_ids', 'extracted_cves'])
    
    # Remove any rows where cve_id is null after explosion
    df_exploded = df_exploded[df_exploded['cve_id'].notna()].copy()
    
    # Keep ALL CVE mentions (no deduplication)
    print(f"\nTotal CVE mentions preserved: {len(df_exploded):,} rows")
    
    # Sort by CVE ID for easier analysis
    df_exploded = df_exploded.sort_values(['cve_id', 'created_at_dt']).reset_index(drop=True)
    
    return df_exploded

# Load and clean data
df = pd.read_csv('data/mastodon/mastodon_raw.csv')
df_clean = df[df['cve_ids'].notna()].copy()

# Remove duplicate rows (exact same posts)
original_count = len(df_clean)
df_clean = df_clean.drop_duplicates().copy()
duplicates_removed = original_count - len(df_clean)
print(f"🧹 DEDUPLICATION: Removed {duplicates_removed:,} duplicate posts")

df_clean['created_at_dt'] = pd.to_datetime(df_clean['created_at'])
df_clean['hour'] = df_clean['created_at_dt'].dt.hour
df_clean['day_of_week'] = df_clean['created_at_dt'].dt.dayofweek

print("🤖 ADVANCED BOT CLASSIFICATION SYSTEM")
print("=" * 60)

# Classify all accounts
account_stats = df_clean['acct'].value_counts()
classifications = {}
classification_details = {}

for account, post_count in account_stats.items():
    account_data = df_clean[df_clean['acct'] == account].copy()
    features = extract_advanced_features(account_data)
    classification, factors = classify_account_advanced(features, account)
    classifications[account] = classification
    classification_details[account] = {
        'classification': classification,
        'factors': factors,
        'features': features
    }

# Add classification to dataset
df_clean['account_type'] = df_clean['acct'].map(classifications)

# Display detailed results for top accounts
print("\nDETAILED CLASSIFICATION RESULTS (Top 10):")
print("-" * 60)

for i, (account, post_count) in enumerate(account_stats.head(10).items(), 1):
    details = classification_details[account]
    classification = details['classification']
    factors = details['factors']
    
    print(f"\n{i:2d}. {classification}")
    print(f"    Account: {account}")
    print(f"    Posts: {post_count:,}")
    if details['features']:
        features = details['features']
        print(f"    Rate: {features['posts_per_day']:.1f}/day | Time var: {features['hour_variance']:.1f}")
    print(f"    Indicators: {', '.join(factors[:2]) if factors else 'Baseline human patterns'}")

# Summary statistics
classification_counts = df_clean['account_type'].value_counts()
print(f"\n" + "=" * 60)
print("POST-CENTRIC CLASSIFICATION SUMMARY:")
for class_type, count in classification_counts.items():
    percentage = (count / len(df_clean)) * 100
    print(f"{class_type}: {count:,} posts ({percentage:.1f}%)")

# EXPLODE CVE IDS INTO SEPARATE ROWS
print(f"\n" + "=" * 60)
df_cve_centric = explode_cve_rows(df_clean)

# CVE-centric analysis
print(f"\n" + "=" * 60)
print("CVE-CENTRIC ANALYSIS RESULTS:")
print(f"Unique CVEs: {df_cve_centric['cve_id'].nunique():,}")
print(f"Total CVE mentions: {len(df_cve_centric):,}")
print(f"Average mentions per CVE: {len(df_cve_centric) / df_cve_centric['cve_id'].nunique():.1f}")

# CVE-centric classification distribution
cve_classification_counts = df_cve_centric['account_type'].value_counts()
print(f"\nCVE-CENTRIC CLASSIFICATION DISTRIBUTION:")
for class_type, count in cve_classification_counts.items():
    percentage = (count / len(df_cve_centric)) * 100
    print(f"{class_type}: {count:,} CVE mentions ({percentage:.1f}%)")

# Most mentioned CVEs
print(f"\nTOP 10 MOST MENTIONED CVEs:")
top_cves = df_cve_centric['cve_id'].value_counts().head(10)
for i, (cve, count) in enumerate(top_cves.items(), 1):
    # Show account type breakdown for this CVE
    cve_accounts = df_cve_centric[df_cve_centric['cve_id'] == cve]['account_type'].value_counts()
    breakdown = ', '.join([f"{acct_type}: {count}" for acct_type, count in cve_accounts.head(2).items()])
    print(f"{i:2d}. {cve}: {count} mentions ({breakdown})")

print(f"\nTotal accounts classified: {len(classifications)}")
print(f"CVE-centric dataset saved with {len(df_cve_centric):,} rows")


# rename cve_id column to cve and rename created_at to date
df_cve_centric = df_cve_centric.rename(columns={'cve_id': 'cve', 'created_at': 'date'})
# Save both datasets
df_clean.to_csv('data/mastodon/mastodon_post_centric_classifications.csv', index=False)
df_cve_centric.to_csv('data/mastodon/mastodon_cve_centric_classifications.csv', index=False) 
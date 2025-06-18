import pandas as pd
import os

def merge_social_media_with_catalog(
    cve_catalog_path='processed_data/enriched_catalog.csv',
    mastodon_path='processed_data/mastodon_catalog.csv',
    reddit_path='processed_data/reddit_catalog.csv',
    mastodon_output_path='parquet_preprocessing/merged_mastodon.csv',
    reddit_output_path='parquet_preprocessing/merged_reddit.csv'
):

    # Load CSV files
    df1 = pd.read_csv(cve_catalog_path)
    df2 = pd.read_csv(mastodon_path)
    df3 = pd.read_csv(reddit_path) 

    #df1.drop(columns=['vendor'], inplace=True)

    # Clean column names (strip any leading/trailing whitespace)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()


    # Merge CVE catalog with Mastodon data
    merged_mastodon = pd.merge(df1, df2, left_on='CVE_ID', right_on='cve_ids', how='inner')
    merged_mastodon.drop(columns=['cve_ids'], inplace=True)

    # Merge CVE catalog with Reddit data
    merged_reddit = pd.merge(df1, df3, left_on='CVE_ID', right_on='CVE_ID', how='inner')
    #merged_reddit.drop(columns=['CVE_ID'], inplace=True)

    # Save the merged data
    merged_mastodon.to_csv(mastodon_output_path, index=False)
    merged_reddit.to_csv(reddit_output_path, index=False)

    print(f"Merge completed. Output saved as:\n- {mastodon_output_path}\n- {reddit_output_path}")



# ---------------- PIPELINE EXECUTION ----------------
merge_social_media_with_catalog()

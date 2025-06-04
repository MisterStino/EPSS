import os
import shutil
import time
import pandas as pd
from data.general_utils.utils import summarize_cve_time_ranges
from data.general_utils.utils import cast_common_columns
from data.general_utils.create_subsets import create_high_score_above_x_subset
from data.general_utils.utils import get_spark_session

def distribute_high_score_cves(
    x,
    team_members,
    date_range=None,
    full_data_path='data/full_db/processed/final_full_data.parquet',
    base_output_dir='data/general_utils/files/by_team',
    max_chunk_size=10000
):
    """
    Orchestrates:
      1) High-score subset creation
      2) Time-range summarization
      3) Splitting into 4 team parts
      4) Chunking each part into ≤max_chunk_size CSVs

    Parameters
    ----------
    x : float
        epss threshold (e.g. 0.7)
    team_members : list of str
        Folder names—one per team member (length should be 4)
    date_range : tuple(str, str) or None
        Optional ("YYYY-MM-DD","YYYY-MM-DD") filter
    full_data_path : str
        Path to your full Parquet folder
    base_output_dir : str
        Base folder under which per-team dirs will be created
    max_chunk_size : int
        Maximum CVEs per CSV chunk
    """
    # 1) Build the high-score subset
    create_high_score_above_x_subset(x, date_range)
    subset_path = os.path.join(
        'data', 'general_utils', 'files', f'high_score_above_{x}.parquet'
    )

    # 2) Summarize date ranges to a single CSV
    range_csv = os.path.join(base_output_dir, 'cve_time_ranges.csv')
    summarize_cve_time_ranges(
        interesting_cves_path=subset_path,
        full_data_path=full_data_path,
        output_file=range_csv
    )

    # 3) Load into pandas, sort by CVE
    df = pd.read_csv(range_csv)
    df = df.sort_values('cve').reset_index(drop=True)

    # 4) Compute how many CVEs per team (evenly + remainder)
    total = len(df)
    nteams = len(team_members)
    base_sz = total // nteams
    rem = total % nteams
    splits = []

    # print for sanity check
    print(f"Total CVEs: {total:,} | Base size: {base_sz:,} | Remainder: {rem:,}")
    print(f"Splitting into {nteams} teams: {team_members}") 
    print(f"Base size per team: {base_sz:,} | Extra for first {rem} teams")
    print(f"Max chunk size: {max_chunk_size:,}")
    
    start = 0
    for i in range(nteams):
        sz = base_sz + (1 if i < rem else 0)
        splits.append((start, start + sz))
        start += sz

    # 5) For each member, create folder and write chunked CSVs
    os.makedirs(base_output_dir, exist_ok=True)
    for idx, member in enumerate(team_members):
        member_dir = os.path.join(base_output_dir, member)
        os.makedirs(member_dir, exist_ok=True)

        part_df = df.iloc[splits[idx][0]:splits[idx][1]].reset_index(drop=True)
        nchunks = (len(part_df) + max_chunk_size - 1) // max_chunk_size

        for chunk_i in range(nchunks):
            chunk_df = part_df.iloc[
                chunk_i*max_chunk_size : (chunk_i+1)*max_chunk_size
            ]
            chunk_file = os.path.join(member_dir, f'chunk_{chunk_i+1}.csv')
            chunk_df.to_csv(chunk_file, index=False)
            print(f"Wrote {len(chunk_df)} rows to {chunk_file}")

    print(f"All chunks written under '{base_output_dir}'")

if __name__ == "__main__":
    # Example invocation with four team members
    members = ["federico", "andre", "stijnuno", "stijndos"]
    distribute_high_score_cves(
        x=0.7,
        team_members=members,
    )

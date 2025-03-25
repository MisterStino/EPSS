from data.general_utils.base_keys import load_base_keys
import pandas as pd
small_base_keys_df = load_base_keys(return_format="dataframe", small=True)
small_sample_df = pd.read_csv('data/general_utils/files/small_sampled_data.csv')
print("Base keys as DataFrame:")        
print(small_base_keys_df.head(100), '\n')

print("Sampled data as DataFrame:") 
print(small_sample_df.head(100))
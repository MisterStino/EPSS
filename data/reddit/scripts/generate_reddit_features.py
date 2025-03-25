from data.general_utils.base_keys import load_base_keys

base_keys_df = load_base_keys(return_format="dataframe", small=True)
print("Base keys as DataFrame:")        
print(base_keys_df.head(100))
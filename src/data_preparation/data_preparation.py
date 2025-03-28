import pandas as pd
import os

root_directory = "data/biosignals_filtered"

columns_to_drop = ['emg_corrugator', 'emg_zygomaticus']  

for root, dirs, files in os.walk(root_directory):
    for file_name in files:
        if file_name.endswith('.csv'): 
            input_file_path = os.path.join(root, file_name)
            
            df = pd.read_csv(input_file_path, sep='\t')
            
            df = df.drop(columns=columns_to_drop, errors='ignore')

            df.to_csv(input_file_path, sep='\t', index=False)
            
            print(f"Processed file: {input_file_path}")

print("Processing of all files complete.")

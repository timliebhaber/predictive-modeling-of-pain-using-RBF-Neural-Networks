import pandas as pd
import os

root_directory = "data/biosignals_filtered"

# Ensure the root directory exists
os.makedirs(root_directory, exist_ok=True)

columns_to_drop = ['emg_corrugator', 'emg_zygomaticus']  

for root, dirs, files in os.walk(root_directory):
    for file_name in files:
        if file_name.endswith('.csv'):  # Only process CSV files
            input_file_path = os.path.join(root, file_name)
            
            # Load the CSV file into a DataFrame with tab separator
            df = pd.read_csv(input_file_path, sep='\t')
            
            # Drop the specified columns
            df = df.drop(columns=columns_to_drop, errors='ignore')  # Ignore errors for missing columns
            
            # Save the updated DataFrame back to the same file with tab separator
            df.to_csv(input_file_path, sep='\t', index=False)
            
            print(f"Processed file: {input_file_path}")

print("Processing of all files complete.")

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:59:41 2024

@author: Ayesha Nadeem
"""

import pandas as pd

# List of input CSV files
input_files = ["train_disease_data_preprocessed.csv", "train_health_data_preprocessed.csv", "train_travel_data.csv"]

# Output CSV file
output_file = "merged_data.csv"

# Merge CSV files
dfs = []
for file in input_files:
    df = pd.read_csv(file)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

# Save merged DataFrame to CSV
merged_df.to_csv(output_file, index=False)

# Check merged DataFrame
print(merged_df.head())

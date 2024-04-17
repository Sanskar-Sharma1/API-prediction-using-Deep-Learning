import pandas as pd
import os

# Assuming your CSV files are named file1.csv, file2.csv, file3.csv, and file4.csv
file_paths = ['EastSikkim_processed.csv', 'WestSikkim_processed.csv', 'SouthSikkim_processed.csv', 'NorthSikkim_processed.csv']

# Create an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Iterate through each file and concatenate the data
for file_path in file_paths:
    df = pd.read_csv(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)

# Display the first few rows of the combined DataFrame
print(combined_df.head())

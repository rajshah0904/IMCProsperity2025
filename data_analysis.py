import pandas as pd

# Load the CSV files into DataFrames
file_paths = [
    '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_1.csv',
    '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_2.csv',
    '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_3.csv'
]

dataframes = [pd.read_csv(file_path, delimiter=';') for file_path in file_paths]

# Explore the data
for i, df in enumerate(dataframes, start=1):
    print(f"\nDataFrame for Day {i}:")
    print(df.head())  # Display the first few rows
    print(df.info())  # Display information about the DataFrame
    print(df.describe())  # Display basic statistics

# Note: Adjust the file paths if necessary to match your directory structure. 
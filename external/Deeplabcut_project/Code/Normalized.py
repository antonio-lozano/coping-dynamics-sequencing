# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:32:03 2025

@author: jsangui
"""

import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.formula.api as smf
#%%

 ##Rename CSVs
# Define the directory containing the CSV files
folder_path = r"C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Freezing_predictions_light"

# Process each CSV file in the directory
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Rename columns
        df.columns = ["Frames", "Freezing_predictions"]
        
        # Save the updated CSV, overwriting the original
        df.to_csv(file_path, index=False)
        
        print(f"Updated: {file}")

print("All CSV files have been updated.")
#%%

# Define the directory containing the CSV files
folder_path = r"C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Freezing_predictions_light"

# Mapping from Animal No. to Condition and Experiment
animal_info = {
    "2.4":  ("ELS", 1), "5.4":  ("Control", 1), "11.4": ("Control", 1), "11.6": ("Control", 1),
    "15.4": ("ELS", 1), "15.6": ("ELS", 1), "23.4": ("ELS", 1), "25.1": ("ELS", 1),
    "26.4": ("ELS", 1), "32.2": ("ELS", 1), "32.4": ("ELS", 1), "34.4": ("Control", 1),
    "34.5": ("Control", 1), "40.4": ("Control", 1), "40.6": ("Control", 1), "41.4": ("Control", 1),
    "41.5": ("Control", 1), "41.6": ("Control", 1), "43.3": ("ELS", 1), "43.5": ("ELS", 1),
    "47.5": ("Control", 1), "48.6": ("Control", 1), "49.5": ("ELS", 1), "49.6": ("ELS", 1),
    "50.5": ("Control", 1), "53.5": ("Control", 1), "54.3": ("Control", 1), "54.4": ("Control", 1),
    "54.6": ("Control", 1), "55.3": ("ELS", 1), "55.5": ("ELS", 1), "56.3": ("ELS", 1),
    "58.5": ("Control", 1), "62.1": ("Control", 1), "62.2": ("Control", 1), "62.3": ("Control", 1),
    "62.4": ("Control", 1), "62.5": ("Control", 1), "63.3": ("ELS", 1), "68.1": ("Control", 1),
    "69.1": ("ELS", 1), "69.2": ("ELS", 1), "69.3": ("ELS", 1), "69.5": ("ELS", 1),
    "73.1": ("ELS", 1), "73.4": ("ELS", 1), "74.1": ("Control", 1), "75.1": ("ELS", 1),
    "80.3": ("Control", 1), "88.2": ("ELS", 1), "88.3": ("ELS", 1), "123.1": ("ELS", 3),
    "123.3": ("ELS", 3), "128.4": ("ELS", 3), "129.1": ("Control", 3), "129.4": ("Control", 3),
    "133.3": ("ELS", 3), "134.2": ("ELS", 3), "134.4": ("ELS", 3), "136.2": ("Control", 3),
    "136.3": ("Control", 3), "136.5": ("Control", 3), "137.3": ("Control", 3), "137.5": ("Control", 3),
    "144.3": ("ELS", 3), "144.5": ("ELS", 3), "145.5": ("Control", 3), "146.6": ("Control", 3),
    "147.4": ("Control", 3), "148.4": ("ELS", 3), "150.3": ("ELS", 3), "150.5": ("ELS", 3),
    "152.3": ("Control", 3), "152.5": ("Control", 3), "153.4": ("ELS", 3), "154.4": ("ELS", 3),
    "157.1": ("Control", 3), "157.4": ("Control", 3), "158.5": ("Control", 3), "159.3": ("ELS", 3),
    "159.4": ("ELS", 3), "159.5": ("ELS", 3), "161.1": ("Control", 3), "161.3": ("Control", 3),
    "192.2": ("ELS", 5), "192.3": ("ELS", 5), "193.1": ("ELS", 5), "194.1": ("Control", 5),
    "194.2": ("Control", 5), "195.1": ("Control", 5), "195.3": ("Control", 5), "195.4": ("Control", 5),
    "197.1": ("Control", 5), "200.1": ("Control", 5), "201.1": ("ELS", 5), "203.2": ("ELS", 5),
    "203.3": ("ELS", 5), "204.4": ("ELS", 5)
}

# Process each CSV file in the directory
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        
        # Extract Animal number from the filename
        match = re.search(r"(\d+\.\d+)", file.replace("_", "."))  # Convert _ to . for proper matching
        if match:
            animal_no = match.group(1)
            if animal_no in animal_info:
                condition, experiment = animal_info[animal_no]
                
                # Load CSV without deleting existing data
                df = pd.read_csv(file_path)
                
                # Rename first two columns, if applicable
                if df.shape[1] >= 2:
                    df.rename(columns={df.columns[0]: "Frames", df.columns[1]: "Freezing_predictions"}, inplace=True)
                
                # Add new columns without removing existing data
                df["Animal_No"] = animal_no
                df["Condition"] = condition
                df["Experiment"] = experiment
                
                # Save the updated CSV
                df.to_csv(file_path, index=False)
                
                print(f"Updated: {file} -> Animal No: {animal_no}, Condition: {condition}, Experiment: {experiment}")
            else:
                print(f"Warning: No match found for {file}")

print("All CSV files have been updated.")

print("All CSV files have been updated.")
#%%

# Define frame rate and time bin size
FPS = 25
BIN_SIZE_SECONDS = 0.5
BIN_SIZE_FRAMES = FPS * BIN_SIZE_SECONDS 

# Process each CSV file in the directory
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        
        # Extract Animal number from the filename
        match = re.search(r"(\d+\.\d+)", file.replace("_", "."))  # Convert _ to . for proper matching
        if match:
            animal_no = match.group(1)
            if animal_no in animal_info:
                condition, experiment = animal_info[animal_no]
                
                # Load CSV without modifying original data
                df = pd.read_csv(file_path)
                
                # Rename first two columns, if applicable
                if df.shape[1] >= 2:
                    df.rename(columns={df.columns[0]: "Frames", df.columns[1]: "Freezing_predictions"}, inplace=True)

                # Add new columns
                df["Animal_No"] = animal_no
                df["Condition"] = condition
                df["Experiment"] = experiment
                
                # Create time bins (every 750 frames)
                df["Time_Bin"] = df["Frames"] // BIN_SIZE_FRAMES  # Integer division to group by bins

                # Compute freezing percentage per bin
                freezing_summary = df.groupby("Time_Bin")["Freezing_predictions"].mean() * 100  # Convert to %

                # Print result for verification (you can use this summary in further processing)
                print(f"\n{file} - Freezing % per 30 sec bin:")
                print(freezing_summary)

print("All CSV files processed with 30-second bins (in memory).")

#%%

# Define frame rate and time bin size
FPS = 25
BIN_SIZE_SECONDS = 30
BIN_SIZE_FRAMES = FPS * BIN_SIZE_SECONDS
BASELINE_FRAMES = 3 * 60 * FPS  # First 3 minutes


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define frame rate and time bin size
FPS = 25
BIN_SIZE_SECONDS = 30
BIN_SIZE_FRAMES = FPS * BIN_SIZE_SECONDS
BASELINE_FRAMES = 3 * 60 * FPS  # First 3 minutes

# Folder path containing CSV files
folder_path = r"C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Freezing_predictions_light"

# Store results
all_results = []

# Process each CSV file in the directory
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        
        # Load CSV
        df = pd.read_csv(file_path)

        # Rename first two columns if necessary
        if df.shape[1] >= 2:
            df.rename(columns={df.columns[0]: "Frames", df.columns[1]: "Freezing_predictions"}, inplace=True)

        # Ensure Condition column exists
        if "Condition" in df.columns:
            group = df["Condition"].iloc[0]  # Extract group label from file
        else:
            raise ValueError(f"Condition column not found in {file}")

        # Create time bins
        df["Time_Bin"] = df["Frames"] // BIN_SIZE_FRAMES
        
        # Compute freezing percentage per bin
        freezing_summary = df.groupby("Time_Bin")["Freezing_predictions"].mean() * 100  # Convert to %
        
        # Determine baseline per individual from the first 3 minutes (first 6 bins)
        individual_baseline = freezing_summary[freezing_summary.index < (BASELINE_FRAMES // BIN_SIZE_FRAMES)].mean()
        
        # Normalize freezing percentages (baseline = 100%)
        normalized_freezing = (freezing_summary / individual_baseline) * 100
        
        # Store results
        result_df = pd.DataFrame({
            "Time_Bin": normalized_freezing.index,
            "Normalized_Freezing_%": normalized_freezing.values,
            "Group": group
        })
        
        all_results.append(result_df)
        

# Combine all results into a single DataFrame
final_df = pd.concat(all_results, ignore_index=True)

# Compute mean and SEM per group
summary_df = final_df.groupby(["Group", "Time_Bin"]).agg(
    Mean_Freezing_Percent=("Normalized_Freezing_%", "mean"),
    SEM_Freezing_Percent=("Normalized_Freezing_%", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
).reset_index()

# Plot both groups together with shaded SEM
plt.figure(figsize=(6,5))
for group in summary_df["Group"].unique():
    group_data = summary_df[summary_df["Group"] == group]
    plt.plot(group_data["Time_Bin"], group_data["Mean_Freezing_Percent"], label=group, marker='o')
    plt.fill_between(group_data["Time_Bin"], 
                     group_data["Mean_Freezing_Percent"] - group_data["SEM_Freezing_Percent"], 
                     group_data["Mean_Freezing_Percent"] + group_data["SEM_Freezing_Percent"], 
                     alpha=0.3, label=f"{group} SEM")
plt.xlabel("Time Bin (30s each)")
plt.ylabel("Normalized Freezing %")
plt.title("Normalized Freezing Percentage Over Time (Both Groups)")
plt.legend()
plt.show()
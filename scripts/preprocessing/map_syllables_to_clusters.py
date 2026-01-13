from copy import deepcopy
import numpy as np

from src.config import RESULTS_RAW_PKL, RESULTS_CLUSTERS_PKL

# Define the new mapping from group names to numeric codes
new_numeric_mapping = {
    "Freezing": 1,
    "Turn": 4,
    "Locomotion": 5,
    "Sniffing": 2,
    "Climbing": 6,
    "Jump": 7,
    "Grooming": 3,
    "Mix": 8,
    "Basura": 9
}

# Define the new syllable grouping (from your JSON)
new_syllable_mapping = {
    "Freezing": [0, 28],
    "Jump": [23, 29, 30, 34],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Basura": [2, 4, 8, 9, 22, 31, 32, 33],
    "Sniffing": [18, 20],
    "Grooming": [24],
    "Mix": [7, 13, 17],
    "Climbing": [111]
}

# Build a reverse mapping from each old syllable to its new group name.
old_to_group = {}
for group_label, syll_list in new_syllable_mapping.items():
    for syl in syll_list:
        old_to_group[syl] = group_label

# =============================================================================
# Load old results and create new_mapped_results
# =============================================================================
import pickle

save_path = RESULTS_RAW_PKL
print(f"Loading results from {save_path}")

with open(save_path, "rb") as f:
    results_dict = pickle.load(f)

# Create a new dictionary that will store the mapped results.
new_mapped_results = {}
for rec_key, rec_data in results_dict.items():
    new_rec_data = deepcopy(rec_data)
    if "syllable" in new_rec_data:
        old_arr = new_rec_data["syllable"]
        # Ensure we have a standard ndarray by explicitly converting.
        old_arr = np.array(old_arr)
        # Initialize a new syllable array of the same shape filled with -1.
        new_arr = np.full(old_arr.shape, 888)
        # Loop over each element.
        for i, syl in enumerate(old_arr):
            if syl in old_to_group:
                new_arr[i] = new_numeric_mapping[old_to_group[syl]]
        # **Explicitly convert to a standard ndarray before assignment**
        new_rec_data["syllable"] = np.array(new_arr)
    new_mapped_results[rec_key] = new_rec_data

# =============================================================================
# Save the new mapped results dictionary
# =============================================================================
output_path = RESULTS_CLUSTERS_PKL
with open(output_path, "wb") as f:
    pickle.dump(new_mapped_results, f)
print("Saved new_mapped_results to", output_path)
#%%

# Reload to verify
with open(output_path, "rb") as f:
    new_mapped_results = pickle.load(f)
print("Loaded new_mapped_results from", output_path)
#%%
# =============================================================================
# Simple check: print one example entry
# =============================================================================
example_key = list(new_mapped_results.keys())[0]
print("Example new syllable array for", example_key, ":", results_dict[example_key]["syllable"][0:1000:50])
print("Example new syllable array for", example_key, ":", new_mapped_results[example_key]["syllable"][0:1000:50])
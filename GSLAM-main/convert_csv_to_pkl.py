import pandas as pd
import numpy as np
import pickle
import os

# Path to your CSV directory
csv_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET"

# List of CSV base names and their PKL equivalents
files = {
    "OVERLAP.csv": "OVERLAP.pkl",
    "IMGRPOSX.csv": "IMGRPOSX.pkl",
    "IMGRPOSY.csv": "IMGRPOSY.pkl",
    "IMGRPOSO.csv": "IMGRPOSO.pkl",
    "ODOM.csv": "ODOM.pkl"
}

# Loop through and convert each
for csv_name, pkl_name in files.items():
    csv_path = os.path.join(csv_dir, csv_name)
    pkl_path = os.path.join(csv_dir, pkl_name)

    # Load and convert to float32 numpy array
    df = pd.read_csv(csv_path, header=None)
    array = df.values.astype(np.float32)

    # Save as pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(array, f)

    print(f"✅ Converted {csv_name} → {pkl_name}")

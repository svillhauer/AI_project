import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# User should replace these with actual paths to their folders
folders = [
    '/path/to/run1',
    '/path/to/run2',
    '/path/to/run3'
]
metric_file = 'training_metrics.csv'

# Load all CSVs
all_dfs = []
for folder in folders:
    path = os.path.join(folder, metric_file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_dfs.append(df)
    else:
        print(f"Warning: {path} does not exist.")

if len(all_dfs) == 0:
    raise ValueError("No valid CSV files found in the provided folders.")

# Concatenate with keys for each run
all_metrics = pd.concat(all_dfs, keys=range(len(all_dfs)))

# Compute mean and std grouped by epoch
mean_metrics = all_metrics.groupby('epoch').mean()
std_metrics = all_metrics.groupby('epoch').std()

# Plotting
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold'})

pastel_purple = '#c9a0dc'
pastel_pink = '#f7a1c4'
pastel_blue = '#a1c8f7'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loss plot
axes[0].plot(mean_metrics.index, mean_metrics['train_loss'], label='Train Loss', color=pastel_purple, lw=2)
axes[0].fill_between(mean_metrics.index,
                    mean_metrics['train_loss'] - std_metrics['train_loss'],
                    mean_metrics['train_loss'] + std_metrics['train_loss'],
                    color=pastel_purple, alpha=0.3)
axes[0].plot(mean_metrics.index, mean_metrics['val_loss'], label='Val Loss', color=pastel_pink, lw=2)
axes[0].fill_between(mean_metrics.index,
                    mean_metrics['val_loss'] - std_metrics['val_loss'],
                    mean_metrics['val_loss'] + std_metrics['val_loss'],
                    color=pastel_pink, alpha=0.3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Loss')
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(mean_metrics.index, mean_metrics['train_acc'], label='Train Acc', color=pastel_blue, lw=2)
axes[1].fill_between(mean_metrics.index,
                    mean_metrics['train_acc'] - std_metrics['train_acc'],
                    mean_metrics['train_acc'] + std_metrics['train_acc'],
                    color=pastel_blue, alpha=0.3)
axes[1].plot(mean_metrics.index, mean_metrics['val_acc'], label='Val Acc', color=pastel_purple, lw=2)
axes[1].fill_between(mean_metrics.index,
                    mean_metrics['val_acc'] - std_metrics['val_acc'],
                    mean_metrics['val_acc'] + std_metrics['val_acc'],
                    color=pastel_purple, alpha=0.3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].set_title('Accuracy')
axes[1].grid(True, alpha=0.3)

# Computational time plot
axes[2].plot(mean_metrics.index, mean_metrics['epoch_time_sec'], label='Epoch Time (s)', color=pastel_pink, lw=2)
axes[2].fill_between(mean_metrics.index,
                    mean_metrics['epoch_time_sec'] - std_metrics['epoch_time_sec'],
                    mean_metrics['epoch_time_sec'] + std_metrics['epoch_time_sec'],
                    color=pastel_pink, alpha=0.3)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Time (seconds)')
axes[2].legend()
axes[2].set_title('Computational Time per Epoch')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

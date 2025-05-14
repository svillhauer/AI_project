import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Full paths to each metrics CSV file
csv_files = [
    '/home/svillhauer/Desktop/data/CNN_REALDATA/training_metrics_01.csv',
    '/home/svillhauer/Desktop/data/CNN_REALDATA/training_metrics_02.csv',
    '/home/svillhauer/Desktop/data/CNN_REALDATA/training_metrics_03.csv',
]

# Load all CSVs
all_dfs = []
for path in csv_files:
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_dfs.append(df)
    else:
        print(f"Warning: {path} does not exist.")

if len(all_dfs) == 0:
    raise ValueError("No valid CSV files found in the provided list.")

# Concatenate with keys for each run and flatten MultiIndex
all_metrics = pd.concat(all_dfs, keys=range(len(all_dfs)))
if isinstance(all_metrics.index, pd.MultiIndex):
    all_metrics = all_metrics.reset_index(level=0, drop=True)

# Compute mean and std grouped by epoch
mean_metrics = all_metrics.groupby('epoch').mean().reset_index()
std_metrics = all_metrics.groupby('epoch').std().reset_index()

# Plotting setup
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold'})

pastel_purple = '#c9a0dc'
pastel_pink = '#f7a1c4'
pastel_blue = '#a1c8f7'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Convert to numpy arrays
epoch = mean_metrics['epoch'].to_numpy()
train_loss = mean_metrics['train_loss'].to_numpy()
val_loss = mean_metrics['val_loss'].to_numpy()
train_loss_std = std_metrics['train_loss'].to_numpy()
val_loss_std = std_metrics['val_loss'].to_numpy()

train_acc = mean_metrics['train_accuracy'].to_numpy()
val_acc = mean_metrics['val_accuracy'].to_numpy()
train_acc_std = std_metrics['train_accuracy'].to_numpy()
val_acc_std = std_metrics['val_accuracy'].to_numpy()

epoch_time = mean_metrics['epoch_time_sec'].to_numpy()
epoch_time_std = std_metrics['epoch_time_sec'].to_numpy()

print("Epoch times (mean):", epoch_time)
print("Epoch time std dev :", epoch_time_std)

# Loss plot
axes[0].plot(epoch, train_loss, label='Train Loss', color=pastel_purple, lw=2)
axes[0].fill_between(epoch, train_loss - train_loss_std, train_loss + train_loss_std,
                     color=pastel_purple, alpha=0.3)
axes[0].plot(epoch, val_loss, label='Val Loss', color=pastel_pink, lw=2)
axes[0].fill_between(epoch, val_loss - val_loss_std, val_loss + val_loss_std,
                     color=pastel_pink, alpha=0.3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Loss')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(np.arange(1, 16))  # <-- Added
axes[0].set_xlim(0.5, 15.5)  # <-- Added

# Accuracy plot
axes[1].plot(epoch, train_acc, label='Train Acc', color=pastel_purple, lw=2)
axes[1].fill_between(epoch, train_acc - train_acc_std, train_acc + train_acc_std,
                     color=pastel_purple, alpha=0.3)
axes[1].plot(epoch, val_acc, label='Val Acc', color=pastel_pink, lw=2)
axes[1].fill_between(epoch, val_acc - val_acc_std, val_acc + val_acc_std,
                     color=pastel_pink, alpha=0.3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].set_title('Accuracy')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(np.arange(1, 16))  # <-- Added
axes[1].set_xlim(0.5, 15.5)  # <-- Added

# Computational time plot
axes[2].plot(epoch, epoch_time, label='Epoch Time (s)', color=pastel_blue, lw=2)
axes[2].fill_between(epoch, epoch_time - epoch_time_std, epoch_time + epoch_time_std,
                     color=pastel_blue, alpha=0.3)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Time (seconds)')
axes[2].legend()
axes[2].set_title('Computational Time per Epoch')
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(np.arange(1, 16))  # <-- Added
axes[2].set_xlim(0.5, 15.5)  # <-- Added


plt.tight_layout()
plt.show()

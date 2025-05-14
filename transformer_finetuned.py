
import pandas as pd
from sklearn.model_selection import train_test_split

# ====== Config ======
csv_path = "/home/sami/Documents/AI2/finetuned/SAMPLE_RANDOM/OVERLAP_PAIRS.csv"
output_dir = "/home/sami/Documents/AI2/finetuned/SAMPLE_RANDOM/"  # Folder to save split csvs
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1


# ====== Load Data ======
df = pd.read_csv(csv_path)

# ====== Shuffle and Split ======
train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

# ====== Save Splits ======
train_df.to_csv(f"{output_dir}/train.csv", index=False)
val_df.to_csv(f"{output_dir}/val.csv", index=False)
test_df.to_csv(f"{output_dir}/test.csv", index=False)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print("Dataset split and saved.")

import numpy as np
from time import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, SwinModel
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import copy

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch

from torchvision.utils import make_grid

from collections import defaultdict

# ====== Config ======
model_name = "microsoft/swin-tiny-patch4-window7-224"
batch_size = 10
num_epochs = 10
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Image Preprocessing ======
processor = AutoImageProcessor.from_pretrained(model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ====== Dataset Class ======
class OverlapDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img1_path = os.path.join(self.root_dir, row['img1'])
        img2_path = os.path.join(self.root_dir, row['img2'])

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        label = torch.tensor(row['label'], dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# ====== Model ======
class SwinSiamese(nn.Module):
    def __init__(self):
        super(SwinSiamese, self).__init__()
        self.swin = SwinModel.from_pretrained(model_name)

        # First freeze all layers
        for param in self.swin.parameters():
            param.requires_grad = False

        # Then unfreeze selected layers
        for name, param in self.swin.named_parameters():
            if "embeddings" in name or "encoder.layers.3" in name:
                param.requires_grad = True

        hidden_size = self.swin.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        emb1 = self.swin(pixel_values=img1).pooler_output
        emb2 = self.swin(pixel_values=img2).pooler_output
        combined = torch.cat([emb1, emb2], dim=1)
        return self.fc(combined)


# ====== Helper Functions ======
def log_trainable_params(model):
    print("\nTrainable parameters:\n" + "-"*40)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<80} | shape: {tuple(param.shape)} | requires_grad: {param.requires_grad}")

def get_param_stats(model):
    param_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.data.norm().item()
            weight_norm = param.data.norm().item()
            param_stats[name] = {
                "grad_norm": grad_norm,
                "weight_norm": weight_norm
            }
    return param_stats

# ====== Dataloader ======
train_csv = os.path.join(output_dir, "train.csv")
val_csv = os.path.join(output_dir, "val.csv")
test_csv = os.path.join(output_dir, "test.csv")
img_dir = os.path.join(output_dir, "IMAGES")

train_dataset = OverlapDataset(train_csv, root_dir=img_dir, transform=transform)
val_dataset = OverlapDataset(val_csv, root_dir=img_dir, transform=transform)
test_dataset = OverlapDataset(test_csv, root_dir=img_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

threshold = 0.5

# ====== PARAM DIFF TRACKING ======
param_history = defaultdict(lambda: {"grad": [], "weight": [], "diff": []})

# ====== Model, Loss, Optimizer ======
model = SwinSiamese().to(device)
criterion = nn.BCELoss()

# ====== Separate params for different learning rates ======
swin_params = []
classifier_params = []
for name, param in model.named_parameters():
    if "swin" in name:
        swin_params.append(param)
    else:
        classifier_params.append(param)

optimizer = torch.optim.Adam([ # Separate Optimizer with different learning rates for better learning
    {'params': swin_params, 'lr': 1e-5},
    {'params': classifier_params, 'lr': 1e-4}
])

# ====== Training Loop + Validation ======
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

log_trainable_params(model)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # ====== PARAM DIFF TRACKING: Store previous params at epoch start ======
    prev_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for img1, img2, label in progress_bar:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img1, img2).squeeze()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (output > threshold).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

        progress_bar.set_postfix(loss=loss.item())

        # Collect grad and weight norms
        stats = get_param_stats(model)
        for name, s in stats.items():
            param_history[name]["grad"].append(s["grad_norm"])
            param_history[name]["weight"].append(s["weight_norm"])

    # ====== PARAM DIFF TRACKING: Compute param differences at epoch end ======
    for name, param in model.named_parameters():
        if param.requires_grad:
            diff = (param.detach() - prev_params[name]).norm().item()
            param_history[name]["diff"].append(diff)

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # ====== Validation ======
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2).squeeze()
            loss = criterion(output, label)

            val_loss += loss.item()
            preds = (output > threshold).float()
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # ====== Save best model ======
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

# ====== Load 'Best' Model Before Test ======
model.load_state_dict(best_model_wts)

# ====== Test Evaluation ======
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for img1, img2, label in tqdm(test_loader, desc="Testing"):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        output = model(img1, img2).squeeze()
        loss = criterion(output, label)

        test_loss += loss.item()
        preds = (output > threshold).float()
        test_correct += (preds == label).sum().item()
        test_total += label.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = test_correct / test_total
print(f"\nTest Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# ====== Plotting Loss & Accuracy ======
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ====== CONFIG ======
save_dir = "/home/sami/Documents/AI2/finetuned/results"
# os.makedirs(save_dir, exist_ok=True)

# ====== SELECTED PARAM GROUPS ======
layer_groups = {
    "layer3": [k for k in param_history if "swin.encoder.layers.3" in k and "weight" in k],
    # "layer2": [k for k in param_history if "swin.encoder.layers.2" in k and "weight" in k],
    # "layer1": [k for k in param_history if "swin.encoder.layers.1" in k and "weight" in k],
    # "layer0": [k for k in param_history if "swin.encoder.layers.0" in k and "weight" in k],
    "embeddings": [k for k in param_history if "swin.embeddings" in k and "weight" in k],
    # "layernorm": [k for k in param_history if "swin.layernorm." in k and "weight" in k],
    "classifier": [k for k in param_history if "fc." in k and "weight" in k],
}

# ====== SAVE PARAM HISTORY TO CSV ======
for group_name, keys in layer_groups.items():
    for metric in ["weight", "grad", "diff"]:
        # Filter out keys that actually exist in param_history
        valid_keys = [k for k in keys if k in param_history and param_history[k][metric]]
        if not valid_keys:
            print(f"[SKIP] No valid data for {group_name} - {metric}")
            continue  # Skip if nothing valid

        file_path = os.path.join(save_dir, f"{group_name}_{metric}.csv")
        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch"] + valid_keys
            writer.writerow(header)
            max_len = max(len(param_history[k][metric]) for k in valid_keys)
            for i in range(max_len):
                row = [i]
                for k in valid_keys:
                    val_list = param_history[k][metric]
                    row.append(val_list[i] if i < len(val_list) else "")
                writer.writerow(row)

print(f"[INFO] CSV logs saved to {save_dir}")

# ====== SAVE PLOTS TO PNG ======
def plot_and_save(title, ylabel, series_dict, filename):
    plt.figure(figsize=(12, 5))

    for name, series in series_dict.items():
        plt.plot(series, label=name)

    plt.title(title)
    plt.xlabel("Epoch" if len(series) == len(train_losses) else "Batch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# Plot weight norms
for group_name, keys in layer_groups.items():
    weight_series = {k: param_history[k]["weight"] for k in keys[:4]}
    plot_and_save(f"{group_name}: Weight Norms", "Weight Norm", weight_series, f"{group_name}_weights.png")

# Plot grad norms
for group_name, keys in layer_groups.items():
    grad_series = {k: param_history[k]["grad"] for k in keys[:4]}
    plot_and_save(f"{group_name}: Gradient Norms", "Grad Norm", grad_series, f"{group_name}_grads.png")

# Plot param diffs
for group_name, keys in layer_groups.items():
    diff_series = {k: param_history[k]["diff"] for k in keys[:4]}
    plot_and_save(f"{group_name}: Param Differences", "Param Diff", diff_series, f"{group_name}_diffs.png")

# Plot train/val loss and accuracy
plot_and_save("Training and Validation Loss", "Loss", {
    "Train Loss": train_losses, "Val Loss": val_losses
}, "loss_curve.png")

plot_and_save("Training and Validation Accuracy", "Accuracy", {
    "Train Accuracy": train_accuracies, "Val Accuracy": val_accuracies
}, "accuracy_curve.png")

print(f"[INFO] All plots saved to {save_dir}")



# ====== Save Best Model ======
torch.save(model.state_dict(), "/home/sami/Documents/AI2/finetuned/models/swin_siamese_best_tuned.pth")
print("Model is saved")
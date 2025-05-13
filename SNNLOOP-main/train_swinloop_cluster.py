import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, SwinModel
import copy
import time

# ====== CONFIG ======
csv_path = "/home/mundus/svillhaue213/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
output_dir = "/home/mundus/svillhaue213/AI_project/SNNLOOP-main/TRAN_RESULTS_CLUSTER"
os.makedirs(output_dir, exist_ok=True)

image_dir = "/home/mundus/svillhaue213/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
model_name = "microsoft/swin-tiny-patch4-window7-224"
batch_size = 16
num_epochs = 5
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== DATA SPLIT ======
df = pd.read_csv(csv_path)
df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})
#df = df.sample(frac=0.01, random_state=42)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

# ====== IMAGE PROCESSING ======
processor = AutoImageProcessor.from_pretrained(model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ====== DATASET CLASS ======
class OverlapDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img1 = Image.open(os.path.join(self.root_dir, row['img1'])).convert("RGB")
        img2 = Image.open(os.path.join(self.root_dir, row['img2'])).convert("RGB")
        label = torch.tensor(row['label'], dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# ====== MODEL DEFINITION ======
class SwinSiamese(nn.Module):
    def __init__(self):
        super(SwinSiamese, self).__init__()
        self.swin = SwinModel.from_pretrained(model_name)
        for param in self.swin.parameters():
            param.requires_grad = False
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
        return self.fc(torch.cat([emb1, emb2], dim=1))

# ====== DATALOADERS ======
train_dataset = OverlapDataset(os.path.join(output_dir, "train.csv"), image_dir, transform)
val_dataset = OverlapDataset(os.path.join(output_dir, "val.csv"), image_dir, transform)
test_dataset = OverlapDataset(os.path.join(output_dir, "test.csv"), image_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ====== TRAINING SETUP ======
model = SwinSiamese().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_losses, val_losses, train_accuracies, val_accuracies, epoch_durations = [], [], [], [], []
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

# ====== TRAINING LOOP ======
for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for img1, img2, label in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img1, img2).squeeze()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (output > 0.5).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2).squeeze()
            loss = criterion(output, label)

            val_loss += loss.item()
            preds = (output > 0.5).float()
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_correct / val_total)
    duration = time.time() - start_time
    epoch_durations.append(duration)

    print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, Time: {duration:.2f}s")

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model_wts = copy.deepcopy(model.state_dict())

# ====== SAVE MODEL & METRICS ======
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), os.path.join(output_dir, "swin_siamese_best.pth"))

# Save predictions
model.eval()
all_labels, all_preds, all_scores = [], [], []
with torch.no_grad():
    for img1, img2, label in test_loader:
        img1, img2 = img1.to(device), img2.to(device)
        output = model(img1, img2).squeeze().cpu()
        all_scores.extend(output.numpy())
        all_preds.extend((output > 0.5).float().numpy())
        all_labels.extend(label.numpy())

pd.DataFrame({"label": all_labels, "prediction": all_preds, "score": all_scores}).to_csv(
    os.path.join(output_dir, "test_results.csv"), index=False)

# Save training metrics
pd.DataFrame({
    "epoch": list(range(1, num_epochs + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accuracies,
    "val_acc": val_accuracies,
    "epoch_time_sec": epoch_durations
}).to_csv(os.path.join(output_dir, "training_metrics.csv"), index=False)

# Plot curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss over Epochs"); plt.legend(); plt.grid()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Accuracy over Epochs"); plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_curves.png"))
print("Training complete and saved to output directory.")


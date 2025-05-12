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
matplotlib.use('Agg')  # Prevents GUI-related hanging during save
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, SwinModel
import copy

# ====== CONFIG ======
# csv_path = "/Users/sarahvillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"  # Your original dataset CSV
# output_dir = "/Users/sarahvillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_TEST"         # Where to save train/val/test CSVs
# image_dir = "/Users/sarahvillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"    # Path to images folder

csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"  # Your original dataset CSV
output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_TEST"         # Where to save train/val/test CSVs
image_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"    # Path to images folder
model_name = "microsoft/swin-tiny-patch4-window7-224"
batch_size = 16
num_epochs = 2
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== LOAD & SPLIT DATA ======
# df = pd.read_csv(csv_path)
# df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})

# train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
# val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)

# train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
# val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
# test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

# print(f"[INFO] Split complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")


df = pd.read_csv(csv_path)

# Force rename
df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})
df = df.sample(frac=0.01, random_state=42)  # Use 20% of data

# Now overwrite clean splits
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)

# Save with corrected columns
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


# ====== IMAGE PREPROCESSING ======
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
        img1_path = os.path.join(self.root_dir, row['img1'])
        img2_path = os.path.join(self.root_dir, row['img2'])

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
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
        combined = torch.cat([emb1, emb2], dim=1)
        return self.fc(combined)

# ====== DATALOADERS ======
# train_dataset = OverlapDataset("train.csv", output_dir, transform)
# val_dataset = OverlapDataset("val.csv", output_dir, transform)
# test_dataset = OverlapDataset("test.csv", output_dir, transform)

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

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

# ====== TRAINING LOOP ======
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for img1, img2, label in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img1, img2).squeeze()
        #output = model(img1, img2).squeeze(dim=1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (output > 0.5).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # ====== VALIDATION ======
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            #output = model(img1, img2).squeeze()
            output = model(img1, img2).squeeze(dim=1)
            loss = criterion(output, label)

            val_loss += loss.item()
            preds = (output > 0.5).float()
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_correct / val_total)

    print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

    # Save best model
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model_wts = copy.deepcopy(model.state_dict())

# ====== SAVE MODEL ======
model.load_state_dict(best_model_wts)
#torch.save(model.state_dict(), "swin_siamese_best.pth")
torch.save(model.state_dict(), os.path.join(output_dir, "swin_siamese_best.pth"))
print("Best model saved to 'swin_siamese_best.pth'")

# ====== SAVE TEST PREDICTIONS AND GROUND TRUTH ======
print("Evaluating on test set...")
model.eval()
all_labels, all_preds, all_scores = [], [], []

with torch.no_grad():
    for img1, img2, label in test_loader:
        img1, img2 = img1.to(device), img2.to(device)
        #output = model(img1, img2).squeeze().cpu()
        output = model(img1, img2).squeeze(dim=1).cpu()
        score = output.numpy()
        pred = (output > 0.5).float().numpy()
        label = label.cpu().numpy()

        all_scores.extend(score)
        all_preds.extend(pred)
        all_labels.extend(label)

# Save to CSV
results_df = pd.DataFrame({
    "label": all_labels,
    "prediction": all_preds,
    "score": all_scores
})
results_path = os.path.join(output_dir, "test_results.csv")
results_df.to_csv(results_path, index=False)
print(f"Saved test predictions to '{results_path}'")


# ====== SAVE TRAINING METRICS ======
metrics_df = pd.DataFrame({
    "epoch": list(range(1, num_epochs + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accuracies,
    "val_acc": val_accuracies
})
metrics_path = os.path.join(output_dir, "training_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"Training metrics saved to '{metrics_path}'")

# ====== PLOT LOSS & ACCURACY ======
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Val Loss")
# plt.legend(); plt.grid(); plt.title("Loss over Epochs")
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label="Train Acc")
# plt.plot(val_accuracies, label="Val Acc")
# plt.legend(); plt.grid(); plt.title("Accuracy over Epochs")
# plt.tight_layout()
# plt.savefig("training_curves.png")
# print("Training plot saved as 'training_curves.png'")

print("Saving training curves...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(); plt.grid(); plt.title("Loss over Epochs")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.legend(); plt.grid(); plt.title("Accuracy over Epochs")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()
print("Training plot saved as 'training_curves.png'")


# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import matplotlib
# matplotlib.use('Agg')  # Prevents GUI-related hanging during save
# import matplotlib.pyplot as plt
# from transformers import AutoImageProcessor, SwinModel
# import copy

# # ====== CONFIG ======
# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"  # Your original dataset CSV
# output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS"         # Where to save train/val/test CSVs
# image_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"    # Path to images folder
# model_name = "microsoft/swin-tiny-patch4-window7-224"
# # csv_path = "/Users/sarahvillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"  # Your original dataset CSV
# # output_dir = "/Users/sarahvillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS"         # Where to save train/val/test CSVs
# # image_dir = "/Users/sarahvillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"    # Path to images folder
# # model_name = "microsoft/swin-tiny-patch4-window7-224"
# batch_size = 16
# num_epochs = 10
# learning_rate = 1e-4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ====== LOAD & SPLIT DATA ======
# # df = pd.read_csv(csv_path)
# # df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})

# # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
# # val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)

# # train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
# # val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
# # test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

# # print(f"[INFO] Split complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")


# df = pd.read_csv(csv_path)

# # Force rename
# df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})
# #df = df.sample(frac=0.01, random_state=42)  # Use 20% of data

# # Now overwrite clean splits
# train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
# val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)

# # Save with corrected columns
# train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
# val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
# test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


# # ====== IMAGE PREPROCESSING ======
# processor = AutoImageProcessor.from_pretrained(model_name)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
# ])

# # ====== DATASET CLASS ======
# class OverlapDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         img1_path = os.path.join(self.root_dir, row['img1'])
#         img2_path = os.path.join(self.root_dir, row['img2'])

#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")
#         label = torch.tensor(row['label'], dtype=torch.float32)

#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)

#         return img1, img2, label

# # ====== MODEL DEFINITION ======
# class SwinSiamese(nn.Module):
#     def __init__(self):
#         super(SwinSiamese, self).__init__()
#         self.swin = SwinModel.from_pretrained(model_name)
#         for param in self.swin.parameters():
#             param.requires_grad = False

#         hidden_size = self.swin.config.hidden_size
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, img1, img2):
#         emb1 = self.swin(pixel_values=img1).pooler_output
#         emb2 = self.swin(pixel_values=img2).pooler_output
#         combined = torch.cat([emb1, emb2], dim=1)
#         return self.fc(combined)

# # ====== DATALOADERS ======
# # train_dataset = OverlapDataset("train.csv", output_dir, transform)
# # val_dataset = OverlapDataset("val.csv", output_dir, transform)
# # test_dataset = OverlapDataset("test.csv", output_dir, transform)

# train_dataset = OverlapDataset(os.path.join(output_dir, "train.csv"), image_dir, transform)
# val_dataset = OverlapDataset(os.path.join(output_dir, "val.csv"), image_dir, transform)
# test_dataset = OverlapDataset(os.path.join(output_dir, "test.csv"), image_dir, transform)


# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # ====== TRAINING SETUP ======
# model = SwinSiamese().to(device)
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train_losses, val_losses = [], []
# train_accuracies, val_accuracies = [], []
# best_val_loss = float('inf')
# best_model_wts = copy.deepcopy(model.state_dict())

# # ====== TRAINING LOOP ======
# for epoch in range(num_epochs):
#     model.train()
#     running_loss, correct, total = 0.0, 0, 0

#     for img1, img2, label in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
#         img1, img2, label = img1.to(device), img2.to(device), label.to(device)
#         optimizer.zero_grad()
#         output = model(img1, img2).squeeze()
#         #output = model(img1, img2).squeeze(dim=1)
#         loss = criterion(output, label)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         preds = (output > 0.5).float()
#         correct += (preds == label).sum().item()
#         total += label.size(0)

#     train_losses.append(running_loss / len(train_loader))
#     train_accuracies.append(correct / total)

#     # ====== VALIDATION ======
#     model.eval()
#     val_loss, val_correct, val_total = 0.0, 0, 0
#     with torch.no_grad():
#         for img1, img2, label in val_loader:
#             img1, img2, label = img1.to(device), img2.to(device), label.to(device)
#             #output = model(img1, img2).squeeze()
#             output = model(img1, img2).squeeze(dim=1)
#             loss = criterion(output, label)

#             val_loss += loss.item()
#             preds = (output > 0.5).float()
#             val_correct += (preds == label).sum().item()
#             val_total += label.size(0)

#     val_losses.append(val_loss / len(val_loader))
#     val_accuracies.append(val_correct / val_total)

#     print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

#     # Save best model
#     if val_losses[-1] < best_val_loss:
#         best_val_loss = val_losses[-1]
#         best_model_wts = copy.deepcopy(model.state_dict())

# # ====== SAVE MODEL ======
# model.load_state_dict(best_model_wts)
# torch.save(model.state_dict(), "swin_siamese_best.pth")
# print("Best model saved to 'swin_siamese_best.pth'")

# # ====== SAVE TEST PREDICTIONS AND GROUND TRUTH ======
# print("Evaluating on test set...")
# model.eval()
# all_labels, all_preds, all_scores = [], [], []

# with torch.no_grad():
#     for img1, img2, label in test_loader:
#         img1, img2 = img1.to(device), img2.to(device)
#         #output = model(img1, img2).squeeze().cpu()
#         output = model(img1, img2).squeeze(dim=1).cpu()
#         score = output.numpy()
#         pred = (output > 0.5).float().numpy()
#         label = label.cpu().numpy()

#         all_scores.extend(score)
#         all_preds.extend(pred)
#         all_labels.extend(label)

# # Save to CSV
# results_df = pd.DataFrame({
#     "label": all_labels,
#     "prediction": all_preds,
#     "score": all_scores
# })
# results_path = os.path.join(output_dir, "test_results.csv")
# results_df.to_csv(results_path, index=False)
# print(f"Saved test predictions to '{results_path}'")


# # ====== SAVE TRAINING METRICS ======
# metrics_df = pd.DataFrame({
#     "epoch": list(range(1, num_epochs + 1)),
#     "train_loss": train_losses,
#     "val_loss": val_losses,
#     "train_acc": train_accuracies,
#     "val_acc": val_accuracies
# })
# metrics_path = os.path.join(output_dir, "training_metrics.csv")
# metrics_df.to_csv(metrics_path, index=False)
# print(f"Training metrics saved to '{metrics_path}'")

# # ====== PLOT LOSS & ACCURACY ======
# # plt.figure(figsize=(12, 5))
# # plt.subplot(1, 2, 1)
# # plt.plot(train_losses, label="Train Loss")
# # plt.plot(val_losses, label="Val Loss")
# # plt.legend(); plt.grid(); plt.title("Loss over Epochs")
# # plt.subplot(1, 2, 2)
# # plt.plot(train_accuracies, label="Train Acc")
# # plt.plot(val_accuracies, label="Val Acc")
# # plt.legend(); plt.grid(); plt.title("Accuracy over Epochs")
# # plt.tight_layout()
# # plt.savefig("training_curves.png")
# # print("Training plot saved as 'training_curves.png'")

# print("Saving training curves...")
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Val Loss")
# plt.legend(); plt.grid(); plt.title("Loss over Epochs")

# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label="Train Acc")
# plt.plot(val_accuracies, label="Val Acc")
# plt.legend(); plt.grid(); plt.title("Accuracy over Epochs")

# plt.tight_layout()
# plt.savefig("training_curves.png")
# plt.close()
# print("Training plot saved as 'training_curves.png'")


# # import os
# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import transforms
# # from PIL import Image
# # from tqdm import tqdm
# # from sklearn.model_selection import train_test_split
# # import matplotlib.pyplot as plt
# # from transformers import AutoImageProcessor, SwinModel
# # import copy

# # # ====== CONFIG ======
# # csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"  # Your original dataset CSV
# # output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS"         # Where to save train/val/test CSVs
# # image_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"    # Path to images folder
# # model_name = "microsoft/swin-tiny-patch4-window7-224"
# # batch_size = 16
# # num_epochs = 10
# # learning_rate = 1e-4
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ====== LOAD & SPLIT DATA ======
# # # df = pd.read_csv(csv_path)
# # # df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})

# # # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
# # # val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)

# # # train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
# # # val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
# # # test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

# # # print(f"[INFO] Split complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")


# # df = pd.read_csv(csv_path)

# # # Force rename
# # df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})

# # # Now overwrite clean splits
# # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
# # val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)

# # # Save with corrected columns
# # train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
# # val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
# # test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


# # # ====== IMAGE PREPROCESSING ======
# # processor = AutoImageProcessor.from_pretrained(model_name)
# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
# # ])

# # # ====== DATASET CLASS ======
# # class OverlapDataset(Dataset):
# #     def __init__(self, csv_file, root_dir, transform=None):
# #         self.data = pd.read_csv(csv_file)
# #         self.root_dir = root_dir
# #         self.transform = transform

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         row = self.data.iloc[idx]
# #         img1_path = os.path.join(self.root_dir, row['img1'])
# #         img2_path = os.path.join(self.root_dir, row['img2'])

# #         img1 = Image.open(img1_path).convert("RGB")
# #         img2 = Image.open(img2_path).convert("RGB")
# #         label = torch.tensor(row['label'], dtype=torch.float32)

# #         if self.transform:
# #             img1 = self.transform(img1)
# #             img2 = self.transform(img2)

# #         return img1, img2, label

# # # ====== MODEL DEFINITION ======
# # class SwinSiamese(nn.Module):
# #     def __init__(self):
# #         super(SwinSiamese, self).__init__()
# #         self.swin = SwinModel.from_pretrained(model_name)
# #         for param in self.swin.parameters():
# #             param.requires_grad = False

# #         hidden_size = self.swin.config.hidden_size
# #         self.fc = nn.Sequential(
# #             nn.Linear(hidden_size * 2, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 1),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, img1, img2):
# #         emb1 = self.swin(pixel_values=img1).pooler_output
# #         emb2 = self.swin(pixel_values=img2).pooler_output
# #         combined = torch.cat([emb1, emb2], dim=1)
# #         return self.fc(combined)

# # # ====== DATALOADERS ======
# # # train_dataset = OverlapDataset("train.csv", output_dir, transform)
# # # val_dataset = OverlapDataset("val.csv", output_dir, transform)
# # # test_dataset = OverlapDataset("test.csv", output_dir, transform)

# # train_dataset = OverlapDataset(os.path.join(output_dir, "train.csv"), image_dir, transform)
# # val_dataset = OverlapDataset(os.path.join(output_dir, "val.csv"), image_dir, transform)
# # test_dataset = OverlapDataset(os.path.join(output_dir, "test.csv"), image_dir, transform)


# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # # ====== TRAINING SETUP ======
# # model = SwinSiamese().to(device)
# # criterion = nn.BCELoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # train_losses, val_losses = [], []
# # train_accuracies, val_accuracies = [], []
# # best_val_loss = float('inf')
# # best_model_wts = copy.deepcopy(model.state_dict())

# # # ====== TRAINING LOOP ======
# # for epoch in range(num_epochs):
# #     model.train()
# #     running_loss, correct, total = 0.0, 0, 0

# #     for img1, img2, label in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
# #         img1, img2, label = img1.to(device), img2.to(device), label.to(device)
# #         optimizer.zero_grad()
# #         output = model(img1, img2).squeeze()
# #         loss = criterion(output, label)
# #         loss.backward()
# #         optimizer.step()

# #         running_loss += loss.item()
# #         preds = (output > 0.5).float()
# #         correct += (preds == label).sum().item()
# #         total += label.size(0)

# #     train_losses.append(running_loss / len(train_loader))
# #     train_accuracies.append(correct / total)

# #     # ====== VALIDATION ======
# #     model.eval()
# #     val_loss, val_correct, val_total = 0.0, 0, 0
# #     with torch.no_grad():
# #         for img1, img2, label in val_loader:
# #             img1, img2, label = img1.to(device), img2.to(device), label.to(device)
# #             output = model(img1, img2).squeeze()
# #             loss = criterion(output, label)

# #             val_loss += loss.item()
# #             preds = (output > 0.5).float()
# #             val_correct += (preds == label).sum().item()
# #             val_total += label.size(0)

# #     val_losses.append(val_loss / len(val_loader))
# #     val_accuracies.append(val_correct / val_total)

# #     print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

# #     # Save best model
# #     if val_losses[-1] < best_val_loss:
# #         best_val_loss = val_losses[-1]
# #         best_model_wts = copy.deepcopy(model.state_dict())

# # # ====== SAVE MODEL ======
# # model.load_state_dict(best_model_wts)
# # torch.save(model.state_dict(), "swin_siamese_best.pth")
# # print("Best model saved to 'swin_siamese_best.pth'")

# # # ====== PLOT LOSS & ACCURACY ======
# # plt.figure(figsize=(12, 5))
# # plt.subplot(1, 2, 1)
# # plt.plot(train_losses, label="Train Loss")
# # plt.plot(val_losses, label="Val Loss")
# # plt.legend(); plt.grid(); plt.title("Loss over Epochs")
# # plt.subplot(1, 2, 2)
# # plt.plot(train_accuracies, label="Train Acc")
# # plt.plot(val_accuracies, label="Val Acc")
# # plt.legend(); plt.grid(); plt.title("Accuracy over Epochs")
# # plt.tight_layout()
# # plt.savefig("training_curves.png")
# # print("Training plot saved as 'training_curves.png'")

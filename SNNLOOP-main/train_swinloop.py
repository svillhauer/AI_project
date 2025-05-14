import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import time

# ====== CONFIG ======
csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL"
embedding_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/embeddings"
batch_size = 10
num_epochs = 15
learning_rate = 1e-4 # change to 1e-4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== DATA PREPARATION ======
print("\n[1/4] Preparing datasets...")
df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Save splits
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(f"{output_dir}/train.csv", index=False)
val_df.to_csv(f"{output_dir}/val.csv", index=False)
test_df.to_csv(f"{output_dir}/test.csv", index=False)

# ====== EMBEDDING DATASET ======
class EmbeddingDataset(Dataset):
    def __init__(self, csv_file, embedding_dir):
        self.data = pd.read_csv(csv_file)
        self.embedding_dir = embedding_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emb1 = np.load(f"{self.embedding_dir}/{os.path.splitext(row['img1'])[0]}.npy")
        emb2 = np.load(f"{self.embedding_dir}/{os.path.splitext(row['img2'])[0]}.npy")
        return (
            torch.FloatTensor(emb1),
            torch.FloatTensor(emb2),
            torch.tensor(row['label'], dtype=torch.float32)
        )

# ====== MODEL ARCHITECTURE ======
print("\n[2/4] Building model...")
class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim=768*2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2):
        combined = torch.cat([emb1, emb2], dim=1)
        return self.classifier(combined).squeeze(1)

# ====== TRAINING SETUP ======
print("\n[3/4] Initializing training...")
train_dataset = EmbeddingDataset(f"{output_dir}/train.csv", embedding_dir)
val_dataset = EmbeddingDataset(f"{output_dir}/val.csv", embedding_dir)
test_dataset = EmbeddingDataset(f"{output_dir}/test.csv", embedding_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model = EmbeddingClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ====== TRAINING LOOP ======
print("\n[4/4] Training classifier...")
best_val_acc = 0
history = {
    'train_loss': [], 
    'val_loss': [], 
    'train_acc': [], 
    'val_acc': [],
    'epoch_time': []
}

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss, train_correct, total = 0, 0, 0
    for emb1, emb2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(emb1, emb2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = (outputs > 0.5).float()
        train_correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for emb1, emb2, labels in val_loader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            outputs = model(emb1, emb2)
            val_loss += criterion(outputs, labels).item()
            val_correct += ((outputs > 0.5).float() == labels).sum().item()
            val_total += labels.size(0)
    
    # Calculate metrics
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_acc = train_correct / total
    val_acc = val_correct / val_total
    epoch_time = time.time() - epoch_start
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['epoch_time'].append(epoch_time)
    
    print(f"\nEpoch {epoch+1:02d}/{num_epochs} | "
          f"Time: {epoch_time:.1f}s | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{output_dir}/swin_classifier_best.pth")

# ====== SAVE METRICS ======
metrics_df = pd.DataFrame({
    'epoch': range(1, num_epochs+1),
    'train_loss': history['train_loss'],
    'val_loss': history['val_loss'],
    'train_acc': history['train_acc'],
    'val_acc': history['val_acc'],
    'epoch_time_sec': history['epoch_time']
})
metrics_df.to_csv(f"{output_dir}/training_metrics.csv", index=False)

# ====== EVALUATION ======
model.load_state_dict(torch.load(f"{output_dir}/swin_classifier_best.pth"))
model.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
    for emb1, emb2, labels in test_loader:
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
        outputs = (model(emb1, emb2) > 0.5).float()
        test_correct += (outputs == labels).sum().item()
        test_total += labels.size(0)

print(f"\nFinal Test Accuracy: {test_correct/test_total:.3f}")

# ====== VISUALIZATION ======
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train')
plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(metrics_df['epoch'], metrics_df['train_acc'], label='Train')
plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{output_dir}/training_curves.png")
plt.close()

print("\nTraining complete! Saved:")
print(f"- Best model: {output_dir}/swin_classifier_best.pth")
print(f"- Training metrics: {output_dir}/training_metrics.csv")
print(f"- Test results: {test_correct/test_total:.3f} accuracy")
print(f"- Training curves: {output_dir}/training_curves.png")


# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import copy
# import time

# # ====== CONFIG ======
# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL"
# embedding_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/embeddings"
# batch_size = 64  # Can use larger batches with embeddings
# num_epochs = 10  # Train longer since it's faster
# learning_rate = 1e-3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ====== DATA PREPARATION ======
# print("\n[1/4] Preparing datasets...")
# df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})

# # Split data
# train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
# val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# # Save splits
# os.makedirs(output_dir, exist_ok=True)
# train_df.to_csv(f"{output_dir}/train.csv", index=False)
# val_df.to_csv(f"{output_dir}/val.csv", index=False)
# test_df.to_csv(f"{output_dir}/test.csv", index=False)

# # ====== EMBEDDING DATASET ======
# class EmbeddingDataset(Dataset):
#     def __init__(self, csv_file, embedding_dir):
#         self.data = pd.read_csv(csv_file)
#         self.embedding_dir = embedding_dir

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
        
#         # Load embeddings
#         emb1 = np.load(f"{self.embedding_dir}/{os.path.splitext(row['img1'])[0]}.npy")
#         emb2 = np.load(f"{self.embedding_dir}/{os.path.splitext(row['img2'])[0]}.npy")
        
#         return (
#             torch.FloatTensor(emb1),
#             torch.FloatTensor(emb2),
#             torch.tensor(row['label'], dtype=torch.float32)
#         )

# # ====== MODEL ARCHITECTURE ======
# print("\n[2/4] Building model...")
# class EmbeddingClassifier(nn.Module):
#     def __init__(self, input_dim=768*2):  # 768 for Swin-Tiny
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, emb1, emb2):
#         combined = torch.cat([emb1, emb2], dim=1)
#         return self.classifier(combined).squeeze(1)

# # ====== TRAINING SETUP ======
# print("\n[3/4] Initializing training...")
# train_dataset = EmbeddingDataset(f"{output_dir}/train.csv", embedding_dir)
# val_dataset = EmbeddingDataset(f"{output_dir}/val.csv", embedding_dir)
# test_dataset = EmbeddingDataset(f"{output_dir}/test.csv", embedding_dir)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# model = EmbeddingClassifier().to(device)
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # ====== TRAINING LOOP ======
# print("\n[4/4] Training classifier...")
# best_val_acc = 0
# history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# for epoch in range(num_epochs):
#     start_time = time.time()
    
#     # Training
#     model.train()
#     train_loss, train_correct, total = 0, 0, 0
#     for emb1, emb2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(emb1, emb2)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
#         preds = (outputs > 0.5).float()
#         train_correct += (preds == labels).sum().item()
#         total += labels.size(0)
    
#     # Validation
#     model.eval()
#     val_loss, val_correct, val_total = 0, 0, 0
#     with torch.no_grad():
#         for emb1, emb2, labels in val_loader:
#             emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            
#             outputs = model(emb1, emb2)
#             val_loss += criterion(outputs, labels).item()
#             val_correct += ((outputs > 0.5).float() == labels).sum().item()
#             val_total += labels.size(0)
    
#     # Metrics
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
#     train_acc = train_correct / total
#     val_acc = val_correct / val_total
    
#     history['train_loss'].append(train_loss)
#     history['val_loss'].append(val_loss)
#     history['train_acc'].append(train_acc)
#     history['val_acc'].append(val_acc)
    
#     print(f"\nEpoch {epoch+1:02d}/{num_epochs} | "
#           f"Time: {time.time()-start_time:.1f}s | "
#           f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
#           f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
    
#     # Save best model
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), f"{output_dir}/swin_classifier_best.pth")

# # ====== EVALUATION ======
# model.load_state_dict(torch.load(f"{output_dir}/swin_classifier_best.pth"))
# model.eval()

# test_correct, test_total = 0, 0
# with torch.no_grad():
#     for emb1, emb2, labels in test_loader:
#         emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
#         outputs = (model(emb1, emb2) > 0.5).float()
#         test_correct += (outputs == labels).sum().item()
#         test_total += labels.size(0)

# print(f"\nFinal Test Accuracy: {test_correct/test_total:.3f}")

# # Save training curves
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history['train_loss'], label='Train')
# plt.plot(history['val_loss'], label='Validation')
# plt.title('Loss Curves')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(history['train_acc'], label='Train')
# plt.plot(history['val_acc'], label='Validation')
# plt.title('Accuracy Curves')
# plt.legend()
# plt.savefig(f"{output_dir}/training_curves.png")
# plt.close()


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
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from transformers import AutoImageProcessor, SwinModel
# import copy
# import time

# # ====== CONFIGURATION ======
# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL"
# image_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
# model_name = "microsoft/swin-tiny-patch4-window7-224"
# batch_size = 16
# num_epochs = 2
# learning_rate = 1e-4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ====== DATA PREPARATION ======
# print("\n[1/6] Preparing datasets...")
# df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# df = df.sample(frac=0.01, random_state=42)  # Reduced dataset for quick testing

# # Split data with stratification
# train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'], shuffle=True)
# val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42, stratify=temp_df['label'])

# # Save splits
# os.makedirs(output_dir, exist_ok=True)
# train_df.to_csv(f"{output_dir}/train.csv", index=False)
# val_df.to_csv(f"{output_dir}/val.csv", index=False)
# test_df.to_csv(f"{output_dir}/test.csv", index=False)

# # ====== IMAGE PROCESSING ======
# print("\n[2/6] Configuring image transforms...")
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
#         img1 = Image.open(f"{self.root_dir}/{row['img1']}").convert("RGB")
#         img2 = Image.open(f"{self.root_dir}/{row['img2']}").convert("RGB")
#         label = torch.tensor(row['label'], dtype=torch.float32)
#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#         return img1, img2, label  # label is scalar float tensor

# # ====== MODEL ARCHITECTURE ======
# print("\n[3/6] Building model...")
# class SwinSiamese(nn.Module):
#     def __init__(self, freeze_swin=True):
#         super().__init__()
#         self.swin = SwinModel.from_pretrained(model_name)
#         if freeze_swin:
#             for param in self.swin.parameters():
#                 param.requires_grad = False
#         hidden_size = self.swin.config.hidden_size
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size*2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, img1, img2):
#         emb1 = self.swin(pixel_values=img1).pooler_output
#         emb2 = self.swin(pixel_values=img2).pooler_output
#         return self.classifier(torch.cat([emb1, emb2], dim=1)).squeeze(1)  # shape: [batch_size]

# # ====== TRAINING SETUP ======
# print("\n[4/6] Initializing training components...")
# model = SwinSiamese().to(device)
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Data loaders
# train_dataset = OverlapDataset(f"{output_dir}/train.csv", image_dir, transform)
# val_dataset = OverlapDataset(f"{output_dir}/val.csv", image_dir, transform)
# test_dataset = OverlapDataset(f"{output_dir}/test.csv", image_dir, transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# # ====== TRAINING LOOP ======
# print("\n[5/6] Starting training...")
# best_val_loss = float('inf')
# history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# for epoch in range(num_epochs):
#     start_time = time.time()
#     model.train()
#     train_loss, train_correct, total = 0, 0, 0
#     for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
#         optimizer.zero_grad()
#         outputs = model(img1, img2)  # shape: [batch_size]
#         labels = labels.view(-1)     # shape: [batch_size]
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         preds = (outputs > 0.5).float()
#         train_correct += (preds == labels).sum().item()
#         total += labels.size(0)
#     # Validation phase
#     model.eval()
#     val_loss, val_correct, val_total = 0, 0, 0
#     with torch.no_grad():
#         for img1, img2, labels in val_loader:
#             img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
#             outputs = model(img1, img2)  # shape: [batch_size]
#             labels = labels.view(-1)     # shape: [batch_size]
#             val_loss += criterion(outputs, labels).item()
#             preds = (outputs > 0.5).float()
#             val_correct += (preds == labels).sum().item()
#             val_total += labels.size(0)
#     # Record metrics
#     epoch_time = time.time() - start_time
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
#     train_acc = train_correct / total
#     val_acc = val_correct / val_total
#     history['train_loss'].append(train_loss)
#     history['val_loss'].append(val_loss)
#     history['train_acc'].append(train_acc)
#     history['val_acc'].append(val_acc)
#     print(f"\nEpoch {epoch+1}/{num_epochs} | "
#           f"Time: {epoch_time:.1f}s | "
#           f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
#           f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
#     # Save best model
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), f"{output_dir}/swin_siamese_best.pth")

# # ====== EVALUATION & SAVING ======
# print("\n[6/6] Final evaluation and saving...")
# model.load_state_dict(torch.load(f"{output_dir}/swin_siamese_best.pth"))
# model.eval()
# all_preds, all_labels, all_scores = [], [], []
# with torch.no_grad():
#     for img1, img2, labels in test_loader:
#         img1, img2 = img1.to(device), img2.to(device)
#         outputs = model(img1, img2).cpu().numpy()
#         all_scores.extend(outputs)
#         all_preds.extend((outputs > 0.5).astype(float))
#         all_labels.extend(labels.numpy())
# results_df = pd.DataFrame({
#     'label': all_labels,
#     'prediction': all_preds,
#     'score': all_scores
# })
# results_df.to_csv(f"{output_dir}/test_results.csv", index=False)
# history_df = pd.DataFrame({
#     'epoch': range(1, num_epochs+1),
#     'train_loss': history['train_loss'],
#     'val_loss': history['val_loss'],
#     'train_acc': history['train_acc'],
#     'val_acc': history['val_acc']
# })
# history_df.to_csv(f"{output_dir}/training_history.csv", index=False)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['val_loss'], label='Val Loss')
# plt.title('Loss Curves')
# plt.legend()
# plt.grid(True)
# plt.subplot(1, 2, 2)
# plt.plot(history['train_acc'], label='Train Accuracy')
# plt.plot(history['val_acc'], label='Val Accuracy')
# plt.title('Accuracy Curves')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{output_dir}/training_curves.png")
# plt.close()
# print("\nTraining complete! Saved:")
# print(f"- Best model: {output_dir}/swin_siamese_best.pth")
# print(f"- Test results: {output_dir}/test_results.csv")
# print(f"- Training history: {output_dir}/training_history.csv")
# print(f"- Training curves: {output_dir}/training_curves.png")


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
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from transformers import AutoImageProcessor, SwinModel
# import copy
# import time

# # ====== CONFIG ======
# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL"
# image_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
# model_name = "microsoft/swin-tiny-patch4-window7-224"
# batch_size = 16
# num_epochs = 2
# learning_rate = 1e-4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ====== DATA SPLIT ======
# df = pd.read_csv(csv_path)
# df = df.rename(columns={"first": "img1", "second": "img2", "match": "label"})
# df = df.sample(frac=0.01, random_state=42)
# train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
# val_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)
# train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
# val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
# test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

# # ====== IMAGE PROCESSING ======
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
#         img1 = Image.open(os.path.join(self.root_dir, row['img1'])).convert("RGB")
#         img2 = Image.open(os.path.join(self.root_dir, row['img2'])).convert("RGB")
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
#         return self.fc(torch.cat([emb1, emb2], dim=1))

# # ====== DATALOADERS ======
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
# train_losses, val_losses, train_accuracies, val_accuracies, epoch_durations = [], [], [], [], []
# best_val_loss = float('inf')
# best_model_wts = copy.deepcopy(model.state_dict())

# # ====== TRAINING LOOP ======
# for epoch in range(num_epochs):
#     start_time = time.time()

#     model.train()
#     running_loss, correct, total = 0.0, 0, 0

#     for img1, img2, label in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
#         img1, img2, label = img1.to(device), img2.to(device), label.to(device)
#         optimizer.zero_grad()
#         output = model(img1, img2).squeeze()
#         loss = criterion(output, label)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         preds = (output > 0.5).float()
#         correct += (preds == label).sum().item()
#         total += label.size(0)

#     train_losses.append(running_loss / len(train_loader))
#     train_accuracies.append(correct / total)

#     model.eval()
#     val_loss, val_correct, val_total = 0.0, 0, 0
#     with torch.no_grad():
#         for img1, img2, label in val_loader:
#             img1, img2, label = img1.to(device), img2.to(device), label.to(device)
#             output = model(img1, img2).squeeze()
#             loss = criterion(output, label)

#             val_loss += loss.item()
#             preds = (output > 0.5).float()
#             val_correct += (preds == label).sum().item()
#             val_total += label.size(0)

#     val_losses.append(val_loss / len(val_loader))
#     val_accuracies.append(val_correct / val_total)
#     duration = time.time() - start_time
#     epoch_durations.append(duration)

#     print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, Time: {duration:.2f}s")

#     if val_losses[-1] < best_val_loss:
#         best_val_loss = val_losses[-1]
#         best_model_wts = copy.deepcopy(model.state_dict())

# # ====== SAVE MODEL & METRICS ======
# model.load_state_dict(best_model_wts)
# torch.save(model.state_dict(), os.path.join(output_dir, "swin_siamese_best.pth"))

# # Save predictions
# model.eval()
# all_labels, all_preds, all_scores = [], [], []
# with torch.no_grad():
#     for img1, img2, label in test_loader:
#         img1, img2 = img1.to(device), img2.to(device)
#         output = model(img1, img2).squeeze().cpu()
#         all_scores.extend(output.numpy())
#         all_preds.extend((output > 0.5).float().numpy())
#         all_labels.extend(label.numpy())

# pd.DataFrame({"label": all_labels, "prediction": all_preds, "score": all_scores}).to_csv(
#     os.path.join(output_dir, "test_results.csv"), index=False)

# # Save training metrics
# pd.DataFrame({
#     "epoch": list(range(1, num_epochs + 1)),
#     "train_loss": train_losses,
#     "val_loss": val_losses,
#     "train_acc": train_accuracies,
#     "val_acc": val_accuracies,
#     "epoch_time_sec": epoch_durations
# }).to_csv(os.path.join(output_dir, "training_metrics.csv"), index=False)

# # Plot curves
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Val Loss")
# plt.title("Loss over Epochs"); plt.legend(); plt.grid()
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label="Train Acc")
# plt.plot(val_accuracies, label="Val Acc")
# plt.title("Accuracy over Epochs"); plt.legend(); plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "training_curves.png"))
# print("Training complete and saved to output directory.")


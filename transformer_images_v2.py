# -*- coding: utf-8 -*-

# Add this before ANY other imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silences TensorFlow logging

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, SwinModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import copy

# ====== CONFIGURATION ======
class Config:
    # Paths
    data_root = "/home/sami/Documents/AI2/v2/SAMPLE_RANDOM_3images/"
    csv_root = os.path.join(data_root)
    image_dir = os.path.join(data_root, "IMAGES/")
    model_save_path = "/home/sami/Documents/AI2/v2/models/output_3imgs_2.pth"
    
    # Model
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    embedding_dim = 768  # swin-tiny hidden size
    
    # Training
    batch_size = 20
    num_epochs = 15
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

# ====== PREPROCESSING & EMBEDDING COMPUTATION ======
def precompute_embeddings(config):
    """Precompute embeddings for all unique images in dataset"""
    # Collect all image paths
    dfs = [
        pd.read_csv(os.path.join(config.csv_root, f"{split}.csv"))
        for split in ["train", "val", "test"]
    ]
    all_images = set()
    for df in dfs:
        all_images.update(df['img1'].tolist())
        all_images.update(df['img2'].tolist())
    
    # Initialize model and transforms
    swin = SwinModel.from_pretrained(config.model_name).to(config.device)
    swin.eval()
    processor = AutoImageProcessor.from_pretrained(config.model_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    # Compute embeddings
    embeddings = {}
    with torch.no_grad():
        for img_path in tqdm(all_images, desc="Precomputing embeddings"):
            img = Image.open(os.path.join(config.image_dir, img_path)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(config.device)
            emb = swin(pixel_values=tensor).pooler_output.cpu().squeeze()
            embeddings[img_path] = emb
    
    # Save embeddings
    embedding_path = os.path.join(config.data_root, "precomputed_embeddings.pt")
    torch.save(embeddings, embedding_path)
    return embedding_path

# ====== DATASET CLASSES ======
class EmbeddingDataset(Dataset):
    def __init__(self, csv_path, embeddings):
        self.data = pd.read_csv(csv_path)
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emb1 = self.embeddings[row['img1']]
        emb2 = self.embeddings[row['img2']]
        label = torch.tensor(row['label'], dtype=torch.float32)
        return emb1, emb2, label

# ====== MODEL ARCHITECTURE ======
class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    # Sigmoid model
    '''def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim*2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ) ''' 
    
    def forward(self, emb1, emb2):
        combined = torch.cat([emb1, emb2], dim=1)
        return self.fc(combined)

# ====== TRAINING UTILITIES ======
def create_dataloaders(embedding_path, config):
    embeddings = torch.load(embedding_path)
    splits = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(config.data_root, f"{split}.csv")
        dataset = EmbeddingDataset(csv_path, embeddings)
        shuffle = (split == "train")
        splits[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            pin_memory=True
        )
    return splits['train'], splits['val'], splits['test']

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for emb1, emb2, labels in tqdm(loader, desc="Training"):
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(emb1, emb2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss/len(loader), correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for emb1, emb2, labels in loader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            outputs = model(emb1, emb2).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss/len(loader), correct/total

# ====== MAIN TRAINING PIPELINE ======
import matplotlib.pyplot as plt

def main():
    print("Main started")
    config = Config()
    print("Config loaded")

    embedding_path = os.path.join(config.data_root, "precomputed_embeddings.pt")
    if not os.path.exists(embedding_path):
        print("Precomputing embeddings...")
        embedding_path = precompute_embeddings(config)
    print("Embeddings ready")

    train_loader, val_loader, test_loader = create_dataloaders(embedding_path, config)
    print("Dataloaders created")

    model = EmbeddingClassifier(config.embedding_dim).to(config.device)
    print("Model initialized")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # For tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs} starting")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)

        # Save to history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, config.model_save_path)
            print("Best model saved.")

    # After training, plot the results
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("/home/sami/Documents/AI2/v2/results/3imgs_2/training_curves.png"))
    plt.show()

    # Final evaluation on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(config.model_save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, config.device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()


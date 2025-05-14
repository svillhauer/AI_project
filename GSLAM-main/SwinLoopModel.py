import torch
import torch.nn as nn
from transformers import SwinModel, AutoImageProcessor
from torchvision import transforms
from PIL import Image
import numpy as np

class SwinLoopModel(nn.Module):
    def __init__(self, model_path, model_name="microsoft/swin-tiny-patch4-window7-224", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Swin backbone
        self.swin = SwinModel.from_pretrained(model_name)
        for param in self.swin.parameters():
            param.requires_grad = False

        hidden_size = self.swin.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Load classifier weights
        classifier_state = torch.load(model_path, map_location=self.device)
        classifier_state = {k.replace("classifier.", ""): v for k, v in classifier_state.items()}
        self.classifier.load_state_dict(classifier_state)
        self.classifier.to(self.device)
        self.swin.to(self.device)

        # Preprocessing
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])

    def preprocess_batch(self, imgs):
        # imgs: NumPy array of shape (B, H, W, C)
        tensors = []
        for img in imgs:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            tensors.append(self.transform(img_pil))
        return torch.stack(tensors).to(self.device)

    def extract_features(self, imgs):
        # imgs: Tensor of shape (B, C, H, W)
        with torch.no_grad():
            return self.swin(imgs).last_hidden_state.mean(dim=1)  # (B, hidden_size)

    def forward(self, img1_batch, img2_batch):
        # img1_batch and img2_batch: NumPy arrays (B, H, W, C)
        x1 = self.preprocess_batch(img1_batch)
        x2 = self.preprocess_batch(img2_batch)
        f1 = self.extract_features(x1)
        f2 = self.extract_features(x2)
        combined = torch.cat([f1, f2], dim=1)
        with torch.no_grad():
            return self.classifier(combined).squeeze()  # (B,)

    def predict(self, img1_batch, img2_batch):
        # Return binary predictions: 0 or 1
        probs = self.forward(img1_batch, img2_batch)
        return (probs > 0.5).int().cpu().numpy()

import torch
import torch.nn as nn
from transformers import SwinModel, AutoImageProcessor
from torchvision import transforms
from PIL import Image
import numpy as np

class SwinLoopModel:
    def __init__(self, model_path, model_name="microsoft/swin-tiny-patch4-window7-224", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Swin model
        self.swin = SwinModel.from_pretrained(model_name)
        for param in self.swin.parameters():
            param.requires_grad = False

        hidden_size = self.swin.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # # Combine
        # self.model = nn.Module()
        # self.model.swin = self.swin
        # self.model.fc = self.classifier
        # self.model.to(self.device)

        # # Load weights
        # state_dict = torch.load(model_path, map_location=self.device)
        # self.model.fc.load_state_dict(state_dict)

        class SwinWrapper(nn.Module):
            def __init__(self, swin, classifier):
                super().__init__()
                self.swin = swin
                self.fc = classifier

            def forward(self, img1, img2):
                emb1 = self.swin(pixel_values=img1).pooler_output
                emb2 = self.swin(pixel_values=img2).pooler_output
                combined = torch.cat([emb1, emb2], dim=1)
                return self.fc(combined)

        # Wrap everything
        self.model = SwinWrapper(self.swin, self.classifier).to(self.device)

        # Load entire model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


        # Setup processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
        self.model.eval()

    # def _prepare(self, img):
    #     # Assumes image is a numpy array
    #     if img.ndim == 2:  # grayscale
    #         img = np.stack([img] * 3, axis=-1)
    #     img = Image.fromarray((img * 255).astype(np.uint8)) if img.max() <= 1.0 else Image.fromarray(img.astype(np.uint8))
    #     return self.transform(img).unsqueeze(0).to(self.device)
    
    def _prepare(self, img):
        # Convert PIL to numpy array
        if isinstance(img, Image.Image):
            img = np.array(img)

        if img.ndim == 2:  # grayscale
            img = np.stack([img] * 3, axis=-1)

        img = Image.fromarray((img * 255).astype(np.uint8)) if img.max() <= 1.0 else Image.fromarray(img.astype(np.uint8))
        return self.transform(img).unsqueeze(0).to(self.device)


    def predict(self, img_pair):
        img1, img2 = img_pair
        img1_tensor = self._prepare(img1)
        img2_tensor = self._prepare(img2)

        with torch.no_grad():
            emb1 = self.model.swin(pixel_values=img1_tensor).pooler_output
            emb2 = self.model.swin(pixel_values=img2_tensor).pooler_output
            combined = torch.cat([emb1, emb2], dim=1)
            output = self.model.fc(combined)

        return output.cpu().numpy()


# import torch
# import torch.nn as nn
# from transformers import AutoImageProcessor, SwinModel
# from torchvision import transforms
# from PIL import Image
# import numpy as np

# class SwinLoopModel(nn.Module):  # Inherit from nn.Module so we can use .to(device)
#     def __init__(self, model_name="microsoft/swin-tiny-patch4-window7-224"):
#         super(SwinLoopModel, self).__init__()  # Initialize base nn.Module
#         # Load a pretrained Swin Transformer from Hugging Face Transformers
#         self.swin = SwinModel.from_pretrained(model_name)

#         # Freeze Swin model weights (you can set to True if you want to finetune it)
#         for param in self.swin.parameters():
#             param.requires_grad = False

#         # Get the output feature size of the Swin model (e.g., 768 for swin-tiny)
#         hidden_size = self.swin.config.hidden_size

#         # Define a simple classifier that takes concatenated features from two images
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size * 2, 256),  # Input is two embeddings concatenated
#             nn.ReLU(),
#             nn.Linear(256, 1),               # Output a single value
#             nn.Sigmoid()                     # Apply sigmoid for binary classification
#         )

#     def forward(self, img1, img2):
#         """
#         Forward pass: Takes two images and returns a loop prediction score between 0 and 1.
#         """
#         # Extract features from both images using the pretrained Swin backbone
#         emb1 = self.swin(pixel_values=img1).pooler_output
#         emb2 = self.swin(pixel_values=img2).pooler_output

#         # Concatenate both embeddings along the feature dimension
#         combined = torch.cat([emb1, emb2], dim=1)

#         # Pass through classifier to get similarity prediction
#         return self.classifier(combined)


#         # Initialize the actual Siamese network model
#         self.model = SwinSiamese(model_name).to(self.device)
#         self.model.eval()  # Set model to evaluation mode (disables dropout, etc.)

#     def load(self, model_path):
#         """
#         Load the trained model weights from a file.
#         """
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()  # Always set to eval mode after loading

#     def preprocess(self, image):
#         """
#         Convert a NumPy image to a tensor and apply the same transform used during training.
#         """
#         # Convert NumPy array to PIL Image (make sure it's RGB)
#         if isinstance(image, np.ndarray):
#             # If image is normalized [0, 1], rescale it to [0, 255]
#             if image.max() <= 1.0:
#                 image = Image.fromarray((image * 255).astype(np.uint8))
#             else:
#                 image = Image.fromarray(image.astype(np.uint8))
#         image = image.convert("RGB")
#         return self.transform(image)

#     def predict(self, image_pair_list):
#         """
#         Predict whether a pair of images is a loop closure.
#         Input: list of two NumPy images [img1, img2]
#         Output: probability (between 0 and 1)
#         """
#         assert len(image_pair_list) == 2, "Expected [img1, img2]"

#         img1, img2 = image_pair_list
#         img1 = self.preprocess(img1).unsqueeze(0).to(self.device)  # Add batch dimension
#         img2 = self.preprocess(img2).unsqueeze(0).to(self.device)

#         with torch.no_grad():  # Disable gradient tracking for inference
#             prob = self.model(img1, img2)  # Output is a probability
#         return prob.cpu().numpy()  # Convert to NumPy for compatibility with existing code

# class SwinSiamese(nn.Module):
#     """
#     A simple Siamese network using Swin Transformer as the backbone.
#     Takes two images, embeds them using Swin, then classifies based on their similarity.
#     """
#     def __init__(self, model_name):
#         super(SwinSiamese, self).__init__()

#         # Load pretrained Swin model
#         self.backbone = SwinModel.from_pretrained(model_name)

#         # Freeze the Swin backbone to avoid fine-tuning (optional)
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#         # Swin output size (depends on model config, usually 768)
#         hidden_size = self.backbone.config.hidden_size

#         # Classification head: combines embeddings from both images
#         self.head = nn.Sequential(
#             nn.Linear(hidden_size * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()  # Outputs a probability between 0 and 1
#         )

#     def forward(self, img1, img2):
#         """
#         Forward pass for the Siamese network:
#         - Extract features for both images
#         - Concatenate features
#         - Run through classifier head
#         """
#         emb1 = self.backbone(pixel_values=img1).pooler_output
#         emb2 = self.backbone(pixel_values=img2).pooler_output
#         combined = torch.cat([emb1, emb2], dim=1)
#         return self.head(combined)

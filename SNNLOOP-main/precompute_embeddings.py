import os
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, SwinModel
from torchvision import transforms  # <-- MISSING IMPORT
import numpy as np

# Configuration (match your training script)
model_name = "microsoft/swin-tiny-patch4-window7-224"
image_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/embeddings"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
processor = AutoImageProcessor.from_pretrained(model_name)
model = SwinModel.from_pretrained(model_name).to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process all images
img_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
for img_file in tqdm(img_files, desc="Precomputing embeddings"):
    img_path = os.path.join(image_dir, img_file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(img_tensor).pooler_output.squeeze().cpu().numpy()
    
    # Save as .npy file
    np.save(os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.npy"), embedding)

print(f"Embeddings saved to {output_dir}")

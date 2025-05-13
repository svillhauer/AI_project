# save_swin_embeddings.py
from transformers import AutoImageProcessor, SwinModel
from torchvision import transforms
from PIL import Image
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/swin-tiny-patch4-window7-224"
image_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
output_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_EMBEDDINGS/embedding_output.npz"

processor = AutoImageProcessor.from_pretrained(model_name)
swin = SwinModel.from_pretrained(model_name).to(device)
swin.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

df = pd.read_csv(csv_path)
unique_images = pd.unique(df[['first', 'second']].values.ravel())

embeddings = {}

for img_name in tqdm(unique_images):
    img_path = os.path.join(image_dir, img_name)
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = swin(pixel_values=img).pooler_output.cpu().numpy()
    embeddings[img_name] = emb[0]

# Save all as a NumPy zip
np.savez(output_path, **embeddings)

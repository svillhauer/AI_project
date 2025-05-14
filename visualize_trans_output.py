import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformer_images_v2 import EmbeddingClassifier, EmbeddingDataset # Only the class is imported
from torch.utils.data import DataLoader

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 768
data_path = "/home/sami/Documents/AI2/v2"
model_path = os.path.join(data_path, "models/output_3imgs.pth")
embedding_path = os.path.join(data_path, "SAMPLE_RANDOM_3images/precomputed_embeddings.pt")
test_csv = os.path.join(data_path, "SAMPLE_RANDOM_3images/test.csv")
image_dir = os.path.join(data_path, "SAMPLE_RANDOM_3images/IMAGES")
output_dir = os.path.join(data_path, "results/3imgs")
os.makedirs(output_dir, exist_ok=True)

# --- Load model ---
model = EmbeddingClassifier(embedding_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Load embeddings and test data ---
embeddings = torch.load(embedding_path, map_location=device)
test_df = pd.read_csv(test_csv)

def show_embedding_predictions(
    model,
    test_df,
    embeddings,
    image_dir,
    threshold=0.3,
    max_loop=10,
    max_no_loop=10,
    output_dir="results",
    csv_name="predictions.csv"
):
    shown_loop = 0
    shown_no_loop = 0
    predictions_list = []

    for idx, row in test_df.iterrows():
        #if shown_loop >= max_loop and shown_no_loop >= max_no_loop:
         #   break

        img1_path = os.path.join(image_dir, row['img1'])
        img2_path = os.path.join(image_dir, row['img2'])
        emb1 = embeddings[row['img1']].unsqueeze(0).to(device)
        emb2 = embeddings[row['img2']].unsqueeze(0).to(device)
        label = row['label']

        with torch.no_grad():
            output = model(emb1, emb2).squeeze()
            prob = torch.sigmoid(output).item()
            pred = int(prob > threshold)

        pred_label = "LOOP" if pred == 1 else "NO LOOP"
        true_label = "LOOP" if label == 1 else "NO LOOP"

        # Only collect up to max_loop and max_no_loop pairs
        if pred_label == "LOOP" and shown_loop < max_loop:
            shown_loop += 1
        elif pred_label == "NO LOOP" and shown_no_loop < max_no_loop:
            shown_no_loop += 1
        else:
            continue  # Skip this pair if we've reached the limit for this class

        # Load images for display
        img1_disp = Image.open(img1_path).convert("RGB")
        img2_disp = Image.open(img2_path).convert("RGB")

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(img1_disp)
        ax[1].imshow(img2_disp)
        for a in ax:
            a.axis('off')

        plt.suptitle(f"Predicted: {pred_label} ({prob:.2f}) | True: {true_label}", fontsize=10)
        plt.tight_layout()

        filename = f"{output_dir}/pair_{shown_loop + shown_no_loop - 1:04d}_{pred_label.lower()}.png"
        plt.savefig(filename)
        plt.close()

        predictions_list.append({
            "pair_id": shown_loop + shown_no_loop - 1,
            "img1": row['img1'],
            "img2": row['img2'],
            "predicted_label": pred_label,
            "confidence": float(prob),
            "true_label": true_label,
            "filename": os.path.basename(filename)
        })

    # Export CSV
    df = pd.DataFrame(predictions_list)
    df.to_csv(os.path.join(output_dir, csv_name), index=False)
    print(f"[INFO] Saved {len(predictions_list)} predictions and CSV at: {output_dir}")

## Positives table
TP = FP = TN = FN = 0
threshold = 0.6

# Use existing variables from your config ▼
test_csv_path = test_csv  # Defined earlier as os.path.join(data_path, "SAMPLE_RANDOM_3images/test.csv")
img_dir = image_dir  # Already defined in your config

# Use EmbeddingDataset with precomputed embeddings
test_dataset2 = EmbeddingDataset(test_csv_path, embeddings)
test_loader = DataLoader(test_dataset2, batch_size=10, shuffle=False)  # Add batch_size to config if needed

model.eval()
with torch.no_grad():
    for img1, img2, label in tqdm(test_loader, desc="Calculating results"):  # test_loader / val_loader
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        output = model(img1, img2).squeeze()
        preds = (output > threshold).float()

        TP += ((preds == 1) & (label == 1)).sum().item()
        TN += ((preds == 0) & (label == 0)).sum().item()
        FP += ((preds == 1) & (label == 0)).sum().item()
        FN += ((preds == 0) & (label == 1)).sum().item()


accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
fallout = FP / (FP + TN) if FP + TN > 0 else 0

print("Results are calculated")
print(f"True Positives:  {TP}")
print(f"True Negatives:  {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Accuracy:        {accuracy}")
print(f"Precision:       {precision}")
print(f"Recall:          {recall}")
print(f"Fallout:         {fallout}")


if __name__ == "__main__":
    show_embedding_predictions(
    model,
    test_df,
    embeddings,
    image_dir,
    threshold=threshold,
    max_loop=20,
    max_no_loop=20,
    output_dir=output_dir,
    csv_name="predictions.csv"
)


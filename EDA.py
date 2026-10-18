import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# --- NEW: PyTorch Imports ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
HOME = os.path.expanduser("~")
DEFAULT_DATASET_DIR = os.path.join(HOME, "Downloads", "archive(1)", "resized_train_cropped")
CSV_LABELS_PATH = os.path.join(HOME, "Downloads", "archive(1)", "trainLabels_cropped_2Lines.csv")

EMBED_LIMIT = 40000  
PIXEL_STATS_SAMPLE_SIZE = 100
OUTPUT_DIR = os.path.join(HOME, "EDA_Results_DeepLearning") # <-- New folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" Using DEEP LEARNING features (ResNet50 on PyTorch)")

# --- MODIFIED: ImageNet Normalization ---
# We MUST use the normalization stats the model was pre-trained on.
DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]

# --- NEW: PyTorch Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {DEVICE}")

# ------------------------------------------------------------
# 1. Load Image Paths and Labels
# ------------------------------------------------------------
def load_image_paths_and_labels(dataset_dir, csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'image' in df.columns and 'label' in df.columns:
            df.rename(columns={'image': 'id_code', 'label': 'diagnosis'}, inplace=True)
        if 'id_code' in df.columns:
            df['path'] = df['id_code'].apply(lambda x: os.path.join(dataset_dir, f"{x}.jpeg"))
            df['exists'] = df['path'].apply(os.path.exists)
            df = df[df['exists']].reset_index(drop=True)
            print(f" Loaded {len(df)} labeled images from CSV.")
            df['diagnosis'] = df['diagnosis'].astype(int)
            return df
    paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return pd.DataFrame({'path': paths, 'diagnosis': 0})

# ------------------------------------------------------------
# 2. Corruption Detection
# ------------------------------------------------------------
def check_corruption(path):
    try:
        img = Image.open(path)
        img.verify()
        return False
    except (UnidentifiedImageError, OSError):
        return True

def detect_corrupted(df):
    print("🔎 Starting image corruption check (using 16 threads)...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        corrupted = list(tqdm(ex.map(check_corruption, df['path']),
                              total=len(df),
                              desc="Corrupt Check",
                              ncols=100))
    df['is_corrupted'] = corrupted
    corrupted_df = df[df['is_corrupted']].copy()
    print(f"🧩 Found {len(corrupted_df)} corrupted images.")
    return df[~df['is_corrupted']], corrupted_df

# ------------------------------------------------------------
# 3. Deep Learning Feature Extraction (ResNet50)
# ------------------------------------------------------------

class ImageDataset(Dataset):
    """Custom PyTorch Dataset for loading images from paths."""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['diagnosis']
        
        try:
            # Open image and convert to RGB
            img = Image.open(img_path).convert("RGB")
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
                
            return img, label, img_path
        except Exception as e:
            # Return a dummy tensor if image is corrupt
            print(f"Warning: Could not load {img_path}. Error: {e}")
            return torch.zeros((3, 224, 224)), -1, img_path

def extract_dl_features_batch(df, max_samples=None):
    """
    Extracts deep learning features using a pre-trained ResNet50
    in batches for maximum GPU efficiency.
    """
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"\n🔹 Extracting Deep Learning features from {len(df)} images...")
    print(" Model: ResNet50 (pre-trained on ImageNet)")
    print(" This will be fast on a GPU.\n")

    # 1. Define the image transformations
    # 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
    ])
    
    # 2. Create Dataset and DataLoader
    dataset = ImageDataset(df, transform=transform)
    # Use num_workers > 0 to speed up data loading
    # Use batch_size 32 or 64 to leverage the GPU
    data_loader = DataLoader(dataset, 
                             batch_size=32, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True)
    
    # 3. Load pre-trained ResNet50 model
    # We use the latest, recommended weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # 4. Modify model to get features (embeddings)
    # We replace the final classification layer (model.fc)
    # with an Identity layer. The model's output will
    # now be the 2048-dim features from the layer before it.
    model.fc = nn.Identity()
    
    model = model.to(DEVICE).eval() # Set model to evaluation mode

    # 5. Feature extraction loop
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad(): # Disable gradient calculations
        for batch_imgs, batch_labels, batch_paths in tqdm(data_loader, desc="Feature Extraction", ncols=100):
            # Send images to the GPU
            batch_imgs = batch_imgs.to(DEVICE)
            
            # Get features from the model
            features = model(batch_imgs)
            
            # Store features, labels, and paths
            # Move features back to CPU for numpy conversion
            all_features.append(features.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_paths.extend(batch_paths)

    # Combine all batches
    features_np = np.concatenate(all_features)
    labels_np = np.concatenate(all_labels)
    paths_np = np.array(all_paths)
    
    # Filter out any images that failed to load
    valid_mask = labels_np != -1
    features_np = features_np[valid_mask]
    labels_np = labels_np[valid_mask]
    paths_np = paths_np[valid_mask]
    
    print(f" Extracted {features_np.shape[1]}-dimensional features from {len(features_np)} valid images")
    
    return features_np, labels_np, paths_np


# ------------------------------------------------------------
# 4. Class Distribution (Unchanged)
# ------------------------------------------------------------
def analyze_class_distribution(df):
    counts = df['diagnosis'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    bars = plt.bar(counts.index, counts.values, color=colors[:len(counts)])
    
    plt.title("Class Distribution - Diabetic Retinopathy", fontsize=14, fontweight='bold')
    plt.xlabel("DR Severity Level", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    total = counts.sum()
    balance_ratio = counts.max() / counts.min() if counts.min() > 0 else np.nan
    summary = pd.DataFrame({
        'class_label': counts.index,
        'severity': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'][:len(counts)],
        'count': counts.values,
        'percentage': (counts.values / total * 100).round(2)
    })
    print("\n" + "="*60)
    print(" CLASS DISTRIBUTION:")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\n Balance Ratio (max/min): {balance_ratio:.2f}")
    if balance_ratio > 10:
        print(f" HIGH CLASS IMBALANCE! Consider weighted loss or oversampling.")
    print("="*60)
    return summary

# ------------------------------------------------------------
# 5. Pixel Statistics (Unchanged, but now just for reference)
# ------------------------------------------------------------
def pixel_statistics(df):
    """
    Calculates the dataset's native pixel stats.
    NOTE: For pre-trained models, we use ImageNet stats,
    but this is still useful info to have.
    """
    sampled = df.sample(min(PIXEL_STATS_SAMPLE_SIZE, len(df)), random_state=42)
    
    total_sum = np.zeros(3, dtype=np.float64)
    total_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    all_pixels_for_plot = []

    print(f"\n Calculating *dataset's native* pixel statistics (sampling {len(sampled)} images)...")
    
    for p in tqdm(sampled['path'], desc="Pixel Stats", ncols=100):
        try:
            img = np.array(Image.open(p).convert("RGB")).astype(np.float64)
            current_pixels = img.reshape(-1, 3)
            total_sum += current_pixels.sum(axis=0)
            total_sum_sq += np.sum(current_pixels**2, axis=0)
            total_pixels += current_pixels.shape[0]
            all_pixels_for_plot.append(current_pixels)
        except Exception:
            continue

    if total_pixels == 0:
        print(" No images loaded for pixel statistics.")
        return [0, 0, 0], [1, 1, 1]

    # Calculate dataset-specific stats
    means_255 = total_sum / total_pixels
    variance_255 = (total_sum_sq / total_pixels) - (means_255 ** 2)
    stds_255 = np.sqrt(np.maximum(0, variance_255))
    dataset_mean_01 = (means_255 / 255.0).round(4).tolist()
    dataset_std_01 = (stds_255 / 255.0).round(4).tolist()

    pixels_for_plot = np.concatenate(all_pixels_for_plot, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['red', 'green', 'blue']
    for i, c in enumerate(colors):
        ax1.hist(pixels_for_plot[:, i], bins=50, color=c, alpha=0.5, label=c.upper(), density=True)
    ax1.legend()
    ax1.set_title("Pixel Intensity Distribution")
    ax1.set_xlabel("Pixel Value (0-255)")
    ax1.set_ylabel("Density")
    ax1.grid(alpha=0.3)
    
    ax2.boxplot([pixels_for_plot[:, 0], pixels_for_plot[:, 1], pixels_for_plot[:, 2]],
                labels=['Red', 'Green', 'Blue'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.5))
    ax2.set_title("Pixel Value Distribution by Channel")
    ax2.set_ylabel("Pixel Value (0-255)")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pixel_distribution.png"), dpi=300)
    plt.close()

    del all_pixels_for_plot, pixels_for_plot

    print(f"\n *Dataset-Native* Pixel Statistics:")
    print(f"   Mean (0-1): {dataset_mean_01}")
    print(f"   Std (0-1):  {dataset_std_01}")
    print(f"\n *Model Required* Pixel Statistics (ImageNet):")
    print(f"   Mean (0-1): {DATASET_MEAN}")
    print(f"   Std (0-1):  {DATASET_STD}")
    
    # We return the ImageNet stats, as that's what the model uses
    return DATASET_MEAN, DATASET_STD

# ------------------------------------------------------------
# 6. Outlier Detection (Unchanged)
# ------------------------------------------------------------
def detect_outliers(embeddings, paths, labels):
    if embeddings.size == 0:
        print(" No embeddings to analyze for outliers")
        return pd.DataFrame({'path': [], 'label': [], 'distance': []})

    print("\n🔍 Computing distances for outlier detection...")
    print(f"   Using {embeddings.shape[1]} features per image")
    
    # 
    dists = cosine_distances(embeddings)
    mean_dists = dists.mean(axis=1)
    
    median_dist = np.median(mean_dists)
    mad = np.median(np.abs(mean_dists - median_dist)) # Median Absolute Deviation
    threshold = median_dist + 3 * mad # 3 MADs is a robust outlier threshold
    outlier_mask = mean_dists > threshold
    outliers = paths[outlier_mask]
    outlier_labels = labels[outlier_mask]

    print(f"✅ Detected {len(outliers)} outliers out of {len(paths)} images ({len(outliers)/len(paths)*100:.2f}%)")
    print(f"📏 Threshold (Median + 3×MAD): {threshold:.4f}")
    
    if len(outliers) > 0:
        outlier_class_dist = pd.Series(outlier_labels).value_counts().sort_index()
        print(f"\n🔬 Outliers by class:")
        for cls, count in outlier_class_dist.items():
            print(f"   Class {cls}: {count} images")

    return pd.DataFrame({
        'path': outliers,
        'label': outlier_labels,
        'distance': mean_dists[outlier_mask]
    }).sort_values('distance', ascending=False)

# ------------------------------------------------------------
# 7. Feature Visualization (PCA) (Unchanged)
# ------------------------------------------------------------
def visualize_feature_space_pca(embeddings_scaled, labels):
    """Create 2D visualization of feature space using PCA"""
    print("\n Creating feature space visualization with PCA...")
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(embeddings_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    for i in range(5):
        mask = labels == i
        if mask.any():
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=colors[i], label=class_names[i], alpha=0.6, s=20)
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Feature Space Visualization (PCA)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_space_pca.png"), dpi=300)
    plt.close()
    
    print(f" Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# ------------------------------------------------------------
# 7.5. Feature Visualization (UMAP) (Unchanged)
# ------------------------------------------------------------
def visualize_feature_space_umap(embeddings_scaled, labels):
    """Create 2D visualization of feature space using UMAP"""
    print("\n Creating feature space visualization with UMAP...")
    print(" (This may take a moment...)")
    
    # 
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    features_2d = reducer.fit_transform(embeddings_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    for i in range(5):
        mask = labels == i
        if mask.any():
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=colors[i], label=class_names[i], alpha=0.6, s=20)
    
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title("Feature Space Visualization (UMAP)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_space_umap.png"), dpi=300)
    plt.close()
    
    print(" UMAP visualization saved.")

# ------------------------------------------------------------
# 8. Save Summary Report (Text Modified)
# ------------------------------------------------------------
def save_summary_report(df, valid_df, corrupted_df, class_summary, outlier_df, n_features):
    report_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DIABETIC RETINOPATHY - EDA SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature Extraction: Deep Learning ({n_features} features)\n") # <-- MODIFIED
        f.write(f"   - Model: ResNet50 (pre-trained on ImageNet)\n")            # <-- MODIFIED
        f.write(f"   - Features: 2048-dim vector from final avgpool layer\n\n") # <-- MODIFIED
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"   Total images: {len(df):,}\n")
        f.write(f"   Valid images: {len(valid_df):,}\n")
        f.write(f"   Corrupted: {len(corrupted_df)}\n\n")
        
        f.write("CLASS DISTRIBUTION:\n")
        f.write(class_summary.to_string(index=False) + "\n\n")
        
        f.write(f"NORMALIZATION VALUES (ImageNet):\n") # <-- MODIFIED
        f.write(f"   Mean (RGB): {DATASET_MEAN}\n")
        f.write(f"   Std (RGB):  {DATASET_STD}\n\n")
        
        f.write(f"OUTLIER ANALYSIS (Deep Features):\n") # <-- MODIFIED
        f.write(f"   Total outliers: {len(outlier_df)}\n")
        if len(outlier_df) > 0:
            f.write(f"\n   Top 10 outliers (most dissimilar to dataset):\n")
            for idx, row in outlier_df.head(10).iterrows():
                filename = os.path.basename(row['path'])
                f.write(f"   {filename[:40]:40s} class={row['label']} dist={row['distance']:.4f}\n")
        
        f.write(f"\n" + "="*60 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*60 + "\n")
        
        balance_ratio = class_summary['count'].max() / class_summary['count'].min()
        if balance_ratio > 10:
            f.write(" SEVERE CLASS IMBALANCE!\n")
            f.write(f"   Ratio: {balance_ratio:.1f}:1\n")
            f.write("   → Use class_weight='balanced' or oversample minority classes\n\n")
        
        if len(outlier_df) > 0:
            f.write(f" {len(outlier_df)} OUTLIERS DETECTED.\n")
            f.write("   → These are semantically different (e.g., black images, artifacts).\n")
            f.write("   → Manual review of 'outliers.csv' is highly recommended.\n\n")
        
        f.write(" Dataset ready for modeling\n")
        f.write(" Use ImageNet normalization values in your training pipeline\n")
    
    print(f"\n📄 Comprehensive report saved to: {report_path}")

# ------------------------------------------------------------
# 9. Main Execution (Modified)
# ------------------------------------------------------------
def run_eda():
    start_time = time.time()
    
    print("\n" + "="*60)
    print(" DIABETIC RETINOPATHY - EDA PIPELINE")
    print(" Using Deep Learning Features (ResNet50 + PyTorch)") # <-- MODIFIED
    print("="*60 + "\n")
    
    df = load_image_paths_and_labels(DEFAULT_DATASET_DIR, CSV_LABELS_PATH)
    valid_df, corrupted_df = detect_corrupted(df)
    print(f"Valid: {len(valid_df):,} | Corrupted: {len(corrupted_df)}")
    
    class_summary = analyze_class_distribution(valid_df)
    
    # This now just prints stats, it doesn't set the global ones
    pixel_statistics(valid_df) 
    
    # --- MODIFIED SECTION ---
    # Sample the *valid* dataframe for feature extraction
    sampled = valid_df.sample(min(EMBED_LIMIT, len(valid_df)), random_state=42).reset_index(drop=True)
    
    # Use the new Deep Learning batch feature extractor
    embeddings, labels, paths = extract_dl_features_batch(sampled)
    
    # Scale features (good practice for PCA/UMAP)
    print("\n Scaling features before visualization...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Visualize feature space (PCA)
    visualize_feature_space_pca(embeddings_scaled, labels)
    
    # Visualize feature space (UMAP)
    visualize_feature_space_umap(embeddings_scaled, labels)
    # --- END MODIFIED SECTION ---
    
    # Detect outliers (use *unscaled* embeddings for cosine distance)
    outlier_df = detect_outliers(embeddings, paths, labels)
    outlier_df.to_csv(os.path.join(OUTPUT_DIR, "outliers.csv"), index=False)
    
    # Save report
    save_summary_report(df, valid_df, corrupted_df, class_summary, outlier_df, embeddings.shape[1])
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*6)
    print(" EDA COMPLETE!")
    print("="*60)
    print(f"⏱ Total time: {elapsed:.1f}s")
    print(f" Results: {OUTPUT_DIR}")
    print("\n Generated files:")
    print("   - class_distribution.png     (class balance)")
    print("   - pixel_distribution.png     (dataset's native RGB stats)")
    print("   - feature_space_pca.png      (2D visualization - Linear)")
    print("   - feature_space_umap.png     (2D visualization - Non-Linear)")
    print("   - outliers.csv               (semantically different images)")
    print("   - eda_summary.txt            (full report)")
    print("="*60 + "\n")

if __name__ == "__main__":
    # This is required for PyTorch multiprocessing on Windows
    torch.multiprocessing.freeze_support() 
    run_eda()

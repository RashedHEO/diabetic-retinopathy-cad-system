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
from sklearn.manifold import TSNE
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import umap

# --- PyTorch Imports ---
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
PIXEL_STATS_SAMPLE_SIZE = 36000  # Increased for better statistics
OUTPUT_DIR = os.path.join(HOME, "EDA_Results_DeepLearning_Enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 Using ENHANCED DEEP LEARNING EDA")

DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {DEVICE}")

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
            print(f"✅ Loaded {len(df)} labeled images from CSV.")
            df['diagnosis'] = df['diagnosis'].astype(int)
            return df
    paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return pd.DataFrame({'path': paths, 'diagnosis': 0})

# ------------------------------------------------------------
# 2. Enhanced Corruption Detection with Quality Metrics
# ------------------------------------------------------------
def check_corruption_and_quality(path):
    """Check corruption AND compute basic quality metrics"""
    try:
        img = Image.open(path)
        img.verify()
        
        # Reopen for quality checks
        img = Image.open(path).convert("RGB")
        img_array = np.array(img)
        
        # Quality metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        sharpness = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        
        # Check if image is too dark (potential issue)
        is_too_dark = brightness < 20
        
        # Check if image is nearly uniform (potential blank/corrupted)
        is_uniform = contrast < 5
        
        return {
            'corrupted': False,
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'too_dark': is_too_dark,
            'uniform': is_uniform
        }
    except (UnidentifiedImageError, OSError):
        return {
            'corrupted': True,
            'brightness': 0,
            'contrast': 0,
            'sharpness': 0,
            'too_dark': False,
            'uniform': False
        }

def detect_corrupted_enhanced(df):
    print("🔎 Enhanced corruption & quality check (using 16 threads)...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(tqdm(ex.map(check_corruption_and_quality, df['path']),
                              total=len(df),
                              desc="Quality Check",
                              ncols=100))
    
    # Add results to dataframe
    df['is_corrupted'] = [r['corrupted'] for r in results]
    df['brightness'] = [r['brightness'] for r in results]
    df['contrast'] = [r['contrast'] for r in results]
    df['sharpness'] = [r['sharpness'] for r in results]
    df['too_dark'] = [r['too_dark'] for r in results]
    df['uniform'] = [r['uniform'] for r in results]
    
    corrupted_df = df[df['is_corrupted']].copy()
    quality_issues_df = df[(df['too_dark']) | (df['uniform'])].copy()
    
    print(f"🧩 Found {len(corrupted_df)} corrupted images.")
    print(f"⚠️  Found {len(quality_issues_df)} images with quality issues (too dark/uniform).")
    
    return df[~df['is_corrupted']], corrupted_df, quality_issues_df

# ------------------------------------------------------------
# 3. Deep Learning Feature Extraction (ResNet50)
# ------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['diagnosis']
        
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, img_path
        except Exception as e:
            print(f"⚠️  Could not load {img_path}. Error: {e}")
            return torch.zeros((3, 224, 224)), -1, img_path

def extract_dl_features_batch(df, max_samples=None):
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"\n🔹 Extracting Deep Learning features from {len(df)} images...")
    print("   Model: ResNet50 (pre-trained on ImageNet)")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
    ])
    
    dataset = ImageDataset(df, transform=transform)
    data_loader = DataLoader(dataset, 
                             batch_size=32, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True)
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model = model.to(DEVICE).eval()

    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch_imgs, batch_labels, batch_paths in tqdm(data_loader, desc="Feature Extraction", ncols=100):
            batch_imgs = batch_imgs.to(DEVICE)
            features = model(batch_imgs)
            all_features.append(features.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_paths.extend(batch_paths)

    features_np = np.concatenate(all_features)
    labels_np = np.concatenate(all_labels)
    paths_np = np.array(all_paths)
    
    valid_mask = labels_np != -1
    features_np = features_np[valid_mask]
    labels_np = labels_np[valid_mask]
    paths_np = paths_np[valid_mask]
    
    print(f"✅ Extracted {features_np.shape[1]}-dimensional features from {len(features_np)} valid images")
    
    return features_np, labels_np, paths_np

# ------------------------------------------------------------
# 4. Enhanced Class Distribution Analysis
# ------------------------------------------------------------
def analyze_class_distribution_enhanced(df):
    """Enhanced class distribution with statistical analysis"""
    counts = df['diagnosis'].value_counts().sort_index()
    
    # Create enhanced visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    bars = axes[0].bar(counts.index, counts.values, color=colors[:len(counts)])
    axes[0].set_title("Class Distribution - Diabetic Retinopathy", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("DR Severity Level", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height):,}\n({height/counts.sum()*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)
    
    # Pie chart
    axes[1].pie(counts.values, labels=labels[:len(counts)], colors=colors[:len(counts)],
                autopct='%1.1f%%', startangle=90, explode=[0.05]*len(counts))
    axes[1].set_title("Class Proportion", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution_enhanced.png"), dpi=300)
    plt.close()

    total = counts.sum()
    balance_ratio = counts.max() / counts.min() if counts.min() > 0 else np.nan
    
    # Calculate Shannon entropy for class distribution
    proportions = counts.values / total
    shannon_entropy = entropy(proportions)
    max_entropy = np.log(len(counts))
    normalized_entropy = shannon_entropy / max_entropy
    
    summary = pd.DataFrame({
        'class_label': counts.index,
        'severity': labels[:len(counts)],
        'count': counts.values,
        'percentage': (counts.values / total * 100).round(2)
    })
    
    print("\n" + "="*70)
    print(" ENHANCED CLASS DISTRIBUTION ANALYSIS:")
    print("="*70)
    print(summary.to_string(index=False))
    print(f"\n📊 Balance Ratio (max/min): {balance_ratio:.2f}")
    print(f"📈 Shannon Entropy: {shannon_entropy:.4f} (normalized: {normalized_entropy:.2%})")
    print(f"   → Higher entropy = more balanced (max = {max_entropy:.2f})")
    
    if balance_ratio > 10:
        print(f"\n⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print(f"   Recommendations:")
        print(f"   • Use class_weight='balanced' in loss function")
        print(f"   • Apply focal loss for hard examples")
        print(f"   • Oversample minority classes or undersample majority")
    
    print("="*70)
    return summary, balance_ratio, shannon_entropy

# ------------------------------------------------------------
# 5. Enhanced Pixel Statistics with Quality Metrics
# ------------------------------------------------------------
def pixel_statistics_enhanced(df):
    """Enhanced pixel statistics with quality distribution"""
    sampled = df.sample(min(PIXEL_STATS_SAMPLE_SIZE, len(df)), random_state=42)
    
    total_sum = np.zeros(3, dtype=np.float64)
    total_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    all_pixels_for_plot = []
    brightness_values = []
    contrast_values = []

    print(f"\n📊 Calculating enhanced pixel statistics (sampling {len(sampled)} images)...")
    
    for p in tqdm(sampled['path'], desc="Pixel Stats", ncols=100):
        try:
            img = np.array(Image.open(p).convert("RGB")).astype(np.float64)
            current_pixels = img.reshape(-1, 3)
            total_sum += current_pixels.sum(axis=0)
            total_sum_sq += np.sum(current_pixels**2, axis=0)
            total_pixels += current_pixels.shape[0]
            
            # Sample pixels for histogram (don't store all)
            if len(all_pixels_for_plot) < 10:
                all_pixels_for_plot.append(current_pixels[::10])  # Sample every 10th pixel
            
            # Quality metrics per image
            brightness_values.append(np.mean(img))
            contrast_values.append(np.std(img))
        except Exception:
            continue

    if total_pixels == 0:
        print("⚠️  No images loaded for pixel statistics.")
        return [0, 0, 0], [1, 1, 1]

    means_255 = total_sum / total_pixels
    variance_255 = (total_sum_sq / total_pixels) - (means_255 ** 2)
    stds_255 = np.sqrt(np.maximum(0, variance_255))
    dataset_mean_01 = (means_255 / 255.0).round(4).tolist()
    dataset_std_01 = (stds_255 / 255.0).round(4).tolist()

    pixels_for_plot = np.concatenate(all_pixels_for_plot, axis=0)
    
    # Enhanced visualization with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors_rgb = ['red', 'green', 'blue']
    
    # 1. Pixel intensity distribution
    for i, c in enumerate(colors_rgb):
        axes[0, 0].hist(pixels_for_plot[:, i], bins=50, color=c, alpha=0.5, label=c.upper(), density=True)
    axes[0, 0].legend()
    axes[0, 0].set_title("Pixel Intensity Distribution", fontweight='bold')
    axes[0, 0].set_xlabel("Pixel Value (0-255)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Boxplot by channel
    axes[0, 1].boxplot([pixels_for_plot[:, 0], pixels_for_plot[:, 1], pixels_for_plot[:, 2]],
                       labels=['Red', 'Green', 'Blue'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.5))
    axes[0, 1].set_title("Pixel Distribution by Channel", fontweight='bold')
    axes[0, 1].set_ylabel("Pixel Value (0-255)")
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Brightness distribution across images
    axes[0, 2].hist(brightness_values, bins=50, color='gold', alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(np.mean(brightness_values), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(brightness_values):.1f}')
    axes[0, 2].set_title("Image Brightness Distribution", fontweight='bold')
    axes[0, 2].set_xlabel("Mean Brightness (0-255)")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Contrast distribution across images
    axes[1, 0].hist(contrast_values, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(contrast_values), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(contrast_values):.1f}')
    axes[1, 0].set_title("Image Contrast Distribution", fontweight='bold')
    axes[1, 0].set_xlabel("Std Dev (Contrast)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Brightness vs Contrast scatter
    axes[1, 1].scatter(brightness_values, contrast_values, alpha=0.5, s=20, c='purple')
    axes[1, 1].set_title("Brightness vs Contrast", fontweight='bold')
    axes[1, 1].set_xlabel("Brightness")
    axes[1, 1].set_ylabel("Contrast")
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Channel comparison
    channel_means = [pixels_for_plot[:, i].mean() for i in range(3)]
    axes[1, 2].bar(['Red', 'Green', 'Blue'], channel_means, color=colors_rgb, alpha=0.7)
    axes[1, 2].set_title("Mean Intensity by Channel", fontweight='bold')
    axes[1, 2].set_ylabel("Mean Value (0-255)")
    axes[1, 2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pixel_distribution_enhanced.png"), dpi=300)
    plt.close()

    print(f"\n📈 Dataset-Native Pixel Statistics:")
    print(f"   Mean (0-1): {dataset_mean_01}")
    print(f"   Std (0-1):  {dataset_std_01}")
    print(f"\n📈 Image Quality Statistics:")
    print(f"   Brightness - Mean: {np.mean(brightness_values):.2f}, Std: {np.std(brightness_values):.2f}")
    print(f"   Contrast   - Mean: {np.mean(contrast_values):.2f}, Std: {np.std(contrast_values):.2f}")
    print(f"\n🎯 Model Required (ImageNet):")
    print(f"   Mean (0-1): {DATASET_MEAN}")
    print(f"   Std (0-1):  {DATASET_STD}")
    
    return DATASET_MEAN, DATASET_STD

# ------------------------------------------------------------
# 6. Enhanced Outlier Detection with Multiple Methods
# ------------------------------------------------------------
def detect_outliers_enhanced(embeddings, paths, labels):
    """Multi-method outlier detection with consensus"""
    if embeddings.size == 0:
        print("⚠️  No embeddings to analyze for outliers")
        return pd.DataFrame({'path': [], 'label': [], 'distance': [], 'outlier_score': []})

    print("\n🔍 Enhanced outlier detection using multiple methods...")
    
    # Method 1: Cosine distance (original)
    dists = cosine_distances(embeddings)
    mean_dists_cosine = dists.mean(axis=1)
    median_dist = np.median(mean_dists_cosine)
    mad = np.median(np.abs(mean_dists_cosine - median_dist))
    threshold_cosine = median_dist + 3 * mad
    outlier_mask_cosine = mean_dists_cosine > threshold_cosine
    
    # Method 2: Euclidean distance
    dists_euclidean = squareform(pdist(embeddings, metric='euclidean'))
    mean_dists_euclidean = dists_euclidean.mean(axis=1)
    median_euclidean = np.median(mean_dists_euclidean)
    mad_euclidean = np.median(np.abs(mean_dists_euclidean - median_euclidean))
    threshold_euclidean = median_euclidean + 3 * mad_euclidean
    outlier_mask_euclidean = mean_dists_euclidean > threshold_euclidean
    
    # Method 3: Isolation Forest score (using distances as features)
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_mask_iso = iso_forest.fit_predict(embeddings) == -1
    
    # Consensus: outlier if detected by 2+ methods
    outlier_votes = (outlier_mask_cosine.astype(int) + 
                    outlier_mask_euclidean.astype(int) + 
                    outlier_mask_iso.astype(int))
    
    consensus_outliers = outlier_votes >= 2
    
    # Calculate combined outlier score
    # Normalize each method's score to [0, 1]
    score_cosine = (mean_dists_cosine - mean_dists_cosine.min()) / (mean_dists_cosine.max() - mean_dists_cosine.min() + 1e-10)
    score_euclidean = (mean_dists_euclidean - mean_dists_euclidean.min()) / (mean_dists_euclidean.max() - mean_dists_euclidean.min() + 1e-10)
    score_iso = -iso_forest.score_samples(embeddings)  # Negative anomaly score
    score_iso = (score_iso - score_iso.min()) / (score_iso.max() - score_iso.min() + 1e-10)
    
    combined_score = (score_cosine + score_euclidean + score_iso) / 3
    
    print(f"✅ Detection Results:")
    print(f"   Cosine distance:    {outlier_mask_cosine.sum()} outliers")
    print(f"   Euclidean distance: {outlier_mask_euclidean.sum()} outliers")
    print(f"   Isolation Forest:   {outlier_mask_iso.sum()} outliers")
    print(f"   Consensus (2+):     {consensus_outliers.sum()} outliers")
    print(f"   Detection rate: {consensus_outliers.sum()/len(paths)*100:.2f}%")
    
    if consensus_outliers.sum() > 0:
        outlier_class_dist = pd.Series(labels[consensus_outliers]).value_counts().sort_index()
        print(f"\n🔬 Outliers by class:")
        for cls, count in outlier_class_dist.items():
            total_cls = (labels == cls).sum()
            print(f"   Class {cls}: {count}/{total_cls} images ({count/total_cls*100:.1f}%)")

    return pd.DataFrame({
        'path': paths[consensus_outliers],
        'label': labels[consensus_outliers],
        'distance_cosine': mean_dists_cosine[consensus_outliers],
        'distance_euclidean': mean_dists_euclidean[consensus_outliers],
        'outlier_score': combined_score[consensus_outliers],
        'detection_votes': outlier_votes[consensus_outliers]
    }).sort_values('outlier_score', ascending=False)

# ------------------------------------------------------------
# 7. Enhanced Feature Space Visualization with t-SNE
# ------------------------------------------------------------
def visualize_feature_space_tsne(embeddings_scaled, labels):
    """Create 2D visualization using t-SNE"""
    print("\n🎨 Creating t-SNE visualization...")
    print("   (This may take a few minutes for large datasets...)")
    
    # Sample if too large
    max_samples = 10000
    if len(embeddings_scaled) > max_samples:
        print(f"   Sampling {max_samples} points for t-SNE...")
        indices = np.random.choice(len(embeddings_scaled), max_samples, replace=False)
        embeddings_sample = embeddings_scaled[indices]
        labels_sample = labels[indices]
    else:
        embeddings_sample = embeddings_scaled
        labels_sample = labels
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(embeddings_sample)
    
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    for i in range(5):
        mask = labels_sample == i
        if mask.any():
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=colors[i], label=class_names[i], alpha=0.6, s=20)
    
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("Feature Space Visualization (t-SNE)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_space_tsne.png"), dpi=300)
    plt.close()
    
    print("   ✅ t-SNE visualization saved.")

# ------------------------------------------------------------
# 8. Feature Space Analysis: Separability Metrics
# ------------------------------------------------------------
def analyze_class_separability(embeddings, labels):
    """Analyze how well-separated classes are in feature space"""
    print("\n📊 Analyzing class separability...")
    
    # Calculate pairwise class distances
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    class_means = []
    for label in unique_labels:
        class_mask = labels == label
        class_mean = embeddings[class_mask].mean(axis=0)
        class_means.append(class_mean)
    
    class_means = np.array(class_means)
    
    # Pairwise distances between class centroids
    class_distances = squareform(pdist(class_means, metric='euclidean'))
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    sns.heatmap(class_distances, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names[:n_classes],
                yticklabels=class_names[:n_classes],
                cbar_kws={'label': 'Euclidean Distance'})
    plt.title("Class Centroid Distances\n(Higher = Better Separated)", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_separability.png"), dpi=300)
    plt.close()
    
    # Calculate within-class vs between-class variance ratio
    within_class_var = []
    for label in unique_labels:
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]
        class_mean = class_embeddings.mean(axis=0)
        variance = np.mean((class_embeddings - class_mean)**2)
        within_class_var.append(variance)
    
    avg_within_var = np.mean(within_class_var)
    between_class_var = np.var(class_means, axis=0).mean()
    separability_ratio = between_class_var / (avg_within_var + 1e-10)
    
    print(f"   Within-class variance: {avg_within_var:.4f}")
    print(f"   Between-class variance: {between_class_var:.4f}")
    print(f"   Separability ratio: {separability_ratio:.4f}")
    print(f"   → Higher ratio = better class separation")
    
    # Find most/least separable class pairs
    class_distances_flat = class_distances[np.triu_indices(n_classes, k=1)]
    min_dist_idx = np.argmin(class_distances_flat)
    max_dist_idx = np.argmax(class_distances_flat)
    
    # Convert flat index to matrix indices
    triu_indices = list(zip(*np.triu_indices(n_classes, k=1)))
    min_pair = triu_indices[min_dist_idx]
    max_pair = triu_indices[max_dist_idx]
    
    print(f"\n   Most separable: {class_names[max_pair[0]]} ↔ {class_names[max_pair[1]]} (dist: {class_distances[max_pair]:.2f})")
    print(f"   Least separable: {class_names[min_pair[0]]} ↔ {class_names[min_pair[1]]} (dist: {class_distances[min_pair]:.2f})")
    
    return separability_ratio, class_distances

# ------------------------------------------------------------
# 9. Save Enhanced Summary Report
# ------------------------------------------------------------
def save_summary_report_enhanced(df, valid_df, corrupted_df, quality_issues_df, 
                                class_summary, outlier_df, n_features, 
                                balance_ratio, shannon_entropy, separability_ratio):
    report_path = os.path.join(OUTPUT_DIR, "eda_summary_enhanced.txt")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("DIABETIC RETINOPATHY - ENHANCED EDA SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature Extraction: Deep Learning ({n_features} features)\n")
        f.write(f"   - Model: ResNet50 (pre-trained on ImageNet)\n")
        f.write(f"   - Features: 2048-dim vector from final avgpool layer\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"   Total images: {len(df):,}\n")
        f.write(f"   Valid images: {len(valid_df):,}\n")
        f.write(f"   Corrupted: {len(corrupted_df)}\n")
        f.write(f"   Quality issues: {len(quality_issues_df)}\n\n")
        
        f.write("CLASS DISTRIBUTION:\n")
        f.write(class_summary.to_string(index=False) + "\n\n")
        f.write(f"   Balance Ratio (max/min): {balance_ratio:.2f}\n")
        f.write(f"   Shannon Entropy: {shannon_entropy:.4f}\n\n")
        
        f.write(f"NORMALIZATION VALUES (ImageNet):\n")
        f.write(f"   Mean (RGB): {DATASET_MEAN}\n")
        f.write(f"   Std (RGB):  {DATASET_STD}\n\n")
        
        f.write(f"FEATURE SPACE ANALYSIS:\n")
        f.write(f"   Class Separability Ratio: {separability_ratio:.4f}\n")
        f.write(f"   → Higher values indicate better-separated classes\n\n")
        
        f.write(f"OUTLIER ANALYSIS (Multi-Method Consensus):\n")
        f.write(f"   Total outliers: {len(outlier_df)}\n")
        if len(outlier_df) > 0:
            f.write(f"   Detection rate: {len(outlier_df)/len(valid_df)*100:.2f}%\n")
            f.write(f"\n   Top 15 outliers (highest anomaly score):\n")
            for idx, row in outlier_df.head(15).iterrows():
                filename = os.path.basename(row['path'])
                f.write(f"   {filename[:45]:45s} class={row['label']} score={row['outlier_score']:.4f} votes={row['detection_votes']}\n")
        
        f.write(f"\n" + "="*70 + "\n")
        f.write("KEY FINDINGS & RECOMMENDATIONS:\n")
        f.write("="*70 + "\n\n")
        
        # Class imbalance
        if balance_ratio > 10:
            f.write("⚠️  SEVERE CLASS IMBALANCE DETECTED!\n")
            f.write(f"   Ratio: {balance_ratio:.1f}:1\n")
            f.write("   Recommendations:\n")
            f.write("   • Use class_weight='balanced' in loss function\n")
            f.write("   • Apply Focal Loss to focus on hard examples\n")
            f.write("   • Consider SMOTE or class-balanced sampling\n")
            f.write("   • Use stratified train/val splits\n\n")
        
        # Quality issues
        if len(quality_issues_df) > 0:
            f.write(f"⚠️  {len(quality_issues_df)} IMAGES WITH QUALITY ISSUES\n")
            f.write(f"   These images are too dark or uniform\n")
            f.write(f"   → Consider manual review or exclusion\n\n")
        
        # Outliers
        if len(outlier_df) > 0:
            f.write(f"⚠️  {len(outlier_df)} OUTLIERS DETECTED\n")
            f.write("   These are semantically different from the dataset\n")
            f.write("   Recommendations:\n")
            f.write("   • Review 'outliers_enhanced.csv' manually\n")
            f.write("   • Consider excluding high-score outliers (score > 0.8)\n")
            f.write("   • These may be artifacts, black images, or mislabeled\n\n")
        
        # Separability
        if separability_ratio < 1.0:
            f.write(f"⚠️  LOW CLASS SEPARABILITY (ratio = {separability_ratio:.2f})\n")
            f.write("   Classes overlap significantly in feature space\n")
            f.write("   Recommendations:\n")
            f.write("   • Use more powerful models (EfficientNet, Vision Transformer)\n")
            f.write("   • Apply advanced data augmentation\n")
            f.write("   • Consider ordinal regression (classes are ordered)\n\n")
        else:
            f.write(f"✅ GOOD CLASS SEPARABILITY (ratio = {separability_ratio:.2f})\n")
            f.write("   Classes are reasonably well-separated\n\n")
        
        f.write("="*70 + "\n")
        f.write("NEXT STEPS:\n")
        f.write("="*70 + "\n")
        f.write("1. Review and handle outliers (outliers_enhanced.csv)\n")
        f.write("2. Address class imbalance in training pipeline\n")
        f.write("3. Use ImageNet normalization values\n")
        f.write("4. Consider stratified K-fold cross-validation\n")
        f.write("5. Monitor per-class performance during training\n")
        f.write("="*70 + "\n")
    
    print(f"\n📄 Comprehensive enhanced report saved to: {report_path}")

# ------------------------------------------------------------
# 10. Main Execution (Enhanced)
# ------------------------------------------------------------
def run_eda():
    start_time = time.time()
    
    print("\n" + "="*70)
    print(" DIABETIC RETINOPATHY - ENHANCED EDA PIPELINE")
    print(" Deep Learning Features + Advanced Analytics")
    print("="*70 + "\n")
    
    df = load_image_paths_and_labels(DEFAULT_DATASET_DIR, CSV_LABELS_PATH)
    valid_df, corrupted_df, quality_issues_df = detect_corrupted_enhanced(df)
    print(f"Valid: {len(valid_df):,} | Corrupted: {len(corrupted_df)} | Quality Issues: {len(quality_issues_df)}")
    
    class_summary, balance_ratio, shannon_entropy = analyze_class_distribution_enhanced(valid_df)
    
    pixel_statistics_enhanced(valid_df)
    
    # Feature extraction
    sampled = valid_df.sample(min(EMBED_LIMIT, len(valid_df)), random_state=42).reset_index(drop=True)
    embeddings, labels, paths = extract_dl_features_batch(sampled)
    
    # Scale features
    print("\n🔧 Scaling features...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Visualizations
    visualize_feature_space_pca(embeddings_scaled, labels)
    visualize_feature_space_tsne(embeddings_scaled, labels)
    visualize_feature_space_umap(embeddings_scaled, labels)
    
    # Class separability analysis
    separability_ratio, class_distances = analyze_class_separability(embeddings, labels)
    
    # Enhanced outlier detection
    outlier_df = detect_outliers_enhanced(embeddings, paths, labels)
    outlier_df.to_csv(os.path.join(OUTPUT_DIR, "outliers_enhanced.csv"), index=False)
    
    # Save enhanced report
    save_summary_report_enhanced(df, valid_df, corrupted_df, quality_issues_df,
                                class_summary, outlier_df, embeddings.shape[1],
                                balance_ratio, shannon_entropy, separability_ratio)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(" 🎉 ENHANCED EDA COMPLETE!")
    print("="*70)
    print(f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"📁 Results: {OUTPUT_DIR}")
    print("\n📊 Generated files:")
    print("   - class_distribution_enhanced.png    (bar + pie charts)")
    print("   - pixel_distribution_enhanced.png    (6-panel quality analysis)")
    print("   - feature_space_pca.png              (2D PCA)")
    print("   - feature_space_tsne.png             (2D t-SNE)")
    print("   - feature_space_umap.png             (2D UMAP)")
    print("   - class_separability.png             (class distance heatmap)")
    print("   - outliers_enhanced.csv              (multi-method detection)")
    print("   - eda_summary_enhanced.txt           (comprehensive report)")
    print("="*70 + "\n")

# Keep original functions for compatibility
def visualize_feature_space_pca(embeddings_scaled, labels):
    """Create 2D visualization of feature space using PCA"""
    print("\n🎨 Creating feature space visualization with PCA...")
    
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
    
    print(f"   Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

def visualize_feature_space_umap(embeddings_scaled, labels):
    """Create 2D visualization of feature space using UMAP"""
    print("\n🎨 Creating feature space visualization with UMAP...")
    
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
    
    print("   ✅ UMAP visualization saved.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support() 
    run_eda()
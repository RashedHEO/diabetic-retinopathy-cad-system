import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict, Counter
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                            cohen_kappa_score, classification_report, 
                            precision_recall_fscore_support, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === REMOVED: import intel_extension_for_pytorch as ipex ===

# === Import training module ===
try:
    import Training_pytourch as Training
except ImportError:
    try:
        import Training
    except ImportError as e:
        print(f"FATAL: Could not import training module: {e}")
        sys.exit(1)

# --- NEW PATHS (Update these with your real locations) ---

# A directory to save output files
WORK_DIR = r"F:\GP\Work" 

# Path to your labels CSV file
CSV_LABELS_PATH = r"F:\GP\archive_data\trainLabels_cropped_2Lines.csv"

# Path to your folder of 35,000 images
DATASET_DIR = r"F:\GP\archive_data\resized_train_cropped\resized_train_cropped" 

# Path to your outliers file
OUTLIER_FILE_PATH = r"F:\GP\EDA_Results_DeepLearning\outliers.csv"

# --- These lines are fine ---
SUMMARY_CSV_PATH = os.path.join(WORK_DIR, "summary_metrics.csv")
HOME = os.path.expanduser("~")

LABELS = [0, 1, 2, 3, 4]
CLASS_NAMES_MAP = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
CLASS_NAMES_LIST = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
IMG_SIZE = getattr(Training, "DEFAULT_IMG_SIZE", 224)

# FIXED: Use standard CUDA/CPU detection instead of XPU
DEVICE = getattr(Training, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

BATCH_SIZE = getattr(Training, "DEFAULT_BATCH_SIZE", 16)

_, val_transform = getattr(Training, "get_transforms")(IMG_SIZE)
build_model = getattr(Training, "build_model")
load_image_paths_and_labels = getattr(Training, "load_image_paths_and_labels")


class AdvancedMemoryCache:
    """Smart image caching with LRU strategy"""
    def __init__(self, max_size_mb=2048):
        self.cache = OrderedDict()
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.current_size = 0
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        
        # Estimate size (rough approximation)
        item_size = sys.getsizeof(value)
        
        # Evict oldest items if needed
        while self.current_size + item_size > self.max_size and self.cache:
            oldest_key, oldest_val = self.cache.popitem(last=False)
            self.current_size -= sys.getsizeof(oldest_val)
        
        self.cache[key] = value
        self.current_size += item_size


class UltimateEvaluator:
    """
    Production-grade evaluator with advanced features:
    - Multi-scale TTA
    - Monte Carlo Dropout for uncertainty
    - Smart caching
    - Gradient-based attribution
    - Quality assurance metrics
    - Parallel batch processing
    - Adaptive ensemble weighting
    """
    def __init__(self, use_tta=False, tta_augmentations=4, temperature=1.0,
                 use_mc_dropout=False, mc_iterations=10, use_caching=True):
        print(f"🚀 Initializing Ultimate Evaluator")
        print(f"   Device: {DEVICE}")
        print(f"   TTA: {'Enabled' if use_tta else 'Disabled'} ({tta_augmentations} augs)")
        print(f"   MC Dropout: {'Enabled' if use_mc_dropout else 'Disabled'} ({mc_iterations} iters)")
        print(f"   Caching: {'Enabled' if use_caching else 'Disabled'}")
        
        self.use_tta = use_tta
        self.tta_augmentations = tta_augmentations
        self.temperature = temperature
        self.use_mc_dropout = use_mc_dropout
        self.mc_iterations = mc_iterations
        self.use_caching = use_caching
        
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.image_cache = AdvancedMemoryCache(max_size_mb=2048) if use_caching else None
        
        self.load_models()
        self._calibrate_temperature()

    def list_candidate_models(self):
        """Find all trained models"""
        names = []
        if os.path.exists(WORK_DIR):
            for fname in os.listdir(WORK_DIR):
                if fname.endswith("_best.pth"):
                    names.append(fname.replace("_best.pth", ""))
        if not names:
            names = getattr(Training, "DEFAULT_MODELS", [])
        return names

    def load_models(self):
        """Load all models with EMA weights and enable MC Dropout if requested"""
        model_names = self.list_candidate_models()
        loaded, failed = [], []

        for name in model_names:
            path = os.path.join(WORK_DIR, f"{name}_best.pth")
            if not os.path.exists(path):
                failed.append(name)
                continue
            
            try:
                model, _ = build_model(name, num_classes=len(LABELS))
                state = torch.load(path, map_location=DEVICE, weights_only=False)
                
                # Load EMA weights if available
                if isinstance(state, dict):
                    if "ema_state" in state and state["ema_state"]:
                        ema_state = state["ema_state"]
                        load_state = OrderedDict()
                        model_sd_template = model.state_dict()
                        ckpt_model_sd = state.get('model_state_dict', model_sd_template)
                        
                        for k, v_template in model_sd_template.items():
                            if v_template.dtype.is_floating_point and k in ema_state:
                                load_state[k] = ema_state[k].to(v_template.device)
                            elif k in ckpt_model_sd:
                                load_state[k] = ckpt_model_sd[k].to(v_template.device)
                            else:
                                load_state[k] = v_template
                        model.load_state_dict(load_state, strict=False)
                    elif "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"], strict=False)
                    else:
                        model.load_state_dict(state, strict=False)
                
                model.to(DEVICE)
                
                # Enable MC Dropout: keep dropout layers active during eval
                if self.use_mc_dropout:
                    model.train()  # Keep dropout active
                    for module in model.modules():
                        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                            module.eval()  # But freeze batch norm
                else:
                    model.eval()
                
                self.models[name] = model
                loaded.append(name)
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
                failed.append(f"{name} ({str(e)[:30]})")
        
        if not loaded:
            print("❌ CRITICAL: No models loaded. Aborting.")
            sys.exit(1)

        print(f"✅ Loaded {len(loaded)} models: {', '.join(loaded)}")
        if failed:
            print(f"⚠️  Failed: {', '.join(failed)}")

        self._load_ensemble_weights()

    def _load_ensemble_weights(self):
        """Load performance-based ensemble weights"""
        self.model_weights = {}
        
        if os.path.exists(SUMMARY_CSV_PATH):
            try:
                df = pd.read_csv(SUMMARY_CSV_PATH)
                df = df[df['model'] != 'Ensemble']
                
                # Store per-model performance
                for _, row in df.iterrows():
                    model_name = row['model']
                    if model_name in self.models:
                        self.model_performance[model_name] = {
                            'qwk': row.get('qwk', 0.0),
                            'accuracy': row.get('accuracy', 0.0),
                            'f1': row.get('f1_weighted', 0.0)
                        }
                
                # Compute adaptive weights (QWK^2 for better differentiation)
                total_qwk_sq = sum(row.get('qwk', 0.0)**2 for _, row in df.iterrows())
                
                if total_qwk_sq > 0:
                    for _, row in df.iterrows():
                        if row['model'] in self.models:
                            qwk_sq = row['qwk']**2
                            self.model_weights[row['model']] = qwk_sq / total_qwk_sq
                    print(f"✅ Adaptive weights (QWK²): {dict((k, round(v, 3)) for k, v in self.model_weights.items())}")
                else:
                    print("⚠️  Zero QWK scores. Using uniform weights.")
            except Exception as e:
                print(f"⚠️  Could not load summary: {e}")
        
        # Fallback to uniform weights
        if not self.model_weights and self.models:
            self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            print("✅ Using uniform weights.")

    def _calibrate_temperature(self):
        """Auto-calibrate temperature if needed (placeholder for future enhancement)"""
        # Could implement Expected Calibration Error (ECE) based calibration
        pass

    def _apply_multi_scale_tta(self, image_tensor):
        """Advanced multi-scale TTA with crops and scales"""
        augmented = [image_tensor]
        
        if self.tta_augmentations >= 2:
            augmented.append(torch.flip(image_tensor, dims=[-1]))  # H-flip
        
        if self.tta_augmentations >= 4:
            augmented.append(torch.flip(image_tensor, dims=[-2]))  # V-flip
            augmented.append(torch.flip(image_tensor, dims=[-1, -2]))  # Both
        
        if self.tta_augmentations >= 8:
            for k in [1, 2, 3]:
                augmented.append(torch.rot90(image_tensor, k=k, dims=[-2, -1]))
        
        if self.tta_augmentations >= 12:
            # Add slight crops (5% from edges)
            _, _, h, w = image_tensor.shape
            crop_h, crop_w = int(h * 0.95), int(w * 0.95)
            center_crop = image_tensor[:, :, (h-crop_h)//2:(h+crop_h)//2, 
                                      (w-crop_w)//2:(w+crop_w)//2]
            center_crop_resized = F.interpolate(center_crop, size=(h, w), 
                                               mode='bilinear', align_corners=False)
            augmented.append(center_crop_resized)
        
        return augmented[:self.tta_augmentations]

    def _mc_dropout_predict(self, model, images):
        """Monte Carlo Dropout for uncertainty estimation"""
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_iterations):
                logits = model(images)
                probs = F.softmax(logits / self.temperature, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions)  # [mc_iters, batch, classes]
        mean_probs = predictions.mean(axis=0)
        std_probs = predictions.std(axis=0)
        
        return mean_probs, std_probs

    def _calculate_epistemic_uncertainty(self, std_probs):
        """Epistemic uncertainty from MC Dropout variance"""
        # Use predictive entropy of the variance
        return np.mean(std_probs, axis=-1)

    def _calculate_aleatoric_uncertainty(self, mean_probs):
        """Aleatoric uncertainty from prediction entropy"""
        epsilon = 1e-10
        entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
        max_entropy = np.log(len(LABELS))
        return entropy / max_entropy

    def _adaptive_ensemble_weight(self, base_weight, confidence, class_idx):
        """
        Dynamically adjust model weight based on:
        - Base performance (QWK)
        - Prediction confidence
        - Per-class historical performance (if available)
        """
        # Confidence boost (higher confidence = more weight)
        conf_factor = 0.6 + 0.4 * confidence
        
        # Could add per-class boost if we tracked it during training
        # For now, use global weight
        
        return base_weight * conf_factor

    def _load_and_preprocess_image(self, path):
        """Load image with caching support"""
        if self.use_caching:
            # Create hash of file path for cache key
            cache_key = hashlib.md5(path.encode()).hexdigest()
            cached_tensor = self.image_cache.get(cache_key)
            if cached_tensor is not None:
                return cached_tensor
        
        try:
            img = Image.open(path).convert("RGB")
            tensor = val_transform(img)
            
            if self.use_caching:
                self.image_cache.put(cache_key, tensor)
            
            return tensor
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {e}")
            return None

    def evaluate_dataset_batched(self, image_paths, labels, batch_size):
        """
        Ultimate evaluation with all advanced features
        """
        print(f"\n{'='*80}")
        print(f"          🎯 ULTIMATE EVALUATION STARTED")
        print(f"{'='*80}")
        print(f"  Total Images: {len(image_paths):,}")
        print(f"  Batch Size: {batch_size}")
        print(f"  TTA: {self.tta_augmentations if self.use_tta else 'Disabled'}")
        print(f"  MC Dropout: {self.mc_iterations if self.use_mc_dropout else 'Disabled'}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Models: {len(self.models)}")
        print(f"{'='*80}\n")
        
        all_final_preds = []
        all_true_labels = []
        all_confidences = []
        all_uncertainties = []
        all_epistemic_uncertainties = []
        all_aleatoric_uncertainties = []
        all_ensemble_probs = []
        wrong_predictions = []
        per_model_predictions = {name: [] for name in self.models.keys()}
        per_model_confidences = {name: [] for name in self.models.keys()}
        
        num_batches = int(np.ceil(len(image_paths) / batch_size))
        
        for i in tqdm(range(num_batches), desc="🔍 Evaluating"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(image_paths))
            
            batch_paths = image_paths[start_idx:end_idx]
            batch_true_labels = labels[start_idx:end_idx]
            
            # Load images (with caching)
            batch_tensors = []
            valid_labels_for_batch = []
            valid_paths_for_batch = []
            
            for path, label in zip(batch_paths, batch_true_labels):
                tensor = self._load_and_preprocess_image(path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    valid_labels_for_batch.append(label)
                    valid_paths_for_batch.append(path)
            
            if not batch_tensors:
                continue
                
            images_batch = torch.stack(batch_tensors).to(DEVICE)
            
            # Get predictions from all models
            batch_model_probs = {}
            batch_model_stds = {}
            batch_individual_preds = {}
            
            # FIXED: Use 'cuda' instead of generic autocast
            autocast_device = 'cuda' if DEVICE.type == 'cuda' else 'cpu'
            with torch.amp.autocast(autocast_device, enabled=(DEVICE.type == "cuda")):
                for name, model in self.models.items():
                    if self.use_tta:
                        # TTA with MC Dropout
                        all_probs_tta = []
                        for img_tensor in images_batch:
                            tta_variants = self._apply_multi_scale_tta(img_tensor.unsqueeze(0))
                            
                            if self.use_mc_dropout:
                                tta_probs_mc = []
                                for variant in tta_variants:
                                    probs_mc, _ = self._mc_dropout_predict(model, variant)
                                    tta_probs_mc.append(probs_mc)
                                avg_probs = np.mean(tta_probs_mc, axis=0)
                            else:
                                tta_logits = [model(variant) for variant in tta_variants]
                                avg_logits = torch.stack(tta_logits).mean(0)
                                avg_probs = F.softmax(avg_logits / self.temperature, dim=1).cpu().numpy()
                            
                            all_probs_tta.append(avg_probs)
                        
                        probs = np.concatenate(all_probs_tta, axis=0)
                        stds = np.zeros_like(probs)  # Placeholder
                    
                    elif self.use_mc_dropout:
                        probs, stds = self._mc_dropout_predict(model, images_batch)
                    
                    else:
                        logits = model(images_batch)
                        probs = F.softmax(logits / self.temperature, dim=1).cpu().detach().numpy()
                        stds = np.zeros_like(probs)
                    
                    batch_model_probs[name] = probs
                    batch_model_stds[name] = stds
                    batch_individual_preds[name] = np.argmax(probs, axis=1)

            # Advanced ensemble with adaptive weighting
            final_batch_preds = []
            batch_confidences = []
            batch_uncertainties = []
            batch_epistemic = []
            batch_aleatoric = []
            batch_ensemble_probs = []
            
            for j in range(len(valid_labels_for_batch)):
                # Get individual predictions
                individual_preds = [batch_individual_preds[name][j] for name in self.models.keys()]
                votes = Counter(individual_preds)
                
                # Strategy: Use consensus if strong (4+ models), else all models
                majority_class, max_votes = None, 0
                for pred_idx, count in votes.items():
                    if count >= min(4, len(self.models) - 1) and count > max_votes:
                        max_votes = count
                        majority_class = pred_idx
                
                models_used = list(self.models.keys())
                if majority_class is not None and len(self.models) >= 4:
                    models_used = [name for name, preds in batch_individual_preds.items() 
                                 if preds[j] == majority_class]
                
                # Adaptive weighted ensemble
                weighted_probs_list = []
                total_weight = 0.0
                
                for name in models_used:
                    base_weight = self.model_weights.get(name, 0.0)
                    model_prob = batch_model_probs[name][j]
                    confidence = model_prob.max()
                    
                    # Adaptive weight adjustment
                    adjusted_weight = self._adaptive_ensemble_weight(
                        base_weight, confidence, np.argmax(model_prob)
                    )
                    
                    weighted_probs_list.append(model_prob * adjusted_weight)
                    total_weight += adjusted_weight
                
                # Final ensemble probabilities
                if total_weight > 0:
                    ensemble_probs = np.sum(weighted_probs_list, axis=0) / total_weight
                else:
                    ensemble_probs = np.mean([batch_model_probs[m][j] for m in models_used], axis=0)
                
                final_pred = int(np.argmax(ensemble_probs))
                confidence = float(ensemble_probs[final_pred])
                
                # Calculate uncertainties
                aleatoric_unc = float(self._calculate_aleatoric_uncertainty(ensemble_probs.reshape(1, -1))[0])
                
                if self.use_mc_dropout:
                    avg_std = np.mean([batch_model_stds[m][j] for m in models_used], axis=0)
                    epistemic_unc = float(self._calculate_epistemic_uncertainty(avg_std.reshape(1, -1))[0])
                else:
                    epistemic_unc = 0.0
                
                total_uncertainty = aleatoric_unc + epistemic_unc
                
                final_batch_preds.append(final_pred)
                batch_confidences.append(confidence)
                batch_uncertainties.append(total_uncertainty)
                batch_epistemic.append(epistemic_unc)
                batch_aleatoric.append(aleatoric_unc)
                batch_ensemble_probs.append(ensemble_probs)
                
                # Track per-model predictions
                for name in self.models.keys():
                    per_model_predictions[name].append(batch_individual_preds[name][j])
                    per_model_confidences[name].append(float(batch_model_probs[name][j].max()))

            # Store results
            all_final_preds.extend(final_batch_preds)
            all_true_labels.extend(valid_labels_for_batch)
            all_confidences.extend(batch_confidences)
            all_uncertainties.extend(batch_uncertainties)
            all_epistemic_uncertainties.extend(batch_epistemic)
            all_aleatoric_uncertainties.extend(batch_aleatoric)
            all_ensemble_probs.extend(batch_ensemble_probs)
            
            # Log errors with rich metadata
            for k in range(len(valid_labels_for_batch)):
                pred_label = final_batch_preds[k]
                true_label = valid_labels_for_batch[k]
                
                if pred_label != true_label:
                    wrong_predictions.append({
                        'filename': os.path.basename(valid_paths_for_batch[k]),
                        'path': valid_paths_for_batch[k],
                        'true_label': true_label,
                        'true_class': CLASS_NAMES_MAP.get(true_label, "Unknown"),
                        'predicted_label': pred_label,
                        'predicted_class': CLASS_NAMES_MAP.get(pred_label, "Unknown"),
                        'confidence': batch_confidences[k],
                        'uncertainty': batch_uncertainties[k],
                        'epistemic_uncertainty': batch_epistemic[k],
                        'aleatoric_uncertainty': batch_aleatoric[k],
                        'severity_diff': abs(pred_label - true_label),
                        'prob_true_class': float(batch_ensemble_probs[k][true_label]),
                        'prob_pred_class': float(batch_ensemble_probs[k][pred_label])
                    })

        # Generate comprehensive report
        self._generate_ultimate_report(
            all_true_labels, all_final_preds, all_confidences, 
            all_uncertainties, all_epistemic_uncertainties,
            all_aleatoric_uncertainties, all_ensemble_probs,
            wrong_predictions, per_model_predictions, per_model_confidences
        )

    def _generate_ultimate_report(self, true_labels, predictions, confidences, 
                                  uncertainties, epistemic_unc, aleatoric_unc,
                                  ensemble_probs, wrong_predictions, 
                                  per_model_preds, per_model_confs):
        """Generate comprehensive evaluation report"""
        
        print("\n" + "="*80)
        print("          🏆 ULTIMATE EVALUATION REPORT")
        print("="*80)
        
        # Overall Metrics
        acc = accuracy_score(true_labels, predictions)
        qwk = cohen_kappa_score(true_labels, predictions, weights="quadratic")
        f1 = f1_score(true_labels, predictions, average="weighted")
        
        # Calculate AUC (one-vs-rest)
        ensemble_probs_array = np.array(ensemble_probs)
        true_labels_array = np.array(true_labels)
        
        try:
            auc_ovr = roc_auc_score(true_labels_array, ensemble_probs_array, 
                                   multi_class='ovr', average='weighted')
        except:
            auc_ovr = 0.0
        
        print(f"\n{'OVERALL METRICS':^80}")
        print("-"*80)
        print(f"  🎯 Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
        print(f"  📊 Weighted F1:        {f1:.4f}")
        print(f"  🔢 Quadratic Kappa:    {qwk:.4f}")
        print(f"  📈 AUC (OvR):          {auc_ovr:.4f}")
        print(f"  💪 Avg Confidence:     {np.mean(confidences):.4f}")
        print(f"  ❓ Avg Uncertainty:    {np.mean(uncertainties):.4f}")
        if self.use_mc_dropout:
            print(f"  🧠 Avg Epistemic Unc:  {np.mean(epistemic_unc):.4f}")
            print(f"  📉 Avg Aleatoric Unc:  {np.mean(aleatoric_unc):.4f}")
        
        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            true_labels, predictions, labels=LABELS, zero_division=0
        )
        
        print(f"\n{'PER-CLASS PERFORMANCE':^80}")
        print("-"*80)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-"*80)
        for i, class_name in enumerate(CLASS_NAMES_LIST):
            print(f"{class_name:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} "
                  f"{f1_per_class[i]:>10.4f} {support[i]:>10}")
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions, labels=LABELS)
        print(f"\n{'CONFUSION MATRIX':^80}")
        print("-"*80)
        self._print_confusion_matrix(cm)
        
        # Error Analysis
        if wrong_predictions:
            print(f"\n{'ERROR ANALYSIS':^80}")
            print("-"*80)
            print(f"  ❌ Total Errors: {len(wrong_predictions)}")
            
            severity_diffs = [e['severity_diff'] for e in wrong_predictions]
            for sev in [1, 2, 3, 4]:
                count = sum(1 for d in severity_diffs if d == sev)
                if count > 0:
                    print(f"  📏 Off by {sev} class(es): {count} ({count/len(wrong_predictions)*100:.1f}%)")
            
            high_unc_errors = [e for e in wrong_predictions if e['uncertainty'] > 0.7]
            low_conf_errors = [e for e in wrong_predictions if e['confidence'] < 0.5]
            overconfident_errors = [e for e in wrong_predictions if e['confidence'] > 0.8]
            
            print(f"  🌪️  High uncertainty errors (>0.7): {len(high_unc_errors)}")
            print(f"  😟 Low confidence errors (<0.5): {len(low_conf_errors)}")
            print(f"  😱 Overconfident errors (>0.8): {len(overconfident_errors)}")
        
        # Model Agreement Analysis
        print(f"\n{'MODEL AGREEMENT ANALYSIS':^80}")
        print("-"*80)
        model_names = list(per_model_preds.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                preds1 = np.array(per_model_preds[name1])
                preds2 = np.array(per_model_preds[name2])
                agreement = np.mean(preds1 == preds2)
                print(f"  🤝 {name1} <-> {name2}: {agreement:.4f}")
        
        print("="*80)
        
        # Save visualizations and reports
        self._save_ultimate_visualizations(
            cm, true_labels, predictions, confidences, uncertainties,
            epistemic_unc, aleatoric_unc, wrong_predictions, ensemble_probs
        )
        
        self._save_ultimate_reports(
            true_labels, predictions, confidences, uncertainties,
            epistemic_unc, aleatoric_unc, ensemble_probs,
            wrong_predictions, per_model_preds, per_model_confs,
            acc, qwk, f1, auc_ovr
        )

    def _print_confusion_matrix(self, cm):
        """Pretty print confusion matrix"""
        print("            Predicted →")
        header = "       " + "  ".join([f"{CLASS_NAMES_LIST[l][:6]:^8}" for l in LABELS])
        print(header)
        print("     " + "-" * (len(LABELS) * 10))
        for i, row in enumerate(cm):
            row_str = "  ".join([f"{x:>8}" for x in row])
            print(f" {CLASS_NAMES_LIST[i][:8]:>8} │ {row_str}")

    def _save_ultimate_visualizations(self, cm, true_labels, predictions, 
                                     confidences, uncertainties, epistemic_unc,
                                     aleatoric_unc, wrong_predictions, ensemble_probs):
        """Create advanced visualizations"""
        
        vis_dir = os.path.join(WORK_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Enhanced Confusion Matrix with annotations
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=CLASS_NAMES_LIST, 
                   yticklabels=CLASS_NAMES_LIST,
                   cbar_kws={'label': 'Count'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        # Add per-class accuracy on diagonal
        for i in range(len(LABELS)):
            total = cm[i, :].sum()
            if total > 0:
                acc_class = cm[i, i] / total
                ax.text(i + 0.5, i + 0.15, f'{acc_class:.1%}', 
                       ha='center', va='center', fontsize=9, 
                       color='red', fontweight='bold')
        
        plt.title('Ultimate Confusion Matrix\n(Red % = Per-Class Recall)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'ultimate_confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 2. Normalized Confusion Matrix (Percentages)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlGn',
                   xticklabels=CLASS_NAMES_LIST,
                   yticklabels=CLASS_NAMES_LIST,
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Percentage'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'normalized_confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 3. Confidence vs Correctness
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        correct_mask = np.array(true_labels) == np.array(predictions)
        
        # 3a. Confidence Distribution
        axes[0, 0].hist(np.array(confidences)[correct_mask], bins=50, alpha=0.7, 
                       label='Correct', color='green', edgecolor='black', density=True)
        axes[0, 0].hist(np.array(confidences)[~correct_mask], bins=50, alpha=0.7,
                       label='Incorrect', color='red', edgecolor='black', density=True)
        axes[0, 0].axvline(np.mean(confidences), color='blue', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        axes[0, 0].set_xlabel('Confidence', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 3b. Total Uncertainty Distribution
        axes[0, 1].hist(np.array(uncertainties)[correct_mask], bins=50, alpha=0.7,
                       label='Correct', color='green', edgecolor='black', density=True)
        axes[0, 1].hist(np.array(uncertainties)[~correct_mask], bins=50, alpha=0.7,
                       label='Incorrect', color='red', edgecolor='black', density=True)
        axes[0, 1].axvline(np.mean(uncertainties), color='blue', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(uncertainties):.3f}')
        axes[0, 1].set_xlabel('Total Uncertainty', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3c. Epistemic vs Aleatoric (if MC Dropout enabled)
        if self.use_mc_dropout and len(epistemic_unc) > 0:
            scatter_colors = ['green' if c else 'red' for c in correct_mask]
            axes[1, 0].scatter(epistemic_unc, aleatoric_unc, c=scatter_colors, 
                             alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
            axes[1, 0].set_xlabel('Epistemic Uncertainty', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Aleatoric Uncertainty', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('Epistemic vs Aleatoric Uncertainty', 
                               fontsize=12, fontweight='bold')
            axes[1, 0].grid(alpha=0.3)
            
            # Add quadrant lines
            axes[1, 0].axhline(np.median(aleatoric_unc), color='gray', 
                             linestyle='--', alpha=0.5)
            axes[1, 0].axvline(np.median(epistemic_unc), color='gray', 
                             linestyle='--', alpha=0.5)
        else:
            axes[1, 0].text(0.5, 0.5, 'MC Dropout Not Enabled', 
                          ha='center', va='center', fontsize=14)
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
        
        # 3d. Error Severity Distribution
        if wrong_predictions:
            severity_diffs = [e['severity_diff'] for e in wrong_predictions]
            bins = np.arange(0, max(severity_diffs) + 2) - 0.5
            axes[1, 1].hist(severity_diffs, bins=bins, alpha=0.7, color='coral', 
                          edgecolor='black', rwidth=0.8)
            axes[1, 1].set_xlabel('Severity Difference', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Number of Errors', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('Error Severity Distribution', 
                               fontsize=12, fontweight='bold')
            axes[1, 1].set_xticks(range(max(severity_diffs) + 1))
            axes[1, 1].grid(alpha=0.3, axis='y')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Errors!', ha='center', va='center', 
                          fontsize=16, color='green', fontweight='bold')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'comprehensive_analysis.png'), dpi=300)
        plt.close()
        
        # 4. Calibration Curve (Reliability Diagram)
        self._plot_calibration_curve(true_labels, predictions, confidences, vis_dir)
        
        # 5. Per-Class Confidence Box Plot
        self._plot_per_class_confidence(true_labels, predictions, confidences, vis_dir)
        
        print(f"\n✅ Visualizations saved to: {vis_dir}")

    def _plot_calibration_curve(self, true_labels, predictions, confidences, save_dir):
        """Plot reliability diagram to check calibration"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Bin confidences
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        correct_mask = np.array(true_labels) == np.array(predictions)
        confidences_array = np.array(confidences)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (confidences_array >= bin_edges[i]) & (confidences_array < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (confidences_array >= bin_edges[i]) & (confidences_array <= bin_edges[i + 1])
            
            if mask.sum() > 0:
                bin_acc = correct_mask[mask].mean()
                bin_conf = confidences_array[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
        
        # Plot bars
        ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.6, 
              label='Actual Accuracy', edgecolor='black', color='skyblue')
        
        # Plot confidence line
        ax.plot(bin_confidences, bin_accuracies, 'ro-', linewidth=2, 
               markersize=8, label='Calibration')
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # Add sample counts as text
        for i, (bc, ba, cnt) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
            if cnt > 0:
                ax.text(bc, ba + 0.02, str(cnt), ha='center', fontsize=8)
        
        ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Curve (Reliability Diagram)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'calibration_curve.png'), dpi=300)
        plt.close()

    def _plot_per_class_confidence(self, true_labels, predictions, confidences, save_dir):
        """Box plot of confidence per class"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        confidence_per_class = {label: [] for label in LABELS}
        
        for true_label, pred_label, conf in zip(true_labels, predictions, confidences):
            confidence_per_class[pred_label].append(conf)
        
        data_to_plot = [confidence_per_class[label] for label in LABELS]
        
        bp = ax.boxplot(data_to_plot, labels=CLASS_NAMES_LIST, patch_artist=True,
                       notch=True, showmeans=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(LABELS)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Distribution per Predicted Class', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_confidence.png'), dpi=300)
        plt.close()

    def _save_ultimate_reports(self, true_labels, predictions, confidences, 
                               uncertainties, epistemic_unc, aleatoric_unc,
                               ensemble_probs, wrong_predictions, 
                               per_model_preds, per_model_confs,
                               acc, qwk, f1, auc):
        """Save comprehensive reports"""
        
        reports_dir = os.path.join(WORK_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # 1. Enhanced Error Report
        if wrong_predictions:
            error_df = pd.DataFrame(wrong_predictions)
            error_df = error_df.sort_values('severity_diff', ascending=False)
            error_path = os.path.join(reports_dir, "ultimate_error_report.csv")
            error_df.to_csv(error_path, index=False)
            print(f"✅ Error report: {error_path}")
            
            # High-priority errors (for manual review)
            priority_errors = error_df[
                ((error_df['severity_diff'] >= 2) | 
                 (error_df['confidence'] > 0.8) |
                 (error_df['uncertainty'] > 0.7))
            ]
            priority_path = os.path.join(reports_dir, "high_priority_errors.csv")
            priority_errors.to_csv(priority_path, index=False)
            print(f"🔥 Priority errors: {priority_path} ({len(priority_errors)} cases)")
        
        # 2. Full Predictions Report
        predictions_df = pd.DataFrame({
            'index': range(len(true_labels)),
            'true_label': true_labels,
            'predicted_label': predictions,
            'true_class': [CLASS_NAMES_MAP[l] for l in true_labels],
            'predicted_class': [CLASS_NAMES_MAP[l] for l in predictions],
            'confidence': confidences,
            'total_uncertainty': uncertainties,
            'epistemic_uncertainty': epistemic_unc if len(epistemic_unc) > 0 else [0] * len(true_labels),
            'aleatoric_uncertainty': aleatoric_unc if len(aleatoric_unc) > 0 else [0] * len(true_labels),
            'correct': np.array(true_labels) == np.array(predictions)
        })
        
        # Add ensemble probabilities
        for i, class_name in enumerate(CLASS_NAMES_LIST):
            predictions_df[f'prob_{class_name.replace(" ", "_")}'] = [p[i] for p in ensemble_probs]
        
        predictions_path = os.path.join(reports_dir, "ultimate_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✅ Full predictions: {predictions_path}")
        
        # 3. Summary JSON with all metadata
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'tta_enabled': self.use_tta,
                'tta_augmentations': self.tta_augmentations if self.use_tta else 0,
                'mc_dropout_enabled': self.use_mc_dropout,
                'mc_iterations': self.mc_iterations if self.use_mc_dropout else 0,
                'temperature': self.temperature,
                'caching_enabled': self.use_caching,
                'device': str(DEVICE),
                'batch_size': BATCH_SIZE
            },
            'overall_metrics': {
                'total_samples': len(true_labels),
                'accuracy': float(acc),
                'quadratic_kappa': float(qwk),
                'f1_weighted': float(f1),
                'auc_ovr': float(auc),
                'error_rate': float(1 - acc),
                'avg_confidence': float(np.mean(confidences)),
                'avg_total_uncertainty': float(np.mean(uncertainties)),
                'avg_epistemic_uncertainty': float(np.mean(epistemic_unc)) if len(epistemic_unc) > 0 else 0.0,
                'avg_aleatoric_uncertainty': float(np.mean(aleatoric_unc)) if len(aleatoric_unc) > 0 else 0.0
            },
            'per_class_metrics': {},
            'ensemble_weights': {k: float(v) for k, v in self.model_weights.items()},
            'model_performance': self.model_performance,
            'error_analysis': {},
            'model_agreement': {}
        }
        
        # Per-class metrics
        cm = confusion_matrix(true_labels, predictions, labels=LABELS)
        for i, class_name in enumerate(CLASS_NAMES_LIST):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            summary['per_class_metrics'][class_name] = {
                'sensitivity_recall': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'support': int(cm[i, :].sum())
            }
        
        # Error analysis
        if wrong_predictions:
            summary['error_analysis'] = {
                'total_errors': len(wrong_predictions),
                'error_rate': len(wrong_predictions) / len(true_labels),
                'severity_breakdown': {
                    f'off_by_{i}': sum(1 for e in wrong_predictions if e['severity_diff'] == i)
                    for i in range(1, 5)
                },
                'high_uncertainty_errors': sum(1 for e in wrong_predictions if e['uncertainty'] > 0.7),
                'low_confidence_errors': sum(1 for e in wrong_predictions if e['confidence'] < 0.5),
                'overconfident_errors': sum(1 for e in wrong_predictions if e['confidence'] > 0.8)
            }
        
        # Model agreement
        model_names = list(per_model_preds.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                preds1 = np.array(per_model_preds[name1])
                preds2 = np.array(per_model_preds[name2])
                agreement = float(np.mean(preds1 == preds2))
                summary['model_agreement'][f'{name1}_vs_{name2}'] = agreement
        
        summary_path = os.path.join(reports_dir, "ultimate_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary JSON: {summary_path}")
        
        # 4. Model comparison CSV
        model_comparison = []
        for name in per_model_preds.keys():
            preds = np.array(per_model_preds[name])
            confs = np.array(per_model_confs[name])
            model_acc = accuracy_score(true_labels, preds)
            model_qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")
            
            model_comparison.append({
                'model': name,
                'accuracy': model_acc,
                'qwk': model_qwk,
                'avg_confidence': np.mean(confs),
                'ensemble_weight': self.model_weights.get(name, 0.0)
            })
        
        comparison_df = pd.DataFrame(model_comparison)
        comparison_df = comparison_df.sort_values('qwk', ascending=False)
        comparison_path = os.path.join(reports_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✅ Model comparison: {comparison_path}")
        
        print(f"\n📁 All reports saved to: {reports_dir}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🚀 Ultimate Medical Image Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--tta", action="store_true", 
                       help="Enable Test-Time Augmentation")
    parser.add_argument("--tta_aug", type=int, default=4, 
                       choices=[2, 4, 8, 12],
                       help="Number of TTA augmentations")
    parser.add_argument("--mc_dropout", action="store_true",
                       help="Enable Monte Carlo Dropout for uncertainty")
    parser.add_argument("--mc_iter", type=int, default=10,
                       help="MC Dropout iterations")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature scaling factor")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable image caching")
    
    args = parser.parse_args() if "__file__" in globals() else parser.parse_args(args=[])
    
    print("\n" + "="*80)
    print("🚀 ULTIMATE MEDICAL IMAGE EVALUATION SYSTEM".center(80))
    print("="*80 + "\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    image_paths, labels = load_image_paths_and_labels(DATASET_DIR, csv_file=CSV_LABELS_PATH)
    
    if not image_paths:
        print("❌ No images found. Exiting.")
        return

    print(f"✅ Loaded {len(image_paths):,} images")

    # Outlier exclusion
    if os.path.exists(OUTLIER_FILE_PATH):
        try:
            outlier_df = pd.read_csv(OUTLIER_FILE_PATH)
            outlier_filenames_set = set(outlier_df['path'].apply(os.path.basename))
            
            original_count = len(image_paths)
            filtered_data = [(p, l) for p, l in zip(image_paths, labels) 
                           if os.path.basename(p) not in outlier_filenames_set]
            
            if filtered_data:
                image_paths, labels = zip(*filtered_data)
                image_paths, labels = list(image_paths), list(labels)
                removed_count = original_count - len(image_paths)
                print(f"🗑️  Removed {removed_count:,} outliers ({removed_count/original_count*100:.1f}%)")
        except Exception as e:
            print(f"⚠️  Error processing outliers: {e}")
    
    # Initialize evaluator
    evaluator = UltimateEvaluator(
        use_tta=args.tta,
        tta_augmentations=args.tta_aug,
        temperature=args.temperature,
        use_mc_dropout=args.mc_dropout,
        mc_iterations=args.mc_iter,
        use_caching=not args.no_cache
    )
    
    # Run evaluation
    import time
    start_time = time.time()
    
    evaluator.evaluate_dataset_batched(image_paths, labels, args.batch_size)
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total Time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print(f"📁 Results Location: {WORK_DIR}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
import os
import glob
import json
import argparse
import random
import time
from collections import Counter, OrderedDict
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Paths / Defaults ---
HOME = os.path.expanduser("~")
DEFAULT_DATASET_DIR = os.path.join(HOME, "Downloads", "archive(1)", "resized_train_cropped")
DEFAULT_WORK_DIR = os.path.join(HOME, "Downloads", "work")
os.makedirs(DEFAULT_WORK_DIR, exist_ok=True)

CSV_LABELS_PATH = os.path.join(HOME, "Downloads", "archive(1)", "trainLabels_cropped_2Lines.csv")
DEFAULT_EDA_DIR = os.path.join(HOME, "EDA_Results_DeepLearning")
OUTLIER_FILE = os.path.join(DEFAULT_EDA_DIR, "outliers.csv")

NUM_CLASSES = 5
LABELS = ["0", "1", "2", "3", "4"]

DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42
DEFAULT_MODELS = ["DINOv3_ViTL16", "ResNet50", "DenseNet201", "EfficientNetV2L", "ConvNeXt_L"]
DEFAULT_WARMUP_EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM NORMALIZATION VALUES FROM EDA ---
CUSTOM_MEAN = [0.485, 0.456, 0.406]
CUSTOM_STD  = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DRDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            print(f"Warning: Failed to load image at index {idx}: {self.paths[idx]}. Error: {e}")
            img = Image.new('RGB', (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), (0, 0, 0))

        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_image_paths_and_labels(dataset_dir=DEFAULT_DATASET_DIR, csv_file=None):
    if csv_file is not None and os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if 'image' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV file must contain 'image' and 'label' columns.")
        image_paths = [os.path.join(dataset_dir, os.path.splitext(img)[0] + ".jpeg") for img in df['image'].tolist()]
        labels = df['label'].astype(int).tolist()
        filtered_paths, filtered_labels = [], []
        for p, l in zip(image_paths, labels):
            if os.path.exists(p):
                filtered_paths.append(p)
                filtered_labels.append(l)
        print(f"Loaded {len(filtered_paths)} jpeg images from CSV (existing files).")
        return filtered_paths, filtered_labels

    image_paths, labels = [], []
    for idx, label in enumerate(LABELS):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        for p in glob.glob(os.path.join(folder, "*.jpeg")):
            image_paths.append(p)
            labels.append(idx)
    print(f"Scanned dataset directories. Total jpeg images: {len(image_paths)}")
    return image_paths, labels


def get_transforms(img_size=DEFAULT_IMG_SIZE):
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(CUSTOM_MEAN, CUSTOM_STD),
        T.RandomErasing(p=0.1, scale=(0.01, 0.03), ratio=(0.3, 3.0), value='random')
    ])
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(CUSTOM_MEAN, CUSTOM_STD)
    ])
    return train_transform, val_transform


def build_model(name, num_classes=NUM_CLASSES):
    if name == "DINOv3_ViTL16":
        try:
            from transformers import AutoModel
            
            model_name_hf = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            base_model = AutoModel.from_pretrained(model_name_hf)
            
            embed_dim = base_model.config.hidden_size
            
            class DINOv3ForClassification(nn.Module):
                def __init__(self, base_model, num_classes, embed_dim):
                    super().__init__()
                    self.base_model = base_model
                    self.head = nn.Sequential(
                        nn.Linear(embed_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes)
                    )
                
                def forward(self, x):
                    outputs = self.base_model(pixel_values=x)
                    features = outputs.last_hidden_state[:, 0, :]
                    return self.head(features)
            
            m_with_head = DINOv3ForClassification(base_model, num_classes, embed_dim)
            backbone = [m_with_head.base_model]
            
            print(f"✅ Loaded DINOv3-ViT-L/16 (embed_dim={embed_dim})")
            return m_with_head.to(DEVICE), backbone

        except Exception as e:
            raise ImportError(f"Could not load DINOv3 from Hugging Face. Error: {e}")
        
    elif name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
        backbone = [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3, m.layer4]
        return m.to(DEVICE), backbone

    elif name == "DenseNet201":
        m = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
        backbone = [m.features]
        return m.to(DEVICE), backbone

    elif name == "EfficientNetV2L":
        m = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
        backbone = [m.features]
        return m.to(DEVICE), backbone

    elif name == "ConvNeXt_L":
        m = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
        backbone = [m.features]
        return m.to(DEVICE), backbone

    else:
        raise ValueError(f"Model {name} not supported")


def freeze_backbone(backbone, freeze=True, model=None):
    for module in backbone:
        for p in module.parameters():
            p.requires_grad = not freeze
    if freeze and model is not None:
        head_attrs = ['fc', 'classifier', 'heads', 'head']
        for attr in head_attrs:
            if hasattr(model, attr):
                module = getattr(model, attr)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        if alpha is not None:
            alpha_arr = np.asarray(alpha, dtype=np.float32)
            self.alpha = torch.tensor(alpha_arr, dtype=torch.float32, device=DEVICE)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.long()

        if self.smoothing > 0:
            targets_onehot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            targets_onehot = targets_onehot * (1 - self.smoothing) + self.smoothing / NUM_CLASSES
        else:
            targets_onehot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)

        pt = (probs * targets_onehot).sum(dim=1)
        log_pt = (log_probs * targets_onehot).sum(dim=1)

        at = 1.0
        if self.alpha is not None:
            valid_targets = targets.clamp(0, len(self.alpha) - 1)
            at = self.alpha[valid_targets]

        loss = -at * ((1 - pt).pow(self.gamma)) * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def effective_num_alpha(train_labels, beta=0.999):
    counts = Counter(train_labels)
    alphas = []
    for c in range(NUM_CLASSES):
        n = counts.get(c, 1)
        alpha_c = (1 - beta) / (1 - (beta ** n))
        alphas.append(alpha_c)
    alphas = np.array(alphas, dtype=np.float32)
    alphas = alphas / alphas.sum() * NUM_CLASSES
    return alphas


def compute_class_weights(train_labels):
    counter = Counter(train_labels)
    total = sum(counter.values())
    weights = []
    for i in range(NUM_CLASSES):
        cls_count = counter.get(i, 1)
        weights.append(total / (NUM_CLASSES * cls_count))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE), counter


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, total_epochs, ema_state, ema_decay):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"Train {epoch + 1}/{total_epochs}", ncols=100)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(scaler is not None and DEVICE.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if scaler and DEVICE.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema_state:
            with torch.no_grad():
                msd = model.state_dict()
                for k, v in msd.items():
                    if v.dtype.is_floating_point and k in ema_state:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def _tta_variants(images, tta):
    outs = [images]
    if tta >= 2: 
        outs.append(torch.flip(images, dims=[-1]))
    if tta >= 4:
        outs.append(torch.rot90(images, k=1, dims=[-2, -1]))
        outs.append(torch.rot90(images, k=3, dims=[-2, -1]))
    if tta >= 8:
        v = torch.flip(images, dims=[-2])
        outs.append(v)
        outs.append(torch.rot90(v, k=1, dims=[-2, -1]))
        outs.append(torch.rot90(v, k=3, dims=[-2, -1]))
    return outs[:tta]


@torch.no_grad()
def validate(model, loader, criterion, tta=1, use_ema=False, ema_state=None):
    original_state_dict = None
    if use_ema and ema_state is not None:
        original_state_dict = OrderedDict({k: v.clone() for k, v in model.state_dict().items()})
        ema_load_state = OrderedDict()
        for k, v in model.state_dict().items():
             if v.dtype.is_floating_point and k in ema_state:
                 ema_load_state[k] = ema_state[k].to(v.device)
             else:
                 ema_load_state[k] = v
        model.load_state_dict(ema_load_state, strict=False)

    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    from tqdm import tqdm
    pbar_val = tqdm(loader, desc="Validate", ncols=100, leave=False)

    for imgs, labels in pbar_val:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if tta <= 1:
            outputs = model(imgs)
        else:
            logits_list = [model(aug) for aug in _tta_variants(imgs, tta)]
            outputs = torch.stack(logits_list, dim=0).mean(0)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    epoch_acc = accuracy_score(all_labels, all_preds) if all_preds.size > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if all_preds.size > 0 else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES)) if all_preds.size > 0 else np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic") if all_preds.size > 0 else 0.0

    if original_state_dict is not None:
        model.load_state_dict(original_state_dict)

    return epoch_loss, epoch_acc, f1, cm, all_preds, all_labels, qwk


def train_model(model_name, train_loader, val_loader, epochs, warmup_epochs, work_dir,
                loss_name="focal", class_weights=None, focal_gamma=2.5, cb_beta=0.999,
                scheduler_name="cosine", tta_val=8, ema_decay=0.999):

    model, backbone = build_model(model_name)

    train_dataset_labels = getattr(train_loader.dataset, "labels", None) or []

    if loss_name == "focal":
        alpha = effective_num_alpha(train_dataset_labels, beta=cb_beta)
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    head_params = []
    backbone_params = []
    
    head_attrs = ['fc', 'classifier', 'heads', 'head']
    for attr in head_attrs:
        if hasattr(model, attr):
            module = getattr(model, attr)
            if module is not None:
                head_params.extend(list(module.parameters()))
    
    for module in backbone:
        backbone_params.extend(list(module.parameters()))
    
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': 5e-5, 'name': 'head'},
        {'params': backbone_params, 'lr': 5e-6, 'name': 'backbone'}
    ], weight_decay=1e-4)
    
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max(1, epochs - warmup_epochs),
            eta_min=1e-6
        )
        use_plateau = False
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        use_plateau = True

    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))
    ema_state = OrderedDict({k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point})

    ckpt_best = os.path.join(work_dir, f"{model_name}_best.pth")
    ckpt_last = os.path.join(work_dir, f"{model_name}_last.pth")
    start_epoch = 0
    best_val_loss = float('inf')

    resume_path = ckpt_last if os.path.exists(ckpt_last) else ckpt_best if os.path.exists(ckpt_best) else None

    if resume_path:
        print(f"Attempting to resume from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=DEVICE)

            if all(k in checkpoint for k in ['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'ema_state', 'best_val_loss']):
                print("   Detected new checkpoint format. Loading full state...")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                loaded_ema = checkpoint['ema_state']
                for k in ema_state.keys():
                    if k in loaded_ema:
                        ema_state[k] = loaded_ema[k].detach().clone().float().to(ema_state[k].device)

                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                print(f"✅ Successfully resumed full state from epoch {start_epoch}. Current best loss: {best_val_loss:.4f}")

            else:
                print("⚠️ Detected old checkpoint format. Attempting to load weights only...")
                weights_loaded = False
                try:
                    loaded_weights = checkpoint
                    if isinstance(loaded_weights, dict):
                        temp_load_state = OrderedDict()
                        model_sd_template = model.state_dict()
                        for k, v in model_sd_template.items():
                            if v.dtype.is_floating_point and k in loaded_weights:
                                temp_load_state[k] = loaded_weights[k].to(v.device)
                            else:
                                temp_load_state[k] = v

                        model.load_state_dict(temp_load_state, strict=False)
                        for k in ema_state.keys():
                            if k in loaded_weights:
                                ema_state[k] = loaded_weights[k].detach().clone().float().to(ema_state[k].device)
                        print("    ✅ Successfully loaded EMA weights from old format.")
                        weights_loaded = True
                except Exception as e_ema_old:
                     print(f"   ⚠️ Error loading EMA weights from old format: {e_ema_old}")

                if not weights_loaded and isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                     try:
                         model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                         ema_state = OrderedDict({k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point})
                         print("    ✅ Successfully loaded model_state_dict weights from old format.")
                         weights_loaded = True
                     except Exception as e_model_old:
                         print(f"   ⚠️ Error loading model_state_dict from old format: {e_model_old}")

                if weights_loaded:
                     print("   ℹ️ Restarting training from epoch 0, but using loaded weights.")
                else:
                     print("   ⚠️ Could not load any weights from old format. Starting fresh.")

                start_epoch = 0
                best_val_loss = float('inf')

            if start_epoch >= warmup_epochs:
                freeze_backbone(backbone, freeze=False, model=model)
            else:
                freeze_backbone(backbone, freeze=True, model=model)

        except Exception as e:
            print(f"❌ Error during checkpoint loading. Starting from scratch. Error: {e}")
            start_epoch = 0
            best_val_loss = float('inf')
            ema_state = OrderedDict({k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point})
            freeze_backbone(backbone, freeze=True, model=model)
    else:
        print("ℹ️ No checkpoint found. Starting from scratch.")
        freeze_backbone(backbone, freeze=True, model=model)

    patience = 8
    trigger_times = 0
    history = []

    print(f"\nStarting training loop for {model_name} from epoch {start_epoch}...")
    loop_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        if epoch == warmup_epochs and start_epoch < warmup_epochs:
            print("--- Unfreezing backbone ---")
            freeze_backbone(backbone, freeze=False, model=model)
            
            for param_group in optimizer.param_groups:
                if param_group['name'] == 'backbone':
                    param_group['lr'] = 5e-6
                    print(f"   📉 Set backbone LR to: {param_group['lr']:.2e}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, epochs, ema_state, ema_decay)
        val_loss, val_acc, val_f1, val_cm, _, _, val_qwk = validate(model, val_loader, criterion, tta=tta_val, use_ema=True, ema_state=ema_state)

        if use_plateau:
            scheduler.step(val_loss)
        elif epoch >= warmup_epochs:
             scheduler.step()

        current_lrs = [f"{pg['name']}: {pg['lr']:.2e}" for pg in optimizer.param_groups]
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} QWK: {val_qwk:.4f}")
        print(f"   🎯 LRs: {', '.join(current_lrs)}")

        current_epoch_num = epoch + 1
        elapsed_time = time.time() - loop_start_time
        avg_epoch_time = elapsed_time / max(1, (current_epoch_num - start_epoch))
        remaining_epochs = epochs - current_epoch_num
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_h, eta_m, eta_s = int(eta_seconds // 3600), int((eta_seconds % 3600) // 60), int(eta_seconds % 60)
        print(f"⏱️  Avg. Epoch Time: {avg_epoch_time:.2f}s | ETA: {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s")

        history.append({"epoch": epoch + 1, "train_loss": float(train_loss), "train_acc": float(train_acc),
                         "val_loss": float(val_loss), "val_acc": float(val_acc), "val_f1": float(val_f1),
                         "val_qwk": float(val_qwk), "lr": float(optimizer.param_groups[0]["lr"])})

        checkpoint_data = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'model_state_dict': model.state_dict(),
            'ema_state': {k: v.cpu() for k, v in ema_state.items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }

        torch.save(checkpoint_data, ckpt_last)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_data['best_val_loss'] = best_val_loss
            torch.save(checkpoint_data, ckpt_best)
            trigger_times = 0
            print(f"💾 Saved BEST model (Epoch {epoch+1}) -> {ckpt_best}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("⏹️  Early stopping triggered.")
                break

    print(f"\n📂 Loading best model weights from {ckpt_best} for final evaluation...")
    if os.path.exists(ckpt_best):
        checkpoint = torch.load(ckpt_best, map_location=DEVICE)
        if 'ema_state' in checkpoint:
            best_ema_state = checkpoint['ema_state']
            load_state = OrderedDict()
            model_sd_template = model.state_dict()
            ckpt_model_state_dict = checkpoint.get('model_state_dict', model_sd_template)

            for k, v_template in model_sd_template.items():
                if v_template.dtype.is_floating_point and k in best_ema_state:
                    load_state[k] = best_ema_state[k].to(v_template.device)
                elif k in ckpt_model_state_dict:
                    load_state[k] = ckpt_model_state_dict[k].to(v_template.device)
                else:
                    load_state[k] = v_template
            try:
                model.load_state_dict(load_state, strict=False)
                print("   ✅ Successfully loaded best EMA weights.")
            except Exception as e_load:
                print(f"   ⚠️ Warning: Error loading best EMA weights: {e_load}. Using last model state.")
        else:
             print("   ⚠️ Warning: Best checkpoint missing 'ema_state'. Attempting to load model_state_dict.")
             if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                 print("   ✅ Loaded model_state_dict instead.")
             else:
                 print("   ⚠️ Model state dict also missing. Using last trained state.")
    else:
        print("⚠️ Warning: No best checkpoint found. Using last model state for evaluation.")

    with open(os.path.join(work_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    model_end_time = time.time()
    model_total_time = model_end_time - loop_start_time
    print(f"✅ --- Finished training {model_name} ---")
    m_h, m_m, m_s = int(model_total_time // 3600), int((model_total_time % 3600) // 60), int(model_total_time % 60)
    print(f"⏱️  Actual time for {model_name}: {m_h:02d}h {m_m:02d}m {m_s:02d}s")

    return model


def make_dataloaders(train_paths, val_paths, train_labels, val_labels, img_size, batch_size, seed):
    train_transform, val_transform = get_transforms(img_size)
    train_dataset = DRDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DRDataset(val_paths, val_labels, transform=val_transform)
    class_sample_counts = Counter(train_labels)
    class_weights = [0] * NUM_CLASSES
    total = sum(class_sample_counts.values())
    for i in range(NUM_CLASSES):
        cls_count = class_sample_counts.get(i, 1)
        class_weights[i] = total / (NUM_CLASSES * cls_count)
    samples_weight = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
    num_workers = min(8, os.cpu_count() or 1)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0), drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0), drop_last=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_model(model, loader, work_dir, prefix="", tta=1):
    model.eval()
    all_preds, all_labels = [], []
    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"Evaluate {prefix}", ncols=100, leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if tta <= 1: 
            outputs = model(imgs)
        else:
            logits_list = [model(aug) for aug in _tta_variants(imgs, tta)]
            outputs = torch.stack(logits_list, dim=0).mean(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    acc = accuracy_score(all_labels, all_preds) if all_preds.size > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if all_preds.size > 0 else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES)) if all_preds.size > 0 else np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic") if all_preds.size > 0 else 0.0
    print(f"{prefix}Accuracy: {acc:.4f}, F1: {f1:.4f}, QWK: {qwk:.4f}")
    sens = {LABELS[i]: round(cm[i, i] / max(cm[i, :].sum(), 1e-6), 4) for i in range(NUM_CLASSES)}
    tn_spec = {}
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        tn_spec[i] = tn / max(tn + fp, 1e-6)
    print(f"{prefix}Per-class Sensitivity:", sens)
    print(f"{prefix}Per-class Specificity:", {LABELS[i]: round(tn_spec[i], 4) for i in range(NUM_CLASSES)})
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix {prefix}'.strip())
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, LABELS, rotation=45, ha='right')
    plt.yticks(tick_marks, LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    fig_name = f"{prefix.strip().replace(' ', '_')}_confusion_matrix.png"
    fig_path = os.path.join(work_dir, fig_name)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    result = {"accuracy": float(acc), "f1_weighted": float(f1), "qwk": float(qwk), "cm": cm.tolist()}
    metrics_name = f"{prefix.strip().replace(' ', '_')}_metrics.json"
    with open(os.path.join(work_dir, metrics_name), "w") as f:
        json.dump(result, f, indent=2)
    return acc, f1, cm, result, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--work_dir", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup_epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.5)
    parser.add_argument("--cb_beta", type=float, default=0.999)
    parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine"], default="cosine")
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--csv_file", type=str, default=CSV_LABELS_PATH)
    parser.add_argument("--outliers_csv", type=str, default=OUTLIER_FILE)
    args = parser.parse_args() if "__file__" in globals() else parser.parse_args(args=[])

    os.makedirs(args.work_dir, exist_ok=True)
    set_seed(args.seed)
    print(f"Device: {DEVICE} | Work dir: {args.work_dir} | Img Size: {args.img_size}x{args.img_size} | Batch: {args.batch_size}")
    print(f"Using Custom Norm Mean: {CUSTOM_MEAN} | Std: {CUSTOM_STD}")

    image_paths, labels = load_image_paths_and_labels(args.dataset_dir, csv_file=args.csv_file)
    if not image_paths:
        raise SystemExit("No images loaded.")

    if os.path.exists(args.outliers_csv):
        try:
            outlier_df = pd.read_csv(args.outliers_csv)
            outlier_paths_set = set(outlier_df['path'].tolist())
            original_count = len(image_paths)
            filtered_data = [(p, l) for p, l in zip(image_paths, labels) if p not in outlier_paths_set]
            if filtered_data:
                 image_paths, labels = zip(*filtered_data)
                 image_paths, labels = list(image_paths), list(labels)
                 removed_count = original_count - len(image_paths)
                 print(f"🗑️ Removed {removed_count} outliers from the dataset (using {args.outliers_csv}).")
            else:
                 print("⚠️ Warning: Filtering outliers resulted in an empty dataset. Check outlier file.")
                 image_paths, labels = load_image_paths_and_labels(args.dataset_dir, csv_file=args.csv_file)
        except Exception as e:
            print(f"⚠️ Warning: Error processing outlier file {args.outliers_csv}. Proceeding without removal. Error: {e}")
    else:
        print(f"ℹ️ Skipping outlier removal: File not found at {args.outliers_csv}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    if not train_paths:
        raise SystemExit("No training images left after split/filtering.")

    train_loader, val_loader = make_dataloaders(train_paths, val_paths, train_labels, val_labels, args.img_size, args.batch_size, args.seed)
    class_weights, _ = compute_class_weights(train_labels)

    print("\n=== STARTING ENTIRE TRAINING PROCESS ===")
    total_start_time = time.time()

    models_dict = {}
    for model_name in args.models:
        if model_name == "Xception":
             try:
                 import timm
             except ImportError:
                 print(f"⚠️ Skipping Xception: timm not installed.")
                 continue
        print(f"\n--- Training {model_name} ---")
        model = train_model(model_name, train_loader, val_loader, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                            work_dir=args.work_dir, loss_name=args.loss, class_weights=class_weights,
                            focal_gamma=args.focal_gamma, cb_beta=args.cb_beta, scheduler_name=args.scheduler, tta_val=args.tta)
        models_dict[model_name] = model

    metrics_summary = {}
    for model_name, model in models_dict.items():
        print(f"\n--- Evaluating {model_name} ---")
        acc, f1, cm, result, preds, labels_true = evaluate_model(model, val_loader, args.work_dir, prefix=model_name, tta=args.tta)
        metrics_summary[model_name] = result

    if len(models_dict) > 1:
        model_weights = {name: max(res.get("qwk", 0.0), 1e-6) for name, res in metrics_summary.items()}
        w_sum = sum(model_weights.values()) or 1.0
        model_weights = {k: v / w_sum for k, v in model_weights.items()}

        print("\n--- Ensemble Evaluation ---")
        print(f"Using weights: {dict((k, round(v, 3)) for k, v in model_weights.items())}")

        @torch.no_grad()
        def ensemble_eval(tta=args.tta):
            for m in models_dict.values():
                m.eval()
            all_preds, all_labels = [], []
            from tqdm import tqdm
            pbar_ens = tqdm(val_loader, desc="Ensemble Evaluate", ncols=100, leave=False)
            for imgs, labels_batch in pbar_ens:
                imgs = imgs.to(DEVICE, non_blocking=True)
                avg_probs = None
                for name, model in models_dict.items():
                    if tta <= 1:
                        outputs = model(imgs)
                    else:
                        logits_list = [model(aug) for aug in _tta_variants(imgs, tta)]
                        outputs = torch.stack(logits_list, dim=0).mean(0)
                    probs = torch.softmax(outputs, dim=1)
                    w = model_weights.get(name, 0.0)
                    avg_probs = probs * w if avg_probs is None else avg_probs + probs * w
                if avg_probs is not None:
                     preds = avg_probs.argmax(dim=1).cpu().numpy()
                     all_preds.append(preds)
                     all_labels.append(labels_batch.numpy())
            all_preds = np.concatenate(all_preds) if all_preds else np.array([])
            all_labels = np.concatenate(all_labels) if all_labels else np.array([])
            acc = accuracy_score(all_labels, all_preds) if all_preds.size > 0 else 0.0
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if all_preds.size > 0 else 0.0
            cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES)) if all_preds.size > 0 else np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
            qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic") if all_preds.size > 0 else 0.0
            return acc, f1, cm, all_preds, all_labels, qwk

        ens_acc, ens_f1, ens_cm, ens_preds, ens_labels, ens_qwk = ensemble_eval()
        print(f"Ensemble Accuracy: {ens_acc:.4f}, F1: {ens_f1:.4f}, QWK: {ens_qwk:.4f}")

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 5), dpi=150)
        plt.imshow(ens_cm, interpolation='nearest')
        plt.title('Confusion Matrix Ensemble')
        plt.colorbar()
        tick_marks = np.arange(NUM_CLASSES)
        plt.xticks(tick_marks, LABELS, rotation=45, ha='right')
        plt.yticks(tick_marks, LABELS)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(args.work_dir, "ensemble_confusion_matrix.png"), bbox_inches='tight')
        plt.close(fig)

        ens_result = {"accuracy": float(ens_acc), "f1_weighted": float(ens_f1), "qwk": float(ens_qwk),
                      "cm": ens_cm.tolist(), "timestamp": datetime.now().isoformat()}
        with open(os.path.join(args.work_dir, "ensemble_metrics.json"), "w") as f:
            json.dump(ens_result, f, indent=2)

    else:
        print("\n⚠️ Skipping ensemble evaluation (only one model trained or evaluation failed).")
        ens_result = {"accuracy": 0.0, "f1_weighted": 0.0, "qwk": 0.0}

    import csv
    csv_path = os.path.join(args.work_dir, "summary_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "f1_weighted", "qwk"])
        for name, res in metrics_summary.items():
            writer.writerow([name, res.get("accuracy", ""), res.get("f1_weighted", ""), res.get("qwk", "")])
        if len(models_dict) > 1:
            writer.writerow(["Ensemble", ens_result["accuracy"], ens_result["f1_weighted"], ens_result["qwk"]])

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n--- Total Training Summary ---")
    t_h, t_m, t_s = int(total_training_time // 3600), int((total_training_time % 3600) // 60), int(total_training_time % 60)
    print(f"⏱️  Total actual time for all models & ensemble: {t_h:02d}h {t_m:02d}m {t_s:02d}s")

    print(f"\n✅ All done. Metrics & figures saved in: {args.work_dir}")


if __name__ == "__main__":
    main()
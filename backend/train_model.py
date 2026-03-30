"""
train_model.py — Train skin lesion classifier on HAM10000

Fine-tunes EfficientNet-B0 (pretrained on ImageNet, downloaded automatically by timm).
Handles class imbalance with weighted sampling.
Saves best checkpoint to models/best_skin_model.pth

Usage:
  python train_model.py

On CPU: ~2-3 hours for 15 epochs
On GPU: ~20-30 minutes
"""

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
from sklearn.metrics import classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config ──
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            for img in (Path(root_dir) / cls).glob("*.jpg"):
                self.samples.append((str(img), self.class_to_idx[cls]))

        print(f"  Loaded {len(self.samples)} images, {len(self.class_names)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


# Augmentations
train_aug = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def main():
    print("🏥 MedScan AI — Model Training")
    print(f"Device: {DEVICE}")
    print("=" * 50)

    if not (PROCESSED_DIR / "train").exists():
        print("❌ No training data. Run setup_data.py first.")
        return

    # Load class info
    with open(PROCESSED_DIR / "class_info.json") as f:
        class_info = json.load(f)
    class_names = class_info["class_names"]
    num_classes = class_info["num_classes"]

    # Datasets
    print("\nLoading datasets...")
    train_ds = SkinDataset(PROCESSED_DIR / "train", train_aug)
    val_ds = SkinDataset(PROCESSED_DIR / "val", val_aug)

    # Weighted sampler for class imbalance
    labels = [l for _, l in train_ds.samples]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model — EfficientNet-B0 pretrained on ImageNet (downloaded automatically by timm)
    print("\nLoading EfficientNet-B0 (pretrained weights download automatically)...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss with class weights
    class_weights = torch.tensor(
        [len(labels) / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    best_acc = 0.0
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print("-" * 50)

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, labels_batch in train_loader:
            imgs, labels_batch = imgs.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels_batch).sum().item()
            train_total += imgs.size(0)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs, labels_batch = imgs.to(DEVICE), labels_batch.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels_batch)
                val_loss += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        scheduler.step()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} ({elapsed:.0f}s) | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%", end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE,
            }, MODEL_DIR / "best_skin_model.pth")
            print(f" ✅ BEST", end="")
        print()

    print(f"\n{'=' * 50}")
    print(f"✅ Done! Best val accuracy: {best_acc:.1f}%")
    print(f"Model saved: {MODEL_DIR / 'best_skin_model.pth'}")
    print(f"\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()

"""
setup_data.py — Download HAM10000 dataset from Kaggle

Prerequisites:
  1. Kaggle account (https://www.kaggle.com)
  2. API token: Go to https://www.kaggle.com/settings → API → "Create New Token"
  3. Either:
     a) Place downloaded kaggle.json in ~/.kaggle/kaggle.json
     b) Or set KAGGLE_USERNAME and KAGGLE_KEY in your .env file

Usage:
  python setup_data.py
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

DATA_DIR = Path("data")
HAM_DIR = DATA_DIR / "ham10000"
PROCESSED_DIR = DATA_DIR / "processed"


def setup_kaggle_credentials():
    """Set up Kaggle credentials from .env or existing kaggle.json."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Check if kaggle.json already exists
    if kaggle_json.exists():
        print("✅ Found existing ~/.kaggle/kaggle.json")
        return True

    # Try from .env
    username = os.getenv("KAGGLE_USERNAME", "")
    key = os.getenv("KAGGLE_KEY", "")

    if username and key and username != "your_kaggle_username":
        kaggle_dir.mkdir(exist_ok=True)
        creds = {"username": username, "key": key}
        with open(kaggle_json, "w") as f:
            json.dump(creds, f)
        os.chmod(kaggle_json, 0o600)
        print(f"✅ Created ~/.kaggle/kaggle.json from .env (user: {username})")
        return True

    print("❌ No Kaggle credentials found!")
    print()
    print("Set up in ONE of these ways:")
    print()
    print("  Option A: Download kaggle.json")
    print("    1. Go to https://www.kaggle.com/settings")
    print("    2. Scroll to 'API' section")
    print("    3. Click 'Create New Token' — downloads kaggle.json")
    print(f"    4. Move it to: {kaggle_json}")
    print()
    print("  Option B: Add to .env file")
    print("    1. Open .env in your project folder")
    print("    2. Set KAGGLE_USERNAME=your_username")
    print("    3. Set KAGGLE_KEY=your_api_key")
    print()
    return False


def download_ham10000():
    """Download HAM10000 from Kaggle."""
    print("=" * 60)
    print("DOWNLOADING HAM10000 DATASET")
    print("=" * 60)

    HAM_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        print("\nDownloading from Kaggle (this may take 2-5 minutes)...")
        print("Dataset: kmader/skin-cancer-mnist-ham10000")
        print(f"Destination: {HAM_DIR}")
        print()

        api.dataset_download_files(
            "kmader/skin-cancer-mnist-ham10000",
            path=str(HAM_DIR),
            unzip=True,
        )
        print("✅ Download complete!")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTry manually:")
        print("  1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("  2. Click 'Download' button")
        print(f"  3. Extract the zip into: {HAM_DIR}/")
        return False


def organize_into_splits():
    """Organize downloaded images into train/val splits by class."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("ORGANIZING DATASET INTO TRAIN/VAL SPLITS")
    print("=" * 60)

    # Find metadata CSV
    metadata_path = None
    for f in HAM_DIR.rglob("*metadata*"):
        if f.suffix == ".csv":
            metadata_path = f
            break

    if not metadata_path:
        print("❌ Cannot find HAM10000_metadata.csv")
        print(f"Files in {HAM_DIR}:")
        for f in sorted(HAM_DIR.rglob("*"))[:20]:
            print(f"  {f.relative_to(HAM_DIR)}")
        return False

    print(f"Found metadata: {metadata_path}")
    df = pd.read_csv(metadata_path)
    print(f"Total entries: {len(df)}")
    print(f"\nClass distribution:")
    print(df["dx"].value_counts().to_string())

    # Find all images
    image_files = {}
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in HAM_DIR.rglob(ext):
            image_files[f.stem] = f

    print(f"\nFound {len(image_files)} images")

    # Match images to metadata
    df["image_path"] = df["image_id"].map(image_files)
    matched = df.dropna(subset=["image_path"])
    print(f"Matched: {len(matched)} images to metadata")

    if len(matched) == 0:
        print("❌ No images matched!")
        return False

    # 80/20 stratified split
    train_df, val_df = train_test_split(
        matched, test_size=0.2, stratify=matched["dx"], random_state=42
    )

    # Copy images into class folders
    class_names = sorted(matched["dx"].unique())
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for cls in class_names:
            (PROCESSED_DIR / split_name / cls).mkdir(parents=True, exist_ok=True)

        for _, row in split_df.iterrows():
            src = Path(row["image_path"])
            dst = PROCESSED_DIR / split_name / row["dx"] / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    # Save metadata
    train_df.to_csv(PROCESSED_DIR / "train_metadata.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val_metadata.csv", index=False)

    # Save class info
    class_info = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "class_descriptions": {
            "akiec": "Actinic Keratoses (Pre-cancerous)",
            "bcc": "Basal Cell Carcinoma",
            "bkl": "Benign Keratosis",
            "df": "Dermatofibroma",
            "mel": "Melanoma",
            "nv": "Melanocytic Nevi (Moles)",
            "vasc": "Vascular Lesions",
        },
    }
    with open(PROCESSED_DIR / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)

    print(f"\n✅ Dataset organized!")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Classes: {class_names}")
    print(f"  Output: {PROCESSED_DIR}/")

    return True


if __name__ == "__main__":
    print("🏥 MedScan AI — Dataset Setup")
    print()

    # Step 1: Check/setup Kaggle credentials
    if not setup_kaggle_credentials():
        sys.exit(1)

    # Step 2: Download if not already present
    if not HAM_DIR.exists() or not any(HAM_DIR.glob("*")):
        if not download_ham10000():
            sys.exit(1)
    else:
        print(f"\n📁 Dataset already exists at {HAM_DIR}, skipping download")

    # Step 3: Organize into train/val
    if organize_into_splits():
        print("\n" + "=" * 60)
        print("✅ ALL DONE! Next step:")
        print("   python train_model.py")
        print("=" * 60)
    else:
        print("\n❌ Failed to organize dataset")
        sys.exit(1)

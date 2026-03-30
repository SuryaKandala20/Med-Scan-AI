"""
modules/skin_classifier.py — Skin lesion classification (inference)

Loads the trained EfficientNet-B0 and predicts top-3 conditions.
"""

import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_PATH = Path("models/best_skin_model.pth")

CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses",
        "full_name": "Actinic Keratoses (Pre-cancerous)",
        "description": "Rough, scaly patches from sun exposure. Pre-cancerous — can progress to squamous cell carcinoma.",
        "triage": "Routine",
        "specialist": "Dermatologist",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "full_name": "Basal Cell Carcinoma",
        "description": "Most common skin cancer. Slow-growing, rarely spreads, but needs treatment.",
        "triage": "Same-day",
        "specialist": "Dermatologist",
    },
    "bkl": {
        "name": "Benign Keratosis",
        "full_name": "Benign Keratosis (Seborrheic Keratosis)",
        "description": "Common non-cancerous growths. Generally harmless.",
        "triage": "Routine",
        "specialist": "Dermatologist",
    },
    "df": {
        "name": "Dermatofibroma",
        "full_name": "Dermatofibroma",
        "description": "Benign firm skin nodule. Usually harmless, no treatment needed.",
        "triage": "Routine",
        "specialist": "Dermatologist",
    },
    "mel": {
        "name": "Melanoma",
        "full_name": "Melanoma (Skin Cancer)",
        "description": "Most serious skin cancer. Early detection is critical. Check ABCDE rule.",
        "triage": "Urgent",
        "specialist": "Dermatologist / Oncologist",
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "full_name": "Melanocytic Nevi (Common Mole)",
        "description": "Common benign moles. Monitor for changes using ABCDE rule.",
        "triage": "Routine",
        "specialist": "Dermatologist",
    },
    "vasc": {
        "name": "Vascular Lesions",
        "full_name": "Vascular Lesions (Angiomas)",
        "description": "Blood vessel abnormalities. Most are benign (cherry angiomas, etc.).",
        "triage": "Routine",
        "specialist": "Dermatologist",
    },
}


class SkinClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        self.temperature = 1.5

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.class_names = checkpoint["class_names"]
            self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=checkpoint["num_classes"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Skin model loaded (val acc: {checkpoint.get('val_acc', '?'):.1f}%)")

    def predict(self, image: Image.Image, top_k: int = 3) -> list:
        if self.model is None:
            return None  # No model — caller handles this

        img = np.array(image.convert("RGB"))
        tensor = self.transform(image=img)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits / self.temperature, dim=1).squeeze().cpu().numpy()

        top_idx = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_idx:
            cls = self.class_names[idx]
            info = CLASS_INFO.get(cls, {})
            results.append({
                "class_id": cls,
                "class_name": info.get("name", cls),
                "full_name": info.get("full_name", cls),
                "confidence": float(probs[idx]),
                "description": info.get("description", ""),
                "triage": info.get("triage", "Routine"),
                "specialist": info.get("specialist", "Dermatologist"),
            })
        return results

    @property
    def is_loaded(self):
        return self.model is not None

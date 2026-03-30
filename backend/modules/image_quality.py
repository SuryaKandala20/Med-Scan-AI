"""
modules/image_quality.py — Image quality gate for uploaded photos
"""

import numpy as np
from PIL import Image
import cv2


class ImageQualityChecker:
    def check(self, image: Image.Image) -> dict:
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape[:2]
        issues = []

        # Resolution
        if min(h, w) < 100:
            issues.append("Image resolution too low. Minimum 100px recommended.")

        # Blur
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 80:
            issues.append(f"Image appears blurry (score: {blur:.0f}). Please retake with camera focused.")

        # Brightness
        brightness = gray.mean()
        if brightness < 40:
            issues.append("Image is too dark. Please retake in better lighting.")
        elif brightness > 240:
            issues.append("Image is overexposed. Consider retaking.")

        return {"passed": len(issues) == 0, "issues": issues}

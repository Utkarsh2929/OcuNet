#!/usr/bin/env python3
"""
Fundus Image Preprocessing Transforms for OcuNet.

Implements:
- Fundus ROI (Region of Interest) detection and cropping
- CLAHE (Contrast Limited Adaptive Histogram Equalization) normalization
- Both are implemented as torchvision-compatible transforms
"""

import numpy as np
from PIL import Image
import cv2


class FundusROICrop:
    """
    Detect and crop the circular fundus region, removing black borders.

    Fundus images often have large black borders around the circular retinal
    image. This transform detects the bright circular region and crops tightly
    around it with a small padding margin.

    This reduces shortcut learning (model using border shape to classify)
    and focuses the model on actual retinal content.
    """

    def __init__(self, padding_ratio: float = 0.02, min_radius_ratio: float = 0.2):
        """
        Args:
            padding_ratio: Extra padding around detected ROI as fraction of diameter
            min_radius_ratio: Minimum radius as fraction of image size to be valid ROI
        """
        self.padding_ratio = padding_ratio
        self.min_radius_ratio = min_radius_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        """Crop to fundus ROI."""
        img_np = np.array(img)

        # Convert to grayscale for detection
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()

        h, w = gray.shape[:2]
        min_dim = min(h, w)
        min_radius = int(min_dim * self.min_radius_ratio)

        # Method 1: Threshold + contour-based detection
        # Use Otsu's thresholding to separate fundus from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If Otsu gives bad result (e.g., too much or too little), use fixed threshold
        foreground_ratio = binary.sum() / (255 * h * w)
        if foreground_ratio < 0.1 or foreground_ratio > 0.95:
            _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find the largest contour (should be the fundus circle)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)

            # Validate: ROI should be reasonably large
            if cw >= min_radius * 2 and ch >= min_radius * 2:
                # Add padding
                pad = int(max(cw, ch) * self.padding_ratio)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)

                cropped = img_np[y1:y2, x1:x2]
                return Image.fromarray(cropped)

        # Fallback: return original image if detection fails
        return img

    def __repr__(self):
        return f"FundusROICrop(padding={self.padding_ratio}, min_radius={self.min_radius_ratio})"


class CLAHETransform:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Normalizes illumination variations across different fundus cameras
    and imaging conditions. Applied to the green channel (which has the
    best contrast for retinal structures) or to the L channel of LAB.

    This stabilizes features across different data sources (RFMiD vs Phase 1).
    """

    def __init__(
            self,
            clip_limit: float = 2.0,
            tile_grid_size: int = 8,
            channel: str = "green"
    ):
        """
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            channel: "green" for green-channel CLAHE, "lab" for L-channel CLAHE
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.channel = channel

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE to image."""
        img_np = np.array(img)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size)
        )

        if len(img_np.shape) == 2:
            # Grayscale
            result = clahe.apply(img_np)
        elif self.channel == "green":
            # Apply CLAHE to green channel only
            result = img_np.copy()
            result[:, :, 1] = clahe.apply(img_np[:, :, 1])
        elif self.channel == "lab":
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Apply to all channels independently
            result = img_np.copy()
            for c in range(img_np.shape[2]):
                result[:, :, c] = clahe.apply(img_np[:, :, c])

        return Image.fromarray(result)

    def __repr__(self):
        return (f"CLAHETransform(clip={self.clip_limit}, "
                f"grid={self.tile_grid_size}, channel={self.channel})")

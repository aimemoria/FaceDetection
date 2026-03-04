#!/usr/bin/env python3
"""
C. Preprocessing and Augmentation Pipeline

This script handles:
1. Loading raw images from dataset folder
2. Face detection and cropping (optional)
3. Resizing to 96x96 grayscale
4. Data augmentation
5. Train/validation/test split
6. Saving processed data as NPZ files

Usage:
    python C_preprocess_and_augment.py --dataset_dir dataset --output_dir processed
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from typing import Tuple, List, Dict
import json

# =============================================================================
# Configuration
# =============================================================================
IMG_SIZE = 96  # Target image size (96x96 pixels)
RANDOM_SEED = 42

# =============================================================================
# Image Loading
# =============================================================================
def load_image(path: str) -> np.ndarray:
    """Load image and convert to grayscale."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def resize_image(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] range."""
    return img.astype(np.float32) / 255.0


# =============================================================================
# Face Detection (Optional - improves accuracy)
# =============================================================================
def detect_and_crop_face(img: np.ndarray, cascade_path: str = None) -> np.ndarray:
    """
    Detect face in image and crop to face region.
    Falls back to center crop if no face detected.
    
    Args:
        img: Grayscale input image
        cascade_path: Path to Haar cascade XML file
        
    Returns:
        Cropped face region or center-cropped image
    """
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        # Take largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding (20% on each side)
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img.shape[1], x + w + pad_w)
        y2 = min(img.shape[0], y + h + pad_h)
        
        return img[y1:y2, x1:x2]
    else:
        # No face detected - center crop
        h, w = img.shape
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return img[y1:y1+size, x1:x1+size]


# =============================================================================
# Data Augmentation
# =============================================================================
class Augmentor:
    """Data augmentation class for training images."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        random.seed(seed)
        np.random.seed(seed)
    
    def random_brightness(self, img: np.ndarray,
                          low: float = 0.6, high: float = 1.4) -> np.ndarray:
        """Randomly adjust brightness."""
        factor = random.uniform(low, high)
        return np.clip(img * factor, 0, 1)
    
    def random_contrast(self, img: np.ndarray,
                        low: float = 0.8, high: float = 1.2) -> np.ndarray:
        """Randomly adjust contrast."""
        factor = random.uniform(low, high)
        mean = np.mean(img)
        return np.clip((img - mean) * factor + mean, 0, 1)
    
    def random_rotation(self, img: np.ndarray,
                        max_angle: float = 20) -> np.ndarray:
        """Randomly rotate image."""
        angle = random.uniform(-max_angle, max_angle)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def random_shift(self, img: np.ndarray, 
                     max_shift: float = 0.1) -> np.ndarray:
        """Randomly shift image horizontally and vertically."""
        h, w = img.shape
        tx = random.uniform(-max_shift, max_shift) * w
        ty = random.uniform(-max_shift, max_shift) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def random_zoom(self, img: np.ndarray,
                    low: float = 0.9, high: float = 1.1) -> np.ndarray:
        """Randomly zoom in/out."""
        factor = random.uniform(low, high)
        h, w = img.shape
        new_h, new_w = int(h * factor), int(w * factor)
        
        if factor > 1:
            # Zoom in - crop center
            resized = cv2.resize(img, (new_w, new_h))
            y1 = (new_h - h) // 2
            x1 = (new_w - w) // 2
            return resized[y1:y1+h, x1:x1+w]
        else:
            # Zoom out - pad
            resized = cv2.resize(img, (new_w, new_h))
            result = np.zeros((h, w), dtype=img.dtype)
            y1 = (h - new_h) // 2
            x1 = (w - new_w) // 2
            result[y1:y1+new_h, x1:x1+new_w] = resized
            return result
    
    def horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(img, 1)
    
    def add_gaussian_noise(self, img: np.ndarray,
                           mean: float = 0, std: float = 0.03) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(mean, std, img.shape)
        return np.clip(img + noise, 0, 1)

    def gaussian_blur(self, img: np.ndarray,
                      sigma_range: tuple = (0.5, 2.0)) -> np.ndarray:
        """Apply Gaussian blur to simulate defocus or motion blur."""
        from PIL import Image as PILImage, ImageFilter
        sigma = random.uniform(*sigma_range)
        uint8_img = (img * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(uint8_img, mode='L')
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(blurred).astype(np.float32) / 255.0

    def motion_blur(self, img: np.ndarray, max_length: int = 10) -> np.ndarray:
        """Apply directional motion blur — simulates OV2640 camera shake."""
        length = random.randint(4, max_length)
        angle = random.uniform(0, 360)
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        M = cv2.getRotationMatrix2D((length / 2.0, length / 2.0), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        s = kernel.sum()
        if s > 0:
            kernel /= s
        uint8_img = (img * 255).astype(np.uint8)
        blurred = cv2.filter2D(uint8_img, -1, kernel)
        return blurred.astype(np.float32) / 255.0

    def random_brightness_extreme(self, img: np.ndarray) -> np.ndarray:
        """Apply extreme brightness: very dark (shadows) or very bright (backlight)."""
        if random.random() > 0.5:
            factor = random.uniform(0.2, 0.5)   # deep shadow
        else:
            factor = random.uniform(1.5, 2.0)   # harsh backlight
        return np.clip(img * factor, 0, 1)

    def simulate_backlight(self, img: np.ndarray) -> np.ndarray:
        """Radial brightness map: dark center, bright edges — simulates window backlight."""
        h, w = img.shape
        cy, cx = h / 2.0, w / 2.0
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        dist = np.clip(dist, 0, 1.0)
        center_factor = random.uniform(0.2, 0.5)
        edge_factor   = random.uniform(1.6, 2.5)
        backlight_map = center_factor + (edge_factor - center_factor) * dist
        return np.clip(img * backlight_map, 0, 1).astype(np.float32)

    def random_occlusion(self, img: np.ndarray,
                         min_frac: float = 0.10,
                         max_frac: float = 0.30) -> np.ndarray:
        """Black rectangle covering 10–30% of image — simulates partial face obstruction."""
        result = img.copy()
        h, w = img.shape
        frac = random.uniform(min_frac, max_frac)
        side = max(int((frac * h * w) ** 0.5), 8)
        y1 = random.randint(0, h - side)
        x1 = random.randint(0, w - side)
        result[y1:y1+side, x1:x1+side] = 0.0
        return result

    def histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalization — matches firmware preprocessing."""
        # Convert to uint8 for histogram
        img_u8 = (img * 255).astype(np.uint8)
        # Build histogram and CDF
        hist, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        # Normalize CDF
        cdf_norm = (cdf - cdf_min) * 255 / (cdf[-1] - cdf_min)
        cdf_norm = cdf_norm.astype(np.uint8)
        # Apply equalization
        equalized = cdf_norm[img_u8]
        return equalized.astype(np.float32) / 255.0

    def simulate_deep_shadow(self, img: np.ndarray) -> np.ndarray:
        """Simulate very dark conditions (deep shadows, dim rooms)."""
        factor = random.uniform(0.1, 0.35)  # Very dark
        darkened = np.clip(img * factor, 0, 1)
        # Add some noise that appears in low-light cameras
        noise = np.random.normal(0, 0.02, img.shape)
        return np.clip(darkened + noise, 0, 1).astype(np.float32)

    def augment(self, img: np.ndarray,
                num_augmentations: int = 8) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of an image.
        
        Args:
            img: Normalized grayscale image (0-1 range)
            num_augmentations: Number of augmented images to generate
            
        Returns:
            List of augmented images including original
        """
        augmented = [img]  # Always include original
        
        for _ in range(num_augmentations):
            aug_img = img.copy()
            
            # Random combination of augmentations
            if random.random() > 0.5:
                aug_img = self.random_brightness(aug_img)
            
            if random.random() > 0.5:
                aug_img = self.random_contrast(aug_img)
            
            if random.random() > 0.5:
                aug_img = self.random_rotation(aug_img)
            
            if random.random() > 0.5:
                aug_img = self.random_shift(aug_img)
            
            if random.random() > 0.5:
                aug_img = self.random_zoom(aug_img)
            
            if random.random() > 0.3:  # Less frequent flip
                aug_img = self.horizontal_flip(aug_img)
            
            if random.random() > 0.5:  # More frequent noise for robustness
                aug_img = self.add_gaussian_noise(aug_img)

            # Real-world degradation augmentations
            if random.random() > 0.4:   # ~60% — simulate defocus blur
                aug_img = self.gaussian_blur(aug_img)

            if random.random() > 0.7:   # ~30% — simulate camera shake / motion blur
                aug_img = self.motion_blur(aug_img)

            if random.random() > 0.55:  # ~45% — simulate shadows/backlighting
                aug_img = self.random_brightness_extreme(aug_img)

            if random.random() > 0.7:   # ~30% — simulate window backlight
                aug_img = self.simulate_backlight(aug_img)

            if random.random() > 0.55:  # ~45% — simulate partial face obstruction
                aug_img = self.random_occlusion(aug_img)

            # NEW: Apply histogram equalization (matches firmware preprocessing)
            # This is CRITICAL — the firmware applies this, so training data should too
            if random.random() > 0.3:   # ~70% — apply histogram equalization
                aug_img = self.histogram_equalization(aug_img)

            # NEW: Deep shadow simulation (very dark rooms)
            if random.random() > 0.7:   # ~30% — simulate very dark conditions
                aug_img = self.simulate_deep_shadow(aug_img)

            augmented.append(aug_img)
        
        return augmented


# =============================================================================
# Dataset Loading
# =============================================================================
def load_stage_a_data(dataset_dir: str, 
                      use_face_detection: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Stage A (person detection) data.
    
    Classes:
        0: no_person (background)
        1: person (any person)
    """
    print("\n=== Loading Stage A Data ===")
    
    stage_a_dir = Path(dataset_dir) / 'stage_a'
    
    images = []
    labels = []
    
    # Load no_person class (label = 0)
    no_person_dir = stage_a_dir / 'no_person'
    if no_person_dir.exists():
        for img_path in no_person_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = load_image(str(img_path))
                    if use_face_detection:
                        # For no_person, just center crop
                        h, w = img.shape
                        size = min(h, w)
                        y1 = (h - size) // 2
                        x1 = (w - size) // 2
                        img = img[y1:y1+size, x1:x1+size]
                    img = resize_image(img)
                    img = normalize_image(img)
                    images.append(img)
                    labels.append(0)
                except Exception as e:
                    print(f"Warning: Could not process {img_path}: {e}")
    
    print(f"  Loaded {len(images)} no_person images")
    
    # Load person class (label = 1) from stage_a/person directory
    person_dir = stage_a_dir / 'person'
    person_count = 0

    if person_dir.exists():
        for img_path in person_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = load_image(str(img_path))
                    if use_face_detection:
                        img = detect_and_crop_face(img)
                    img = resize_image(img)
                    img = normalize_image(img)
                    images.append(img)
                    labels.append(1)
                    person_count += 1
                except Exception as e:
                    print(f"Warning: Could not process {img_path}: {e}")
    
    print(f"  Loaded {person_count} person images")
    print(f"  Total: {len(images)} images")
    
    return np.array(images), np.array(labels)



# =============================================================================
# Train/Val/Test Split
# =============================================================================
def split_data(X: np.ndarray, y: np.ndarray,
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               seed: int = RANDOM_SEED) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train/validation/test sets.
    
    Args:
        X: Image data
        y: Labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1"
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    
    # Second split: train vs val
    val_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_adjusted, 
        random_state=seed, stratify=y_trainval
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


# =============================================================================
# Apply Augmentation
# =============================================================================
def augment_dataset(X: np.ndarray, y: np.ndarray,
                    augmentations_per_image: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation to training data.
    
    Args:
        X: Training images
        y: Training labels
        augmentations_per_image: Number of augmented copies per original
        
    Returns:
        Augmented images and labels
    """
    print(f"\n=== Augmenting Data ===")
    print(f"  Original size: {len(X)}")
    
    augmentor = Augmentor()
    
    augmented_X = []
    augmented_y = []
    
    for img, label in zip(X, y):
        aug_images = augmentor.augment(img, augmentations_per_image)
        augmented_X.extend(aug_images)
        augmented_y.extend([label] * len(aug_images))
    
    print(f"  Augmented size: {len(augmented_X)}")
    
    return np.array(augmented_X), np.array(augmented_y)


# =============================================================================
# Save Processed Data
# =============================================================================
def save_processed_data(output_dir: str, stage_a_data: Dict):
    """Save processed Stage A datasets as NPZ files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez(output_path / 'stage_a_train.npz',
             X=stage_a_data['train'][0], y=stage_a_data['train'][1])
    np.savez(output_path / 'stage_a_val.npz',
             X=stage_a_data['val'][0],   y=stage_a_data['val'][1])
    np.savez(output_path / 'stage_a_test.npz',
             X=stage_a_data['test'][0],  y=stage_a_data['test'][1])

    print(f"\nStage A saved:")
    print(f"  Train: {len(stage_a_data['train'][0])} samples")
    print(f"  Val:   {len(stage_a_data['val'][0])} samples")
    print(f"  Test:  {len(stage_a_data['test'][0])} samples")

    metadata = {
        'img_size': IMG_SIZE,
        'num_classes_stage_a': 2,
        'class_names_stage_a': ['no_person', 'person'],
        'stage_a_train_size': len(stage_a_data['train'][0]),
        'stage_a_val_size':   len(stage_a_data['val'][0]),
        'stage_a_test_size':  len(stage_a_data['test'][0]),
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_path / 'metadata.json'}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Preprocess and augment face detection dataset'
    )
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Path to raw dataset directory')
    parser.add_argument('--output_dir', type=str, default='processed',
                        help='Path to save processed data')
    parser.add_argument('--use_face_detection', action='store_true',
                        help='Use OpenCV face detection for cropping')
    parser.add_argument('--augment_train', action='store_true',
                        help='Apply data augmentation to training set')
    parser.add_argument('--augmentations', type=int, default=8,
                        help='Number of augmented copies per image')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Fraction of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Fraction of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Fraction of data for testing')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Face Detection Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dataset directory: {args.dataset_dir}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Image size:        {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Face detection:    {args.use_face_detection}")
    print(f"  Augmentation:      {args.augment_train}")
    print(f"  Split:             {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")

    X_a, y_a = load_stage_a_data(args.dataset_dir, args.use_face_detection)

    if len(X_a) == 0:
        print("\nERROR: No Stage A data found!")
        print("Run: python3 download_larger_dataset.py")
        return

    print("\n=== Splitting Stage A Data ===")
    stage_a_splits = split_data(X_a, y_a,
                                args.train_ratio, args.val_ratio, args.test_ratio)
    print(f"  Train: {len(stage_a_splits['train'][0])}")
    print(f"  Val:   {len(stage_a_splits['val'][0])}")
    print(f"  Test:  {len(stage_a_splits['test'][0])}")

    if args.augment_train:
        X_train_a, y_train_a = augment_dataset(
            stage_a_splits['train'][0],
            stage_a_splits['train'][1],
            args.augmentations
        )
        stage_a_splits['train'] = (X_train_a, y_train_a)

    for key in stage_a_splits:
        X, y = stage_a_splits[key]
        stage_a_splits[key] = (X.reshape(-1, IMG_SIZE, IMG_SIZE, 1), y)

    save_processed_data(args.output_dir, stage_a_splits)

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nNext step: python3 E_train_model.py --data_dir {args.output_dir}")


if __name__ == '__main__':
    main()

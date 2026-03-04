#!/usr/bin/env python3
"""
Generate realistic training data that matches what the ArduCAM actually captures.

The problem: Training data has tightly-cropped faces (face fills 80-90% of frame)
            But ArduCAM captures full scenes where face is 30-50% of frame.

Solution: Create synthetic training images that simulate camera captures by:
1. Placing face images on background at realistic sizes
2. Adding realistic camera noise/blur
3. Ensuring the face is in center (where camera points)
"""

import cv2
import numpy as np
from pathlib import Path
import random

IMG_SIZE = 96
DATASET_DIR = Path("dataset/stage_a")
PERSON_DIR = DATASET_DIR / "person"
NO_PERSON_DIR = DATASET_DIR / "no_person"
OUTPUT_DIR = Path("dataset/stage_a_realistic")

def create_realistic_face_image(face_img, background_color=128):
    """
    Create a realistic camera-view image with face at center.
    Face occupies 40-70% of the frame (realistic for someone facing camera).
    """
    h, w = face_img.shape[:2]
    
    # Target face size: 40-70% of frame
    face_scale = random.uniform(0.45, 0.75)
    target_size = int(IMG_SIZE * face_scale)
    
    # Resize face
    face_resized = cv2.resize(face_img, (target_size, target_size))
    
    # Create output with random gray background
    bg_val = random.randint(80, 180)
    output = np.full((IMG_SIZE, IMG_SIZE), bg_val, dtype=np.uint8)
    
    # Add some background noise
    noise = np.random.randint(-20, 20, (IMG_SIZE, IMG_SIZE), dtype=np.int16)
    output = np.clip(output.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Place face at center with small random offset
    offset_x = random.randint(-5, 5)
    offset_y = random.randint(-5, 5)
    start_x = (IMG_SIZE - target_size) // 2 + offset_x
    start_y = (IMG_SIZE - target_size) // 2 + offset_y
    
    # Clamp to valid range
    start_x = max(0, min(start_x, IMG_SIZE - target_size))
    start_y = max(0, min(start_y, IMG_SIZE - target_size))
    
    # Place face
    output[start_y:start_y+target_size, start_x:start_x+target_size] = face_resized
    
    # Add slight blur (camera is not perfectly sharp)
    if random.random() > 0.5:
        output = cv2.GaussianBlur(output, (3, 3), 0)
    
    return output

def create_no_person_image():
    """Create realistic background-only image."""
    bg_val = random.randint(60, 200)
    output = np.full((IMG_SIZE, IMG_SIZE), bg_val, dtype=np.uint8)
    
    # Add texture patterns
    pattern_type = random.randint(0, 4)
    
    if pattern_type == 0:
        # Random noise
        noise = np.random.randint(-40, 40, (IMG_SIZE, IMG_SIZE), dtype=np.int16)
        output = np.clip(output.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif pattern_type == 1:
        # Gradient
        for i in range(IMG_SIZE):
            output[i, :] = np.clip(bg_val + i - IMG_SIZE//2, 0, 255)
    elif pattern_type == 2:
        # Vertical stripes
        for i in range(0, IMG_SIZE, random.randint(8, 20)):
            output[:, i:i+random.randint(4, 10)] = random.randint(40, 220)
    elif pattern_type == 3:
        # Blotchy pattern
        for _ in range(random.randint(3, 8)):
            cx, cy = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
            radius = random.randint(10, 30)
            color = random.randint(40, 220)
            cv2.circle(output, (cx, cy), radius, color, -1)
        output = cv2.GaussianBlur(output, (15, 15), 0)
    else:
        # Just noise
        output = np.random.randint(40, 220, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    return output

def main():
    print("=" * 60)
    print("Creating Realistic Training Dataset")
    print("=" * 60)
    
    # Create output directories
    out_person = OUTPUT_DIR / "person"
    out_no_person = OUTPUT_DIR / "no_person"
    out_person.mkdir(parents=True, exist_ok=True)
    out_no_person.mkdir(parents=True, exist_ok=True)
    
    # Load existing face images
    face_files = list(PERSON_DIR.glob("*.png"))
    print(f"Found {len(face_files)} face images")
    
    # Generate realistic person images
    print("\nGenerating realistic person images...")
    count = 0
    for i, face_file in enumerate(face_files):
        face = cv2.imread(str(face_file), cv2.IMREAD_GRAYSCALE)
        if face is None:
            continue
        
        # Generate 3 variations per face
        for j in range(3):
            realistic = create_realistic_face_image(face)
            out_path = out_person / f"realistic_{count:04d}.png"
            cv2.imwrite(str(out_path), realistic)
            count += 1
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(face_files)} faces...")
    
    print(f"  Created {count} realistic person images")
    
    # Generate no-person images
    print("\nGenerating no-person images...")
    for i in range(count):  # Same number as person images
        no_person = create_no_person_image()
        out_path = out_no_person / f"bg_{i:04d}.png"
        cv2.imwrite(str(out_path), no_person)
    
    print(f"  Created {count} no-person images")
    
    print("\n" + "=" * 60)
    print("Dataset created!")
    print(f"  Person: {count} images")
    print(f"  No-person: {count} images")
    print(f"  Location: {OUTPUT_DIR}")
    print("=" * 60)
    
    print("\nNext: Run preprocessing and training:")
    print("  python3 C_preprocess_and_augment.py --dataset_dir dataset/stage_a_realistic --output_dir processed --augment_train --augmentations 8")
    print("  python3 E_train_model.py --data_dir processed --output_dir models --epochs 30")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
D. Model Architectures for Two-Stage Face Recognition

This module defines:
1. Stage A Model: Binary classifier (Person vs No-Person)
2. Stage B Model: Multi-class classifier (5 known persons)

Both models are optimized for TinyML deployment:
- Small memory footprint (<50KB each)
- INT8 quantization compatible
- Efficient inference on ARM Cortex-M4

Usage:
    from D_model_architecture import create_stage_a_model, create_stage_b_model
    
    model_a = create_stage_a_model()
    model_b = create_stage_b_model(num_classes=5)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple

# =============================================================================
# Configuration
# =============================================================================
IMG_SIZE = 96
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)  # Grayscale

# =============================================================================
# Stage A Model: Person Detection (Binary)
# =============================================================================
def create_stage_a_model(input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                         name: str = 'PersonDetector') -> Model:
    """
    Create Stage A model for binary person detection.
    
    Architecture: Tiny CNN optimized for detecting presence of any person
    
    Target specs:
        - Parameters: ~15,000
        - Model size (INT8): ~15-20 KB
        - Input: 96x96x1 grayscale
        - Output: 2 classes (no_person, person)
    
    Args:
        input_shape: Input image shape (H, W, C)
        name: Model name
        
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Block 1: Initial convolution
    # Input: 96x96x1 -> Output: 48x48x8
    x = layers.Conv2D(8, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    
    # Block 2: Feature extraction
    # Input: 48x48x8 -> Output: 24x24x16
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    
    # Block 3: Deeper features
    # Input: 24x24x16 -> Output: 12x12x24
    x = layers.Conv2D(24, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    
    # Block 4: Final convolution
    # Input: 12x12x24 -> Output: 6x6x32
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.ReLU(name='relu4')(x)
    
    # Global pooling
    # Input: 6x6x32 -> Output: 32
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.2, name='dropout')(x)
    
    # Output layer
    # Binary classification: person vs no_person
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    return model


# =============================================================================
# Stage B Model: Face Recognition (5-class)
# =============================================================================
def create_stage_b_model(num_classes: int = 5,
                         input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                         name: str = 'FaceRecognizer') -> Model:
    """
    Create Stage B model for face recognition.
    
    Architecture: Tiny CNN optimized for distinguishing between known faces
    
    Target specs:
        - Parameters: ~25,000
        - Model size (INT8): ~25-30 KB
        - Input: 96x96x1 grayscale
        - Output: 5 classes (person1-5)
    
    Args:
        num_classes: Number of known persons (default: 5)
        input_shape: Input image shape (H, W, C)
        name: Model name
        
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Block 1: Initial convolution with more filters
    # Input: 96x96x1 -> Output: 48x48x16
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    
    # Block 2: Depthwise separable convolution (efficient)
    # Input: 48x48x16 -> Output: 24x24x32
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                               use_bias=False, name='dw_conv2')(x)
    x = layers.BatchNormalization(name='bn2a')(x)
    x = layers.ReLU(name='relu2a')(x)
    x = layers.Conv2D(32, kernel_size=1, padding='same',
                      use_bias=False, name='pw_conv2')(x)
    x = layers.BatchNormalization(name='bn2b')(x)
    x = layers.ReLU(name='relu2b')(x)
    
    # Block 3: Depthwise separable convolution
    # Input: 24x24x32 -> Output: 12x12x48
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                               use_bias=False, name='dw_conv3')(x)
    x = layers.BatchNormalization(name='bn3a')(x)
    x = layers.ReLU(name='relu3a')(x)
    x = layers.Conv2D(48, kernel_size=1, padding='same',
                      use_bias=False, name='pw_conv3')(x)
    x = layers.BatchNormalization(name='bn3b')(x)
    x = layers.ReLU(name='relu3b')(x)
    
    # Block 4: Depthwise separable convolution
    # Input: 12x12x48 -> Output: 6x6x64
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                               use_bias=False, name='dw_conv4')(x)
    x = layers.BatchNormalization(name='bn4a')(x)
    x = layers.ReLU(name='relu4a')(x)
    x = layers.Conv2D(64, kernel_size=1, padding='same',
                      use_bias=False, name='pw_conv4')(x)
    x = layers.BatchNormalization(name='bn4b')(x)
    x = layers.ReLU(name='relu4b')(x)
    
    # Global pooling
    # Input: 6x6x64 -> Output: 64
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.3, name='dropout')(x)

    # Embedding layer (for potential future triplet loss training)
    embedding = layers.Dense(64, activation='relu', name='embedding')(x)

    # Second dropout after embedding to reduce overfitting on minority classes
    embedding = layers.Dropout(0.2, name='dropout2')(embedding)

    # Output layer
    # Multi-class classification with softmax
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(embedding)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    return model


# =============================================================================
# Alternative: Ultra-Tiny Models (if memory is critical)
# =============================================================================
def create_stage_a_ultratiny(input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                             name: str = 'PersonDetectorTiny') -> Model:
    """
    Ultra-tiny version of Stage A for extremely constrained memory.
    
    Target specs:
        - Parameters: ~5,000
        - Model size (INT8): ~5-7 KB
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Block 1: 96x96x1 -> 24x24x8
    x = layers.Conv2D(8, kernel_size=5, strides=4, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    
    # Block 2: 24x24x8 -> 6x6x16
    x = layers.Conv2D(16, kernel_size=3, strides=4, padding='same',
                      use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    
    # Global pooling: 6x6x16 -> 16
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Output
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    return model


def create_stage_b_ultratiny(num_classes: int = 5,
                             input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                             name: str = 'FaceRecognizerTiny') -> Model:
    """
    Ultra-tiny version of Stage B for extremely constrained memory.
    
    Target specs:
        - Parameters: ~10,000
        - Model size (INT8): ~10-12 KB
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Block 1: 96x96x1 -> 24x24x12
    x = layers.Conv2D(12, kernel_size=5, strides=4, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    
    # Block 2: 24x24x12 -> 6x6x24
    x = layers.Conv2D(24, kernel_size=3, strides=4, padding='same',
                      use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    
    # Global pooling: 6x6x24 -> 24
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    return model


# =============================================================================
# Model Summary and Analysis
# =============================================================================
def analyze_model(model: Model) -> dict:
    """
    Analyze model parameters and estimate size.
    
    Returns:
        Dictionary with model statistics
    """
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Estimate sizes
    # Float32: 4 bytes per param, INT8: 1 byte per param
    float32_size_kb = (total_params * 4) / 1024
    int8_size_kb = (total_params * 1) / 1024
    
    return {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'float32_size_kb': float32_size_kb,
        'int8_size_kb': int8_size_kb
    }


def print_model_analysis(model: Model):
    """Print formatted model analysis."""
    stats = analyze_model(model)
    
    print(f"\n{'='*50}")
    print(f"Model: {stats['name']}")
    print(f"{'='*50}")
    print(f"Total parameters:     {stats['total_params']:,}")
    print(f"Trainable params:     {stats['trainable_params']:,}")
    print(f"Non-trainable params: {stats['non_trainable_params']:,}")
    print(f"{'='*50}")
    print(f"Estimated sizes:")
    print(f"  Float32: {stats['float32_size_kb']:.1f} KB")
    print(f"  INT8:    {stats['int8_size_kb']:.1f} KB")
    print(f"{'='*50}")


# =============================================================================
# Main: Print Model Summaries
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TinyML Face Recognition Model Architectures")
    print("=" * 60)
    
    # Stage A Model
    print("\n\n>>> STAGE A: Person Detection (Binary) <<<")
    model_a = create_stage_a_model()
    model_a.summary()
    print_model_analysis(model_a)
    
    # Stage B Model
    print("\n\n>>> STAGE B: Face Recognition (5-class) <<<")
    model_b = create_stage_b_model(num_classes=5)
    model_b.summary()
    print_model_analysis(model_b)
    
    # Ultra-tiny alternatives
    print("\n\n>>> ALTERNATIVE: Ultra-Tiny Stage A <<<")
    model_a_tiny = create_stage_a_ultratiny()
    model_a_tiny.summary()
    print_model_analysis(model_a_tiny)
    
    print("\n\n>>> ALTERNATIVE: Ultra-Tiny Stage B <<<")
    model_b_tiny = create_stage_b_ultratiny(num_classes=5)
    model_b_tiny.summary()
    print_model_analysis(model_b_tiny)
    
    # Total memory estimate
    print("\n\n" + "=" * 60)
    print("COMBINED MEMORY ESTIMATE (Standard Models)")
    print("=" * 60)
    
    stats_a = analyze_model(model_a)
    stats_b = analyze_model(model_b)
    
    total_int8 = stats_a['int8_size_kb'] + stats_b['int8_size_kb']
    total_float32 = stats_a['float32_size_kb'] + stats_b['float32_size_kb']
    
    print(f"Stage A (INT8):  {stats_a['int8_size_kb']:.1f} KB")
    print(f"Stage B (INT8):  {stats_b['int8_size_kb']:.1f} KB")
    print(f"{'='*40}")
    print(f"Total (INT8):    {total_int8:.1f} KB")
    print(f"Total (Float32): {total_float32:.1f} KB")
    
    # Arduino Nano 33 BLE Sense memory
    print(f"\nArduino Nano 33 BLE Sense Rev2 Flash: 1,024 KB")
    print(f"Available for models: ~700 KB (after bootloader + code)")
    print(f"Model usage: {total_int8:.1f} KB ({total_int8/700*100:.1f}%)")
    
    print(f"\nArduino Nano 33 BLE Sense Rev2 RAM: 256 KB")
    print(f"Tensor Arena estimate: ~100 KB")
    print(f"Image buffer (96x96): ~9 KB")
    print(f"Free for variables: ~147 KB")
    
    print("\n" + "=" * 60)
    print("Architecture Design Rationale")
    print("=" * 60)
    print("""
Stage A (Person Detection):
  - Simple 4-block CNN with increasing filters (8→16→24→32)
  - Aggressive stride (2) for fast downsampling
  - BatchNorm for stable training without bias terms
  - GlobalAveragePooling reduces overfitting
  - Binary output: [no_person, person]
  - Optimized for speed (gating function)

Stage B (Face Recognition):  
  - MobileNet-style depthwise separable convolutions
  - More filters (16→32→48→64) for identity discrimination
  - Embedding layer (32-dim) for potential metric learning
  - Higher dropout (0.3) to prevent overfitting on small dataset
  - 5-class output: [person1, person2, person3, person4, person5]
  - Optimized for accuracy (identity matters)

Unknown Detection:
  - NOT a separate class in Stage B
  - Detected via confidence thresholding on Stage B output
  - If max(softmax) < threshold (e.g., 0.7), predict "Unknown"
  - Threshold tuned using unknown_test data (not in training)
""")

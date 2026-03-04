#!/usr/bin/env python3
"""
F. INT8 Quantization Pipeline for TinyML Deployment

This script converts trained Keras models to INT8 quantized TFLite format
for deployment on Arduino Nano 33 BLE Sense Rev2.

Quantization benefits:
- 4x smaller model size (Float32 → INT8)
- Faster inference on ARM Cortex-M4
- Lower memory requirements

Usage:
    python F_quantize_model.py --model_dir models --data_dir processed
"""

import os
import json
import tempfile
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
from typing import Callable, Generator

# =============================================================================
# Configuration
# =============================================================================
IMG_SIZE = 96


# =============================================================================
# Representative Dataset Generator
# =============================================================================
def create_representative_dataset(data_path: str, 
                                  num_samples: int = 100) -> Callable:
    """
    Create a representative dataset generator for quantization calibration.
    
    INT8 quantization requires a representative dataset to calibrate
    the quantization ranges for each layer.
    
    Args:
        data_path: Path to NPZ file with calibration data
        num_samples: Number of samples to use for calibration
        
    Returns:
        Generator function for representative dataset
    """
    data = np.load(data_path)
    X = data['X']
    
    # Select samples (shuffle and take first num_samples)
    indices = np.random.permutation(len(X))[:num_samples]
    samples = X[indices]
    
    def representative_dataset() -> Generator:
        for sample in samples:
            # Ensure correct shape: (1, H, W, C)
            sample = sample.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            # Convert to float32 (TFLite expects float32 input)
            yield [sample.astype(np.float32)]
    
    return representative_dataset


# =============================================================================
# Quantization Functions
# =============================================================================
def convert_to_tflite_int8(model_path: str,
                           rep_dataset_path: str,
                           output_path: str,
                           num_calibration_samples: int = 100) -> dict:
    """
    Convert Keras model to INT8 quantized TFLite format.
    
    This performs full integer quantization (weights AND activations are INT8).
    
    Args:
        model_path: Path to saved Keras model (.keras)
        rep_dataset_path: Path to calibration data (.npz)
        output_path: Path to save TFLite model
        num_calibration_samples: Number of samples for calibration
        
    Returns:
        Dictionary with conversion statistics
    """
    print(f"\nConverting: {model_path}")
    print(f"Calibration data: {rep_dataset_path}")
    
    # Load Keras model
    model = tf.keras.models.load_model(model_path)

    # Build representative dataset
    rep_dataset = create_representative_dataset(
        rep_dataset_path,
        num_calibration_samples
    )

    # model.export() produces an inference-optimised SavedModel whose graph has
    # BatchNorm folded — this avoids the Keras 3 BatchNorm Cast issue in
    # TF 2.16's MLIR quantizer that breaks from_keras_model / from_saved_model.
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.export(tmp_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)

        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert
        print("Converting to INT8 TFLite...")
        tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Calculate sizes
    original_size = os.path.getsize(model_path)
    quantized_size = os.path.getsize(output_path)
    
    stats = {
        'original_path': model_path,
        'quantized_path': output_path,
        'original_size_bytes': original_size,
        'quantized_size_bytes': quantized_size,
        'original_size_kb': original_size / 1024,
        'quantized_size_kb': quantized_size / 1024,
        'compression_ratio': original_size / quantized_size
    }
    
    print(f"\nConversion complete!")
    print(f"Original size: {stats['original_size_kb']:.2f} KB")
    print(f"Quantized size: {stats['quantized_size_kb']:.2f} KB")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Saved to: {output_path}")
    
    return stats


def convert_to_tflite_float(model_path: str,
                            output_path: str) -> dict:
    """
    Convert Keras model to Float32 TFLite (for comparison).
    
    Args:
        model_path: Path to saved Keras model
        output_path: Path to save TFLite model
        
    Returns:
        Dictionary with conversion statistics
    """
    print(f"\nConverting (Float32): {model_path}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create converter (no quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    original_size = os.path.getsize(model_path)
    float_size = os.path.getsize(output_path)
    
    return {
        'original_path': model_path,
        'float_path': output_path,
        'original_size_kb': original_size / 1024,
        'float_size_kb': float_size / 1024
    }


# =============================================================================
# TFLite Model Validation
# =============================================================================
def validate_tflite_model(tflite_path: str,
                          test_data_path: str,
                          num_test_samples: int = 100) -> dict:
    """
    Validate TFLite model accuracy on test data.
    
    Args:
        tflite_path: Path to TFLite model
        test_data_path: Path to test data NPZ
        num_test_samples: Number of samples to test
        
    Returns:
        Dictionary with validation results
    """
    print(f"\nValidating: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input/output details
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Get quantization parameters
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]
    
    print(f"\nQuantization parameters:")
    print(f"  Input scale: {input_scale}, zero_point: {input_zero_point}")
    print(f"  Output scale: {output_scale}, zero_point: {output_zero_point}")
    
    # Load test data
    test_data = np.load(test_data_path)
    X_test = test_data['X']
    y_test = test_data['y']
    
    # Select samples
    indices = np.random.permutation(len(X_test))[:num_test_samples]
    X_test = X_test[indices]
    y_test = y_test[indices]
    
    # Run inference
    correct = 0
    for i in range(len(X_test)):
        # Prepare input
        sample = X_test[i].reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
        
        # Quantize input if needed
        if input_details[0]['dtype'] == np.int8:
            sample = (sample / input_scale + input_zero_point).astype(np.int8)
        
        # Set input
        interpreter.set_tensor(input_details[0]['index'], sample)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize if needed
        if output_details[0]['dtype'] == np.int8:
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # Get prediction
        pred = np.argmax(output)
        
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / len(X_test)
    
    print(f"\nValidation Results:")
    print(f"  Samples tested: {len(X_test)}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    
    return {
        'model_path': tflite_path,
        'num_samples': len(X_test),
        'correct': correct,
        'accuracy': accuracy,
        'input_scale': float(input_scale),
        'input_zero_point': int(input_zero_point),
        'output_scale': float(output_scale),
        'output_zero_point': int(output_zero_point)
    }


# =============================================================================
# Generate C Header File
# =============================================================================
def generate_c_header(tflite_path: str,
                      header_path: str,
                      array_name: str):
    """
    Convert TFLite model to C header file for Arduino inclusion.
    
    Args:
        tflite_path: Path to TFLite model
        header_path: Path to save C header file
        array_name: Name for the C array
    """
    print(f"\nGenerating C header: {header_path}")
    
    # Read TFLite model
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # Generate C array
    model_size = len(model_data)
    
    header_content = f"""// Auto-generated TFLite model header
// Model: {os.path.basename(tflite_path)}
// Size: {model_size} bytes ({model_size/1024:.2f} KB)

#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

#include <Arduino.h>

alignas(8) const unsigned char {array_name}[] = {{
"""
    
    # Add hex data (16 bytes per line)
    for i in range(0, len(model_data), 16):
        chunk = model_data[i:i+16]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        header_content += f"  {hex_values},\n"
    
    header_content += f"""
}};

const unsigned int {array_name}_len = {model_size};

#endif // {array_name.upper()}_H
"""
    
    # Save header file
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"Header file saved: {header_path}")
    print(f"Array name: {array_name}")
    print(f"Array length: {model_size} bytes")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Quantize models for TinyML')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory with trained Keras models')
    parser.add_argument('--data_dir', type=str, default='processed',
                        help='Directory with calibration data')
    parser.add_argument('--output_dir', type=str, default='tflite',
                        help='Directory to save TFLite models')
    parser.add_argument('--num_calibration', type=int, default=100,
                        help='Number of calibration samples')
    parser.add_argument('--validate', action='store_true',
                        help='Validate converted models')
    args = parser.parse_args()
    
    model_path = Path(args.model_dir)
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TFLite INT8 Quantization Pipeline")
    print("=" * 60)
    
    all_stats = {}
    
    # =================================
    # Stage A Quantization
    # =================================
    stage_a_keras = model_path / 'stage_a_final.keras'
    if stage_a_keras.exists():
        print("\n" + "=" * 60)
        print("STAGE A: Person Detection Quantization")
        print("=" * 60)
        
        # INT8 quantization
        stats_a = convert_to_tflite_int8(
            str(stage_a_keras),
            str(data_path / 'stage_a_train.npz'),
            str(output_path / 'stage_a_int8.tflite'),
            args.num_calibration
        )
        all_stats['stage_a'] = stats_a
        
        # Generate C header
        generate_c_header(
            str(output_path / 'stage_a_int8.tflite'),
            str(output_path / 'stage_a_model.h'),
            'stage_a_model'
        )
        
        # Validate
        if args.validate:
            val_a = validate_tflite_model(
                str(output_path / 'stage_a_int8.tflite'),
                str(data_path / 'stage_a_test.npz')
            )
            all_stats['stage_a_validation'] = val_a
    else:
        print(f"\nStage A model not found: {stage_a_keras}")
    
    # =================================
    # Summary
    # =================================
    print("\n" + "=" * 60)
    print("Quantization Summary")
    print("=" * 60)

    if 'stage_a' in all_stats:
        size_kb = all_stats['stage_a']['quantized_size_kb']
        print(f"\nStage A model: {size_kb:.2f} KB (INT8)")
        print(f"Arduino flash: 1,024 KB available")
        print(f"Model usage:   {size_kb:.2f} KB ({size_kb/700*100:.1f}%)")

        if 'stage_a_validation' in all_stats:
            print(f"Validation accuracy: {all_stats['stage_a_validation']['accuracy']*100:.2f}%")

    # Save quantization stats
    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_path / 'quantization_stats.json', 'w') as f:
        json.dump(json.loads(json.dumps(all_stats, default=convert_numpy)), f, indent=2)

    # Append to accuracy log
    log_path = Path(args.model_dir).parent / 'accuracy_log.json'
    metrics_path = Path(args.model_dir) / 'stage_a_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        log = []
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)

        from datetime import date
        log.append({
            "run": len(log) + 1,
            "date": str(date.today()),
            "test_accuracy": round(metrics.get('test_accuracy', 0), 4),
            "model_size_kb": round(all_stats.get('stage_a', {}).get('quantized_size_kb', 0), 2),
            "notes": ""
        })

        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"\nAccuracy log updated: {log_path}")

    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - {output_path / 'stage_a_int8.tflite'}")
    print(f"  - {output_path / 'stage_a_model.h'}")
    print(f"\nNext steps:")
    print(f"  1. Copy stage_a_model.h to G_arduino_firmware/")
    print(f"  2. Upload firmware: G_arduino_firmware.ino")


if __name__ == '__main__':
    main()

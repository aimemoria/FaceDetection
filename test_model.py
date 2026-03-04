#!/usr/bin/env python3
"""Test the TFLite model to diagnose detection issues."""
import tensorflow as tf
import numpy as np
import cv2

# Load model
interp = tf.lite.Interpreter(model_path="tflite/stage_a_int8.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print("=== MODEL SPECS ===")
print(f"Input shape: {inp['shape']}")
print(f"Input dtype: {inp['dtype']}")
print(f"Input scale: {inp['quantization_parameters']['scales'][0]}")
print(f"Input zero_point: {inp['quantization_parameters']['zero_points'][0]}")

in_scale = inp['quantization_parameters']['scales'][0]
in_zp = inp['quantization_parameters']['zero_points'][0]
out_scale = out['quantization_parameters']['scales'][0]
out_zp = out['quantization_parameters']['zero_points'][0]

def test_image(path, name):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load: {path}")
        return
    img = cv2.resize(img, (96, 96))
    img_float = img.astype(np.float32) / 255.0
    
    # Quantize
    img_int8 = np.clip(np.round(img_float / in_scale) + in_zp, -128, 127).astype(np.int8)
    
    interp.set_tensor(inp['index'], img_int8.reshape(1, 96, 96, 1))
    interp.invoke()
    output = interp.get_tensor(out['index'])[0]
    
    p_no = (output[0] - out_zp) * out_scale
    p_yes = (output[1] - out_zp) * out_scale
    
    result = "PERSON" if p_yes > p_no else "NO PERSON"
    print(f"{name}: p_no={p_no:.3f}, p_yes={p_yes:.3f} -> {result}")

print("\n=== TESTING FACE IMAGES ===")
test_image("dataset/stage_a/person/olivetti_0001.png", "Olivetti face 1")
test_image("dataset/stage_a/person/olivetti_0050.png", "Olivetti face 50")
test_image("dataset/stage_a/person/person_0001.png", "LFW face 1")
test_image("dataset/stage_a/person/person_0100.png", "LFW face 100")

print("\n=== TESTING NO-PERSON IMAGES ===")
test_image("dataset/stage_a/no_person/bg_0001.png", "Background 1")
test_image("dataset/stage_a/no_person/cifar_bg_0001.png", "CIFAR bg 1")

print("\n=== TEST WITH SIMULATED CAMERA CAPTURE ===")
# Simulate what camera sees: face is small in frame
# Create a 160x120 "scene" with face in center
scene = np.ones((120, 160), dtype=np.uint8) * 128  # gray background
face = cv2.imread("dataset/stage_a/person/olivetti_0001.png", cv2.IMREAD_GRAYSCALE)
face_small = cv2.resize(face, (40, 40))  # face is small in scene
# Place face in center
scene[40:80, 60:100] = face_small

# Now crop center 48x48 and resize to 96x96 (what firmware does)
crop = scene[36:84, 56:104]  # 48x48 around center
crop_resized = cv2.resize(crop, (96, 96))

# Save for inspection
cv2.imwrite("test_simulated_capture.png", crop_resized)
print("Saved simulated capture to: test_simulated_capture.png")

img_float = crop_resized.astype(np.float32) / 255.0
img_int8 = np.clip(np.round(img_float / in_scale) + in_zp, -128, 127).astype(np.int8)
interp.set_tensor(inp['index'], img_int8.reshape(1, 96, 96, 1))
interp.invoke()
output = interp.get_tensor(out['index'])[0]
p_no = (output[0] - out_zp) * out_scale
p_yes = (output[1] - out_zp) * out_scale
result = "PERSON" if p_yes > p_no else "NO PERSON"
print(f"Simulated camera (small face): p_no={p_no:.3f}, p_yes={p_yes:.3f} -> {result}")

print("\n=== KEY FINDING ===")
print("If cropped faces work but simulated camera fails,")
print("the issue is that training data has TIGHT CROPS")
print("but camera captures FULL SCENES with small faces.")

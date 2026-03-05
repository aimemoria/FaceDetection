#!/usr/bin/env python3
"""
Test the TFLite model with your webcam to diagnose detection issues.
This simulates what the Arduino sees and helps identify problems.

Run: python3 test_webcam.py
Press 'q' to quit, 's' to save a frame for debugging.
"""
import tensorflow as tf
import numpy as np
import cv2
import time

# Load model
print("Loading model...")
interp = tf.lite.Interpreter(model_path="tflite/stage_a_int8.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

in_scale = inp['quantization_parameters']['scales'][0]
in_zp = inp['quantization_parameters']['zero_points'][0]
out_scale = out['quantization_parameters']['scales'][0]
out_zp = out['quantization_parameters']['zero_points'][0]

print(f"Model input: {inp['shape']}, scale={in_scale:.6f}, zero_point={in_zp}")

# Haar cascade for face detection (same as preview_server.py uses)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_like_arduino(frame):
    """
    Preprocess frame EXACTLY like the Arduino firmware does:
    1. Convert to grayscale
    2. Take center 96x96 crop (from 160x120 equivalent)
    3. Apply histogram equalization
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Scale down to 160x120 (what Arduino captures)
    h, w = gray.shape
    # Maintain aspect ratio, resize to fit in 160x120
    scale = min(160/w, 120/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(gray, (new_w, new_h))
    
    # Pad to 160x120 if needed
    full_frame = np.full((120, 160), 128, dtype=np.uint8)
    start_x = (160 - new_w) // 2
    start_y = (120 - new_h) // 2
    full_frame[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    # Center crop 96x96 (like Arduino does)
    crop_w, crop_h = 96, 96
    start_x = (160 - crop_w) // 2  # 32
    start_y = (120 - crop_h) // 2  # 12
    
    cropped = full_frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Apply histogram equalization (like Arduino's preprocessImage())
    equalized = cv2.equalizeHist(cropped)
    
    return equalized, cropped  # Return both equalized and raw

def preprocess_face_crop(frame):
    """
    Alternative: Detect face and crop like training data was prepared.
    This should give BETTER results since training images are face-cropped.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add 20% padding
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(gray.shape[1], x + w + pad_w)
        y2 = min(gray.shape[0], y + h + pad_h)
        
        face_crop = gray[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (96, 96))
        return face_resized, (x, y, w, h)
    
    return None, None

def run_inference(img_96x96):
    """Run the TFLite model on a 96x96 grayscale image."""
    img_float = img_96x96.astype(np.float32) / 255.0
    img_int8 = np.clip(np.round(img_float / in_scale) + in_zp, -128, 127).astype(np.int8)
    
    interp.set_tensor(inp['index'], img_int8.reshape(1, 96, 96, 1))
    interp.invoke()
    output = interp.get_tensor(out['index'])[0]
    
    p_no = (output[0] - out_zp) * out_scale
    p_yes = (output[1] - out_zp) * out_scale
    
    return p_yes, p_no

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    print("Try: python3 test_webcam.py")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("\n" + "="*60)
print("FACE DETECTION DIAGNOSTIC TEST")
print("="*60)
print("This tests your model with webcam input.")
print("")
print("CONTROLS:")
print("  'q' - Quit")
print("  's' - Save current frame for debugging")
print("  'm' - Toggle between Arduino-style and Face-crop mode")
print("")
print("WHAT TO LOOK FOR:")
print("  - Does 'Face Crop' mode detect faces better than 'Arduino' mode?")
print("  - If yes, the problem is that training data was face-cropped")
print("    but Arduino sends center-cropped full scenes.")
print("="*60 + "\n")

use_face_crop = True  # Start with face-crop mode (what training data looks like)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    
    # Create display window with multiple views
    display = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Show original frame (scaled)
    frame_small = cv2.resize(frame, (320, 240))
    display[0:240, 0:320] = frame_small
    cv2.putText(display, "Webcam Input", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Process with current mode
    if use_face_crop:
        face_img, face_bbox = preprocess_face_crop(frame)
        mode_text = "Mode: FACE CROP (like training data)"
        
        if face_img is not None:
            p_yes, p_no = run_inference(face_img)
            
            # Draw face bbox on original
            x, y, w, h = face_bbox
            scale_x = 320 / frame.shape[1]
            scale_y = 240 / frame.shape[0]
            cv2.rectangle(display, 
                         (int(x*scale_x), int(y*scale_y)), 
                         (int((x+w)*scale_x), int((y+h)*scale_y)),
                         (0, 255, 0), 2)
            
            # Show face crop (enlarged)
            face_display = cv2.resize(face_img, (192, 192))
            face_display_color = cv2.cvtColor(face_display, cv2.COLOR_GRAY2BGR)
            display[240:432, 0:192] = face_display_color
            cv2.putText(display, "Face Crop (96x96)", (10, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            p_yes, p_no = 0, 1
            cv2.putText(display, "No face detected by Haar", (10, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        arduino_img, raw_crop = preprocess_like_arduino(frame)
        mode_text = "Mode: ARDUINO (center crop)"
        
        p_yes, p_no = run_inference(arduino_img)
        
        # Show arduino-style crop (enlarged)
        arduino_display = cv2.resize(arduino_img, (192, 192))
        arduino_display_color = cv2.cvtColor(arduino_display, cv2.COLOR_GRAY2BGR)
        display[240:432, 0:192] = arduino_display_color
        cv2.putText(display, "Center Crop + HistEq", (10, 260), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show result
    result = "FACE DETECTED" if p_yes > p_no else "NO FACE"
    conf = int(max(p_yes, p_no) * 100)
    color = (0, 255, 0) if p_yes > p_no else (0, 0, 255)
    
    cv2.putText(display, mode_text, (330, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, f"Result: {result}", (330, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(display, f"Confidence: {conf}%", (330, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, f"p_yes={p_yes:.3f}, p_no={p_no:.3f}", (330, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Instructions
    cv2.putText(display, "Press 'm' to toggle mode", (330, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(display, "Press 's' to save frame", (330, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(display, "Press 'q' to quit", (330, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    cv2.imshow("Face Detection Test", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        use_face_crop = not use_face_crop
        print(f"Switched to: {'Face Crop' if use_face_crop else 'Arduino'} mode")
    elif key == ord('s'):
        cv2.imwrite(f"debug_frame_{frame_count}.png", frame)
        cv2.imwrite(f"debug_display_{frame_count}.png", display)
        print(f"Saved debug_frame_{frame_count}.png")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)
print("""
If FACE CROP mode detects your face but ARDUINO mode doesn't,
the problem is a TRAIN-TEST MISMATCH:

  TRAINING DATA: Tightly cropped face images (Olivetti, LFW)
  ARDUINO INPUT: Center crop from full camera scene

SOLUTIONS:

1. QUICK FIX - Improve Arduino preprocessing:
   - Use face detection on Arduino before inference
   - Currently implemented in captureAndCropFace() but may not work well

2. BETTER FIX - Retrain with realistic data:
   - Use create_realistic_dataset.py to generate webcam-like training data
   - Re-run the training pipeline

3. BEST FIX - Add YOUR face to training data:
   - Take 50-100 photos of your face in various lighting/angles
   - Add to dataset/stage_a/person/
   - Retrain the model

Run this to retrain with better data:
   python3 create_realistic_dataset.py
   python3 C_preprocess_and_augment.py --dataset_dir dataset_realistic --output_dir processed --augment_train
   python3 E_train_model.py --data_dir processed --output_dir models
   python3 F_quantize_model.py --model_dir models --data_dir processed --validate
""")

# Face Detection System

TinyML face detection running on Arduino Nano 33 BLE Sense Rev2 with ArduCAM OV2640.

## Hardware Requirements

- Arduino Nano 33 BLE Sense Rev2
- ArduCAM Mini 2MP OV2640 (with 8MB FIFO)
- Piezo buzzer (optional, for audio feedback)

### Wiring

| ArduCAM Pin | Arduino Pin |
|-------------|-------------|
| CS          | D7          |
| MOSI        | D11         |
| MISO        | D12         |
| SCK         | D13         |
| SDA         | A4          |
| SCL         | A5          |
| VCC         | 3.3V        |
| GND         | GND         |

Buzzer: Connect positive to D6, negative to GND.

## Quick Start

### 1. Upload Firmware

1. Open `G_arduino_firmware/G_arduino_firmware.ino` in Arduino IDE
2. Install required libraries:
   - ArduCAM
   - Arduino_TensorFlowLite
3. Select board: **Arduino Nano 33 BLE**
4. Upload

### 2. Run Live Preview

```bash
cd face_detection_system
pip install -r requirements.txt
python3 preview_server.py
```

Open **http://localhost:7654** in your browser.

### 3. Serial Monitor

Open Arduino IDE Serial Monitor at **115200 baud** to see:
- Detection results (Face Detected / No face detected)
- Confidence percentage
- Inference timing

## Output Format

**Serial output:**
```
RESULT: Face Detected
CONFIDENCE: 95%
```

**Buzzer feedback:**
- 2 high beeps (1000 Hz) = Face detected
- 1 low beep (200 Hz) = No face

## Retraining the Model

If you want to retrain with your own data:

```bash
# 1. Add images to dataset/stage_a/person/ and dataset/stage_a/no_person/

# 2. Preprocess and augment
python3 C_preprocess_and_augment.py --dataset_dir dataset --output_dir processed --augment_train --augmentations 12

# 3. Train
python3 E_train_model.py --data_dir processed --output_dir models

# 4. Quantize for Arduino
python3 F_quantize_model.py --model_dir models --data_dir processed --validate

# 5. Copy model to firmware folder
cp tflite/stage_a_model.h G_arduino_firmware/

# 6. Re-upload firmware
```

## Project Structure

```
face_detection_system/
├── G_arduino_firmware/     # Arduino code
│   ├── G_arduino_firmware.ino
│   └── stage_a_model.h     # Quantized model
├── C_preprocess_and_augment.py
├── D_model_architecture.py
├── E_train_model.py
├── F_quantize_model.py
├── preview_server.py       # Live browser preview
├── requirements.txt
└── ACCURACY_REPORT.md
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No face detected" always | Ensure good lighting, face the camera directly |
| Serial port not found | Check port in `preview_server.py` line 28 |
| Camera test failed | Verify wiring, check SPI connections |
| Model too large | Reduce model complexity in `D_model_architecture.py` |

## Specs

- Input: 96×96 grayscale
- Model size: ~17 KB (INT8 quantized)
- Inference time: ~150ms
- Accuracy: 98.97% (lab), ~96% (real-world estimate)

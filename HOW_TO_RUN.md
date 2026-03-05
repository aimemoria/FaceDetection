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

### 1. Compile & Upload Firmware (arduino-cli)

```bash
cd "/Users/aimemoria/Documents/dynamic/face detectors/face_detection_system"

arduino-cli compile --fqbn arduino:mbed_nano:nano33ble G_arduino_firmware/

arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/cu.usbmodem101 G_arduino_firmware/
```

> **Note:** If upload fails with "Device unsupported", double-press the reset button on the Arduino to enter bootloader mode (LED will pulse), then run the upload command again immediately.

### 2. Monitor Serial Output

```bash
arduino-cli monitor --port /dev/cu.usbmodem101 --config baudrate=115200
```

### 3. Run Live Camera Preview

```bash
cd "/Users/aimemoria/Documents/dynamic/face detectors/face_detection_system"
python3 preview_server.py
```

Open **http://localhost:7654** in your browser.

### 4. Test Model on Webcam (no Arduino needed)

```bash
python3 test_webcam.py
```

Press `m` to toggle between Face Crop and Arduino mode. Press `q` to quit.

## Output Format

**Serial output:**
```
----------------------------------------
Center crop: x=32 y=12 w=96 h=96
Applied histogram equalization
Stage A: 215 ms — person (92.3%)
RESULT: Face Detected
CONFIDENCE: 92%
Inference time: 2651 ms
----------------------------------------
```

**Buzzer feedback:**
- 2 high beeps (1000 Hz) = Face detected
- 1 low beep (200 Hz) = No face

## Retraining the Model

Full pipeline to retrain from scratch with realistic camera-scale data:

```bash
# 1. Generate realistic training dataset (faces at 45-75% frame scale)
python3 create_realistic_dataset.py

# 2. Preprocess and augment (52920 samples with 8x augmentation)
python3 C_preprocess_and_augment.py --dataset_dir dataset_realistic --output_dir processed --augment_train --augmentations 8

# 3. Train (early stopping, best weights saved automatically)
python3 E_train_model.py --data_dir processed --output_dir models --epochs 60

# 4. Quantize to INT8 TFLite and validate
python3 F_quantize_model.py --model_dir models --data_dir processed --validate

# 5. Copy new model into firmware and re-upload
cp tflite/stage_a_model.h G_arduino_firmware/
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble G_arduino_firmware/
arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/cu.usbmodem101 G_arduino_firmware/
```

## Project Structure

```
face_detection_system/
├── G_arduino_firmware/          # Arduino firmware
│   ├── G_arduino_firmware.ino
│   └── stage_a_model.h          # INT8 quantized model (deployed)
├── dataset/stage_a/             # Original cropped face images
├── dataset_realistic/stage_a/   # Camera-scale synthetic training data
├── processed/                   # Preprocessed NPZ files for training
├── models/                      # Trained Keras models + metrics
├── tflite/                      # Converted TFLite models + C headers
├── A_download_dataset.py
├── C_preprocess_and_augment.py
├── D_model_architecture.py
├── E_train_model.py
├── F_quantize_model.py
├── create_realistic_dataset.py  # Generates camera-scale training data
├── test_webcam.py               # Webcam inference diagnostic tool
├── test_model.py                # Model validation on saved images
├── preview_server.py            # Live browser preview (port 7654)
├── accuracy_log.json            # Training run history
└── ACCURACY_REPORT.md
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No face detected" always | Ensure bright front lighting; face centered in camera |
| Upload fails "Device unsupported" | Double-press reset button to enter bootloader, retry upload |
| `^[[200~` appears when pasting | Run: `echo 'unset zle_bracketed_paste' >> ~/.zshrc && source ~/.zshrc` |
| Serial port not found | Run `arduino-cli board list` to find the current port |
| Camera test failed | Verify wiring; check SPI connections on D7/D11/D12/D13 |
| Preview server not loading | Port is 7654 not 8080 — open http://localhost:7654 |
| Image too dark (avg < 50) | Improve room lighting; avoid backlighting |

## Model Specs (Run 2 — Current Deployment)

| Property | Value |
|----------|-------|
| Input | 96×96 grayscale |
| Model size | 17.55 KB (INT8 quantized) |
| Inference time | ~215 ms on Arduino Nano 33 BLE |
| Test accuracy | **99.84%** (1260 test samples) |
| Validation accuracy | **100%** (INT8 TFLite, 100 samples) |
| Training samples | 52,920 (after 8x augmentation) |
| Training dataset | Realistic synthetic — faces at 45–75% frame scale |
| Stopped at epoch | 49 (best weights: epoch 24) |
| Flash usage | 376,736 bytes (38% of 983 KB) |
| RAM usage | 130,456 bytes (49% of 256 KB) |

# Face Detection System

A TinyML face detection system running on Arduino Nano 33 BLE Sense Rev2 with ArduCAM OV2640 camera.

## Features

- Real-time face detection on embedded hardware
- INT8 quantized model (~17 KB)
- Live browser preview via MJPEG stream
- Audio feedback via piezo buzzer
- Works in varied lighting conditions (shadows, backlight)

## Hardware

- Arduino Nano 33 BLE Sense Rev2
- ArduCAM Mini 2MP OV2640
- Piezo buzzer (optional)

## Quick Start

1. **Upload firmware**: Open `G_arduino_firmware/G_arduino_firmware.ino` in Arduino IDE and upload
2. **Run preview server**:
   ```bash
   pip install -r requirements.txt
   python3 preview_server.py
   ```
3. **Open browser**: http://localhost:7654

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed instructions.

## Performance

- **Accuracy**: 98.97% (test set), ~96% (real-world estimate)
- **Model size**: 17.55 KB
- **Inference time**: ~150ms
- **Input**: 96×96 grayscale

## License

MIT

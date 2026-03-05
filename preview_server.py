#!/usr/bin/env python3
"""
Live Camera Preview Server for ArduCAM + Arduino Nano 33 BLE Sense
===================================================================
Reads JPEG frames streamed from the Arduino over Serial and serves
them as a true MJPEG stream.  No page-refresh polling — frames are
pushed directly to the browser as they arrive.

Architecture:
  serial_thread    — reads raw bytes from Arduino, assembles JPEG frames
  annotate_thread  — pre-annotates each new frame once (Haar + status bar)
  HTTP threads     — one per client, serve from the pre-annotated cache

Usage:
    python3 preview_server.py

Then open:  http://localhost:7654
"""

import re
import serial
import threading
import time
import cv2
import numpy as np
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

# ─── Config ───────────────────────────────────────────────────────────────────
SERIAL_PORT   = '/dev/cu.usbmodem101'
BAUD_RATE     = 115200
HTTP_PORT     = 7654
DISPLAY_WIDTH = 480   # px — frames are scaled to this width

FRAME_START = bytes([0xFF, 0xAA])
FRAME_END   = bytes([0xFF, 0xBB])

# Haar cascade for face bounding-box overlay
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade  = cv2.CascadeClassifier(_cascade_path)
# ──────────────────────────────────────────────────────────────────────────────

# Shared state — written by serial / annotation threads, read by HTTP threads
lock                  = threading.Lock()
latest_jpeg           = None          # raw frame from Arduino
latest_annotated_jpeg = None          # pre-annotated frame ready to serve
latest_result         = "Waiting..."
latest_confidence     = "---"

# Signals annotation thread when a new raw frame arrives
new_frame_event = threading.Event()


# ─── Serial Reader Thread ─────────────────────────────────────────────────────
def serial_reader():
    global latest_jpeg, latest_result, latest_confidence

    print(f"Opening serial port {SERIAL_PORT} ...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        ser.dtr = True
    except serial.SerialException as e:
        print(f"ERROR: Cannot open serial port: {e}")
        return

    print("Serial connected. Waiting for frames...")
    buf = bytearray()

    while True:
        try:
            # Read all buffered bytes at once — far fewer syscalls than fixed chunks
            waiting = ser.in_waiting
            chunk   = ser.read(waiting if waiting > 0 else 4096)
            if not chunk:
                continue
            buf.extend(chunk)

            # Parse text lines only from the region before any JPEG marker
            # (JPEG binary data contains 0x0A bytes that corrupt text parsing)
            frame_pos = buf.find(FRAME_START)
            text_end  = frame_pos if frame_pos != -1 else len(buf)
            while b'\n' in buf[:text_end]:
                nl = buf.index(b'\n')
                if nl >= text_end:
                    break
                line     = buf[:nl].decode('utf-8', errors='replace').strip()
                buf      = buf[nl + 1:]
                text_end -= nl + 1
                if line.startswith('RESULT:'):
                    with lock:
                        latest_result = line.replace('RESULT:', '').strip()
                    print(f"  {line}")
                elif line.startswith('CONFIDENCE:'):
                    with lock:
                        latest_confidence = line.replace('CONFIDENCE:', '').strip()
                    print(f"  {line}")
                elif line.startswith('Vote:'):
                    if 'NO FACE' in line:
                        with lock:
                            latest_result = 'No Face Detected'
                        print(f'  {line}')
                    elif 'FACE' in line:
                        with lock:
                            latest_result = 'Face Detected'
                        print(f'  {line}')
                elif line.startswith('Stage A:'):
                    m = re.search(r'\(([0-9.]+)%\)', line)
                    if m:
                        with lock:
                            latest_confidence = m.group(1) + '%'
                    print(f'  {line}')

            # Extract complete JPEG frames
            while True:
                start = buf.find(FRAME_START)
                if start == -1:
                    break
                if len(buf) < start + 6:
                    break
                length     = int.from_bytes(buf[start + 2:start + 6], 'little')
                end_needed = start + 6 + length + 2
                if len(buf) < end_needed:
                    break
                end_marker = buf[start + 6 + length: start + 6 + length + 2]
                if end_marker == FRAME_END:
                    with lock:
                        latest_jpeg = bytes(buf[start + 6: start + 6 + length])
                    new_frame_event.set()   # wake annotation thread
                    buf = buf[end_needed:]
                else:
                    buf = buf[start + 1:]   # bad frame — skip one byte and retry

            # Keep buffer bounded (200 KB max)
            if len(buf) > 200_000:
                buf = buf[-100_000:]

        except Exception as e:
            print(f"Serial error: {e}")
            time.sleep(1)


# ─── Frame Annotation ─────────────────────────────────────────────────────────
def annotate_frame(jpeg_data, result, confidence):
    """
    Decode the JPEG, scale to display width, overlay:
      - green bounding box around detected face (Haar cascade)
      - status bar at the bottom of the frame
    Re-encode and return as JPEG bytes.
    """
    arr = np.frombuffer(jpeg_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_data

    # Scale to display width — INTER_LINEAR is fast and looks good at this size
    h, w  = img.shape[:2]
    new_h = max(1, int(h * DISPLAY_WIDTH / w))
    img   = cv2.resize(img, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)
    h, w  = img.shape[:2]

    is_detected = (result == 'Face Detected')

    # Green bounding box — always run Haar cascade independently of TFLite model
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 210, 80), 2)

    # Semi-transparent status bar at the bottom
    bar_h   = 36
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    if is_detected:
        label = 'Face Detected'
        if confidence != '---':
            label += f'   {confidence}'
        color = (60, 215, 100)          # green
    elif result.startswith('Waiting'):
        label = 'Waiting for frame...'
        color = (100, 100, 100)         # grey
    else:
        label = 'No Face Detected'
        color = (60, 145, 255)          # orange

    cv2.putText(img, label, (10, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ─── Annotation Thread ────────────────────────────────────────────────────────
def annotation_worker():
    """
    Pre-annotates each new raw frame exactly once.
    HTTP threads read from latest_annotated_jpeg — no duplicate Haar runs.
    """
    global latest_annotated_jpeg
    last_raw = None

    while True:
        new_frame_event.wait(timeout=1.0)
        new_frame_event.clear()

        with lock:
            raw    = latest_jpeg
            result = latest_result
            conf   = latest_confidence

        if raw is None or raw is last_raw:
            continue

        last_raw   = raw
        annotated  = annotate_frame(raw, result, conf)

        with lock:
            latest_annotated_jpeg = annotated


# ─── HTML Page ────────────────────────────────────────────────────────────────
HTML_PAGE = b"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Detection \xe2\x80\x94 Live Preview</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #0d0d0d;
      color: #c0c0c0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 22px;
      padding: 40px 20px;
    }

    h1 {
      font-size: 0.8rem;
      font-weight: 400;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #484848;
    }

    .preview {
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #1e1e1e;
      box-shadow: 0 16px 56px rgba(0, 0, 0, 0.75);
    }

    .preview img {
      display: block;
      width: 480px;
      height: auto;
    }

    footer {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.7rem;
      color: #333;
      letter-spacing: 0.07em;
    }

    .live-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #2d6e3a;
      animation: pulse 1.6s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.2; }
    }
  </style>
</head>
<body>
  <h1>TinyML Face Detection &mdash; Live Preview</h1>

  <div class="preview">
    <img src="/stream.mjpeg" alt="Live camera feed">
  </div>

  <footer>
    <div class="live-dot"></div>
    ArduCAM OV2640 &nbsp;&middot;&nbsp; Arduino Nano 33 BLE Sense Rev2
  </footer>
</body>
</html>
"""


# ─── HTTP Handler ─────────────────────────────────────────────────────────────
class PreviewHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self._send(200, 'text/html; charset=utf-8', HTML_PAGE)

        elif self.path == '/stream.mjpeg':
            self._stream_mjpeg()

        elif self.path.startswith('/frame.jpg'):
            # Single-frame fallback (kept for compatibility)
            with lock:
                data = latest_annotated_jpeg or latest_jpeg
            self._send(200, 'image/jpeg', data if data else self._placeholder())

        elif self.path == '/status':
            import json
            with lock:
                payload = json.dumps({
                    'result':     latest_result,
                    'confidence': latest_confidence,
                })
            self._send(200, 'application/json', payload.encode())

        else:
            self._send(404, 'text/plain', b'Not found')

    def _stream_mjpeg(self):
        """Push pre-annotated JPEG frames as a multipart MJPEG stream."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache, no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        last_frame = None
        try:
            while True:
                with lock:
                    annotated = latest_annotated_jpeg

                if annotated is not None and annotated is not last_frame:
                    last_frame = annotated
                    chunk = (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(annotated)).encode() + b'\r\n\r\n'
                        + annotated + b'\r\n'
                    )
                    self.wfile.write(chunk)
                    self.wfile.flush()

                time.sleep(0.005)   # 200 Hz poll; only sends on new pre-annotated frame

        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache, no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _placeholder(self):
        """1×1 grey JPEG placeholder before the first frame arrives."""
        return bytes([
            0xff,0xd8,0xff,0xe0,0x00,0x10,0x4a,0x46,0x49,0x46,0x00,0x01,
            0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xff,0xdb,0x00,0x43,
            0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
            0x09,0x08,0x0a,0x0c,0x14,0x0d,0x0c,0x0b,0x0b,0x0c,0x19,0x12,
            0x13,0x0f,0x14,0x1d,0x1a,0x1f,0x1e,0x1d,0x1a,0x1c,0x1c,0x20,
            0x24,0x2e,0x27,0x20,0x22,0x2c,0x23,0x1c,0x1c,0x28,0x37,0x29,
            0x2c,0x30,0x31,0x34,0x34,0x34,0x1f,0x27,0x39,0x3d,0x38,0x32,
            0x3c,0x2e,0x33,0x34,0x32,0xff,0xc0,0x00,0x0b,0x08,0x00,0x01,
            0x00,0x01,0x01,0x01,0x11,0x00,0xff,0xc4,0x00,0x1f,0x00,0x00,
            0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
            0x09,0x0a,0x0b,0xff,0xc4,0x00,0xb5,0x10,0x00,0x02,0x01,0x03,
            0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7d,
            0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
            0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
            0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,
            0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
            0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,
            0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
            0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,
            0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
            0x8a,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,
            0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,
            0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,
            0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,
            0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,
            0xf6,0xf7,0xf8,0xf9,0xfa,0xff,0xda,0x00,0x08,0x01,0x01,0x00,
            0x00,0x3f,0x00,0xf5,0x0a,0x28,0xa2,0xff,0xd9
        ])

    def log_message(self, *args):
        pass  # suppress per-request logs


# ─── Threaded Server ──────────────────────────────────────────────────────────
class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Each MJPEG client gets its own thread — no head-of-line blocking."""
    daemon_threads = True


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=serial_reader,    daemon=True).start()
    threading.Thread(target=annotation_worker, daemon=True).start()

    server = ThreadedHTTPServer(('localhost', HTTP_PORT), PreviewHandler)
    print(f"\n{'='*55}")
    print(f"  ArduCAM Live Preview Server")
    print(f"{'='*55}")
    print(f"  HTTP server: http://localhost:{HTTP_PORT}")
    print(f"\n  In VS Code:  Ctrl+Shift+P")
    print(f"               → 'Simple Browser: Show'")
    print(f"               → http://localhost:{HTTP_PORT}")
    print(f"\n  Press Ctrl+C to stop")
    print(f"{'='*55}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

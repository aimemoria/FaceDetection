/*
 * G. Arduino Firmware — Face Detection Only
 *
 * Target Hardware:
 *   - Arduino Nano 33 BLE Sense Rev2
 *   - ArduCam Mini 2MP OV2640 with 8MB FIFO
 *   - Piezo buzzer on pin D6  (optional, for audio feedback)
 *
 * Pipeline:
 *   Stage A: Person Detection — binary: person vs no_person
 *
 * Flow:
 *   1. Capture 96x96 grayscale image from ArduCam
 *   2. Run Stage A → no person? → Serial + 1 short beep, stop
 *                  → person?    → Serial + 2 beeps
 *
 * Serial feedback format (115200 baud):
 *   RESULT: Face Detected
 *   CONFIDENCE: 95%
 *
 *   RESULT: No face detected
 *   CONFIDENCE: ---
 *
 * Buzzer patterns:
 *   No person — 1 short low beep  (200 Hz)
 *   Person    — 2 short high beeps (1000 Hz)
 */

// =============================================================================
// Libraries
// =============================================================================
#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>
#include "memorysaver.h"

// mbed defines swap(a,b,size) as a 3-arg macro; undefine it so
// the TFLite/STL std::swap templates compile without conflict.
#ifdef swap
#undef swap
#endif

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "stage_a_model.h"

// =============================================================================
// Configuration
// =============================================================================
#define CS_PIN              7       // ArduCam chip-select
#define BUZZER_PIN          6       // Piezo buzzer (optional)

#define IMG_WIDTH           96
#define IMG_HEIGHT          96
#define IMG_CHANNELS        1

#define ARENA_A_SIZE  (48 * 1024)   // 48 KB for Stage A
#define INFERENCE_INTERVAL_MS  2000  // ms gap between inference cycles (preview streams during gap)

// =============================================================================
// Buzzer patterns  {freq, duration_ms, pause_ms, ...}  — zero-terminated
// =============================================================================
const int BEEP_PATTERNS[][10] = {
  // 0: No person — single low beep
  {200,  100, 0,    0,    0,  0, 0, 0, 0, 0},
  // 1: Person    — two short high beeps
  {1000, 100, 50, 1000, 100,  0, 0, 0, 0, 0},
};

#define PATTERN_NO_PERSON   0
#define PATTERN_PERSON      1

// =============================================================================
// Test-mode placeholder JPEG (96x96 gray grid, 879 bytes, stored in flash)
// Streamed instead of a live camera frame when camera_ok == false.
// =============================================================================
static const uint8_t TEST_JPEG[] PROGMEM = {
  0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
  0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x1B, 0x12, 0x14, 0x17, 0x14, 0x11, 0x1B,
  0x17, 0x16, 0x17, 0x1E, 0x1C, 0x1B, 0x20, 0x28, 0x42, 0x2B, 0x28, 0x25, 0x25, 0x28, 0x51, 0x3A,
  0x3D, 0x30, 0x42, 0x60, 0x55, 0x65, 0x64, 0x5F, 0x55, 0x5D, 0x5B, 0x6A, 0x78, 0x99, 0x81, 0x6A,
  0x71, 0x90, 0x73, 0x5B, 0x5D, 0x85, 0xB5, 0x86, 0x90, 0x9E, 0xA3, 0xAB, 0xAD, 0xAB, 0x67, 0x80,
  0xBC, 0xC9, 0xBA, 0xA6, 0xC7, 0x99, 0xA8, 0xAB, 0xA4, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x60,
  0x00, 0x60, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
  0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
  0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
  0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00,
  0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
  0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
  0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
  0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
  0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
  0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94,
  0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2,
  0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
  0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6,
  0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA,
  0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xC3, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4,
  0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24,
  0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1,
  0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x38, 0xC9, 0xE4, 0xFE, 0x54, 0x71, 0x83, 0xC9, 0xFC, 0xA8, 0xE3,
  0xD4, 0xFE, 0x54, 0x1C, 0x64, 0xF2, 0x7F, 0x2A, 0x38, 0xC1, 0xE4, 0xFE, 0x54, 0x71, 0xEA, 0x7F,
  0x2A, 0x0E, 0x32, 0x79, 0x3F, 0x95, 0x1C, 0x60, 0xF2, 0x7F, 0x2A, 0x38, 0xF5, 0x3F, 0x95, 0x07,
  0x19, 0x3C, 0x9F, 0xCA, 0x8E, 0x30, 0x79, 0x3F, 0x95, 0x1C, 0x7A, 0x9F, 0xCA, 0x86, 0xFB, 0xC7,
  0xEB, 0x47, 0xF0, 0x9F, 0xAD, 0x25, 0x2B, 0x7D, 0xE3, 0xF5, 0xA3, 0xF8, 0x4F, 0xD6, 0x92, 0x95,
  0xBE, 0xF1, 0xFA, 0xD1, 0xFC, 0x27, 0xEB, 0x49, 0x4A, 0xDF, 0x78, 0xFD, 0x68, 0xFE, 0x13, 0xF5,
  0xA4, 0xA5, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2,
  0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1,
  0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x38, 0xC9,
  0xE4, 0xFE, 0x54, 0x71, 0x83, 0xC9, 0xFC, 0xA8, 0xE3, 0xD4, 0xFE, 0x54, 0x1C, 0x64, 0xF2, 0x7F,
  0x2A, 0x38, 0xC1, 0xE4, 0xFE, 0x54, 0x71, 0xEA, 0x7F, 0x2A, 0x0E, 0x32, 0x79, 0x3F, 0x95, 0x1C,
  0x60, 0xF2, 0x7F, 0x2A, 0x38, 0xF5, 0x3F, 0x95, 0x07, 0x19, 0x3C, 0x9F, 0xCA, 0x8E, 0x30, 0x79,
  0x3F, 0x95, 0x1C, 0x7A, 0x9F, 0xCA, 0x86, 0xFB, 0xC7, 0xEB, 0x47, 0xF0, 0x9F, 0xAD, 0x25, 0x2B,
  0x7D, 0xE3, 0xF5, 0xA3, 0xF8, 0x4F, 0xD6, 0x92, 0x95, 0xBE, 0xF1, 0xFA, 0xD1, 0xFC, 0x27, 0xEB,
  0x49, 0x4A, 0xDF, 0x78, 0xFD, 0x68, 0xFE, 0x13, 0xF5, 0xA4, 0xA5, 0x24, 0xE4, 0xF2, 0x68, 0xC9,
  0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA,
  0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68,
  0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x38, 0xC9, 0xE4, 0xFE, 0x54, 0x71, 0x83, 0xC9, 0xFC,
  0xA8, 0xE3, 0xD4, 0xFE, 0x54, 0x1C, 0x64, 0xF2, 0x7F, 0x2A, 0x38, 0xC1, 0xE4, 0xFE, 0x54, 0x71,
  0xEA, 0x7F, 0x2A, 0x0E, 0x32, 0x79, 0x3F, 0x95, 0x1C, 0x60, 0xF2, 0x7F, 0x2A, 0x38, 0xF5, 0x3F,
  0x95, 0x07, 0x19, 0x3C, 0x9F, 0xCA, 0x8E, 0x30, 0x79, 0x3F, 0x95, 0x1C, 0x7A, 0x9F, 0xCA, 0x86,
  0xFB, 0xC7, 0xEB, 0x47, 0xF0, 0x9F, 0xAD, 0x25, 0x2B, 0x7D, 0xE3, 0xF5, 0xA3, 0xF8, 0x4F, 0xD6,
  0x92, 0x95, 0xBE, 0xF1, 0xFA, 0xD1, 0xFC, 0x27, 0xEB, 0x49, 0x4A, 0xDF, 0x78, 0xFD, 0x68, 0xFE,
  0x13, 0xF5, 0xA4, 0xA5, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24,
  0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1,
  0xE4, 0xD1, 0x93, 0xEA, 0x68, 0x24, 0xE4, 0xF2, 0x68, 0xC9, 0xC1, 0xE4, 0xD1, 0x93, 0xEA, 0x68,
  0x38, 0xC9, 0xE4, 0xFE, 0x54, 0x71, 0x83, 0xC9, 0xFC, 0xA8, 0xE3, 0xD4, 0xFE, 0x54, 0x1C, 0x64,
  0xF2, 0x7F, 0x2A, 0x38, 0xC1, 0xE4, 0xFE, 0x54, 0x71, 0xEA, 0x7F, 0x2A, 0x0E, 0x32, 0x79, 0x3F,
  0x95, 0x1C, 0x60, 0xF2, 0x7F, 0x2A, 0x38, 0xF5, 0x3F, 0x95, 0x07, 0x19, 0x3C, 0x9F, 0xCA, 0x8E,
  0x30, 0x79, 0x3F, 0x95, 0x1C, 0x7A, 0x9F, 0xCA, 0x86, 0xFB, 0xC7, 0xEB, 0x47, 0xF0, 0x9F, 0xAD,
  0x25, 0x2B, 0x7D, 0xE3, 0xF5, 0xA3, 0xF8, 0x4F, 0xD6, 0x92, 0x95, 0xBE, 0xF1, 0xFA, 0xD1, 0xFC,
  0x27, 0xEB, 0x49, 0x4A, 0xDF, 0x78, 0xFD, 0x68, 0xFE, 0x13, 0xF5, 0xA4, 0xAF, 0xFF, 0xD9
};
static const uint32_t TEST_JPEG_LEN = 879;

// =============================================================================
// Global Variables
// =============================================================================
ArduCAM myCAM(OV2640, CS_PIN);

bool camera_ok = false;

const tflite::Model*      model_stage_a = nullptr;
tflite::MicroInterpreter* interp_a      = nullptr;

alignas(16) uint8_t tensor_arena_a[ARENA_A_SIZE];
uint8_t image_buffer[IMG_WIDTH * IMG_HEIGHT];

unsigned long last_inference_time = 0;

// Temporal smoothing — 3-frame majority vote
#define VOTE_WINDOW 3
int vote_buffer[VOTE_WINDOW] = {0, 0, 0};  // 1=person, 0=no person
int vote_index = 0;
int last_reported = -1;  // -1=never reported

// =============================================================================
// Forward declarations
// =============================================================================
void setupCamera();
void setupTFLite();
bool captureImage();
bool captureAndCropFace();  // NEW: capture + Haar-like face crop
void restoreJPEGMode();
void preprocessImage();
int  runStageA();
void playBeepPattern(int pattern_index);
void printSeparator();
void streamPreviewJPEG();
void recoverI2CBus();

// =============================================================================
// Setup
// =============================================================================
void setup() {
  Serial.begin(115200);
  // Wait for a terminal to open before proceeding — ensures all startup prints
  // are visible and prevents USB CDC TX buffer from filling before any reader.
  delay(1500);  // Give USB CDC time to enumerate — while(!Serial) hangs on mbed

  Serial.println("\n========================================");
  Serial.println("  TinyML Face Detection System");
  Serial.println("  Arduino Nano 33 BLE Sense Rev2");
  Serial.println("========================================");

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  tone(BUZZER_PIN, 1000, 100); delay(150);
  tone(BUZZER_PIN, 1200, 100); delay(150);
  tone(BUZZER_PIN, 1500, 150); delay(200);

  // Recover the I2C bus before initialising Wire — the OV2640 may be holding
  // SDA low if a previous session's transaction was interrupted mid-byte.
  recoverI2CBus();
  Wire.begin();
  SPI.begin();

  Serial.println("Initialising camera ...");
  setupCamera();

  Serial.println("Initialising TensorFlow Lite ...");
  setupTFLite();

  Serial.println("\nSystem ready — detecting ...\n");
}

// =============================================================================
// Main Loop
// =============================================================================
void loop() {
  unsigned long now = millis();
  if (now - last_inference_time >= INFERENCE_INTERVAL_MS) {
    runInference();
    last_inference_time = millis();  // reset AFTER inference — gap starts here
    streamPreviewJPEG();
  } else {
    streamPreviewJPEG();  // stream preview frames during the gap between inferences
  }
}

// =============================================================================
// Inference Pipeline — Stage A with 3-frame majority vote
// =============================================================================
void runInference() {
  printSeparator();
  unsigned long t0 = millis();

  // Warm-up capture: discard one frame so OV2640 auto-exposure settles
  // before the frame we actually run inference on.
  if (camera_ok) {
    captureAndCropFace();
  }

  // Inference capture
  if (!captureAndCropFace()) {
    Serial.println("ERROR: Camera capture failed");
    playBeepPattern(PATTERN_NO_PERSON);
    return;
  }
  preprocessImage();

  int stage_a = runStageA();

  // Store result in rolling vote buffer
  vote_buffer[vote_index] = (stage_a > 0) ? 1 : 0;
  vote_index = (vote_index + 1) % VOTE_WINDOW;

  // Count votes — need 2 of 3 frames to agree
  int votes_yes = 0;
  for (int i = 0; i < VOTE_WINDOW; i++) votes_yes += vote_buffer[i];
  int majority = (votes_yes >= 1) ? 1 : 0;  // 1 of 3 frames enough

  Serial.print("Vote: "); Serial.print(votes_yes);
  Serial.print("/"); Serial.print(VOTE_WINDOW);
  Serial.println(majority ? " -> FACE" : " -> NO FACE");

  // Only beep/report when the decision changes to avoid repeated alerts
  if (majority != last_reported) {
    last_reported = majority;
    if (majority == 0) {
      Serial.println("RESULT: No face detected");
      Serial.println("CONFIDENCE: ---");
      playBeepPattern(PATTERN_NO_PERSON);
    } else {
      TfLiteTensor* output = interp_a->output(0);
      float p_yes;
      if (output->type == kTfLiteInt8) {
        float s   = output->params.scale;
        int32_t z = output->params.zero_point;
        p_yes = (output->data.int8[1] - z) * s;
      } else {
        p_yes = output->data.f[1];
      }
      Serial.println("RESULT: Face Detected");
      Serial.print("CONFIDENCE: ");
      Serial.print((int)(p_yes * 100.0f));
      Serial.println("%");
      playBeepPattern(PATTERN_PERSON);
    }
  }

  Serial.print("Inference time: ");
  Serial.print(millis() - t0);
  Serial.println(" ms");
  printSeparator();
}

// =============================================================================
// I2C Bus Recovery
// Clocks SCL 9 times to release the OV2640 if it is holding SDA low
// from a previously interrupted I2C transaction.  Must be called BEFORE
// Wire.begin() so we can drive the I2C pins manually as GPIO.
// =============================================================================
void recoverI2CBus() {
  // SDA / SCL pin numbers for the Arduino Nano 33 BLE connector (A4=18, A5=19).
  const uint8_t SDA_P = PIN_WIRE_SDA;   // 18
  const uint8_t SCL_P = PIN_WIRE_SCL;   // 19

  pinMode(SDA_P, INPUT_PULLUP);
  pinMode(SCL_P, OUTPUT);

  // Send 9 clock pulses — enough to complete any in-progress byte the slave
  // is waiting for, after which it will release SDA.
  for (int i = 0; i < 9; i++) {
    digitalWrite(SCL_P, HIGH); delayMicroseconds(10);
    digitalWrite(SCL_P, LOW);  delayMicroseconds(10);
  }

  // Generate a STOP condition: SDA LOW → SCL HIGH → SDA HIGH
  pinMode(SDA_P, OUTPUT);
  digitalWrite(SDA_P, LOW);
  delayMicroseconds(10);
  digitalWrite(SCL_P, HIGH);
  delayMicroseconds(10);
  digitalWrite(SDA_P, HIGH);
  delayMicroseconds(10);

  // Release both pins — Wire.begin() will reconfigure them as I2C
  pinMode(SDA_P, INPUT);
  pinMode(SCL_P, INPUT);
}

// =============================================================================
// Camera Setup
// =============================================================================
void setupCamera() {
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  if (myCAM.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("WARNING: SPI interface test failed - camera unavailable");
    Serial.println(">>> Running in TEST MODE with synthetic image <<<");
    camera_ok = false;
    return;
  }

  uint8_t vid, pid;
  myCAM.wrSensorReg8_8(0xFF, 0x01);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW,  &pid);

  if (vid != 0x26 || (pid != 0x41 && pid != 0x42)) {
    Serial.println("WARNING: OV2640 not detected - camera unavailable");
    Serial.println(">>> Running in TEST MODE with synthetic image <<<");
    camera_ok = false;
    return;
  }

  Serial.println("Camera: OV2640 OK");
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);   // smaller JPEG = less serial lag
  delay(100);
  myCAM.clear_fifo_flag();
  camera_ok = true;
}

// =============================================================================
// TFLite Setup — Stage A only
// =============================================================================
void setupTFLite() {
  static tflite::AllOpsResolver resolver;

  model_stage_a = tflite::GetModel(stage_a_model);
  if (model_stage_a->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Stage A schema mismatch!"); while (1);
  }
  Serial.print("Stage A model: ");
  Serial.print(stage_a_model_len);
  Serial.println(" bytes in flash");

  static tflite::MicroInterpreter si_a(
    model_stage_a, resolver, tensor_arena_a, ARENA_A_SIZE);
  interp_a = &si_a;
  if (interp_a->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Stage A tensor allocation failed!"); while (1);
  }
  Serial.print("Stage A arena used: ");
  Serial.print(interp_a->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(ARENA_A_SIZE);
  Serial.println(" bytes");

}

// =============================================================================
// Image Capture
// =============================================================================
void restoreJPEGMode() {
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);   // smaller JPEG = less serial lag
  delay(100);
  myCAM.clear_fifo_flag();
}

bool captureImage() {
  if (!camera_ok) {
    memset(image_buffer, 128, IMG_WIDTH * IMG_HEIGHT);
    return true;
  }

  myCAM.set_format(BMP);
  myCAM.InitCAM();
  // Do NOT call OV2640_set_JPEG_size in BMP mode — it corrupts the output window.
  // BMP mode after InitCAM() defaults to 320x240 (QVGA).
  delay(100);

  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  unsigned long t = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t > 3000) {
      myCAM.CS_HIGH();
      restoreJPEGMode();
      Serial.println("Capture timeout");
      return false;
    }
  }

  // Read FIFO length to detect actual BMP resolution.
  // ArduCAM may add a small header, so accept +64 byte tolerance around expected sizes.
  uint32_t fifo_len = myCAM.read_fifo_length();
  int SRC_W, SRC_H;
  const uint32_t QVGA = 320u * 240u * 2u;   // 153600 bytes (320x240 RGB565)
  const uint32_t QQVGA = 160u * 120u * 2u;  //  38400 bytes (160x120 RGB565)
  if (fifo_len >= QVGA && fifo_len <= QVGA + 64u) {
    SRC_W = 320; SRC_H = 240;
  } else if (fifo_len >= QQVGA && fifo_len <= QQVGA + 64u) {
    SRC_W = 160; SRC_H = 120;
  } else {
    Serial.print("BMP FIFO unexpected: "); Serial.println(fifo_len);
    myCAM.CS_HIGH();
    restoreJPEGMode();
    return false;
  }
  myCAM.CS_LOW();
  myCAM.set_fifo_burst();

  // Read one row at a time using bulk SPI transfer (far fewer API calls than per-byte).
  // Row buffer is static to avoid stack allocation; sent as MOSI (value doesn't matter
  // in burst read mode — only SCLK and CS matter for the FIFO controller).
  static uint8_t row_buf[320 * 2];   // max row width: 320 pixels × 2 bytes

  // Nearest-neighbor downscale: SRC_W × SRC_H → IMG_WIDTH × IMG_HEIGHT.
  // Uses the FULL camera frame (not a centre-crop), so faces at any position/scale
  // are visible to the model.  For each output row, the matching source row is
  //   r_in = (r_out * SRC_H) / IMG_HEIGHT   (integer floor)
  // We track next_r_out and consume source rows sequentially from the FIFO.
  int out = 0;
  int next_r_out = 0;
  for (int r = 0; r < SRC_H; r++) {
    SPI.transfer(row_buf, SRC_W * 2);   // read one full row (must always consume)
    if (next_r_out < IMG_HEIGHT) {
      int r_in_needed = (next_r_out * SRC_H) / IMG_HEIGHT;
      if (r == r_in_needed) {
        for (int c_out = 0; c_out < IMG_WIDTH; c_out++) {
          int c_in = (c_out * SRC_W) / IMG_WIDTH;
          // OV2640 BMP outputs RGB565 in little-endian order (low byte first).
          uint8_t lo = row_buf[c_in * 2];
          uint8_t hi = row_buf[c_in * 2 + 1];
          uint8_t r8 = (hi & 0xF8);
          uint8_t g8 = ((hi & 0x07) << 5) | ((lo & 0xE0) >> 3);
          uint8_t b8 = (lo & 0x1F) << 3;
          image_buffer[out++] = (uint8_t)((77u * r8 + 150u * g8 + 29u * b8) >> 8);
        }
        next_r_out++;
      }
    }
  }

  myCAM.CS_HIGH();
  restoreJPEGMode();

  // Debug: pixel stats + 20x20 ASCII preview of what the model sees
  uint32_t sum = 0;
  uint8_t mn = 255, mx = 0;
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    sum += image_buffer[i];
    if (image_buffer[i] < mn) mn = image_buffer[i];
    if (image_buffer[i] > mx) mx = image_buffer[i];
  }
  Serial.print("Pixels min="); Serial.print(mn);
  Serial.print(" max="); Serial.print(mx);
  Serial.print(" avg="); Serial.println(sum / (IMG_WIDTH * IMG_HEIGHT));

  // 20x20 ASCII art of the 96x96 capture
  Serial.println("--- 20x20 preview (darker=dark, .=mid, #=bright) ---");
  for (int r = 0; r < 20; r++) {
    for (int c = 0; c < 20; c++) {
      uint8_t px = image_buffer[(r * IMG_HEIGHT / 20) * IMG_WIDTH + (c * IMG_WIDTH / 20)];
      char ch;
      if      (px < 40)  ch = ' ';
      else if (px < 80)  ch = '.';
      else if (px < 120) ch = '+';
      else if (px < 160) ch = '*';
      else if (px < 200) ch = '#';
      else               ch = '@';
      Serial.print(ch);
    }
    Serial.println();
  }
  Serial.println("-----");

  return (out == IMG_WIDTH * IMG_HEIGHT);
}

// =============================================================================
// Preprocessing with Histogram Equalization
// This normalizes lighting to handle shadows, backlighting, and dark rooms.
// The Olivetti training data has well-lit faces with good contrast, so we
// need to transform real camera images to match that distribution.
// =============================================================================
void preprocessImage() {
  // 1. Build histogram
  uint16_t hist[256] = {0};
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    hist[image_buffer[i]]++;
  }

  // 2. Build cumulative distribution function (CDF)
  uint32_t cdf[256];
  cdf[0] = hist[0];
  for (int i = 1; i < 256; i++) {
    cdf[i] = cdf[i-1] + hist[i];
  }

  // 3. Find CDF minimum (first non-zero)
  uint32_t cdf_min = 0;
  for (int i = 0; i < 256; i++) {
    if (cdf[i] > 0) {
      cdf_min = cdf[i];
      break;
    }
  }

  // 4. Apply equalization: new_val = (cdf[val] - cdf_min) * 255 / (total - cdf_min)
  uint32_t total = IMG_WIDTH * IMG_HEIGHT;
  uint32_t denom = total - cdf_min;
  if (denom == 0) denom = 1;  // avoid div by zero

  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    uint32_t new_val = ((cdf[image_buffer[i]] - cdf_min) * 255) / denom;
    image_buffer[i] = (uint8_t)(new_val > 255 ? 255 : new_val);
  }

  Serial.println("Applied histogram equalization");
}

// =============================================================================
// Simple contrast-based face region detection
// Scans the captured image for the brightest region (likely a face under
// normal lighting) and crops to that area. This mimics what the training
// pipeline's Haar cascade did: deliver a tightly-cropped face to the model.
// =============================================================================

// Temporary buffer for full-frame capture (we'll crop from this)
static uint8_t full_frame[160 * 120];  // QQVGA grayscale
#define FULL_W 160
#define FULL_H 120

bool captureAndCropFace() {
  if (!camera_ok) {
    // No camera — fill with mid-gray (test mode)
    memset(image_buffer, 128, IMG_WIDTH * IMG_HEIGHT);
    return true;
  }

  // 1. Capture at QQVGA (160x120) for faster processing
  myCAM.set_format(BMP);
  myCAM.InitCAM();
  delay(100);

  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  unsigned long t = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t > 3000) {
      myCAM.CS_HIGH();
      restoreJPEGMode();
      Serial.println("Capture timeout");
      return false;
    }
  }

  uint32_t fifo_len = myCAM.read_fifo_length();
  
  // Determine actual resolution from FIFO length
  int SRC_W, SRC_H;
  const uint32_t QVGA_SIZE  = 320u * 240u * 2u;   // 153600
  const uint32_t QQVGA_SIZE = 160u * 120u * 2u;   //  38400
  
  if (fifo_len >= QVGA_SIZE && fifo_len <= QVGA_SIZE + 64u) {
    SRC_W = 320; SRC_H = 240;
  } else if (fifo_len >= QQVGA_SIZE && fifo_len <= QQVGA_SIZE + 64u) {
    SRC_W = 160; SRC_H = 120;
  } else {
    Serial.print("Unexpected FIFO: "); Serial.println(fifo_len);
    myCAM.CS_HIGH();
    restoreJPEGMode();
    return false;
  }

  myCAM.CS_LOW();
  myCAM.set_fifo_burst();

  // Read and convert to grayscale into full_frame buffer (downscale if QVGA)
  static uint8_t row_buf[320 * 2];
  int ff_idx = 0;
  
  for (int r = 0; r < SRC_H; r++) {
    SPI.transfer(row_buf, SRC_W * 2);
    
    // Map to 160x120 output
    int out_r = (r * FULL_H) / SRC_H;
    if (out_r >= FULL_H) continue;
    
    // Only process rows we need (simple nearest-neighbor)
    int expected_r = (out_r * SRC_H) / FULL_H;
    if (r != expected_r) continue;
    
    for (int c_out = 0; c_out < FULL_W; c_out++) {
      int c_in = (c_out * SRC_W) / FULL_W;
      uint8_t lo = row_buf[c_in * 2];
      uint8_t hi = row_buf[c_in * 2 + 1];
      uint8_t r8 = (hi & 0xF8);
      uint8_t g8 = ((hi & 0x07) << 5) | ((lo & 0xE0) >> 3);
      uint8_t b8 = (lo & 0x1F) << 3;
      full_frame[out_r * FULL_W + c_out] = (uint8_t)((77u * r8 + 150u * g8 + 29u * b8) >> 8);
    }
  }
  myCAM.CS_HIGH();
  restoreJPEGMode();

  // SIMPLIFIED APPROACH: Just take the center 96x96 region directly
  // The model should be trained on full scenes, not cropped faces
  // This avoids complex face detection that may fail
  
  // Calculate center crop coordinates
  // From 160x120, take center region that maps to 96x96
  int crop_w = 96;  // We want a square crop
  int crop_h = 96;
  int start_x = (FULL_W - crop_w) / 2;  // 32
  int start_y = (FULL_H - crop_h) / 2;  // 12
  
  // If source is smaller, adjust
  if (crop_h > FULL_H) {
    crop_h = FULL_H;
    start_y = 0;
  }
  if (crop_w > FULL_W) {
    crop_w = FULL_W;
    start_x = 0;
  }

  Serial.print("Center crop: x="); Serial.print(start_x);
  Serial.print(" y="); Serial.print(start_y);
  Serial.print(" w="); Serial.print(crop_w);
  Serial.print(" h="); Serial.println(crop_h);

  // Extract center region and resize to 96x96
  for (int r = 0; r < IMG_HEIGHT; r++) {
    for (int c = 0; c < IMG_WIDTH; c++) {
      int src_r = start_y + (r * crop_h) / IMG_HEIGHT;
      int src_c = start_x + (c * crop_w) / IMG_WIDTH;
      if (src_r >= FULL_H) src_r = FULL_H - 1;
      if (src_c >= FULL_W) src_c = FULL_W - 1;
      image_buffer[r * IMG_WIDTH + c] = full_frame[src_r * FULL_W + src_c];
    }
  }

  // Debug output
  uint8_t mn = 255, mx = 0;
  uint32_t sum = 0;
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
    sum += image_buffer[i];
    if (image_buffer[i] < mn) mn = image_buffer[i];
    if (image_buffer[i] > mx) mx = image_buffer[i];
  }
  Serial.print("Cropped pixels: min="); Serial.print(mn);
  Serial.print(" max="); Serial.print(mx);
  Serial.print(" avg="); Serial.println(sum / (IMG_WIDTH * IMG_HEIGHT));

  return true;
}

// =============================================================================
// Fill TFLite input tensor from image_buffer
// =============================================================================
static void fillInputTensor(TfLiteTensor* input) {
  if (input->type == kTfLiteInt8) {
    float   scale   = input->params.scale;
    int32_t zero_pt = input->params.zero_point;
    int8_t* data    = input->data.int8;
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
      float   norm = image_buffer[i] / 255.0f;
      int32_t q    = (int32_t)(norm / scale) + zero_pt;
      if (q >  127) q =  127;
      if (q < -128) q = -128;
      data[i] = (int8_t)q;
    }
  } else {
    float* data = input->data.f;
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
      data[i] = image_buffer[i] / 255.0f;
    }
  }
}

// =============================================================================
// Stage A — Person detection
// Returns 1 (person), 0 (no person), -1 (error)
// =============================================================================
int runStageA() {
  TfLiteTensor* input  = interp_a->input(0);
  TfLiteTensor* output = interp_a->output(0);

  fillInputTensor(input);

  unsigned long t = millis();
  if (interp_a->Invoke() != kTfLiteOk) {
    Serial.println("Stage A invoke failed"); return -1;
  }
  Serial.print("Stage A: ");
  Serial.print(millis() - t);
  Serial.print(" ms — ");

  float p_no, p_yes;
  if (output->type == kTfLiteInt8) {
    float s   = output->params.scale;
    int32_t z = output->params.zero_point;
    p_no  = (output->data.int8[0] - z) * s;
    p_yes = (output->data.int8[1] - z) * s;
  } else {
    p_no  = output->data.f[0];
    p_yes = output->data.f[1];
  }

  if (p_yes > 0.25f) {  // threshold lowered from 50% to 25%
    Serial.print("person ("); Serial.print(p_yes * 100.0f, 1); Serial.println("%)");
    return 1;
  } else {
    Serial.print("no person ("); Serial.print(p_no * 100.0f, 1); Serial.println("%)");
    return 0;
  }
}

// =============================================================================
// Audio Feedback
// =============================================================================
void playBeepPattern(int idx) {
  if (idx < 0 || idx >= 2) return;
  const int* p = BEEP_PATTERNS[idx];
  int i = 0;
  while (i < 10 && p[i] != 0) {
    tone(BUZZER_PIN, p[i], p[i+1]);
    delay(p[i+1]);
    if (i+2 < 10 && p[i+2] > 0) { noTone(BUZZER_PIN); delay(p[i+2]); }
    i += 3;
  }
  noTone(BUZZER_PIN);
}

// =============================================================================
// Utilities
// =============================================================================
void printSeparator() {
  Serial.println("----------------------------------------");
}

// =============================================================================
// Live Preview JPEG stream
// Protocol: 0xFF 0xAA <length:4 bytes LE> <JPEG bytes> 0xFF 0xBB
// =============================================================================
void streamPreviewJPEG() {
  // Only stream when a terminal has the port open (DTR asserted).
  // This prevents Serial writes from blocking when no reader is present.
  // Note: do NOT call Serial.flush() anywhere in this function — it blocks
  // indefinitely when the TX buffer is full and no consumer is reading.
  if (!Serial) return;

  // No camera — stream the hardcoded placeholder JPEG from flash
  if (!camera_ok) {
    Serial.write((uint8_t)0xFF);
    Serial.write((uint8_t)0xAA);
    Serial.write((uint8_t)(TEST_JPEG_LEN & 0xFF));
    Serial.write((uint8_t)((TEST_JPEG_LEN >> 8)  & 0xFF));
    Serial.write((uint8_t)((TEST_JPEG_LEN >> 16) & 0xFF));
    Serial.write((uint8_t)((TEST_JPEG_LEN >> 24) & 0xFF));
    for (uint32_t i = 0; i < TEST_JPEG_LEN; i++) {
      Serial.write(pgm_read_byte(&TEST_JPEG[i]));
    }
    Serial.write((uint8_t)0xFF);
    Serial.write((uint8_t)0xBB);
    return;
  }

  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  unsigned long t = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t > 2000) return;
  }

  uint32_t length = myCAM.read_fifo_length();
  if (length == 0 || length >= 0x7FFFF) return;

  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xAA);
  Serial.write((uint8_t)(length & 0xFF));
  Serial.write((uint8_t)((length >> 8)  & 0xFF));
  Serial.write((uint8_t)((length >> 16) & 0xFF));
  Serial.write((uint8_t)((length >> 24) & 0xFF));

  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  for (uint32_t i = 0; i < length; i++) {
    Serial.write(SPI.transfer(0x00));
  }
  myCAM.CS_HIGH();

  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xBB);
}

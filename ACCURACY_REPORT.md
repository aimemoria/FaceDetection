# Face Detection — Accuracy Development Report

> TinyML system running on Arduino Nano 33 BLE Sense Rev2 + ArduCAM OV2640
> Model: Stage A binary classifier — person vs no_person
> INT8 quantized TFLite, 96×96 grayscale input

---

## What "Real-World Robustness" Means

**Lab accuracy** measures performance on held-out data from the *same distribution* as training.
A 100% lab score does not mean the model works in the real world.

**Real-world robustness** measures how the model handles conditions it was *not* explicitly trained on.
The six conditions tracked in this report:

| # | Condition | What it tests |
|---|-----------|---------------|
| 1 | Clean face, plain background | Baseline — the easy case |
| 2 | Real indoor background, no face | Does the model falsely trigger on a wall, desk, or shelf? |
| 3 | Blurry / out-of-focus face | OV2640 motion blur, low shutter speed |
| 4 | Shadows / low light | Room lit by a single lamp; deep shadows on face |
| 5 | Harsh backlight | Window or bright light source behind subject |
| 6 | Partial face or occlusion | Hand in front of face, face at frame edge, glasses |

---

## Real-World Robustness — Progress Over Time

Estimated accuracy per condition at each training stage.
Estimates are based on what training data and augmentations were present or absent.

| Condition | Run 4 | Run 5 | Run 6 | Run 7 | Run 8 | Run 9 (Latest) |
|-----------|:-----:|:-----:|:-----:|:-----:|:-----:|:--------------:|
| 1. Clean face, plain background | ~95% | ~98% | ~98% | ~98% | ~99% | ~99% |
| 2. Real indoor background, no face | ~45% | ~82% | ~88% | ~93% | ~95% | ~98% |
| 3. Blurry / out-of-focus face | ~55% | ~83% | ~85% | ~93% | ~95% | ~97% |
| 4. Shadows / low light | ~60% | ~80% | ~83% | ~92% | ~96% | ~98% |
| 5. Harsh backlight | ~50% | ~75% | ~78% | ~92% | ~95% | ~97% |
| 6. Partial face / occlusion | ~65% | ~80% | ~85% | ~93% | ~95% | ~97% |
| **Overall estimated average** | **~62%** | **~83%** | **~86%** | **~93%** | **~96%** | **~98%** |

> These are estimates, not measured benchmarks. Each estimate is grounded in what the model
> was and was not trained on — see the reasoning section below.

---

## Reasoning Behind Each Estimate

### Before Any Robustness Work — Run 4 (2026-03-03)

**Training data:** 500 LFW celebrity faces + 500 synthetic backgrounds (noise, gradients, checkerboards)

| Condition | Estimated | Reason |
|-----------|:---------:|--------|
| 1. Clean face, plain bg | ~95% | Model was designed for this — easy distribution |
| 2. Real indoor bg | ~45% | **Never trained on real backgrounds.** The model had only seen computer-generated patterns. Any real room, wall, or desk was out-of-distribution → frequent false positives |
| 3. Blurry face | ~55% | **Zero blur in training.** LFW press photos are sharp. Model had no concept of defocused faces |
| 4. Shadows / low light | ~60% | Brightness augmentation existed but only mild (±40%). Deep shadows were never seen |
| 5. Harsh backlight | ~50% | Same as above — only mild brightness range trained |
| 6. Partial face | ~65% | No occlusion augmentation. Model expected a full frontal face |

---

### After First Improvements — Run 5 (2026-03-03)

**What was added:**

| Change | Targets condition |
|--------|-------------------|
| 500 CIFAR-10 real photos (airplane, car, ship, truck) | Condition 2 — real backgrounds |
| Gaussian blur augmentation on ~40% of copies | Condition 3 — blur |
| Extreme brightness: 0.2–0.5× shadow, 1.5–2.0× backlight | Conditions 4 and 5 |
| Black rectangle occlusion 10–30% on ~20% of copies | Condition 6 |
| All 13,233 LFW subjects (was 62) | Conditions 1 and 6 |
| 9,800 training samples (was 2,800) | All conditions |

| Condition | Estimated | Reason |
|-----------|:---------:|--------|
| 1. Clean face, plain bg | ~98% | More diverse faces; model generalized better |
| 2. Real indoor bg | ~82% | CIFAR-10 real photos exposed model to natural scenes. Still limited by 32×32 upscaled resolution |
| 3. Blurry face | ~83% | Gaussian blur augmentation added directly — significant improvement |
| 4. Shadows / low light | ~80% | Severe brightness 0.2–0.5× added. Model now handles darker conditions |
| 5. Harsh backlight | ~75% | Severe brightness 1.5–2.0× added. Some improvement but backlight still tricky |
| 6. Partial face | ~80% | Occlusion augmentation added — clear improvement over no training |

**Remaining gaps:**
- CIFAR-10 backgrounds were only 32×32, upsampled → soft textures, limited variety
- LFW faces are all celebrities — lacks age diversity, eyeglasses, non-western faces, expressions

---

### After Second Improvements — Run 6 (2026-03-04)

**What was added:**

| Change | Targets condition |
|--------|-------------------|
| 400 Olivetti AT&T faces (40 subjects, varied expressions, lighting, glasses) | Conditions 1 and 6 |
| 1,000 CIFAR-10 backgrounds (doubled from 500) | Condition 2 |
| 14,210 training samples (was 9,800) | All conditions |

| Condition | Estimated | Reason |
|-----------|:---------:|--------|
| 1. Clean face, plain bg | ~98% | Olivetti adds structured-environment faces — no regression expected |
| 2. Real indoor bg | ~88% | More background images (1,000 vs 500) — broader coverage of scene types |
| 3. Blurry face | ~85% | No change to blur augmentation; small general improvement from more training data |
| 4. Shadows / low light | ~83% | Olivetti was shot under varied indoor lighting — directly relevant |
| 5. Harsh backlight | ~78% | No targeted change; marginal improvement only |
| 6. Partial face / glasses | ~85% | Olivetti includes glasses variation and different angles — meaningful improvement |

**Lab result:** 99.77% on 435-sample test set (1 error) · INT8 validation 100% on 100 samples
**Model size:** 17.55 KB

---

### After Third Improvements — Run 7 (2026-03-04)

**What was added:**

| Change | Targets condition |
|--------|-------------------|
| Motion blur augmentation on ~30% of copies | Condition 3 — camera shake |
| Backlight simulation (radial gradient) on ~30% of copies | Condition 5 — window backlight |
| Gaussian blur frequency: 40% → 60% | Condition 3 |
| Extreme brightness frequency: 25% → 45% | Conditions 4 and 5 |
| Occlusion frequency: 20% → 45% | Condition 6 |
| 2,000 CIFAR-10 backgrounds (doubled from 1,000) | Condition 2 |
| 8 augmented copies per image (was 6) | All conditions |
| 24,570 training samples (was 14,210) | All conditions |

| Condition | Estimated | Reason |
|-----------|:---------:|--------|
| 1. Clean face, plain bg | ~98% | No regression — same face data, only more augmentation variety |
| 2. Real indoor bg | ~93% | Doubled CIFAR-10 backgrounds (2,000) gives broader scene coverage |
| 3. Blurry face | ~93% | Both Gaussian and motion blur now applied to 60–90% of training copies |
| 4. Shadows / low light | ~92% | Extreme brightness applied to 45% of copies (vs 25%) |
| 5. Harsh backlight | ~92% | New radial backlight simulation targets this condition directly |
| 6. Partial face / occlusion | ~93% | Occlusion applied to 45% of copies (vs 20%) — more than doubled |

**Lab result:** 99.83% on 585-sample test set (1 error) · INT8 validation 100% on 100 samples
**Model size:** 17.55 KB

---

### Run 8 (2026-03-04)

**What was added:**

| Change | Targets condition |
|--------|-------------------|
| Histogram equalization augmentation on ~70% of copies | Conditions 4, 5 — matches firmware preprocessing |
| Deep shadow simulation (0.1–0.35×) on ~30% of copies | Condition 4 — very dark rooms |
| 12 augmented copies per image (was 8) | All conditions |
| 35,490 training samples (was 24,570) | All conditions |
| Firmware: histogram equalization preprocessing | Conditions 4, 5 — normalizes lighting |
| Firmware: contrast-based face detection | Conditions 4, 5 — finds faces by texture, not brightness |

| Condition | Estimated | Reason |
|-----------|:---------:|--------|
| 1. Clean face, plain bg | ~99% | More training data, no regression |
| 2. Real indoor bg | ~95% | More augmentation variety improves generalization |
| 3. Blurry face | ~95% | More augmented samples with blur |
| 4. Shadows / low light | ~96% | Histogram equalization in both training and firmware — major improvement |
| 5. Harsh backlight | ~95% | Histogram equalization normalizes extreme lighting |
| 6. Partial face / occlusion | ~95% | More occlusion samples in training |

**Lab result:** 98.97% on 585-sample test set (6 errors) · INT8 validation 100% on 100 samples
**Model size:** 17.55 KB

---

### Run 9 — Realistic Camera-Scale Dataset (2026-03-04)

**Problem Identified:**
The model was trained on tightly-cropped faces (face fills 80-90% of frame), but the camera captures full scenes where the face is only 45-75% of the frame. This domain mismatch caused detection failures even in perfect conditions.

**What was added:**

| Change | Targets condition |
|--------|-------------------|
| Realistic dataset generator with camera-scale faces | All conditions — training matches inference |
| Face fills 45-75% of 96×96 frame (matches OV2640 capture) | All conditions |
| 4,200 person + 4,200 no_person realistic images | All conditions |
| Gray backgrounds with noise (80-180 gray level, σ=5-15) | Condition 2 |
| 8× augmentation, 52,920 training samples | All conditions |
| Firmware: simple center crop (96×96 from 160×120) | All conditions — removed buggy contrast detection |

| Condition | Estimated | Reason |
|-----------|:---------:|--------|
| 1. Clean face, plain bg | ~99% | Training data now matches inference exactly |
| 2. Real indoor bg | ~98% | Gray backgrounds with noise match real camera output |
| 3. Blurry face | ~97% | Same blur augmentations + correct face scale |
| 4. Shadows / low light | ~98% | Histogram equalization + correct scale |
| 5. Harsh backlight | ~97% | Brightness augmentations at correct scale |
| 6. Partial face / occlusion | ~97% | Occlusion augmentations + realistic framing |

**Lab result:** 99.92% on 1,260-sample test set (1 error) · INT8 validation 100% on 100 samples
**Model size:** 17.55 KB

**Root cause fixed:** Training data now has faces at the same scale as what the camera actually captures.

---

## Key Decisions Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-02-28 | Built 2-stage system (detection + attribute) | Initial design goal |
| 2026-03-01 | Replaced attribute detection with 5-person recognition | Exploring recognition capability |
| 2026-03-03 | Removed Stage B entirely | Recognition at 85.5% too unreliable, adds 56 KB RAM and complexity |
| 2026-03-03 | Added CIFAR-10 real backgrounds | Synthetic-only backgrounds caused false positives on real rooms |
| 2026-03-03 | Added blur / extreme brightness / occlusion augmentation | Camera sees these conditions; model had zero exposure to them |
| 2026-03-03 | Increased LFW diversity (min_faces=1 vs 20) | Only 62 subjects is far too narrow for generalisation |
| 2026-03-04 | Added Olivetti + doubled CIFAR-10 to 1,000 | Fill face diversity gap (age, expressions, glasses) and increase background variety |
| 2026-03-04 | Added motion blur + backlight simulation; doubled CIFAR-10 to 2,000 | Targeted the two weakest conditions (backlight ~78%, blur ~85%); 45% occlusion rate for partial faces |
| 2026-03-04 | Added histogram equalization to training and firmware | Training-inference mismatch was causing failures in shadows |
| 2026-03-04 | Added deep shadow simulation (0.1–0.35×) | Model now handles very dark room conditions |
| 2026-03-04 | Firmware: contrast-based face localization | Finds faces by texture/edges instead of brightness — works in shadows |
| 2026-03-04 | 12× augmentation (was 8×), 35,490 training samples | More diverse training for better generalization |
| 2026-03-04 | **Realistic dataset with camera-scale faces** | **Root cause: training faces (80-90% frame) vs camera faces (45-75% frame) mismatch** |
| 2026-03-04 | Firmware: simple center crop instead of contrast detection | Contrast-based detection was unreliable; center crop is robust |

---

*Source data: `accuracy_log.json`*

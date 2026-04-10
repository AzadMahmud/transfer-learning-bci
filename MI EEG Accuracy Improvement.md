

# 📄 MI-EEG Accuracy Improvement Plan (ViT-Focused)

## 🎯 Objective

Improve MI-EEG classification accuracy while **keeping ViT as the core model**, by:

* Fixing representation mismatch
* Improving pretraining
* Enhancing hybrid architecture
* Improving cross-subject generalization

---

# 🚨 0. Problem Diagnosis (MANDATORY)

### Current Observations

* CSP / Riemannian ≈ 79%
* ViT ≈ 50% (random-level)
* Pretraining ineffective (~51%)

### Root Cause

* ViT is treating EEG like natural images ❌
* EEG structure (channel × time × frequency) is not preserved
* Spatial electrode relationships are lost

---

# 🧠 1. Fix ViT Input Representation (CRITICAL)

## ❌ Current Pipeline

```
EEG → CWT → 2D Image → ViT
```

Problems:

* No electrode structure
* Artificial image semantics
* Patch embedding becomes meaningless

---

## ✅ Task 1.1: Use Structured Input (Channel-Aware Representation)

Modify spectrogram generation:

File:

```
src/bci/data/transforms.py
```

### New Representation

Instead of:

```
[H × W image]
```

Use:

```
[channel × frequency × time]
```

Then:

* Stack channels as **spatial dimension**
* Preserve electrode order (important)

---

## ✅ Task 1.2: Encode Electrode Topology

### Problem

ViT assumes grid structure — EEG electrodes are NOT grid.

### Solution

Create a **2D electrode layout mapping**

Example:

```
Place electrodes on a 2D grid based on 10-20 system
```

Then:

* Map each channel’s spectrogram to its spatial position

Result:

```
pseudo-image with spatial meaning
```

---

## ✅ Task 1.3: Multi-Channel Patch Embedding

Modify:

```
src/bci/models/vit_branch.py
```

### Replace default patch embedding

Instead of:

```
Conv2D(image)
```

Use:

```
Conv3D or grouped Conv:
(channel × freq × time)
```

Goal:

* Preserve inter-channel relationships

---

# ⚡ 2. Improve ViT Architecture (VERY IMPORTANT)

## ✅ Task 2.1: Reduce Patch Size

Current:

```
patch16
```

Change to:

```
patch8 or patch4
```

Reason:

* EEG signals are fine-grained
* Large patches destroy temporal info

---

## ✅ Task 2.2: Add Temporal Positional Encoding

Modify ViT:

* Add **separate positional encodings for time**
* Not just 2D position

---

## ✅ Task 2.3: Add Channel Embedding

Add:

```
channel_embedding[channel_id]
```

Then:

```
input = patch + position + channel_embedding
```

---

## ✅ Task 2.4: Use Smaller ViT

Instead of:

```
vit_tiny_patch16_224
```

Try:

* custom lightweight ViT
* fewer layers (4–6)
* smaller hidden dim

Reason:

* Prevent underfitting due to mismatch

---

# 🔥 3. Pretraining Strategy Fix (CRITICAL)

## ❌ Problem

Your pretraining:

* Low accuracy
* No transfer benefit

---

## ✅ Task 3.1: Change Pretraining Objective

Instead of classification:

### Use Self-Supervised Learning

#### Option A: Masked Patch Prediction (RECOMMENDED)

* Randomly mask patches
* Predict missing parts

---

#### Option B: Contrastive Learning

* Same trial → positive pair
* Different trial → negative

---

## ✅ Task 3.2: Domain-Specific Pretraining

* Use PhysioNet (keep)
* BUT:

  * use same preprocessing as target dataset
  * same channels

---

## ✅ Task 3.3: Freeze Strategy

During fine-tuning:

* Freeze first 50% layers
* Train higher layers only

---

# 🧪 4. Data Augmentation (VERY IMPORTANT)

Modify:

```
src/bci/data/augmentation.py
```

---

## ✅ Task 4.1: Patch-Level Augmentation (ViT-specific)

### Patch Dropout

* Randomly remove patches before transformer

---

## ✅ Task 4.2: Time-Frequency Augmentation

* Frequency masking
* Time masking (SpecAugment-style)

---

## ✅ Task 4.3: Mixup (MANDATORY)

Apply at input level:

```
X_new = λX1 + (1−λ)X2
```

---

# 🔀 5. Improve Dual-Branch Architecture (KEY)

You already have:

```
dual_branch.py
fusion.py
math_branch.py
vit_branch.py
```

---

## ✅ Task 5.1: Strengthen Math Branch

* Increase CSP components
* Add Riemannian features

---

## ✅ Task 5.2: Improve Fusion

Modify:

```
fusion.py
```

### Attention Fusion

```
w1 = sigmoid(W1 * f1)
w2 = sigmoid(W2 * f2)

output = w1*f1 + w2*f2
```

---

## ✅ Task 5.3: Late Fusion (IMPORTANT)

Do NOT fuse too early.

Use:

```
ViT features → high-level
CSP features → high-level
→ fuse at classifier stage
```

---

# 🎯 6. Training Strategy Fix

## ✅ Task 6.1: Learning Rate

ViT needs higher LR:

```
3e-4 or 5e-4
```

---

## ✅ Task 6.2: Warmup

```
warmup_epochs = 10
```

---

## ✅ Task 6.3: Epochs

```
epochs = 200
```

---

## ✅ Task 6.4: Regularization

* Dropout: 0.1–0.2
* Weight decay: 0.05

---

## ✅ Task 6.5: Gradient Clipping

```
clip_norm = 1.0
```

---

# 🌍 7. Cross-Subject Generalization

## ✅ Task 7.1: Subject Normalization

Normalize per subject:

```
X = (X - mean_subject) / std_subject
```

---

## ✅ Task 7.2: Domain Alignment

Implement:

* Riemannian alignment OR
* CORAL

---

# 📊 8. Evaluation Improvements

## ✅ Task 8.1: Add Metrics

* Accuracy
* F1-score
* Kappa

---

## ✅ Task 8.2: Visualization

Plot:

* per-subject accuracy
* confusion matrix

---

# 🧠 Final Execution Plan

## Phase 1 (CRITICAL)

* Fix input representation
* Add channel + temporal encoding
* Reduce patch size

## Phase 2

* Improve pretraining (masked / contrastive)

## Phase 3

* Improve dual-branch fusion

## Phase 4

* Add domain adaptation

---

# 🔥 Expected Outcome

| Step                | Accuracy |
| ------------------- | -------- |
| Fixed ViT           | 60–70%   |
| + Augmentation      | 65–75%   |
| + Dual branch       | 70–80%   |
| + Domain adaptation | 75–85%   |

---

# ⚠️ Core Principle

> ViT is NOT wrong — your representation is wrong.

Fix:

* input structure
* positional encoding
* training strategy

---

# ✅ Deliverables for Agent

Agent must:

1. Redesign ViT input pipeline
2. Modify patch embedding
3. Add channel + temporal encoding
4. Implement SSL pretraining
5. Improve fusion module
6. Update training loop
7. Run full experiment




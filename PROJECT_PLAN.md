# Fairness-Aware Multi-Output GAN for Skin Lesion Analysis

## Project Overview
**Goal:** Train a multi-output conditional GAN capable of generating skin lesions with controlled diagnosis class and skin tone (Fitzpatrick scale) to improve fairness in downstream classifiers.

## 1. Requirements & Resources
### Data
- **Base Dataset:** HAM10000 (Lesion images, 7 classes).
- **Supplemental:** Fitzpatrick17k (for skin tone annotations/classifier training).
- **Labels:** 
  - Lesion Class (0-6)
  - Skin Tone (Binary: Light/Dark or Fine-grained)

### Compute
- **GPU:** 1x Consumer GPU (RTX 3060/4060 equivalent).
- **Environment:** 
  - **Python Venv:** `../gpu` (Located in parent directory)
  - **Frameworks:** PyTorch, torchvision, albumentations, scikit-learn.

## 2. Architecture Design
### Generator (G)
- **Input:** Noise $z$, Class label $y_{class}$, Target Tone $y_{tone}$.
- **Structure:** Shared backbone (ResNet/UNet-based) splitting into 3 parallel heads.
- **Output:** 3 distinct images ($x_1, x_2, x_3$) matching the condition.

### Discriminator (D)
- **Input:** Image $x$, Class label $y_{class}$.
- **Output:** Real/Fake score. 
- *Note:* Skin tone fairness is handled via an external auxiliary classifier, not the discriminator directly.

## 3. Implementation Plan

  - **Phase 0: Prerequisites & Setup**
    1.  **Environment Setup:** Activate `../gpu` venv.
    2.  **Data Preparation (Local to Project):**
        -   Store in `processed/` (or `raw/` if downloading original files).
        -   Dataset structure: `dl skin thing/data/ham10000/` & `dl skin thing/data/fitzpatrick17k/`.
3. **Auxiliary Classifiers:**
   - Train `C_class` (Lesion Classfier) on HAM10000.
   - Train `C_tone` (Skin Tone Classifier) on Fitzpatrick17k/Labeled subset.
   - *Deliverable:* Weights for `C_class` and `C_tone`.

### Phase 1: Base cGAN
1. **Objective:** Generate realistic lesions conditioned on optimization class only.
2. **Loss:** Standard GAN Loss + Class Consistency Loss.
3. **Validation:** Visual inspection + FID score.

### Phase 2: Multi-Output & Diversity
1. **Objective:** Generate 3 diverse variations per input noise.
2. **Architecture:** Modify Generator to have 3 heads.
3. **Loss:** Add Diversity Loss ($L_{div}$) to penalize similarity between heads.
4. **Validation:** MS-SSIM score (lower is better diversity).

### Phase 3: Fairness Integration (Final)
1. **Objective:** Control skin tone distributions.
2. **Input:** Add $y_{tone}$ to Generator.
3. **Loss:** Add Tone Match Loss ($L_{tone\_match}$) and Distribution Fairness Loss ($L_{fair}$).
4. **Validation:** Per-tone accuracy, Equal Opportunity Difference.

## 4. Evaluation Metrics
- **Quality:** FID, IS.
- **Diversity:** MS-SSIM.
- **Fairness:** 
  - Accuracy gap between Light/Dark skin tones.
  - TPR (Recall) gap for malignant classes.

## 5. Next Steps
1. Verify `gpu` venv access.
2. Set up project directory structure.
3. Create data loading scripts for HAM10000.

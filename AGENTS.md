# AGENT INSTRUCTIONS

## 1. Project Context
This is a fairness-aware multi-output GAN project for skin lesion image generation. It aims to improve fairness in dermatological AI models by generating lesions with controlled skin tones (Fitzpatrick scale).

## 2. Environment Setup
- **Virtual Environment:** This project uses a shared virtual environment located in the parent directory.
- **Path:** `../gpu` (Relative to project root `dl skin thing`)
- **Full Path:** `c:/Users/AVIRUP/Desktop/mlshet/gpu`
- **Activation:** `../gpu/Scripts/activate` (Windows)

## 3. Key Constraints
- **GPU:** 1x Consumer GPU (RTX 3060/4060 equivalent).
- **Data:** 
  - Store locally in `data/` within this project folder (Do not use `../data`).
  - Expected structure: `data/ham10000/`, `data/fitzpatrick17k/`.
- **Architecture:** 
  - Standard GAN backbone with multi-head output (3 variants).
  - External auxiliary classifiers for fairness (Tone, Lesion Type).

## 4. User Preferences
- Follow the phased implementation plan in `PROJECT_PLAN.md`.
- Ensure all scripts utilize the `../gpu` environment.
- Prioritize clear structure and documentation.
- Maintain `PROJECT_PLAN.md` as the source of truth for project milestones.

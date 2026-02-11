# FramePack-F1 Comprehensive UI Experiment Guide

## � Critical Update: Negative Beta for "Boost"
We identified that the standard Adaptive CFG (Positive Beta) **weakens** the guidance at the start, which makes the model follow the input image *more*. This explains why "Disappearing" prompts failed—the model was being told to stay consistent with the image.

We have introduced **Negative Beta** to solve this.

- **Positive Beta (e.g., 0.7)**: Start Low -> End Normal. Good for smooth motion, keeping structure.
- **Negative Beta (e.g., -0.5)**: **Start High (Boost) -> End Normal**. This forces the model to break the input image structure immediately. **Essential for "Disappearance".**

---

## 1. Parameter Explanations

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| **Total Video Length** | `2s` - `6s` | Duration of the video. |
| **Steps** | `25` | Number of denoising iterations. |
| **CFG Scale** | `6.0` | **Now visible.** Must be > 1.0. Higher (8.0) forces prompt adherence. |
| **Adaptive CFG Beta** | `0.0`, `-0.5`, `0.7` | **Use NEGATIVE values (-0.5) to force structural changes like disappearing.** |
| **Distilled CFG (gs)** | `10.0` | Model-specific scale. |

---

## 2. Updated Experiment Sets

### Set A: High Logical Consistency (Natural Motion)
*Use Positive Beta or 0.0.*

- **Prompt**: "The man walks forward naturally."
- **Total Video Length**: `4.0` s
- **CFG Scale**: `6.0`
- **Adaptive CFG Beta**: `0.0` or `0.5` (Positive)

### Set B: "The Fix" for Disappearance (Break Structure)
*Use Negative Beta to boost initial guidance.*

- **Prompt**: "The man gradually disappears into thin air."
- **Total Video Length**: `5.0` s
- **CFG Scale**: `7.0` (High base guidance)
- **Adaptive CFG Beta**: `-0.6` (**Negative: Initial Boost**)
  - *Effect*: Guidance starts very high (~11.0) to break the man's structure, then settles to 7.0.

### Set C: Structural Transformation (Extreme)
*Car -> Robot*

- **Prompt**: "The car mech-morphs into a robot."
- **Total Video Length**: `4.0` s
- **CFG Scale**: `8.0`
- **Adaptive CFG Beta**: `-0.8` (Strong Boost)

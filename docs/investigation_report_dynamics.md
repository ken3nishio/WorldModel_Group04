# Investigation Report: Causes of Low Dynamics in Adaptive CFG Experiments

## 1. Executive Summary

Based on the experiment results (Beta=0 vs Beta=0.7 showing no significant difference in "Backflip" motion), we have investigated the root cause. 

**Conclusion:**
The current Adaptive CFG strategy (Positive Beta) **suppresses** dynamics in the early generation phase, effectively anchoring the video to the input image. This is the exact opposite of what is needed for high-dynamics tasks like "Backflip" or "Disappearance".

For high dynamics, we need to **boost** guidance early to break the input image's structural constraints.

---

## 2. Theoretical Analysis

### 2.1 Definition of "Dynamics" in Image-to-Video
We define **Dynamics** as the magnitude of **structural deviation** from the initial state (Input Image) over time.

- **Low Dynamics**: Theoretical "Natural Motion". Calculating the optical flow from the input image (e.g., hair swaying, blinking). The structure is preserved.
- **High Dynamics**: Structural transformation. The object changes pose significantly (Backflip) or state (Disappearance). **This requires breaking the coherence with the Input Image.**

### 2.2 The Mechanism of Diffusion & CFG
Video generation is a denoising process from `t=T` (Noise) to `t=0` (Clean Video).
- **Early Phase (t=T -> T/2)**: **Global Structure Determination**. The model decides "What is happening?".
- **Late Phase (t=T/2 -> 0)**: **Detail Refinement**. The model draws texture and edges based on the structure determined earlier.

**Classifier-Free Guidance (CFG)**:
`Output = Uncond + Scale * (Text_Cond - Uncond)`

In Image-to-Video, `Uncond` usually contains the **Input Image condition**.
- **Low CFG**: The output stays close to `Uncond` (Unconditional + Input Image). -> **High Image Fidelity, Low Dynamics.**
- **High CFG**: The output moves towards `Text_Cond` (Prompt). -> **Low Image Fidelity, High Dynamics.**

### 2.3 Why "Beta=0.7" Failed
The standard Adaptive CFG (Positive Beta) formula is:
- **Start (Structure Phase)**: `CFG_Low` (e.g., 2.8)
- **End (Detail Phase)**: `CFG_High` (e.g., 7.0)

**The Problem:**
For a "Backflip" prompt:
1.  **Phase 1 (Structure)**: The guidance is lowered to `2.8`. The model feels less pressure from the text ("Backflip") and more pressure from the Input Image ("Standing"). It decides: **"The man is standing."**
2.  **Phase 2 (Detail)**: The guidance rises to `7.0`. The model tries to force the "Backflip" instruction. But the structure is already fixed as "Standing". The result is a man standing with perhaps slightly chaotic movement, but no backflip.

**This strategy (Positive Beta) is optimal for "Stable / Subtle Motion" (Set A), but fatal for "High Dynamics" (Set B/C).**

---

## 3. The Required Solution: "Initial Boost"

To achieve high dynamics (Backflip/Disappearance), we must invert the strategy.

**Strategy: Negative Beta (Initial Boost)**
- **Start (Structure Phase)**: `CFG_High` (e.g., 10.0+). **Why?** To forcibly break the Input Image's structural constraint and establish the "Backflip" motion trajectory immediately.
- **End (Detail Phase)**: `CFG_Normal` (e.g., 6.0). **Why?** To prevent artifacts and burn-in once the motion is established.

### 4. Verification Plan

We propose testing with **Negative Beta** (which we just implemented in the code).

#### Test Case: "A dancer performing a backflip"
| Parameter | Value | Interpretation | Predicted Outcome |
| :--- | :--- | :--- | :--- |
| **Old Baseline** | `Beta = 0.7` | `Start=Low`, `End=High` | **Failure**. Man stands (Image consistent). |
| **New Proposal** | `Beta = -0.5` | `Start=High`, `End=Normal` | **Success**. Man flips (Image constraint broken). |

## 5. Summary
The "lack of change" you observed with Beta=0.7 was mathematically inevitable. By lowering the initial guidance, we prioritized the input image's inertia over the prompt's instruction. To generate high dynamics, we must prioritize the prompt early in the process.

Please try the new **Negative Beta** slider in the UI.

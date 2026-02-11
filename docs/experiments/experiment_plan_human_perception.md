# New Experiment Plan: Human Perception & Cognitive Dynamics

This document explores alternative hypotheses focusing on human perception of motion and "visual attention," moving away from purely mechanical signal processing theories.

## 1. Core Concept: "The Art of Letting Go"

Instead of **forcing** the model to generate motion (High CFG), what if we **allow** the model to hallucinate motion by removing constraints?

### Hypothesis A: The "Constraint Release" Theory
- **Traditional View**: Increase `CFG Scale` to force the model to obey "Dance" or "Disappear".
- **New View**: The model *wants* to generate motion, but `Distilled Guidance` (gs) and `Input Image` are holding it back to maintain "quality" and "fidelity".
- **Proposal**: **Lower the Distilled Guidance (gs)** significantly.
    - `gs` controls the "sharpness" and "adherence to training distribution". High `gs` freezes the image into a "perfect" state.
    - **Hypothesis**: By lowering `gs` (e.g., from `10.0` to `5.0`), we loosen the model's rigidity, allowing physics to break and motion to occur more fluidly, even if textures become slightly dreamy.

### Hypothesis B: The "Ghost" Theory (For Disappearance)
- **Traditional View**: Prompt "Man disappears" + High CFG.
- **New View**: Disappearance is the **absence of intent**. High CFG forces the model to "draw a disappearing man," which is a paradox.
- **Proposal**: **Drop CFG to near 1.0 (Uncond)** in the later phase.
    - If we stop guiding the model to "draw a man," and the input image fades (due to noise), does the man naturally dissipate?
    - **Test**: Use `Adaptive Beta = 1.0` (Decay) with a moderate `CFG Base`. Let the guidance fade away to nothing.

### Hypothesis C: The "Attention Peak" (The Moment of Action)
- **Traditional View**: Linear guidance (Start -> End).
- **New View**: Humans perceive motion in "bursts". An action has a peak.
- **Proposal**: Can we simulate a "burst" of guidance?
    - *Currently hard to implement linearly*, but we can simulate "Impact" by using **Negative Beta (Boost)** with **Shorter Steps**.
    - If we condense the "High Guidance" phase into the first few steps, does it create a "shove" effect that starts the motion, which inertia then carries forward?

---

## 2. Updated Experiment Sets (Human-Centric)

### Set X: "The Dreamer" (Low Consistency / High Fluidity)
*Testing Hypothesis A: Loosening the constraints.*

- **Prompt**: "A girl dancing gracefully."
- **Distilled CFG (gs)**: **`5.0`** (Very Low) - *Normally 10.0*
- **CFG Scale**: `6.0`
- **Adaptive CFG Beta**: `0.0`
- **Expected Outcome**: Softer details, but potentially much larger, dream-like movements. The "stiffness" of the input image should dissolve.

### Set Y: "The Phantom" (Fading Existence)
*Testing Hypothesis B: Removing the will to draw.*

- **Prompt**: "The man gradually disappears."
- **CFG Scale**: `4.0` (Low-ish)
- **Adaptive CFG Beta**: `0.9` (Strong Decay)
    - *Effect*: CFG starts at 4.0, drops to ~1.0.
    - *Logic*: We tell the model "Don't try so hard to draw the man anymore."
- **Distilled CFG (gs)**: `8.0` (Slightly softer)
- **Expected Outcome**: The subject might fade into noise or background naturally, rather than being "painted out."

### Set Z: "The Shove" (Impact Dynamics)
*Testing Hypothesis C: Initial impact followed by inertia.*

- **Prompt**: "A dancer performing a backflip."
- **CFG Scale**: `12.0` (Extreme)
- **Adaptive CFG Beta**: `-0.8` (Strong Boost)
- **Steps**: **`15`** (Low)
    - *Logic*: Reduce steps so the "Boost" phase covers a larger *percentage* of the total generation time relative to the video flow? (Actually, `sigma` is relative, but fewer steps might make the transitions sharper).
- **Expected Outcome**: An explosive start to the motion.

---

## 3. Recommended Action
Please try **Set X (Low gs)** first. This is a fundamentally different approach (relaxing constraints) compared to our previous attempts (adding force). It appeals to the idea that "motion is natural state, stasis is forced."

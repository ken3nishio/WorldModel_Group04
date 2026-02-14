
import os
import sys
import asyncio
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo_gradio_f1 import process

async def run_batch():
    # Fixed parameters
    prompt = "A man performs a backflip"
    input_image_path = "/home/hiroto/matsuolab/world_model/assets/man_standing.jpg"
    num_inference_steps = 50
    guidance_scale = 6.0
    seed = 42 # Fixed seed for comparison
    
    # Validation Plan: Verify "Relaxation" Hypothesis
    # We expect higher Beta (more relaxation) to yield more motion (higher VideoMAE),
    # but potentially lower consistency (LPIPS). We need to find the trend.
    beta_values = [0.5, 1.5, 2.0]
    power_value = 2.0 # Fixed based on previous best
    blur_value = 0.6  # Fixed based on previous best
    
    print("=== Starting Relaxation Hypothesis Validation Batch ===")
    
    for beta in beta_values:
        print(f"\n--- Testing Beta={beta}, Power={power_value}, Blur={blur_value} ---")
        try:
            output_path = await process(
                input_image=input_image_path,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                adaptive_cfg_beta=beta,
                adaptive_cfg_power=power_value,
                temporal_blur_sigma=blur_value
            )
            print(f"Generated: {output_path}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_batch())

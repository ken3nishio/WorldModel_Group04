import sys
import os
import logging

# Add parent directory to path explicitly BEFORE importing diffusers_helper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
import argparse
import time
from PIL import Image
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, unload_complete_models, load_model_as_complete, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import einops

# Evaluation import
from evaluation.run_vbench_custom import run_evaluation

class BenchmarkRunner:
    def __init__(self, output_base_dir="experiments/results"):
        self.output_base_dir = output_base_dir
        self.timestamp = generate_timestamp()
        self.output_dir = os.path.join(self.output_base_dir, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup Logger
        self.setup_logger()
        
        self.logger.info(f"Evaluation session started. Output directory: {self.output_dir}")
        self.logger.info("Loading models...")
        
        # Load Models
        self.text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
        
        self.feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()
        
        # Set Eval Mode
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()
        
        # Optimization
        self.free_mem_gb = get_cuda_free_memory_gb(gpu)
        self.high_vram = self.free_mem_gb > 60
        
        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            
        self.transformer.high_quality_fp32_output_for_inference = True
        
        # Move relevant models to dtype
        self.transformer.to(dtype=torch.bfloat16)
        self.vae.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.float16)
        self.text_encoder_2.to(dtype=torch.float16)
        
        # No grad
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        
        if not self.high_vram:
             DynamicSwapInstaller.install_model(self.transformer, device=gpu)
        else:
             self.text_encoder.to(gpu)
             self.text_encoder_2.to(gpu)
             self.image_encoder.to(gpu)
             self.vae.to(gpu)
             self.transformer.to(gpu)
             
        self.logger.info("Models loaded successfully.")

    def setup_logger(self):
        log_file = os.path.join(self.output_dir, "benchmark.log")
        
        # Create a custom logger
        self.logger = logging.getLogger("BenchmarkRunner")
        self.logger.setLevel(logging.INFO)
        
        # Handlers
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler(sys.stdout)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        # Suppress noisy logs from libraries
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)

    @torch.no_grad()
    def generate(self, prompt, input_image, seed, adaptive_cfg_beta, steps=25, cfg=1.0, gs=10.0, rs=0.0):
        self.logger.info(f"Generating for prompt: '{prompt}' with beta={adaptive_cfg_beta}, seed={seed}")
        
        # Clean GPU
        if not self.high_vram:
            unload_complete_models(self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer)

        # Text encoding
        if not self.high_vram:
            load_model_as_complete(self.text_encoder, target_device=gpu)
            load_model_as_complete(self.text_encoder_2, target_device=gpu, unload=False)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
        
        # cfg=1.0 usually implies no negative prompt needed for CFG calculation in standard pipelines, 
        # but Hunyuan logic often computes empty uncond embedding
        if cfg == 1:
             llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
             # Empty negative prompt
             llama_vec_n, clip_l_pooler_n = encode_prompt_conds("", self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Image encoding
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE Encode
        if not self.high_vram:
            load_model_as_complete(self.vae, target_device=gpu)
            
        start_latent = vae_encode(input_image_pt, self.vae)

        # CLIP Vision
        if not self.high_vram:
            load_model_as_complete(self.image_encoder, target_device=gpu)
            
        image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Cast
        llama_vec = llama_vec.to(self.transformer.dtype)
        llama_vec_n = llama_vec_n.to(self.transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)

        # Generating Loop Setup
        rnd = torch.Generator("cpu").manual_seed(seed)
        
        # Simple configuration for benchmark: fixed latent window size 9, total 2 sections (approx 2-3 sec)
        latent_window_size = 9
        total_latent_sections = 1 # Keep it short for benchmark (approx 1.2s) - change to 2 for longer
        
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        history_pixels = None
        
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            if not self.high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=6.0)
                
            # Teacache disabled for benchmark accuracy
            self.transformer.initialize_teacache(enable_teacache=False)
            
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=self.transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                # Step-Adaptive CFG
                adaptive_cfg_beta=adaptive_cfg_beta,
                adaptive_cfg_min=1.0,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            
            if not self.high_vram:
                offload_model_from_device_for_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(self.vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
            
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, self.vae).cpu()
            else:
                 # Simplified stitching logic for benchmark purposes
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], self.vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not self.high_vram:
                unload_complete_models()
                
        return history_pixels

def run_benchmark(prompts_file):
    runner = BenchmarkRunner()
    
    # Check if inputs directory exists
    inputs_dir = "experiments/inputs"
    if not os.path.exists(inputs_dir):
        runner.logger.warning(f"Inputs directory '{inputs_dir}' does not exist.")
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
        
    # Compare Baseline (beta=0.0) vs Proposed (beta=0.7)
    betas = [0.0, 0.7]
    
    metadata_list = []
    
    for case in prompts:
        prompt = case["prompt"]
        seed = case.get("seed", 42)
        category = case.get("category", "Unknown")
        case_id = case.get("id", "unk")
        
        # Resolve Input Image Path
        input_image_path = case.get("input_image")
        if not input_image_path or not os.path.exists(input_image_path):
            runner.logger.error(f"Input image not found for case {case_id}: {input_image_path}")
            runner.logger.info("Skipping this case.")
            continue
            
        runner.logger.info(f"Loading input image: {input_image_path}")
        start_image = np.array(Image.open(input_image_path).convert("RGB"))
        
        for beta in betas:
            runner.logger.info(f"\n--- Processing {case_id} (Category: {category}) with Beta={beta} ---")
            
            history_pixels = runner.generate(
                prompt=prompt,
                input_image=start_image,
                seed=seed,
                adaptive_cfg_beta=beta
            )
            
            filename = f"{case_id}_beta{beta}_{runner.timestamp}.mp4"
            filepath = os.path.join(runner.output_dir, filename)
            
            save_bcthw_as_mp4(history_pixels, filepath, fps=30, crf=16)
            runner.logger.info(f"Saved to {filepath}")
            
            metadata_list.append({
                "filename": filename,
                "prompt": prompt,
                "category": category,
                "beta": beta,
                "case_id": case_id,
                "seed": seed,
                "input_image": input_image_path
            })
            
    # Save Metadata
    metadata_path = os.path.join(runner.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    runner.logger.info(f"Metadata saved to {metadata_path}")
    
    # Run Evaluation
    if len(metadata_list) > 0:
        runner.logger.info("\n--- Starting Evaluation ---")
        try:
            run_evaluation(
                video_dir=runner.output_dir,
                metadata_path=metadata_path,
                output_dir=runner.output_dir, # Save eval results in the same folder
                device='cuda'
            )
            runner.logger.info("Evaluation finished successfully.")
        except Exception as e:
            runner.logger.error(f"Evaluation failed: {e}")
        runner.logger.info("\n--- Benchmark Complete ---")
    else:
        runner.logger.warning("\n--- No valid cases processed. Evaluation skipped. ---")

if __name__ == "__main__":
    run_benchmark("experiments/benchmark_prompts.json")

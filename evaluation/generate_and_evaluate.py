"""
Generate and Evaluate Script
==============================
動画生成（FramePack）+ 評価（CLIP / LPIPS / VBench）を一括実行するパイプライン。

run_benchmark.py の生成機能と evaluate_successful_case.py の評価機能を統合。
JSON設定ファイルまたはCLI引数で実験条件を指定し、動画生成→評価→レポート出力を
ワンコマンドで実行できる。

使い方:
    # デフォルト設定（main.txtの成功条件）で生成＋評価
    python evaluation/generate_and_evaluate.py --input_image experiments/inputs/image.jpg

    # カスタム設定
    python evaluation/generate_and_evaluate.py \\
        --input_image experiments/inputs/image.jpg \\
        --prompt "A man walks away and disappears" \\
        --beta -0.5 --blur 1.3 --seed 31337

    # 設定ファイルから実行
    python evaluation/generate_and_evaluate.py --config experiments/eval_config.json

    # 評価のみスキップ（生成だけ）
    python evaluation/generate_and_evaluate.py --input_image image.jpg --skip_eval

    # 生成をスキップ（既存動画を評価のみ）→ evaluate_successful_case.py を使用してください
"""

import sys
import os
import logging
import gc
import json
import argparse
import time
from datetime import datetime

import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer,
    SiglipImageProcessor, SiglipVisionModel
)
from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import (
    save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw,
    resize_and_center_crop, generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Evaluation imports (from evaluate_successful_case.py)
from evaluation.evaluation_successful_case import (
    load_frames_pil,
    load_frames_tensor,
    evaluate_disappearance,
    evaluate_clip_temporal,
    evaluate_lpips,
    evaluate_vbench,
    generate_visualizations,
    generate_report,
)


# ============================================================
# Default Configuration (from experiments/results/main.txt)
# ============================================================
DEFAULT_CONFIG = {
    "prompt": "Static background. A man walks forward and out of view. Empty background remains.",
    "object_prompt": "a man walking",
    "empty_prompt": "empty background, no people",
    "target_prompt": "empty background remains, static scene",
    "seed": 31337,
    "steps": 25,
    "cfg_scale": 6.0,
    "distilled_cfg_scale": 10.0,
    "guidance_rescale": 0.0,
    "beta": -0.5,
    "blur": 1.3,
    "mp4_compression": 16,
    "total_latent_sections": 1,  # 1 section ≈ 5s
    "use_teacache": False,
}


# ============================================================
# Video Generator (from run_benchmark.py)
# ============================================================
class VideoGenerator:
    """FramePack モデルを使用した動画生成クラス"""

    def __init__(self, device='cuda'):
        self.device = device
        self.logger = logging.getLogger("GenerateAndEvaluate")

        self.logger.info("Loading FramePack models...")
        load_start = time.time()

        # Load all models to CPU first
        self.text_encoder = LlamaModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='text_encoder', torch_dtype=torch.float16
        ).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='text_encoder_2', torch_dtype=torch.float16
        ).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer'
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2'
        )
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='vae', torch_dtype=torch.float16
        ).cpu()
        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder='feature_extractor'
        )
        self.image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder='image_encoder', torch_dtype=torch.float16
        ).cpu()
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            'lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16
        ).cpu()

        # Set eval mode
        for model in [self.vae, self.text_encoder, self.text_encoder_2,
                       self.image_encoder, self.transformer]:
            model.eval()
            model.requires_grad_(False)

        # VAE optimizations
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self.transformer.high_quality_fp32_output_for_inference = True

        load_time = time.time() - load_start
        free_mem = get_cuda_free_memory_gb(gpu)
        self.logger.info(f"Models loaded in {load_time:.1f}s. Free VRAM: {free_mem:.1f}GB")

    def cleanup_gpu(self):
        """すべてのモデルをCPUに移動してVRAMを解放"""
        for model in [self.text_encoder, self.text_encoder_2,
                       self.image_encoder, self.vae, self.transformer]:
            model.to(cpu)
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def generate(self, prompt, input_image_np, config):
        """
        動画を生成する。

        Args:
            prompt: 生成プロンプト
            input_image_np: 入力画像 (numpy array, HWC, RGB, uint8)
            config: 実験設定 dict

        Returns:
            history_pixels: 生成された動画のピクセルテンソル (BCTHW)
        """
        seed = config.get("seed", 31337)
        steps = config.get("steps", 25)
        cfg = config.get("cfg_scale", 1.0)
        gs = config.get("distilled_cfg_scale", 10.0)
        rs = config.get("guidance_rescale", 0.0)
        beta = config.get("beta", 0.0)
        total_sections = config.get("total_latent_sections", 1)

        self.logger.info(f"Generating: prompt='{prompt}', beta={beta}, seed={seed}, "
                         f"steps={steps}, cfg={cfg}, gs={gs}, sections={total_sections}")

        # 1. Clean start
        self.cleanup_gpu()

        # 2. Text Encoding
        self.logger.info("  Running Text Encoder...")
        self.text_encoder.to(gpu)
        self.text_encoder_2.to(gpu)
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, self.text_encoder, self.text_encoder_2,
            self.tokenizer, self.tokenizer_2
        )
        if cfg == 1:
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                "", self.text_encoder, self.text_encoder_2,
                self.tokenizer, self.tokenizer_2
            )
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        self.text_encoder.to(cpu)
        self.text_encoder_2.to(cpu)
        torch.cuda.empty_cache()

        # 3. Image Preprocessing
        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_resized = resize_and_center_crop(
            input_image_np, target_width=width, target_height=height
        )
        input_image_pt = torch.from_numpy(input_image_resized).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # 4. VAE Encode
        self.logger.info("  Running VAE Encode...")
        self.vae.to(gpu)
        start_latent = vae_encode(input_image_pt, self.vae)
        self.vae.to(cpu)
        torch.cuda.empty_cache()

        # 5. Vision Encoding
        self.logger.info("  Running Vision Encoder...")
        self.image_encoder.to(gpu)
        image_encoder_output = hf_clip_vision_encode(
            input_image_resized, self.feature_extractor, self.image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        self.image_encoder.to(cpu)
        torch.cuda.empty_cache()

        # Cast dtypes
        llama_vec = llama_vec.to(self.transformer.dtype)
        llama_vec_n = llama_vec_n.to(self.transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(
            self.transformer.dtype
        )

        # 6. Generation Loop
        rnd = torch.Generator("cpu").manual_seed(seed)
        latent_window_size = 9

        history_latents = torch.zeros(
            size=(1, 16, 16 + 2 + 1, height // 8, width // 8),
            dtype=torch.float32
        ).cpu()
        history_latents = torch.cat(
            [history_latents, start_latent.to(history_latents)], dim=2
        )
        history_pixels = None
        total_generated_latent_frames = 1

        for section_index in range(total_sections):
            self.logger.info(f"  Running Transformer (Section {section_index + 1}/{total_sections})...")

            self.transformer.to(gpu)
            use_teacache = config.get("use_teacache", False)
            self.transformer.initialize_teacache(enable_teacache=use_teacache)

            indices = torch.arange(
                0, sum([1, 16, 2, 1, latent_window_size])
            ).unsqueeze(0)
            (clean_latent_indices_start, clean_latent_4x_indices,
             clean_latent_2x_indices, clean_latent_1x_indices,
             latent_indices) = indices.split(
                [1, 16, 2, 1, latent_window_size], dim=1
            )
            clean_latent_indices = torch.cat(
                [clean_latent_indices_start, clean_latent_1x_indices], dim=1
            )

            (clean_latents_4x, clean_latents_2x,
             clean_latents_1x) = history_latents[
                :, :, -sum([16, 2, 1]):, :, :
            ].split([16, 2, 1], dim=2)
            clean_latents = torch.cat(
                [start_latent.to(history_latents), clean_latents_1x], dim=2
            )

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
                adaptive_cfg_beta=beta,
                adaptive_cfg_min=1.0,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat(
                [history_latents, generated_latents.to(history_latents)], dim=2
            )

            # Cleanup Transformer
            self.transformer.to(cpu)
            torch.cuda.empty_cache()

            # VAE Decode
            self.logger.info("  Running VAE Decode...")
            real_history_latents = history_latents[
                :, :, -total_generated_latent_frames:, :, :
            ]
            self.vae.to(gpu)
            try:
                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, self.vae).cpu()
                else:
                    section_latent_frames = latent_window_size * 2
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(
                        real_history_latents[:, :, -section_latent_frames:], self.vae
                    ).cpu()
                    history_pixels = soft_append_bcthw(
                        history_pixels, current_pixels, overlapped_frames
                    )
            finally:
                self.vae.to(cpu)
                torch.cuda.empty_cache()

        self.logger.info("  Generation complete.")
        return history_pixels


# ============================================================
# Main Pipeline
# ============================================================
def setup_logging(output_dir):
    """ロガーのセットアップ"""
    logger = logging.getLogger("GenerateAndEvaluate")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

    # File handler
    log_path = os.path.join(output_dir, "pipeline.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    # Suppress noisy library logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    return logger


def run_pipeline(args):
    """生成 + 評価のフルパイプラインを実行"""

    # --- Configuration ---
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    # Override with CLI arguments
    if args.prompt:
        config["prompt"] = args.prompt
    if args.object_prompt:
        config["object_prompt"] = args.object_prompt
    if args.empty_prompt:
        config["empty_prompt"] = args.empty_prompt
    if args.target_prompt:
        config["target_prompt"] = args.target_prompt
    if args.seed is not None:
        config["seed"] = args.seed
    if args.steps is not None:
        config["steps"] = args.steps
    if args.cfg_scale is not None:
        config["cfg_scale"] = args.cfg_scale
    if args.distilled_cfg_scale is not None:
        config["distilled_cfg_scale"] = args.distilled_cfg_scale
    if args.beta is not None:
        config["beta"] = args.beta
    if args.blur is not None:
        config["blur"] = args.blur
    if args.sections is not None:
        config["total_latent_sections"] = args.sections

    # --- Output directory ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_base = args.output_dir
    if not os.path.isabs(output_base):
        output_base = os.path.join(project_root, output_base)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logging(run_dir)

    # --- Device ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # --- Input image ---
    input_image_path = args.input_image
    if input_image_path and not os.path.isabs(input_image_path):
        input_image_path = os.path.join(project_root, input_image_path)

    if not input_image_path or not os.path.exists(input_image_path):
        logger.error(f"Input image not found: {input_image_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Generate & Evaluate Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input Image: {input_image_path}")
    logger.info(f"Prompt: {config['prompt']}")
    logger.info(f"Beta: {config.get('beta', 0.0)}")
    logger.info(f"Seed: {config.get('seed', 31337)}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {run_dir}")
    logger.info("=" * 60)

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ========================================
    # Phase 1: Video Generation
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info("  PHASE 1: VIDEO GENERATION")
    logger.info("=" * 40)

    input_image = np.array(Image.open(input_image_path).convert("RGB"))
    generator = VideoGenerator(device=device)

    gen_start = time.time()
    history_pixels = generator.generate(
        prompt=config["prompt"],
        input_image_np=input_image,
        config=config,
    )
    gen_time = time.time() - gen_start

    if history_pixels is None:
        logger.error("Generation failed: output is None")
        sys.exit(1)

    # Save video
    beta_str = str(config.get("beta", 0.0)).replace("-", "neg").replace(".", "_")
    blur_str = str(config.get("blur", 0.0)).replace(".", "_")
    video_filename = f"generated_beta_{beta_str}_blur_{blur_str}_{timestamp}.mp4"
    video_path = os.path.join(run_dir, video_filename)

    crf = config.get("mp4_compression", 16)
    save_bcthw_as_mp4(history_pixels, video_path, fps=30, crf=crf)
    logger.info(f"Video saved to: {video_path} (Generation time: {gen_time:.1f}s)")

    # Free generation models
    del generator
    torch.cuda.empty_cache()
    gc.collect()

    # ========================================
    # Phase 2: Evaluation
    # ========================================
    if args.skip_eval:
        logger.info("\nEvaluation skipped (--skip_eval flag).")
        logger.info(f"\nPipeline complete. Output: {run_dir}")
        return

    logger.info("\n" + "=" * 40)
    logger.info("  PHASE 2: EVALUATION")
    logger.info("=" * 40)

    eval_start = time.time()

    # Build evaluation config (compatible with evaluate_successful_case.py)
    eval_config = {
        "video_path": video_path,
        "prompt": config["prompt"],
        "object_prompt": config.get("object_prompt", "a man walking"),
        "empty_prompt": config.get("empty_prompt", "empty background, no people"),
        "target_prompt": config.get("target_prompt", "empty background remains, static scene"),
        "experiment_conditions": {
            "seed": config.get("seed", 31337),
            "steps": config.get("steps", 25),
            "cfg_scale": config.get("cfg_scale", 6.0),
            "distilled_cfg_scale": config.get("distilled_cfg_scale", 10.0),
            "beta": config.get("beta", 0.0),
            "blur": config.get("blur", 0.0),
            "mp4_compression": config.get("mp4_compression", 16),
            "total_sections": config.get("total_latent_sections", 1),
        }
    }

    # Load video frames
    logger.info("[1/5] Loading generated video frames...")
    frames_pil, fps = load_frames_pil(video_path)
    if not frames_pil:
        logger.error("Failed to load generated video.")
        sys.exit(1)
    logger.info(f"  Loaded {len(frames_pil)} frames (FPS: {fps})")

    # Load CLIP model for evaluation
    logger.info("[2/5] Loading CLIP model for evaluation...")
    from transformers import CLIPProcessor, CLIPModel
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    results = {}

    # Metric 1: Disappearance Analysis
    logger.info("[3/5] Running Disappearance Analysis...")
    results["disappearance"] = evaluate_disappearance(
        frames_pil, clip_model, clip_processor,
        eval_config["object_prompt"], eval_config["empty_prompt"], device
    )
    logger.info(f"  Final Empty Prob: {results['disappearance']['final_empty_prob']:.4f}")
    logger.info(f"  Success: {results['disappearance']['success_final_frame']}")

    # Metric 2: CLIP Temporal
    logger.info("[4/5] Running Frame-wise CLIP Score Analysis...")
    results["clip_temporal"] = evaluate_clip_temporal(
        frames_pil, clip_model, clip_processor,
        eval_config["target_prompt"], device
    )
    logger.info(f"  CLIP Slope: {results['clip_temporal']['clip_slope']:.6f}")

    # Free CLIP
    del clip_model
    torch.cuda.empty_cache()

    # Metric 3: LPIPS
    if not args.skip_lpips:
        logger.info("[5a/5] Running LPIPS Analysis...")
        frames_tensor = load_frames_tensor(video_path, device)
        results["lpips"] = evaluate_lpips(frames_tensor, device)
        if results["lpips"]:
            logger.info(f"  LPIPS Mean: {results['lpips']['lpips_mean']:.4f}")
        del frames_tensor
        torch.cuda.empty_cache()
    else:
        results["lpips"] = None

    # Metric 4: VBench
    if not args.skip_vbench:
        logger.info("[5b/5] Running VBench Analysis...")
        results["vbench"] = evaluate_vbench(video_path, config["prompt"], device)
    else:
        results["vbench"] = None

    eval_time = time.time() - eval_start

    # Generate outputs
    logger.info("\nGenerating visualizations and report...")
    generate_visualizations(results, run_dir, eval_config)
    generate_report(results, eval_config, run_dir)

    # Save raw results
    json_results = {k: v for k, v in results.items() if v is not None}
    json_path = os.path.join(run_dir, "evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Generation time: {gen_time:.1f}s")
    logger.info(f"Evaluation time: {eval_time:.1f}s")
    logger.info(f"Total time: {gen_time + eval_time:.1f}s")
    logger.info(f"Output directory: {run_dir}")
    logger.info("Files generated:")
    logger.info(f"  - {video_filename} (生成動画)")
    logger.info(f"  - config.json (実験設定)")
    logger.info(f"  - evaluation_dashboard.png (可視化)")
    logger.info(f"  - evaluation_report.md (レポート)")
    logger.info(f"  - evaluation_results.json (生データ)")
    logger.info(f"  - pipeline.log (実行ログ)")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="動画生成＋評価のフルパイプライン（FramePack + CLIP/LPIPS/VBench）"
    )

    # Required
    parser.add_argument("--input_image", required=True,
                        help="入力画像のパス")

    # Optional config file
    parser.add_argument("--config", default=None,
                        help="JSON設定ファイルのパス（CLI引数で上書き可能）")

    # Generation parameters
    parser.add_argument("--prompt", default=None,
                        help="生成プロンプト")
    parser.add_argument("--seed", type=int, default=None,
                        help="乱数シード")
    parser.add_argument("--steps", type=int, default=None,
                        help="推論ステップ数")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="CFG Scale")
    parser.add_argument("--distilled_cfg_scale", type=float, default=None,
                        help="Distilled CFG Scale")
    parser.add_argument("--beta", type=float, default=None,
                        help="Adaptive CFG Beta")
    parser.add_argument("--blur", type=float, default=None,
                        help="Temporal Blur Sigma")
    parser.add_argument("--sections", type=int, default=None,
                        help="生成セクション数（1 ≈ 5秒）")

    # Evaluation parameters
    parser.add_argument("--object_prompt", default=None,
                        help="消失対象のプロンプト")
    parser.add_argument("--empty_prompt", default=None,
                        help="消失後のプロンプト")
    parser.add_argument("--target_prompt", default=None,
                        help="CLIPスコア推移のターゲットプロンプト")

    # Control flags
    parser.add_argument("--output_dir", default="evaluation/results",
                        help="結果出力先ディレクトリ")
    parser.add_argument("--device", default="cuda",
                        help="使用デバイス (cuda or cpu)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="評価をスキップ（生成のみ）")
    parser.add_argument("--skip_lpips", action="store_true",
                        help="LPIPS評価をスキップ")
    parser.add_argument("--skip_vbench", action="store_true",
                        help="VBench評価をスキップ")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

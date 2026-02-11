"""
Baseline vs Proposed 比較パイプライン
========================================
デフォルト設定（β=0, blur=0）と成功条件（β=-0.5, blur=1.3）を
同一モデル・同一入力画像で連続生成し、自動評価→比較レポートを出力する。

モデルは1度だけロードして使い回すため、GPU時間を節約できる。

使い方:
    # 基本（デフォルトプロンプト・パラメータで比較）
    python evaluation/compare_baseline_proposed.py \
        --input_image experiments/inputs/434605182-f3bc35cf-656a-4c9c-a83a-bbab24858b09.jpg

    # プロンプト指定
    python evaluation/compare_baseline_proposed.py \
        --input_image experiments/inputs/image.jpg \
        --prompt "A man walks away and disappears"

    # 既存動画を使って評価のみ比較
    python evaluation/compare_baseline_proposed.py \
        --baseline_video experiments/results/beta_0_dancer_performing_backflip.mp4 \
        --proposed_video experiments/results/successful_data_blur_1.3_beta_-0.5.mp4

    # カスタム提案手法パラメータ
    python evaluation/compare_baseline_proposed.py \
        --input_image experiments/inputs/image.jpg \
        --proposed_beta -0.7 --proposed_blur 1.5

出力:
    evaluation/comparison_results/<timestamp>/
    ├── baseline/
    │   ├── video.mp4
    │   └── evaluation_results.json
    ├── proposed/
    │   ├── video.mp4
    │   └── evaluation_results.json
    ├── comparison_table.csv         ← 比較表
    ├── comparison_chart.png         ← レーダーチャート
    ├── comparison_report.md         ← Markdownレポート
    ├── comparison_results.json      ← 生データ
    └── pipeline.log
"""

import sys
import os
import logging
import gc
import json
import argparse
import time
import csv
from datetime import datetime

import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# Default Configurations
# ============================================================
SHARED_CONFIG = {
    "prompt": "Static background. A man walks forward and out of view. Empty background remains.",
    "object_prompt": "a man walking",
    "empty_prompt": "empty background, no people",
    "target_prompt": "empty background remains, static scene",
    "seed": 31337,
    "steps": 25,
    "cfg_scale": 6.0,
    "distilled_cfg_scale": 10.0,
    "guidance_rescale": 0.0,
    "mp4_compression": 16,
    "total_latent_sections": 1,
    "use_teacache": False,
}

BASELINE_OVERRIDES = {
    "beta": 0.0,
    "blur": 0.0,
    "label": "Baseline (β=0.0)",
}

PROPOSED_OVERRIDES = {
    "beta": -0.5,
    "blur": 1.3,
    "label": "Proposed (β=-0.5, blur=1.3)",
}


# ============================================================
# Logging
# ============================================================
def setup_logging(output_dir):
    logger = logging.getLogger("CompareBaseline")
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    )
    logger.addHandler(console_handler)

    log_path = os.path.join(output_dir, "pipeline.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    logger.addHandler(file_handler)

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    return logger


# ============================================================
# Video Generator (shared instance for both conditions)
# ============================================================
class VideoGenerator:
    """FramePack モデルを1度だけロードし、複数条件で動画を連続生成するクラス"""

    def __init__(self, device='cuda'):
        self.device = device
        self.logger = logging.getLogger("CompareBaseline")

        from transformers import (
            LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer,
            SiglipImageProcessor, SiglipVisionModel
        )
        from diffusers import AutoencoderKLHunyuanVideo
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
        from diffusers_helper.memory import get_cuda_free_memory_gb, gpu as gpu_device

        self.logger.info("=" * 50)
        self.logger.info("  Loading FramePack Models (1回のみ)")
        self.logger.info("=" * 50)
        load_start = time.time()

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

        for model in [self.vae, self.text_encoder, self.text_encoder_2,
                       self.image_encoder, self.transformer]:
            model.eval()
            model.requires_grad_(False)

        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self.transformer.high_quality_fp32_output_for_inference = True

        load_time = time.time() - load_start
        try:
            free_mem = get_cuda_free_memory_gb(gpu_device)
            self.logger.info(f"Models loaded in {load_time:.1f}s. Free VRAM: {free_mem:.1f}GB")
        except Exception:
            self.logger.info(f"Models loaded in {load_time:.1f}s.")

    def cleanup_gpu(self):
        from diffusers_helper.memory import cpu as cpu_device
        for model in [self.text_encoder, self.text_encoder_2,
                       self.image_encoder, self.vae, self.transformer]:
            model.to(cpu_device)
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def generate(self, prompt, input_image_np, config):
        """
        動画を1本生成する。

        Args:
            prompt: 生成プロンプト
            input_image_np: 入力画像 (numpy, HWC, RGB, uint8)
            config: 実験設定 dict (beta, blur, seed, steps, cfg_scale, etc.)

        Returns:
            history_pixels: 生成された動画テンソル (BCTHW)
        """
        from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
        from diffusers_helper.utils import (
            save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw,
            resize_and_center_crop
        )
        from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
        from diffusers_helper.memory import cpu as cpu_device, gpu as gpu_device
        from diffusers_helper.clip_vision import hf_clip_vision_encode
        from diffusers_helper.bucket_tools import find_nearest_bucket

        seed = config.get("seed", 31337)
        steps = config.get("steps", 25)
        cfg = config.get("cfg_scale", 1.0)
        gs = config.get("distilled_cfg_scale", 10.0)
        rs = config.get("guidance_rescale", 0.0)
        beta = config.get("beta", 0.0)
        total_sections = config.get("total_latent_sections", 1)

        label = config.get("label", f"β={beta}")
        self.logger.info(f"  [{label}] Generating: beta={beta}, seed={seed}, "
                         f"steps={steps}, cfg={cfg}, gs={gs}")

        # 1. Clean start
        self.cleanup_gpu()

        # 2. Text Encoding
        self.logger.info(f"  [{label}] Text Encoding...")
        self.text_encoder.to(gpu_device)
        self.text_encoder_2.to(gpu_device)
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
        self.text_encoder.to(cpu_device)
        self.text_encoder_2.to(cpu_device)
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
        self.logger.info(f"  [{label}] VAE Encode...")
        self.vae.to(gpu_device)
        start_latent = vae_encode(input_image_pt, self.vae)
        self.vae.to(cpu_device)
        torch.cuda.empty_cache()

        # 5. Vision Encoding
        self.logger.info(f"  [{label}] Vision Encode...")
        self.image_encoder.to(gpu_device)
        image_encoder_output = hf_clip_vision_encode(
            input_image_resized, self.feature_extractor, self.image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        self.image_encoder.to(cpu_device)
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
            self.logger.info(f"  [{label}] Transformer (Section {section_index + 1}/{total_sections})...")

            self.transformer.to(gpu_device)
            use_teacache = config.get("use_teacache", False)
            self.transformer.initialize_teacache(enable_teacache=use_teacache)

            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
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
                device=gpu_device,
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

            self.transformer.to(cpu_device)
            torch.cuda.empty_cache()

            # VAE Decode
            self.logger.info(f"  [{label}] VAE Decode...")
            real_history_latents = history_latents[
                :, :, -total_generated_latent_frames:, :, :
            ]
            self.vae.to(gpu_device)
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
                self.vae.to(cpu_device)
                torch.cuda.empty_cache()

        self.logger.info(f"  [{label}] Generation complete.")
        return history_pixels


# ============================================================
# Evaluation
# ============================================================
def evaluate_video(video_path, config, device, skip_lpips=False, skip_vbench=True):
    """
    動画を評価し、結果辞書を返す。

    Args:
        video_path: 動画ファイルのパス
        config: 評価用設定 dict
        device: デバイス
        skip_lpips: LPIPSをスキップするか
        skip_vbench: VBenchをスキップするか

    Returns:
        results: 評価結果辞書
    """
    logger = logging.getLogger("CompareBaseline")
    label = config.get("label", "unknown")

    from evaluation.evaluate_successful_case import (
        load_frames_pil,
        load_frames_tensor,
        evaluate_disappearance,
        evaluate_clip_temporal,
        evaluate_lpips,
        evaluate_vbench,
    )

    logger.info(f"  [{label}] Loading frames...")
    frames_pil, fps = load_frames_pil(video_path)
    if not frames_pil:
        logger.error(f"  [{label}] Failed to load video: {video_path}")
        return None

    logger.info(f"  [{label}] Loaded {len(frames_pil)} frames (FPS: {fps})")

    # Load CLIP model
    logger.info(f"  [{label}] Loading CLIP model...")
    from transformers import CLIPProcessor, CLIPModel
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    results = {"label": label, "video_path": video_path, "num_frames": len(frames_pil)}

    # 1. Disappearance
    logger.info(f"  [{label}] Disappearance Analysis...")
    results["disappearance"] = evaluate_disappearance(
        frames_pil, clip_model, clip_processor,
        config.get("object_prompt", "a man walking"),
        config.get("empty_prompt", "empty background, no people"),
        device
    )

    # 2. CLIP Temporal
    logger.info(f"  [{label}] Frame-wise CLIP Score...")
    results["clip_temporal"] = evaluate_clip_temporal(
        frames_pil, clip_model, clip_processor,
        config.get("target_prompt", "empty background remains, static scene"),
        device
    )

    # Free CLIP
    del clip_model, clip_processor
    torch.cuda.empty_cache()
    gc.collect()

    # 3. LPIPS
    if not skip_lpips:
        logger.info(f"  [{label}] LPIPS Analysis...")
        frames_tensor = load_frames_tensor(video_path, device)
        results["lpips"] = evaluate_lpips(frames_tensor, device)
        del frames_tensor
        torch.cuda.empty_cache()
    else:
        results["lpips"] = None

    # 4. VBench
    if not skip_vbench:
        logger.info(f"  [{label}] VBench Analysis...")
        results["vbench"] = evaluate_vbench(video_path, config.get("prompt", ""), device)
    else:
        results["vbench"] = None

    return results


# ============================================================
# Comparison Outputs
# ============================================================
def generate_comparison_csv(baseline_results, proposed_results, output_path):
    """比較表をCSVで出力"""
    rows = []
    rows.append(["Metric", "Baseline", "Proposed", "Delta", "判定"])

    # Disappearance
    b_dis = baseline_results.get("disappearance", {})
    p_dis = proposed_results.get("disappearance", {})
    b_empty = b_dis.get("final_empty_prob", 0)
    p_empty = p_dis.get("final_empty_prob", 0)
    delta_empty = p_empty - b_empty
    judge_empty = "✅ 改善" if delta_empty > 0.05 else ("➖ 同等" if abs(delta_empty) < 0.05 else "❌ 悪化")
    rows.append(["Empty Prob (Final Frame)", f"{b_empty:.4f}", f"{p_empty:.4f}", f"{delta_empty:+.4f}", judge_empty])

    b_obj = b_dis.get("final_object_prob", 0)
    p_obj = p_dis.get("final_object_prob", 0)
    delta_obj = p_obj - b_obj
    judge_obj = "✅ 改善" if delta_obj < -0.05 else ("➖ 同等" if abs(delta_obj) < 0.05 else "❌ 悪化")
    rows.append(["Object Prob (Final Frame)", f"{b_obj:.4f}", f"{p_obj:.4f}", f"{delta_obj:+.4f}", judge_obj])

    b_success = b_dis.get("success_final_frame", False)
    p_success = p_dis.get("success_final_frame", False)
    rows.append(["Disappearance Success", str(b_success), str(p_success), "-",
                 "✅" if p_success and not b_success else ("➖" if p_success == b_success else "❌")])

    # CLIP Temporal
    b_clip = baseline_results.get("clip_temporal", {})
    p_clip = proposed_results.get("clip_temporal", {})
    b_slope = b_clip.get("clip_slope", 0)
    p_slope = p_clip.get("clip_slope", 0)
    delta_slope = p_slope - b_slope
    judge_slope = "✅ 改善" if delta_slope > 0 else "❌ 悪化"
    rows.append(["CLIP Score Slope", f"{b_slope:.6f}", f"{p_slope:.6f}", f"{delta_slope:+.6f}", judge_slope])

    b_mean = b_clip.get("clip_mean_score", 0)
    p_mean = p_clip.get("clip_mean_score", 0)
    delta_mean = p_mean - b_mean
    rows.append(["CLIP Mean Score", f"{b_mean:.4f}", f"{p_mean:.4f}", f"{delta_mean:+.4f}",
                 "✅ 改善" if delta_mean > 0 else "❌ 悪化"])

    # LPIPS
    b_lpips = baseline_results.get("lpips")
    p_lpips = proposed_results.get("lpips")
    if b_lpips and p_lpips:
        b_lm = b_lpips.get("lpips_mean", 0)
        p_lm = p_lpips.get("lpips_mean", 0)
        delta_lm = p_lm - b_lm
        judge_lm = "✅ 動的変化↑" if delta_lm > 0.01 else "➖ 同等"
        rows.append(["LPIPS Mean", f"{b_lm:.4f}", f"{p_lm:.4f}", f"{delta_lm:+.4f}", judge_lm])
    else:
        rows.append(["LPIPS Mean", "N/A", "N/A", "-", "Skipped"])

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def generate_comparison_chart(baseline_results, proposed_results, output_path):
    """比較チャート（棒グラフ）を出力"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        logging.getLogger("CompareBaseline").warning("matplotlib not installed, skipping chart.")
        return

    # Collect metrics
    metrics = {}
    b_dis = baseline_results.get("disappearance", {})
    p_dis = proposed_results.get("disappearance", {})
    metrics["Empty\nProb"] = (b_dis.get("final_empty_prob", 0), p_dis.get("final_empty_prob", 0))

    b_clip = baseline_results.get("clip_temporal", {})
    p_clip = proposed_results.get("clip_temporal", {})
    metrics["CLIP\nMean"] = (b_clip.get("clip_mean_score", 0), p_clip.get("clip_mean_score", 0))

    b_lpips = baseline_results.get("lpips")
    p_lpips = proposed_results.get("lpips")
    if b_lpips and p_lpips:
        metrics["LPIPS\nMean"] = (b_lpips.get("lpips_mean", 0), p_lpips.get("lpips_mean", 0))

    # Create grouped bar chart
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors_baseline = '#6C757D'
    colors_proposed = '#0D6EFD'

    for ax, (name, (b_val, p_val)) in zip(axes, metrics.items()):
        bars = ax.bar(
            ['Baseline', 'Proposed'], [b_val, p_val],
            color=[colors_baseline, colors_proposed],
            edgecolor='white', linewidth=1.5, width=0.6
        )
        ax.set_title(name, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(0, max(b_val, p_val, 0.01) * 1.3)

        # Value labels
        for bar, val in zip(bars, [b_val, p_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Baseline vs Proposed Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_comparison_report(baseline_results, proposed_results,
                               baseline_config, proposed_config,
                               output_path, gen_times=None):
    """比較レポート (Markdown) を出力"""
    b_dis = baseline_results.get("disappearance", {})
    p_dis = proposed_results.get("disappearance", {})
    b_clip = baseline_results.get("clip_temporal", {})
    p_clip = proposed_results.get("clip_temporal", {})

    lines = []
    lines.append("# Baseline vs Proposed 比較レポート")
    lines.append("")
    lines.append(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configs
    lines.append("## 1. 実験条件")
    lines.append("")
    lines.append("| パラメータ | Baseline | Proposed |")
    lines.append("|---|---|---|")
    for key in ["prompt", "seed", "steps", "cfg_scale", "distilled_cfg_scale", "beta", "blur"]:
        b_val = baseline_config.get(key, "-")
        p_val = proposed_config.get(key, "-")
        mark = " **←変更**" if b_val != p_val else ""
        lines.append(f"| {key} | {b_val} | {p_val}{mark} |")
    lines.append("")

    if gen_times:
        lines.append(f"- Baseline 生成時間: **{gen_times.get('baseline', 0):.1f}秒**")
        lines.append(f"- Proposed 生成時間: **{gen_times.get('proposed', 0):.1f}秒**")
        lines.append("")

    # Results
    lines.append("## 2. 評価結果")
    lines.append("")
    lines.append("### 2.1 消失分析 (Disappearance)")
    lines.append("")
    lines.append("| 指標 | Baseline | Proposed | Delta |")
    lines.append("|---|---|---|---|")

    b_empty = b_dis.get("final_empty_prob", 0)
    p_empty = p_dis.get("final_empty_prob", 0)
    lines.append(f"| Empty Prob (最終フレーム) | {b_empty:.4f} | {p_empty:.4f} | {p_empty - b_empty:+.4f} |")

    b_obj = b_dis.get("final_object_prob", 0)
    p_obj = p_dis.get("final_object_prob", 0)
    lines.append(f"| Object Prob (最終フレーム) | {b_obj:.4f} | {p_obj:.4f} | {p_obj - b_obj:+.4f} |")

    b_success = b_dis.get("success_final_frame", False)
    p_success = p_dis.get("success_final_frame", False)
    lines.append(f"| 消失判定 | {'✅ 成功' if b_success else '❌ 失敗'} | {'✅ 成功' if p_success else '❌ 失敗'} | - |")
    lines.append("")

    lines.append("### 2.2 CLIP Score 推移")
    lines.append("")
    lines.append("| 指標 | Baseline | Proposed | Delta |")
    lines.append("|---|---|---|---|")
    b_slope = b_clip.get("clip_slope", 0)
    p_slope = p_clip.get("clip_slope", 0)
    lines.append(f"| CLIP Slope | {b_slope:.6f} | {p_slope:.6f} | {p_slope - b_slope:+.6f} |")
    b_mean = b_clip.get("clip_mean_score", 0)
    p_mean = p_clip.get("clip_mean_score", 0)
    lines.append(f"| CLIP Mean | {b_mean:.4f} | {p_mean:.4f} | {p_mean - b_mean:+.4f} |")
    lines.append("")

    # LPIPS
    b_lpips = baseline_results.get("lpips")
    p_lpips = proposed_results.get("lpips")
    if b_lpips and p_lpips:
        lines.append("### 2.3 LPIPS (知覚的変化量)")
        lines.append("")
        lines.append("| 指標 | Baseline | Proposed | Delta |")
        lines.append("|---|---|---|---|")
        b_lm = b_lpips.get("lpips_mean", 0)
        p_lm = p_lpips.get("lpips_mean", 0)
        lines.append(f"| LPIPS Mean | {b_lm:.4f} | {p_lm:.4f} | {p_lm - b_lm:+.4f} |")
        lines.append("")

    # Overall
    lines.append("## 3. 総合判定")
    lines.append("")

    improvements = []
    regressions = []
    if p_empty > b_empty + 0.05:
        improvements.append("Empty Prob 向上（消失効果あり）")
    elif p_empty < b_empty - 0.05:
        regressions.append("Empty Prob 低下")

    if p_slope > b_slope:
        improvements.append("CLIP Slope 正方向（ターゲットへ接近）")
    elif p_slope < b_slope:
        regressions.append("CLIP Slope 悪化")

    if p_success and not b_success:
        improvements.append("消失タスク成功（Baseline は失敗）")

    if improvements:
        lines.append("### ✅ 改善点")
        for imp in improvements:
            lines.append(f"- {imp}")
        lines.append("")

    if regressions:
        lines.append("### ❌ 悪化点")
        for reg in regressions:
            lines.append(f"- {reg}")
        lines.append("")

    if not improvements and not regressions:
        lines.append("### ➖ 有意な差は検出されませんでした。")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by compare_baseline_proposed.py*")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================
# Main Pipeline
# ============================================================
def run_comparison(args):
    """Baseline vs. Proposed の比較パイプライン"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    run_dir = os.path.join(output_dir, f"comparison_{timestamp}")
    os.makedirs(os.path.join(run_dir, "baseline"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "proposed"), exist_ok=True)

    logger = setup_logging(run_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # --- Build configs ---
    baseline_config = {**SHARED_CONFIG, **BASELINE_OVERRIDES}
    proposed_config = {**SHARED_CONFIG, **PROPOSED_OVERRIDES}

    # CLI overrides for shared params
    if args.prompt:
        baseline_config["prompt"] = args.prompt
        proposed_config["prompt"] = args.prompt
    if args.object_prompt:
        baseline_config["object_prompt"] = args.object_prompt
        proposed_config["object_prompt"] = args.object_prompt
    if args.empty_prompt:
        baseline_config["empty_prompt"] = args.empty_prompt
        proposed_config["empty_prompt"] = args.empty_prompt
    if args.target_prompt:
        baseline_config["target_prompt"] = args.target_prompt
        proposed_config["target_prompt"] = args.target_prompt
    if args.seed is not None:
        baseline_config["seed"] = args.seed
        proposed_config["seed"] = args.seed
    if args.steps is not None:
        baseline_config["steps"] = args.steps
        proposed_config["steps"] = args.steps
    if args.cfg_scale is not None:
        baseline_config["cfg_scale"] = args.cfg_scale
        proposed_config["cfg_scale"] = args.cfg_scale

    # CLI overrides for proposed-only params
    if args.proposed_beta is not None:
        proposed_config["beta"] = args.proposed_beta
        proposed_config["label"] = f"Proposed (β={args.proposed_beta})"
    if args.proposed_blur is not None:
        proposed_config["blur"] = args.proposed_blur

    logger.info("=" * 60)
    logger.info("  Baseline vs Proposed 比較パイプライン")
    logger.info("=" * 60)
    logger.info(f"  Prompt: {baseline_config['prompt']}")
    logger.info(f"  Baseline: β={baseline_config['beta']}, blur={baseline_config['blur']}")
    logger.info(f"  Proposed: β={proposed_config['beta']}, blur={proposed_config['blur']}")
    logger.info(f"  Seed: {baseline_config['seed']}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Output: {run_dir}")
    logger.info("=" * 60)

    # Save configs
    with open(os.path.join(run_dir, "baseline", "config.json"), 'w') as f:
        json.dump(baseline_config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "proposed", "config.json"), 'w') as f:
        json.dump(proposed_config, f, indent=2, ensure_ascii=False)

    gen_times = {}

    # ========================================
    # Phase 1: Video Generation (or use existing)
    # ========================================
    baseline_video_path = None
    proposed_video_path = None

    if args.baseline_video and args.proposed_video:
        # --- Evaluate-only mode ---
        baseline_video_path = args.baseline_video
        proposed_video_path = args.proposed_video
        if not os.path.isabs(baseline_video_path):
            baseline_video_path = os.path.join(PROJECT_ROOT, baseline_video_path)
        if not os.path.isabs(proposed_video_path):
            proposed_video_path = os.path.join(PROJECT_ROOT, proposed_video_path)

        logger.info("\n📹 既存動画を使用（生成スキップ）")
        logger.info(f"  Baseline: {baseline_video_path}")
        logger.info(f"  Proposed: {proposed_video_path}")

    else:
        # --- Generate both videos ---
        input_image_path = args.input_image
        if not input_image_path:
            logger.error("--input_image が必要です（--baseline_video / --proposed_video で既存動画を使う場合を除く）")
            sys.exit(1)
        if not os.path.isabs(input_image_path):
            input_image_path = os.path.join(PROJECT_ROOT, input_image_path)
        if not os.path.exists(input_image_path):
            logger.error(f"Input image not found: {input_image_path}")
            sys.exit(1)

        input_image = np.array(Image.open(input_image_path).convert("RGB"))

        # Load models ONCE
        generator = VideoGenerator(device=device)

        from diffusers_helper.utils import save_bcthw_as_mp4

        # --- Generate Baseline ---
        logger.info("\n" + "=" * 50)
        logger.info("  PHASE 1a: BASELINE 動画生成")
        logger.info("=" * 50)

        gen_start = time.time()
        baseline_pixels = generator.generate(
            prompt=baseline_config["prompt"],
            input_image_np=input_image,
            config=baseline_config,
        )
        gen_times["baseline"] = time.time() - gen_start

        baseline_video_path = os.path.join(run_dir, "baseline", "video.mp4")
        crf = baseline_config.get("mp4_compression", 16)
        save_bcthw_as_mp4(baseline_pixels, baseline_video_path, fps=30, crf=crf)
        logger.info(f"  Baseline saved: {baseline_video_path} ({gen_times['baseline']:.1f}s)")

        del baseline_pixels
        torch.cuda.empty_cache()
        gc.collect()

        # --- Generate Proposed ---
        logger.info("\n" + "=" * 50)
        logger.info("  PHASE 1b: PROPOSED 動画生成")
        logger.info("=" * 50)

        gen_start = time.time()
        proposed_pixels = generator.generate(
            prompt=proposed_config["prompt"],
            input_image_np=input_image,
            config=proposed_config,
        )
        gen_times["proposed"] = time.time() - gen_start

        proposed_video_path = os.path.join(run_dir, "proposed", "video.mp4")
        save_bcthw_as_mp4(proposed_pixels, proposed_video_path, fps=30, crf=crf)
        logger.info(f"  Proposed saved: {proposed_video_path} ({gen_times['proposed']:.1f}s)")

        del proposed_pixels
        torch.cuda.empty_cache()
        gc.collect()

        # Free generator models
        del generator
        torch.cuda.empty_cache()
        gc.collect()

    # ========================================
    # Phase 2: Evaluation
    # ========================================
    logger.info("\n" + "=" * 50)
    logger.info("  PHASE 2: 評価")
    logger.info("=" * 50)

    eval_start = time.time()

    logger.info("\n--- Baseline 評価 ---")
    baseline_results = evaluate_video(
        baseline_video_path, baseline_config, device,
        skip_lpips=args.skip_lpips, skip_vbench=args.skip_vbench
    )

    logger.info("\n--- Proposed 評価 ---")
    proposed_results = evaluate_video(
        proposed_video_path, proposed_config, device,
        skip_lpips=args.skip_lpips, skip_vbench=args.skip_vbench
    )

    eval_time = time.time() - eval_start

    if not baseline_results or not proposed_results:
        logger.error("評価に失敗しました。")
        sys.exit(1)

    # Save individual results
    with open(os.path.join(run_dir, "baseline", "evaluation_results.json"), 'w') as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(run_dir, "proposed", "evaluation_results.json"), 'w') as f:
        json.dump(proposed_results, f, indent=2, ensure_ascii=False, default=str)

    # ========================================
    # Phase 3: Comparison Outputs
    # ========================================
    logger.info("\n" + "=" * 50)
    logger.info("  PHASE 3: 比較レポート生成")
    logger.info("=" * 50)

    # CSV
    csv_path = os.path.join(run_dir, "comparison_table.csv")
    generate_comparison_csv(baseline_results, proposed_results, csv_path)
    logger.info(f"  CSV: {csv_path}")

    # Chart
    chart_path = os.path.join(run_dir, "comparison_chart.png")
    generate_comparison_chart(baseline_results, proposed_results, chart_path)
    logger.info(f"  Chart: {chart_path}")

    # Markdown Report
    report_path = os.path.join(run_dir, "comparison_report.md")
    generate_comparison_report(
        baseline_results, proposed_results,
        baseline_config, proposed_config,
        report_path, gen_times=gen_times
    )
    logger.info(f"  Report: {report_path}")

    # Combined JSON
    combined = {
        "timestamp": timestamp,
        "baseline": baseline_results,
        "proposed": proposed_results,
        "baseline_config": baseline_config,
        "proposed_config": proposed_config,
        "generation_times": gen_times,
        "evaluation_time": eval_time,
    }
    combined_path = os.path.join(run_dir, "comparison_results.json")
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  JSON: {combined_path}")

    # ========================================
    # Summary
    # ========================================
    b_dis = baseline_results.get("disappearance", {})
    p_dis = proposed_results.get("disappearance", {})

    logger.info("\n" + "=" * 60)
    logger.info("  比較完了 - サマリー")
    logger.info("=" * 60)
    total_gen = sum(gen_times.values()) if gen_times else 0
    logger.info(f"  生成時間: {total_gen:.1f}s")
    logger.info(f"  評価時間: {eval_time:.1f}s")
    logger.info(f"  合計時間: {total_gen + eval_time:.1f}s")
    logger.info("")
    logger.info(f"  【Baseline】 Empty Prob: {b_dis.get('final_empty_prob', 0):.4f} | "
                f"Success: {b_dis.get('success_final_frame', False)}")
    logger.info(f"  【Proposed】 Empty Prob: {p_dis.get('final_empty_prob', 0):.4f} | "
                f"Success: {p_dis.get('success_final_frame', False)}")
    logger.info("")
    logger.info(f"  出力ディレクトリ: {run_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline vs Proposed 比較パイプライン（生成→評価→比較レポート）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 新規生成して比較
  python evaluation/compare_baseline_proposed.py \\
      --input_image experiments/inputs/434605182-f3bc35cf-656a-4c9c-a83a-bbab24858b09.jpg

  # 既存動画で評価だけ比較
  python evaluation/compare_baseline_proposed.py \\
      --baseline_video experiments/results/beta_0_dancer_performing_backflip.mp4 \\
      --proposed_video experiments/results/successful_data_blur_1.3_beta_-0.5.mp4
        """
    )

    # Input (either image or existing videos)
    parser.add_argument("--input_image", default=None,
                        help="入力画像のパス（新規生成する場合）")
    parser.add_argument("--baseline_video", default=None,
                        help="既存のベースライン動画パス（評価のみのモード）")
    parser.add_argument("--proposed_video", default=None,
                        help="既存の提案手法動画パス（評価のみのモード）")

    # Shared parameters
    parser.add_argument("--prompt", default=None, help="生成プロンプト")
    parser.add_argument("--object_prompt", default=None, help="消失対象のプロンプト")
    parser.add_argument("--empty_prompt", default=None, help="消失後のプロンプト")
    parser.add_argument("--target_prompt", default=None, help="CLIPスコア推移のターゲット")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument("--steps", type=int, default=None, help="推論ステップ数")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG Scale")

    # Proposed-only parameters
    parser.add_argument("--proposed_beta", type=float, default=None,
                        help="Proposed の Adaptive CFG Beta (default: -0.5)")
    parser.add_argument("--proposed_blur", type=float, default=None,
                        help="Proposed の Temporal Blur σ (default: 1.3)")

    # Control flags
    parser.add_argument("--output_dir", default="evaluation/comparison_results",
                        help="結果出力先ディレクトリ")
    parser.add_argument("--device", default="cuda", help="デバイス (cuda or cpu)")
    parser.add_argument("--skip_lpips", action="store_true", help="LPIPS評価をスキップ")
    parser.add_argument("--skip_vbench", action="store_true", default=True,
                        help="VBench評価をスキップ (default: True)")

    args = parser.parse_args()

    # Validate: need either input_image or both existing videos
    if not args.input_image and not (args.baseline_video and args.proposed_video):
        parser.error("--input_image か (--baseline_video + --proposed_video) のどちらかが必要です")

    run_comparison(args)


if __name__ == "__main__":
    main()

"""
Evaluate Successful Case Script
================================
experiments/results/main.txt に記載された成功条件を評価するスクリプト。

既存の評価手法を統合:
1. CLIP消失分析 (evaluate_disappearance.py の手法)
   - Object vs Empty 確率の時間推移
2. Frame-wise CLIP Score (run_vbench_custom.py の手法)
   - ターゲットプロンプトとの類似度の時間推移
3. LPIPS (run_vbench_custom.py の手法)
   - フレーム間の知覚変化率
4. VBench (オプション)
   - Dynamic Degree / Subject Consistency / Motion Smoothness

使い方:
    python evaluation/evaluate_successful_case.py
    python evaluation/evaluate_successful_case.py --video_path path/to/video.mp4
    python evaluation/evaluate_successful_case.py --video_path path/to/video.mp4 --device cpu
"""

import os
import sys
import json
import argparse
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================
# Default Successful Case Configuration (from main.txt)
# ============================================================
DEFAULT_CONFIG = {
    "video_path": "experiments/results/successful_data_blur_1.3_beta_-0.5.mp4",
    "prompt": "Static background. A man walks forward and out of view. Empty background remains.",
    "object_prompt": "a man walking",
    "empty_prompt": "empty background, no people",
    "target_prompt": "empty background remains, static scene",
    "experiment_conditions": {
        "use_cache": True,
        "length": "5s",
        "steps": 25,
        "seed": 31337,
        "cfg_scale": 6,
        "distilled_cfg_scale": 10,
        "mp4_compression": 16,
        "beta": -0.5,
        "blur": 1.3,
    }
}


# ============================================================
# Video Loading Utilities
# ============================================================
def load_frames_pil(video_path):
    """動画からフレームをPIL Imageのリストとして読み込む"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return [], 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames, fps


def load_frames_tensor(video_path, device='cuda'):
    """動画からフレームをTensorとして読み込む（LPIPS用）"""
    import torchvision.transforms as transforms
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        tensor = transform(img).unsqueeze(0)
        frames.append(tensor)
    cap.release()
    
    if len(frames) == 0:
        return None
    return torch.cat(frames, dim=0).to(device)


# ============================================================
# Metric 1: CLIP Disappearance Analysis
# ============================================================
def evaluate_disappearance(frames, model, processor, object_prompt, empty_prompt, device):
    """
    各フレームに対してObject/Emptyのsoftmax確率を計算。
    消失タスクでは、時間経過とともにEmpty確率が上昇することが期待される。
    """
    text_labels = [object_prompt, empty_prompt]
    inputs_text = processor(text=text_labels, return_tensors="pt", padding=True).to(device)
    
    prob_history = []
    batch_size = 8
    
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            inputs_image = processor(images=batch_frames, return_tensors="pt").to(device)
            outputs = model(**inputs_image, **inputs_text)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            prob_history.append(probs.cpu().numpy())
    
    probs_all = np.concatenate(prob_history, axis=0)
    object_probs = probs_all[:, 0]
    empty_probs = probs_all[:, 1]
    
    # 成功判定
    final_empty = float(empty_probs[-1])
    final_object = float(object_probs[-1])
    initial_object = float(object_probs[0])
    
    # Object確率の変化量
    object_drop = initial_object - final_object
    
    # Empty確率のピーク到達フレーム
    peak_empty_frame = int(np.argmax(empty_probs))
    
    return {
        "object_probs": object_probs.tolist(),
        "empty_probs": empty_probs.tolist(),
        "final_empty_prob": final_empty,
        "final_object_prob": final_object,
        "initial_object_prob": initial_object,
        "object_prob_drop": object_drop,
        "peak_empty_frame": peak_empty_frame,
        "peak_empty_prob": float(np.max(empty_probs)),
        "success_final_frame": final_empty > final_object,
    }


# ============================================================
# Metric 2: Frame-wise CLIP Score (Target Prompt Similarity)
# ============================================================
def evaluate_clip_temporal(frames, model, processor, target_prompt, device):
    """
    各フレームとターゲットプロンプトとのCLIP cosine similarityを計算。
    正のslopeは、動画内容がターゲットに近づいていることを示す。
    """
    scores = []
    
    with torch.no_grad():
        inputs_text = processor(text=[target_prompt], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs_text)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            inputs_image = processor(images=batch_frames, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs_image)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).squeeze(-1)
            scores.extend(similarity.cpu().numpy().tolist())
    
    scores_np = np.array(scores)
    
    # 線形回帰でslopeを算出
    x = np.arange(len(scores_np))
    slope, intercept = np.polyfit(x, scores_np, 1)
    
    return {
        "clip_scores": scores,
        "clip_start": scores[0],
        "clip_end": scores[-1],
        "clip_slope": float(slope),
        "clip_mean": float(np.mean(scores_np)),
        "clip_std": float(np.std(scores_np)),
        "clip_max": float(np.max(scores_np)),
        "clip_min": float(np.min(scores_np)),
    }


# ============================================================
# Metric 3: LPIPS Perceptual Change Rate
# ============================================================
def evaluate_lpips(frames_tensor, device):
    """
    連続フレーム間のLPIPS距離を計算。
    高い値 = 大きな知覚変化 / 低い値 = 静的
    """
    try:
        import lpips
    except ImportError:
        print("Warning: lpips not installed. Skipping LPIPS evaluation.")
        return None
    
    if frames_tensor is None or len(frames_tensor) < 2:
        return None
    
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    dists = []
    with torch.no_grad():
        for i in range(len(frames_tensor) - 1):
            d = loss_fn(frames_tensor[i:i+1], frames_tensor[i+1:i+2])
            dists.append(d.item())
    
    dists_np = np.array(dists)
    
    # 前半・後半での変化率比較
    mid = len(dists) // 2
    first_half_mean = float(np.mean(dists_np[:mid])) if mid > 0 else 0.0
    second_half_mean = float(np.mean(dists_np[mid:])) if mid > 0 else 0.0
    
    return {
        "lpips_per_frame": dists,
        "lpips_mean": float(np.mean(dists_np)),
        "lpips_std": float(np.std(dists_np)),
        "lpips_max": float(np.max(dists_np)),
        "lpips_first_half_mean": first_half_mean,
        "lpips_second_half_mean": second_half_mean,
    }


# ============================================================
# Metric 4: VBench (Optional)
# ============================================================
def evaluate_vbench(video_path, prompt, device):
    """VBenchの主要指標を計算（インストールされている場合のみ）"""
    try:
        from vbench import VBench
    except ImportError:
        print("Warning: VBench not installed. Skipping VBench evaluation.")
        return None
    
    try:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="vbench_eval_")
        
        vbench_dims = [
            "subject_consistency",
            "dynamic_degree",
            "motion_smoothness",
            "temporal_flickering",
            "imaging_quality"
        ]
        
        my_vbench = VBench(
            device=device,
            full_info_dir=os.path.join(output_dir, "vbench_info"),
            output_path=output_dir
        )
        
        vb_scores = my_vbench.evaluate(
            videos_path=[video_path],
            prompts=[prompt],
            dimension_list=vbench_dims,
            local=True
        )
        
        result = {}
        import pandas as pd
        if isinstance(vb_scores, pd.DataFrame) and len(vb_scores) > 0:
            for col in vbench_dims:
                if col in vb_scores.columns:
                    result[f"vbench_{col}"] = float(vb_scores.iloc[0][col])
        
        return result if result else None
        
    except Exception as e:
        print(f"VBench evaluation error: {e}")
        return None


# ============================================================
# Visualization
# ============================================================
def generate_visualizations(results, output_dir, config):
    """評価結果の可視化グラフを生成"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Evaluation: Successful Case (beta={config['experiment_conditions']['beta']}, "
        f"blur={config['experiment_conditions']['blur']})",
        fontsize=14, fontweight='bold'
    )
    
    # --- Plot 1: Object vs Empty Probability ---
    ax1 = axes[0, 0]
    if "disappearance" in results and results["disappearance"]:
        d = results["disappearance"]
        frames = range(len(d["object_probs"]))
        ax1.plot(frames, d["object_probs"], 'r-', linewidth=2, label=f'Object ("{config["object_prompt"]}")', alpha=0.8)
        ax1.plot(frames, d["empty_probs"], 'b-', linewidth=2, label=f'Empty ("{config["empty_prompt"]}")', alpha=0.8)
        ax1.fill_between(frames, d["empty_probs"], alpha=0.1, color='blue')
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        ax1.set_title("Disappearance Analysis: Object vs Empty Probability")
        ax1.set_xlabel("Frame Index")
        ax1.set_ylabel("Probability (CLIP Softmax)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
    else:
        ax1.text(0.5, 0.5, "Disappearance data not available", ha='center', va='center')
    
    # --- Plot 2: Frame-wise CLIP Score ---
    ax2 = axes[0, 1]
    if "clip_temporal" in results and results["clip_temporal"]:
        ct = results["clip_temporal"]
        frames = range(len(ct["clip_scores"]))
        ax2.plot(frames, ct["clip_scores"], 'g-', linewidth=2, marker='o', markersize=2, label='CLIP Score')
        
        # Trend line
        x = np.arange(len(ct["clip_scores"]))
        slope = ct["clip_slope"]
        intercept = ct["clip_scores"][0]
        trend_y = slope * x + intercept
        ax2.plot(frames, trend_y, 'r--', linewidth=1, alpha=0.7, 
                 label=f'Trend (slope={slope:.6f})')
        
        ax2.set_title(f'Frame-wise CLIP Score vs Target Prompt\n"{config["target_prompt"]}"')
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("CLIP Cosine Similarity")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "CLIP temporal data not available", ha='center', va='center')
    
    # --- Plot 3: LPIPS Per-Frame ---
    ax3 = axes[1, 0]
    if "lpips" in results and results["lpips"]:
        lp = results["lpips"]
        frames = range(len(lp["lpips_per_frame"]))
        ax3.bar(frames, lp["lpips_per_frame"], color='purple', alpha=0.7, label='LPIPS Distance')
        ax3.axhline(y=lp["lpips_mean"], color='red', linestyle='--', linewidth=1, 
                     label=f'Mean ({lp["lpips_mean"]:.4f})')
        ax3.set_title("Perceptual Change Rate (LPIPS) per Frame Pair")
        ax3.set_xlabel("Frame Pair Index (t → t+1)")
        ax3.set_ylabel("LPIPS Distance")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "LPIPS data not available", ha='center', va='center')
    
    # --- Plot 4: Summary Table ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_rows = []
    summary_rows.append(["Metric", "Value", "Interpretation"])
    
    if "disappearance" in results and results["disappearance"]:
        d = results["disappearance"]
        summary_rows.append(["Final Empty Prob", f'{d["final_empty_prob"]:.4f}', 
                             "✅ Good" if d["final_empty_prob"] > 0.5 else "⚠️ Low"])
        summary_rows.append(["Object Prob Drop", f'{d["object_prob_drop"]:.4f}', 
                             "✅ Decreased" if d["object_prob_drop"] > 0 else "⚠️ No drop"])
        summary_rows.append(["Success (Final)", str(d["success_final_frame"]), 
                             "✅" if d["success_final_frame"] else "❌"])
    
    if "clip_temporal" in results and results["clip_temporal"]:
        ct = results["clip_temporal"]
        summary_rows.append(["CLIP Slope", f'{ct["clip_slope"]:.6f}', 
                             "✅ Positive" if ct["clip_slope"] > 0 else "⚠️ Negative"])
        summary_rows.append(["CLIP Mean", f'{ct["clip_mean"]:.4f}', ""])
    
    if "lpips" in results and results["lpips"]:
        lp = results["lpips"]
        summary_rows.append(["LPIPS Mean", f'{lp["lpips_mean"]:.4f}', 
                             "Dynamic" if lp["lpips_mean"] > 0.05 else "Static"])
    
    if "vbench" in results and results["vbench"]:
        for k, v in results["vbench"].items():
            summary_rows.append([k.replace("vbench_", "VB: "), f'{v:.4f}', ""])
    
    if len(summary_rows) > 1:
        table = ax4.table(cellText=summary_rows[1:], colLabels=summary_rows[0],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        ax4.set_title("Evaluation Summary", fontsize=12, pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_dashboard.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {plot_path}")
    
    return plot_path


# ============================================================
# Report Generation
# ============================================================
def generate_report(results, config, output_dir):
    """Markdownレポートを生成"""
    lines = []
    lines.append("# 成功条件の評価レポート")
    lines.append("")
    lines.append(f"**日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**動画**: `{config['video_path']}`")
    lines.append(f"**プロンプト**: `{config['prompt']}`")
    lines.append("")
    
    # 実験条件テーブル
    lines.append("## 実験条件")
    lines.append("")
    lines.append("| パラメータ | 値 |")
    lines.append("|---|---|")
    for k, v in config["experiment_conditions"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    
    # 消失分析
    if "disappearance" in results and results["disappearance"]:
        d = results["disappearance"]
        lines.append("## 1. 消失分析 (CLIP Object vs Empty)")
        lines.append("")
        lines.append(f"- **最終フレーム Empty確率**: {d['final_empty_prob']:.4f}")
        lines.append(f"- **最終フレーム Object確率**: {d['final_object_prob']:.4f}")
        lines.append(f"- **初期Object確率**: {d['initial_object_prob']:.4f}")
        lines.append(f"- **Object確率の減少量**: {d['object_prob_drop']:.4f}")
        lines.append(f"- **Empty確率ピークフレーム**: {d['peak_empty_frame']}")
        lines.append(f"- **Empty確率ピーク値**: {d['peak_empty_prob']:.4f}")
        lines.append(f"- **消失成功判定（最終フレーム）**: {'✅ 成功' if d['success_final_frame'] else '❌ 失敗'}")
        lines.append("")
        
        if d['success_final_frame']:
            lines.append("> **解釈**: 最終フレームでEmpty確率がObject確率を上回っており、消失タスクが成功していると判断できる。")
        else:
            lines.append("> **解釈**: 最終フレームではまだObject確率がEmpty確率を上回っている。消失が完了していない可能性がある。")
        lines.append("")
    
    # CLIP temporal
    if "clip_temporal" in results and results["clip_temporal"]:
        ct = results["clip_temporal"]
        lines.append("## 2. Frame-wise CLIP Score (ターゲットプロンプトとの整合性)")
        lines.append("")
        lines.append(f"- **ターゲットプロンプト**: `{config['target_prompt']}`")
        lines.append(f"- **開始時スコア**: {ct['clip_start']:.4f}")
        lines.append(f"- **終了時スコア**: {ct['clip_end']:.4f}")
        lines.append(f"- **Slope（傾き）**: {ct['clip_slope']:.6f}")
        lines.append(f"- **平均スコア**: {ct['clip_mean']:.4f}")
        lines.append(f"- **標準偏差**: {ct['clip_std']:.4f}")
        lines.append("")
        
        if ct['clip_slope'] > 0:
            lines.append("> **解釈**: 正のslopeはフレームが時間経過とともにターゲット状態に近づいていることを示す。消失・変化が確認できる。")
        else:
            lines.append("> **解釈**: 負のslopeはフレームがターゲットから離れていることを示す。期待される変化が起きていない可能性がある。")
        lines.append("")
    
    # LPIPS
    if "lpips" in results and results["lpips"]:
        lp = results["lpips"]
        lines.append("## 3. LPIPS (知覚変化率)")
        lines.append("")
        lines.append(f"- **平均LPIPS距離**: {lp['lpips_mean']:.4f}")
        lines.append(f"- **標準偏差**: {lp['lpips_std']:.4f}")
        lines.append(f"- **最大変化**: {lp['lpips_max']:.4f}")
        lines.append(f"- **前半平均**: {lp['lpips_first_half_mean']:.4f}")
        lines.append(f"- **後半平均**: {lp['lpips_second_half_mean']:.4f}")
        lines.append("")
        
        if lp['lpips_first_half_mean'] > lp['lpips_second_half_mean']:
            lines.append("> **解釈**: 前半の変化率が高く、後半は安定している。これは「人が歩き去り、その後は背景が静止する」という期待動作と一致する。")
        else:
            lines.append("> **解釈**: 後半の変化率の方が高い。動画全体を通じて変化が続いている。")
        lines.append("")
    
    # VBench
    if "vbench" in results and results["vbench"]:
        lines.append("## 4. VBench 評価指標")
        lines.append("")
        lines.append("| 指標名 | スコア |")
        lines.append("|---|---|")
        for k, v in results["vbench"].items():
            name = k.replace("vbench_", "")
            lines.append(f"| {name} | {v:.4f} |")
        lines.append("")
        
        # Subject Consistency の逆説的解釈
        sc = results["vbench"].get("vbench_subject_consistency")
        if sc is not None:
            lines.append("> **Subject Consistency の逆説的解釈**: 消失タスクにおいては、Subject Consistencyが**低い**ことが")
            lines.append("> 望ましい。被写体が変化・消失したことを示唆するためである。")
            lines.append("")
    
    # 総合判定
    lines.append("## 総合判定")
    lines.append("")
    
    success_criteria = []
    if "disappearance" in results and results["disappearance"]:
        d = results["disappearance"]
        success_criteria.append(("消失成功（最終フレーム）", d["success_final_frame"]))
        success_criteria.append(("Object確率減少", d["object_prob_drop"] > 0.05))
    
    if "clip_temporal" in results and results["clip_temporal"]:
        ct = results["clip_temporal"]
        success_criteria.append(("CLIP Slope 正", ct["clip_slope"] > 0))
    
    if "lpips" in results and results["lpips"]:
        lp = results["lpips"]
        success_criteria.append(("動的変化あり (LPIPS > 0.02)", lp["lpips_mean"] > 0.02))
    
    lines.append("| 成功基準 | 結果 |")
    lines.append("|---|---|")
    for name, passed in success_criteria:
        icon = "✅" if passed else "❌"
        lines.append(f"| {name} | {icon} |")
    
    passed_count = sum(1 for _, p in success_criteria if p)
    total_count = len(success_criteria)
    lines.append("")
    lines.append(f"**合格: {passed_count}/{total_count}**")
    lines.append("")
    
    report_text = "\n".join(lines)
    
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Report saved to: {report_path}")
    
    return report_text


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="成功条件の動画を一括評価するスクリプト（CLIP消失分析・CLIP Temporal・LPIPS・VBench）"
    )
    parser.add_argument("--video_path", default=None,
                        help="評価する動画のパス（未指定の場合はmain.txtに記載のデフォルトを使用）")
    parser.add_argument("--prompt", default=None,
                        help="動画生成時のプロンプト")
    parser.add_argument("--object_prompt", default=None,
                        help="消失対象のプロンプト")
    parser.add_argument("--empty_prompt", default=None,
                        help="消失後のプロンプト")
    parser.add_argument("--target_prompt", default=None,
                        help="CLIPスコア推移のターゲットプロンプト")
    parser.add_argument("--output_dir", default="evaluation/results",
                        help="結果出力先ディレクトリ")
    parser.add_argument("--device", default="cuda",
                        help="使用デバイス (cuda or cpu)")
    parser.add_argument("--skip_lpips", action="store_true",
                        help="LPIPS評価をスキップする")
    parser.add_argument("--skip_vbench", action="store_true",
                        help="VBench評価をスキップする")
    
    args = parser.parse_args()
    
    # Configuration の構築
    config = DEFAULT_CONFIG.copy()
    if args.video_path:
        config["video_path"] = args.video_path
    if args.prompt:
        config["prompt"] = args.prompt
    if args.object_prompt:
        config["object_prompt"] = args.object_prompt
    if args.empty_prompt:
        config["empty_prompt"] = args.empty_prompt
    if args.target_prompt:
        config["target_prompt"] = args.target_prompt
    
    # Resolve relative paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    video_path = config["video_path"]
    if not os.path.isabs(video_path):
        video_path = os.path.join(project_root, video_path)
    
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    # Validate
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"eval_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("=" * 60)
    print("  成功条件の評価 - Successful Case Evaluation")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Prompt: {config['prompt']}")
    print(f"Device: {device}")
    print(f"Output: {run_output_dir}")
    print("=" * 60)
    
    # --- Load Video ---
    print("\n[1/5] Loading video frames...")
    frames_pil, fps = load_frames_pil(video_path)
    if not frames_pil:
        print("Error: No frames loaded from video.")
        sys.exit(1)
    print(f"  Loaded {len(frames_pil)} frames (FPS: {fps})")
    
    # --- Load CLIP Model ---
    print("\n[2/5] Loading CLIP model...")
    from transformers import CLIPProcessor, CLIPModel
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    print("  CLIP model loaded.")
    
    results = {}
    
    # --- Metric 1: Disappearance Analysis ---
    print("\n[3/5] Running Disappearance Analysis (CLIP Object vs Empty)...")
    results["disappearance"] = evaluate_disappearance(
        frames_pil, clip_model, clip_processor,
        config["object_prompt"], config["empty_prompt"], device
    )
    print(f"  Final Empty Prob: {results['disappearance']['final_empty_prob']:.4f}")
    print(f"  Success: {results['disappearance']['success_final_frame']}")
    
    # --- Metric 2: CLIP Temporal ---
    print("\n[4/5] Running Frame-wise CLIP Score Analysis...")
    results["clip_temporal"] = evaluate_clip_temporal(
        frames_pil, clip_model, clip_processor,
        config["target_prompt"], device
    )
    print(f"  CLIP Slope: {results['clip_temporal']['clip_slope']:.6f}")
    print(f"  CLIP Start→End: {results['clip_temporal']['clip_start']:.4f} → {results['clip_temporal']['clip_end']:.4f}")
    
    # Free CLIP model after use
    del clip_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # --- Metric 3: LPIPS ---
    if not args.skip_lpips:
        print("\n[5a/5] Running LPIPS Analysis...")
        frames_tensor = load_frames_tensor(video_path, device)
        results["lpips"] = evaluate_lpips(frames_tensor, device)
        if results["lpips"]:
            print(f"  LPIPS Mean: {results['lpips']['lpips_mean']:.4f}")
        del frames_tensor
        torch.cuda.empty_cache() if device == "cuda" else None
    else:
        print("\n[5a/5] LPIPS skipped.")
        results["lpips"] = None
    
    # --- Metric 4: VBench ---
    if not args.skip_vbench:
        print("\n[5b/5] Running VBench Analysis...")
        results["vbench"] = evaluate_vbench(video_path, config["prompt"], device)
        if results["vbench"]:
            for k, v in results["vbench"].items():
                print(f"  {k}: {v:.4f}")
    else:
        print("\n[5b/5] VBench skipped.")
        results["vbench"] = None
    
    # --- Generate Visualizations ---
    print("\nGenerating visualizations...")
    generate_visualizations(results, run_output_dir, config)
    
    # --- Generate Report ---
    print("\nGenerating report...")
    report_text = generate_report(results, config, run_output_dir)
    
    # --- Save raw results as JSON ---
    # Remove non-serializable data (lists are fine)
    json_results = {}
    for key, val in results.items():
        if val is not None:
            json_results[key] = val
    
    json_path = os.path.join(run_output_dir, "evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"Raw results saved to: {json_path}")
    
    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {run_output_dir}")
    print(f"Files generated:")
    print(f"  - evaluation_dashboard.png (可視化)")
    print(f"  - evaluation_report.md (レポート)")
    print(f"  - evaluation_results.json (生データ)")
    print("=" * 60)


if __name__ == "__main__":
    main()

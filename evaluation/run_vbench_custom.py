import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# VBench Imports
try:
    from vbench import VBench
    VBENCH_AVAILABLE = True
except ImportError:
    print("Warning: VBench not installed. Skipping VBench metrics.")
    VBENCH_AVAILABLE = False

# LPIPS Import
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not installed. Skipping LPIPS.")
    LPIPS_AVAILABLE = False

# Configuration
CATEGORIES = {
    "B_Disappearance": [
        "disappear", "fade", "melt", "transform", "explode", 
        "vanish", "dissolve", "remove", "turn into", "gone",
        "crumble", "evaporate"
    ],
    "C_LargeChange": [
        "zoom", "pan", "tilt", "transition", "morph", 
        "shift", "change", "move"
    ]
}

def get_category(prompt):
    prompt_lower = prompt.lower()
    for kw in CATEGORIES["B_Disappearance"]:
        if kw in prompt_lower:
            return "B_Disappearance"
    for kw in CATEGORIES["C_LargeChange"]:
        if kw in prompt_lower:
            return "C_LargeChange"
    return "A_Standard"

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def load_video_frames_tensor(video_path, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), # Resize for LPIPS/VGG
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

def calculate_lpips_diversity(frames_tensor, lpips_model):
    """
    Calculate average LPIPS between consecutive frames to measure perceptual change rate.
    High LPIPS = High Dynamics / Change.
    Low LPIPS = Static.
    """
    if frames_tensor is None or len(frames_tensor) < 2:
        return 0.0
    
    # Calculate pairwise distance between t and t+1
    dists = []
    with torch.no_grad():
        for i in range(len(frames_tensor) - 1):
            d = lpips_model(frames_tensor[i], frames_tensor[i+1])
            dists.append(d.item())
            
    return np.mean(dists)

def calculate_framewise_clip_score(frames, target_prompt, model, processor, device):
    """
    Calculates CLIP similarity between each frame and the target prompt.
    Returns a list of scores.
    """
    scores = []
    texts = [target_prompt]
    
    with torch.no_grad():
        # Batch processing could be faster, but per-frame is simpler for now
        # Process text once
        inputs_text = processor(text=texts, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs_text)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        for frame in frames:
            inputs_image = processor(images=frame, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs_image)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (image_features @ text_features.T).item()
            scores.append(similarity)
            
    return scores

def plot_clip_scores(scores_dict, output_path, title="Frame-wise CLIP Score Transition"):
    """
    Plots CLIP scores over frames for multiple runs (e.g. Beta 0.0 vs 0.7).
    scores_dict: { 'beta_0.0': [score, ...], 'beta_0.7': [score, ...] }
    """
    plt.figure(figsize=(10, 6))
    
    for label, scores in scores_dict.items():
        plt.plot(scores, label=label, marker='o', markersize=3)
        
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("CLIP Similarity to Target Prompt")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def run_evaluation(video_dir, metadata_path, output_dir, device='cuda'):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Init VBench
    my_vbench = None
    if VBENCH_AVAILABLE:
        vbench_dims = [
            "subject_consistency", 
            "dynamic_degree", 
            "motion_smoothness", 
            "temporal_flickering",
            "imaging_quality"
        ]
        try:
            my_vbench = VBench(device=device, full_info_dir=os.path.join(output_dir, "vbench_info"), output_path=output_dir)
        except Exception as e:
            print(f"Failed to initialize VBench: {e}")

    # Init LPIPS
    loss_fn_alex = None
    if LPIPS_AVAILABLE:
        try:
            loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        except Exception as e:
            print(f"Failed to initialize LPIPS: {e}")

    # Init CLIP
    try:
        clip_model_name = "openai/clip-vit-large-patch14"
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        print("CLIP model loaded for frame-wise analysis.")
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        clip_model = None

    results = []
    clip_trends = {} # Store clip scores for plotting: { case_id: { beta: [scores] } }
    
    for item in tqdm(metadata, desc="Evaluating videos"):
        video_filename = item.get("filename")
        prompt = item.get("prompt")
        target_prompt = item.get("target_prompt", prompt) # Use prompt if target not specified
        beta = item.get("beta", "unknown")
        case_id = item.get("case_id", "unknown")
        
        if not video_filename: continue
            
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path): continue
            
        category = get_category(prompt)
        row = {
            "filename": video_filename, 
            "prompt": prompt, 
            "category": category, 
            "beta": beta,
            "case_id": case_id
        }
        
        # 1. VBench
        if my_vbench:
            try:
                vb_scores = my_vbench.evaluate(
                    videos_path=[video_path],
                    prompts=[prompt], 
                    dimension_list=vbench_dims,
                    local=True
                )
                if isinstance(vb_scores, pd.DataFrame) and len(vb_scores) > 0:
                    for col in vbench_dims:
                        if col in vb_scores.columns:
                            row[f"vb_{col}"] = vb_scores.iloc[0][col]
            except Exception as e:
                print(f"VBench error for {video_filename}: {e}")
                
        # 2. LPIPS (Average Perceptual Change)
        if loss_fn_alex:
            try:
                frames_tensor = load_video_frames_tensor(video_path, device)
                if frames_tensor is not None:
                    score = calculate_lpips_diversity(frames_tensor, loss_fn_alex)
                    row["lpips_change_rate"] = score
            except Exception as e:
                print(f"LPIPS error for {video_filename}: {e}")
                row["lpips_change_rate"] = np.nan

        # 3. Frame-wise CLIP Score (Temporal Analysis)
        if clip_model:
            try:
                frames_pil = load_video_frames(video_path)
                if len(frames_pil) > 0:
                    scores = calculate_framewise_clip_score(frames_pil, target_prompt, clip_model, clip_processor, device)
                    
                    # Store for plotting
                    if case_id not in clip_trends:
                        clip_trends[case_id] = {}
                    clip_trends[case_id][f"Beta {beta}"] = scores
                    
                    # Add summary stats to row
                    row["clip_start"] = scores[0]
                    row["clip_end"] = scores[-1]
                    row["clip_slope"] = scores[-1] - scores[0] # Positive slope = getting closer to target
            except Exception as e:
                print(f"CLIP analysis error for {video_filename}: {e}")

        results.append(row)
        
    # Generate Plots
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    for case_id, betas_dict in clip_trends.items():
        if len(betas_dict) > 0:
            plot_path = os.path.join(visualization_dir, f"clip_trend_{case_id}.png")
            plot_clip_scores(betas_dict, plot_path, title=f"CLIP Score Transition: {case_id}")
            print(f"Saved visualization to {plot_path}")

    # Save & Report
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(output_dir, f"eval_results_{timestamp}.csv"), index=False)
    
    report = generate_report(df)
    with open(os.path.join(output_dir, f"eval_report_{timestamp}.md"), 'w') as f:
        f.write(report)
    print(f"Report saved to {output_dir}")

def generate_report(df):
    report_lines = ["# FramePack Evaluation Report (VBench + LPIPS + Temporal CLIP)", "", f"Date: {datetime.now()}", ""]
    
    categories = df['category'].unique()
    betas = df['beta'].unique()
    
    for cat in sorted(categories):
        cat_df = df[df['category'] == cat]
        report_lines.append(f"## Category: {cat}")
        
        report_lines.append("| Beta | Subject Consistency | Dynamic Degree | LPIPS (Change) | CLIP Slope (Target Correlation) |")
        report_lines.append("|---|---|---|---|---|")
        
        for beta in sorted(betas):
            beta_df = cat_df[cat_df['beta'] == beta]
            if len(beta_df) == 0: continue
            
            sc = beta_df.get('vb_subject_consistency', pd.Series([0])).mean()
            dd = beta_df.get('vb_dynamic_degree', pd.Series([0])).mean()
            lp = beta_df.get('lpips_change_rate', pd.Series([0])).mean()
            slope = beta_df.get('clip_slope', pd.Series([0])).mean()
            
            report_lines.append(f"| {beta} | {sc:.4f} | {dd:.4f} | {lp:.4f} | {slope:.4f} |")
        
        report_lines.append("")
        
        # New Interpretation Logic based on "clever hacks"
        if cat == "B_Disappearance":
            report_lines.append("### Success Criteria Analysis (Disappearance Task):")
            report_lines.append("1. **Dynamic Degree (Direct)**: Comparison of 'Dynamic Degree'. Higher is better, proving more movement.")
            report_lines.append("2. **Subject Consistency (Paradoxical)**: **LOWER is BETTER**. A drop in consistency implies the object has successfully disappeared/changed, whereas high consistency means it failed to disappear.")
            report_lines.append("3. **Frame-wise CLIP Slope**: **Positive Slope is BETTER**. It indicates the visual content is evolving *towards* the target state (e.g., empty background) over time.")
        else:
            report_lines.append("### Success Criteria Analysis (Standard Task):")
            report_lines.append("1. **Subject Consistency**: Higher is better (Stability).")
            report_lines.append("2. **Dynamic Degree**: Higher is better (if stability maintained).")
            
        report_lines.append("")

    return "\n".join(report_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", default="evaluation/results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_evaluation(args.video_dir, args.metadata, args.output_dir, args.device)

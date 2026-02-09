import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import cv2

# Import custom metrics
try:
    from evaluation.metrics import EvaluationMetrics
except ImportError:
    # Adding parent directory to path to allow import
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from evaluation.metrics import EvaluationMetrics

try:
    from vbench import VBench
    VBENCH_AVAILABLE = True
except ImportError:
    print("Warning: VBench is not installed. VBench metrics will be skipped.")
    print("Please install it using: pip install vbench")
    VBENCH_AVAILABLE = False

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
    
    # Check for Disappearance/Change keywords first
    for kw in CATEGORIES["B_Disappearance"]:
        if kw in prompt_lower:
            return "B_Disappearance"
    
    for kw in CATEGORIES["C_LargeChange"]:
        if kw in prompt_lower:
            return "C_LargeChange"
            
    # Default to Standard (Maintenance)
    return "A_Standard"

def load_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def evaluate_custom_metrics(video_path, prompt, metrics_engine):
    frames = load_video_frames(video_path)
    if not frames:
        return {}
    
    # Determine transition index (hardcoded for now, or estimated)
    # Assuming transition happens around 30% of the video or user-specified
    transition_idx = max(0, int(len(frames) * 0.2)) 
    
    try:
        stf_score = metrics_engine.compute_stf(frames, prompt, transition_index=transition_idx)
    except Exception as e:
        print(f"Error computing STF for {video_path}: {e}")
        stf_score = np.nan
        
    try:
        gas_score = metrics_engine.compute_gas(frames, transition_index=transition_idx)
    except Exception as e:
        print(f"Error computing GAS for {video_path}: {e}")
        gas_score = np.nan
        
    return {
        "custom_stf": stf_score,
        "custom_gas": gas_score
    }

def run_evaluation(video_dir, metadata_path, output_dir, device='cuda'):
    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize VBench
    vbench_dims = ["subject_consistency", "dynamic_degree", "motion_smoothness", "background_consistency", "temporal_flickering"]
    if VBENCH_AVAILABLE:
        # VBench initialization might take time and valid paths
        # Assuming default cache path or environment configuration
        try:
            my_vbench = VBench(device=device, full_info_dir=os.path.join(output_dir, "vbench_info"), output_path=output_dir)
        except Exception as e:
            print(f"Failed to initialize VBench: {e}")
            VBENCH_AVAILABLE = False
    
    # Initialize Custom Metrics
    custom_metrics = EvaluationMetrics(device=device)
    
    results = []
    
    for item in tqdm(metadata, desc="Evaluating videos"):
        video_filename = item.get("filename")
        prompt = item.get("prompt")
        
        if not video_filename or not prompt:
            print(f"Skipping invalid item: {item}")
            continue
            
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        category = get_category(prompt)
        
        row = {
            "filename": video_filename,
            "prompt": prompt,
            "category": category
        }
        
        # 1. Custom Metrics Evaluation
        custom_scores = evaluate_custom_metrics(video_path, prompt, custom_metrics)
        row.update(custom_scores)
        
        # 2. VBench Evaluation
        if VBENCH_AVAILABLE:
            # VBench typically takes a list of videos, but we do it one by one or batch it later?
            # Doing one by one for now to easily merge with custom metrics
            # Note: VBench API might expect list. Adjusting accordingly.
            try:
                # This is a hypothetical call, VBench API usage depends on version
                # If my_vbench.evaluate takes paths list:
                vb_scores = my_vbench.evaluate(
                    videos_path=[video_path],
                    prompts=[prompt], # Some metrics need prompt
                    dimension_list=vbench_dims,
                    local=True
                )
                # vb_scores is likely a DataFrame or dict
                if isinstance(vb_scores, pd.DataFrame):
                    for col in vb_scores.columns:
                        if col in vbench_dims:
                            row[f"vb_{col}"] = vb_scores.iloc[0][col]
                elif isinstance(vb_scores, dict):
                    for k, v in vb_scores.items():
                        row[f"vb_{k}"] = v
            except Exception as e:
                print(f"VBench evaluation failed for {video_filename}: {e}")
        
        results.append(row)
        
    # Save raw results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw results saved to {csv_path}")
    
    # Aggregation and Report
    report = generate_report(df)
    report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

def generate_report(df):
    report_lines = ["# FramePack Evaluation Report", "", f"Date: {datetime.now()}", ""]
    
    categories = df['category'].unique()
    
    for cat in sorted(categories):
        cat_df = df[df['category'] == cat]
        report_lines.append(f"## Category: {cat}")
        report_lines.append(f"Count: {len(cat_df)}")
        
        # Calculate means for numeric columns
        numeric_cols = cat_df.select_dtypes(include=[np.number]).columns
        means = cat_df[numeric_cols].mean()
        
        report_lines.append("| Metric | Mean Score | Std Dev |")
        report_lines.append("|---|---|---|")
        for col in numeric_cols:
            mean_val = means[col]
            std_val = cat_df[col].std()
            report_lines.append(f"| {col} | {mean_val:.4f} | {std_val:.4f} |")
        
        report_lines.append("")
        
        # Success Criteria Check (Mockup logic based on rationale)
        report_lines.append("### Success Criteria Check")
        if cat == "A_Standard":
            report_lines.append("- [ ] **Subject Consistency**: Should be HIGH (Monitor vb_subject_consistency)")
        elif cat == "B_Disappearance":
            report_lines.append("- [ ] **Alignment**: Should be HIGH (Monitor custom_stf)")
            report_lines.append("- [ ] **Subject Consistency**: Can be LOW")
        
        report_lines.append("")

    return "\n".join(report_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FramePack Evaluation Script")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing generated videos")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.json containing filename and prompt")
    parser.add_argument("--output_dir", type=str, default="evaluation/results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    run_evaluation(args.video_dir, args.metadata, args.output_dir, args.device)

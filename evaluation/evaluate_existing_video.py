
import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
import glob

# LPIPSのインポート（インストールされている場合）
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_magnitude = 0.0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        total_magnitude += np.mean(magnitude)
        frame_count += 1
        prev_gray = gray
        
    cap.release()
    return total_magnitude / frame_count if frame_count > 0 else 0.0

def generate_thumbnail(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        base_name = os.path.basename(video_path).split('.')[0]
        thumb_filename = f"{base_name}_thumb.jpg"
        thumb_path = os.path.join(output_dir, thumb_filename)
        cv2.imwrite(thumb_path, frame)
        return thumb_filename
    return None

def analyze_clip_model(frames, target_prompt, object_prompt, empty_prompt, device, model, processor):
    # 1. Disappearance Analysis (Object vs Empty Softmax)
    labels = [object_prompt, empty_prompt]
    inputs_text = processor(text=labels, return_tensors="pt", padding=True).to(device)
    
    probs_history = []
    batch_size = 8
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        inputs_image = processor(images=batch_frames, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs_image, **inputs_text)
            probs = outputs.logits_per_image.softmax(dim=1)
            probs_history.append(probs.cpu().numpy())
    
    probs_np = np.concatenate(probs_history, axis=0)
    object_probs = probs_np[:, 0]
    empty_probs = probs_np[:, 1]
    
    # 2. Target Fidelity Analysis (Cosine Similarity)
    inputs_target = processor(text=[target_prompt], return_tensors="pt", padding=True).to(device)
    target_scores = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        inputs_image = processor(images=batch_frames, return_tensors="pt").to(device)
        with torch.no_grad():
            # Robust embedding extraction
            img_feats = model.get_image_features(**inputs_image)
            if not isinstance(img_feats, torch.Tensor):
                img_feats = getattr(img_feats, "image_embeds", getattr(img_feats, "pooler_output", img_feats))

            txt_feats = model.get_text_features(**inputs_target)
            if not isinstance(txt_feats, torch.Tensor):
                txt_feats = getattr(txt_feats, "text_embeds", getattr(txt_feats, "pooler_output", txt_feats))

            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
            sim = (img_feats @ txt_feats.T).squeeze()
            if sim.ndim == 0:
                target_scores.append(sim.item())
            else:
                target_scores.extend(sim.cpu().numpy().tolist())
            
    target_scores = np.array(target_scores)
    
    # Slope (傾き) 計算
    if len(target_scores) > 1:
        x = np.arange(len(target_scores))
        slope, intercept = np.polyfit(x, target_scores, 1)
    else:
        slope = 0.0
    
    return {
        "object_probs": object_probs,
        "empty_probs": empty_probs,
        "target_scores": target_scores,
        "clip_slope": slope,
        "success": empty_probs[-1] > object_probs[-1]
    }

def analyze_clip(frames, target_prompt, object_prompt, empty_prompt, device):
    print(f"Loading CLIP model...")
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return analyze_clip_model(frames, target_prompt, object_prompt, empty_prompt, device, model, processor)

def process_single_video(args, device):
    print(f"--- Evaluating: {args.video_path} ---")
    frames = load_frames(args.video_path)
    if not frames: return
    
    # Optical Flow calculation
    flow_magnitude = calculate_optical_flow(args.video_path)
    print(f"Optical Flow Magnitude: {flow_magnitude:.4f}")

    results = analyze_clip(frames, args.target_prompt, args.object_prompt, args.empty_prompt, device)
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Probabilities
    plt.subplot(1, 2, 1)
    plt.plot(results["object_probs"], label=f'Object ("{args.object_prompt}")', color='red')
    plt.plot(results["empty_probs"], label=f'Empty ("{args.empty_prompt}")', color='blue')
    plt.title("Disappearance Probability")
    plt.xlabel("Frame Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Target Similarity
    plt.subplot(1, 2, 2)
    plt.plot(results["target_scores"], label="Similarity to Target", color='green')
    plt.title(f"Target Fidelity (Slope: {results['clip_slope']:.6f})")
    plt.xlabel("Frame Index")
    plt.ylabel("CLIP Cosine Similarity")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.output_dir, f"evaluation_{timestamp}.png")
    plt.savefig(plot_path)
    
    print("\n--- RESULTS ---")
    print(f"Disappearance Success: {'YES' if results['success'] else 'NO'}")
    print(f"CLIP Temporal Slope: {results['clip_slope']:.6f} (Positive is good)")
    print(f"Final Empty Probability: {results['empty_probs'][-1]:.4f}")
    print(f"Visualization saved to: {plot_path}")
    print("----------------")

def process_batch_videos(args, device):
    import glob
    
    input_dir = args.input_dir
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    video_files.sort()
    
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting batch processing...")
    
    # Load model once
    print(f"Loading CLIP model...")
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    report_path = os.path.join(args.output_dir, "batch_report.md")
    
    with open(report_path, "w") as f:
        f.write(f"# Batch Evaluation Report\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"| Video File | Motion (Flow) | CLIP Slope | Success | Thumbnail | Manual Score |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            print(f"Processing: {video_name}")
            
            # 1. Optical Flow
            flow = calculate_optical_flow(video_path)
            
            # 2. CLIP Analysis
            frames = load_frames(video_path)
            if not frames:
                continue
                
            results = analyze_clip_model(frames, args.target_prompt, args.object_prompt, args.empty_prompt, device, model, processor)
            
            # 3. Thumbnail
            thumb_name = generate_thumbnail(video_path, args.output_dir)
            thumb_md = f"![](./{thumb_name})" if thumb_name else "N/A"
            
            success_icon = "✅" if results['success'] else "❌"
            
            f.write(f"| {video_name} | {flow:.2f} | {results['clip_slope']:.4f} | {success_icon} | {thumb_md} | |\n")
            f.flush()
            
    print(f"Batch processing complete. Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generated video for the disappearance task.")
    parser.add_argument("--mode", choices=["single", "batch"], default="single", help="Processing mode: 'single' for one video, 'batch' for a directory.")
    parser.add_argument("--video_path", help="Path to the mp4 video file (required for single mode).")
    parser.add_argument("--input_dir", help="Directory containing mp4 files (required for batch mode).")
    parser.add_argument("--object_prompt", default="a man walking", help="Prompt for the object to disappear.")
    parser.add_argument("--empty_prompt", default="empty background", help="Prompt for the background alone.")
    parser.add_argument("--target_prompt", default="an empty street background", help="The goal state prompt.")
    parser.add_argument("--output_dir", default="evaluation/results", help="Directory to save results.")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "single":
        if not args.video_path:
            print("Error: --video_path is required for single mode.")
            return
        process_single_video(args, device)
    elif args.mode == "batch":
        if not args.input_dir:
            print("Error: --input_dir is required for batch mode.")
            return
        process_batch_videos(args, device)

if __name__ == "__main__":
    main()

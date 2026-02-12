
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

def analyze_clip(frames, target_prompt, object_prompt, empty_prompt, device):
    print(f"Loading CLIP model...")
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
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
    # ターゲットプロンプト（例: "empty background"）への近づき具合を測る
    inputs_target = processor(text=[target_prompt], return_tensors="pt", padding=True).to(device)
    target_scores = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        inputs_image = processor(images=batch_frames, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = model.get_image_features(**inputs_image)
            txt_feats = model.get_text_features(**inputs_target)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
            sim = (img_feats @ txt_feats.T).squeeze()
            target_scores.extend(sim.cpu().numpy().tolist())
            
    target_scores = np.array(target_scores)
    
    # Slope (傾き) 計算
    x = np.arange(len(target_scores))
    slope, intercept = np.polyfit(x, target_scores, 1)
    
    return {
        "object_probs": object_probs,
        "empty_probs": empty_probs,
        "target_scores": target_scores,
        "clip_slope": slope,
        "success": empty_probs[-1] > object_probs[-1]
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generated video for the disappearance task.")
    parser.add_argument("--video_path", required=True, help="Path to the mp4 video file.")
    parser.add_argument("--object_prompt", default="a man walking", help="Prompt for the object to disappear.")
    parser.add_argument("--empty_prompt", default="empty background", help="Prompt for the background alone.")
    parser.add_argument("--target_prompt", default="an empty street background", help="The goal state prompt.")
    parser.add_argument("--output_dir", default="evaluation/results", help="Directory to save results.")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"--- Evaluating: {args.video_path} ---")
    frames = load_frames(args.video_path)
    if not frames: return
    
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
    plot_path = os.path.join(args.output_dir, f"evaluation_{timestamp}.png")
    plt.savefig(plot_path)
    
    print("\n--- RESULTS ---")
    print(f"Disappearance Success: {'YES' if results['success'] else 'NO'}")
    print(f"CLIP Temporal Slope: {results['clip_slope']:.6f} (Positive is good)")
    print(f"Final Empty Probability: {results['empty_probs'][-1]:.4f}")
    print(f"Visualization saved to: {plot_path}")
    print("----------------")

if __name__ == "__main__":
    main()

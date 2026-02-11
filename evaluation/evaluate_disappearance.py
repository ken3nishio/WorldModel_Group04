
import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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

def calculate_probabilities(frames, model, processor, text_labels, device):
    """
    各フレームに対して、指定されたテキストラベルの確率（softmax）を計算する。
    Args:
        frames: List of PIL Images
        text_labels: List of strings (e.g. ["a photo of a man", "an empty street"])
    Returns:
        prob_history: List of shape (N_frames, N_labels)
    """
    prob_history = []
    
    # テキスト入力の準備（一度だけ計算）
    inputs_text = processor(text=text_labels, return_tensors="pt", padding=True).to(device)
    
    # バッチ処理のためにフレームを分割処理する（VRAM節約）
    batch_size = 8
    
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            inputs_image = processor(images=batch_frames, return_tensors="pt").to(device)
            
            outputs = model(**inputs_image, **inputs_text)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            
            prob_history.append(probs.cpu().numpy())
            
    return np.concatenate(prob_history, axis=0)

def evaluate_video(video_path, object_prompt="a man", empty_prompt="empty background", device="cuda"):
    print(f"Loading video: {video_path}")
    frames = load_frames(video_path)
    if not frames:
        return None
    
    print(f"Loaded {len(frames)} frames. Analyzing content...")
    
    # CLIPモデルのロード
    model_name = "openai/clip-vit-large-patch14"
    try:
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        return None
        
    labels = [object_prompt, empty_prompt]
    probs = calculate_probabilities(frames, model, processor, labels, device)
    
    # 解析結果
    object_probs = probs[:, 0]
    empty_probs = probs[:, 1]
    
    # 成功判定: 最後のフレームのempty確率が高いか
    final_score = empty_probs[-1]
    success = final_score > object_probs[-1]
    
    return {
        "frames": range(len(frames)),
        "object_probs": object_probs,
        "empty_probs": empty_probs,
        "success": success,
        "final_empty_score": final_score
    }

def plot_comparison(results_dict, output_path="comparison_plot.png"):
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    for idx, (label, res) in enumerate(results_dict.items()):
        if res is None: continue
        
        # Plot Object Probability (Dashed)
        plt.plot(res["frames"], res["object_probs"], linestyle='--', color=colors[idx % len(colors)], alpha=0.5, label=f"{label} (Object)")
        
        # Plot Empty Probability (Solid)
        plt.plot(res["frames"], res["empty_probs"], linestyle='-', linewidth=2, color=colors[idx % len(colors)], label=f"{label} (Empty)")
        
    plt.title("Disappearance Task Analysis: Object vs Empty Probability")
    plt.xlabel("Frame Index")
    plt.ylabel("Probability (CLIP Softmax)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate disappearance task success using CLIP.")
    parser.add_argument("--video_paths", nargs='+', required=True, help="Path(s) to video files to evaluate.")
    parser.add_argument("--labels", nargs='+', default=["Baseline", "Proposed"], help="Labels for the videos in the plot.")
    parser.add_argument("--object_prompt", default="a man walking", help="Prompt describing the object that should disappear.")
    parser.add_argument("--empty_prompt", default="empty street background", help="Prompt describing the scene after disappearance.")
    parser.add_argument("--output_plot", default="evaluation_plot.png", help="Path to save the comparison plot.")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    
    for i, video_path in enumerate(args.video_paths):
        label = args.labels[i] if i < len(args.labels) else f"Video {i+1}"
        print(f"Evaluating {label} ({video_path})...")
        
        res = evaluate_video(video_path, args.object_prompt, args.empty_prompt, device)
        results[label] = res
        
        if res:
            print(f"  Result for {label}:")
            print(f"  - Final Empty Probability: {res['final_empty_score']:.4f}")
            print(f"  - Success (Empty > Object): {res['success']}")
            
    plot_comparison(results, args.output_plot)

if __name__ == "__main__":
    main()


import os
import sys
import argparse
import torch
import cv2
import numpy as np
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime

# Check for LPIPS compatibility
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not found. Falling back to MSE.")

def load_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
    current = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        current += 1
        if len(frames) >= num_frames: break
        
    cap.release()
    return frames

def calculate_consistency(frames, device):
    if len(frames) < 2: return 0.0, 0.0
    
    distances = []
    if LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net='alex').to(device)
        to_tensor = lambda x: torch.from_numpy(np.array(x)).permute(2,0,1).float()/127.5 - 1.0
        
        with torch.no_grad():
            for i in range(len(frames)-1):
                img0 = to_tensor(frames[i]).unsqueeze(0).to(device)
                img1 = to_tensor(frames[i+1]).unsqueeze(0).to(device)
                dist = loss_fn(img0, img1)
                distances.append(dist.item())
    else:
        for i in range(len(frames)-1):
            arr0 = np.array(frames[i]).astype(float)/255.0
            arr1 = np.array(frames[i+1]).astype(float)/255.0
            mse = np.mean((arr0 - arr1)**2)
            distances.append(mse * 10.0)
            
    return np.mean(distances)

def analyze_clip_score(frames, prompt, device):
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    inputs = processor(text=[prompt], images=frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # image_embeds = outputs.image_embeds
        # text_embeds = outputs.text_embeds
        logits_per_image = outputs.logits_per_image # (n_frames, 1)
        probs = logits_per_image.softmax(dim=1) # useless for single text

        # Use cosine similarity directly as score
        # CLIP score uses cosine similarity * 100 usually, but we keep raw here
        # Actually logits_per_image is scaled dot product. To be precise let's normalize.
        
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        
        similarity = (image_embeds @ text_embeds.t()).squeeze() # (n_frames,)
        
    return similarity.mean().item(), similarity.max().item()

def generate_thumb(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        name = os.path.basename(video_path).split('.')[0] + "_thumb.jpg"
        cv2.imwrite(os.path.join(output_dir, name), frame)
        return name
    return None

def process_batch(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    
    if not files:
        print("No files found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"batch_report_{args.task}.md")
    
    print(f"Generating report for {len(files)} videos...")
    
    with open(report_path, "w") as f:
        f.write(f"# Batch Report: {args.task}\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Prompt evaluated: '{args.prompt}'\n\n")
        
        f.write("| File | CLIP Score (Mean) | Consistency (LPIPS) | Trade-off (CLIP/LPIPS) | Thumb |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            
        for path in files:
            print(f"Processing {os.path.basename(path)}...")
            frames = load_frames(path)
            if not frames: continue
            
            thumb = generate_thumb(path, args.output_dir)
            thumb_md = f"![](./{thumb})" if thumb else ""
            
            # 1. CLIP Score (Prompt Adherence)
            clip_mean, clip_max = analyze_clip_score(frames, args.prompt, device)
            
            # 2. Consistency (Visual Stability)
            # Lower is better for LPIPS, but for trade-off we want to maximize adherence while minimizing instability.
            consistency_score = calculate_consistency(frames, device)
            
            # 3. Trade-off Score
            # High CLIP & Low LPIPS is ideal.
            # Avoid division by zero
            trade_off = clip_mean / (consistency_score + 1e-6)
            
            f.write(f"| {os.path.basename(path)} | {clip_mean:.4f} | {consistency_score:.4f} | {trade_off:.2f} | {thumb_md} |\n")
            f.flush()
            
    print(f"Saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="evaluation/results")
    parser.add_argument("--task", default="backflip")
    parser.add_argument("--prompt", default="A man performs a backflip") # Default Prompt
    args = parser.parse_args()
    
    process_batch(args)

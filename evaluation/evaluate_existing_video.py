
import os
import sys
import argparse
import torch
import cv2
import numpy as np
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, VideoMAEImageProcessor, VideoMAEForVideoClassification
from datetime import datetime

# Check for LPIPS compatibility
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not found. Falling back to MSE.")

def load_frames(video_path, num_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    indices = range(total_frames)
    if num_frames:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
    current = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if num_frames is None or current in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        current += 1
        if num_frames and len(frames) >= num_frames: break
        
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
            
    return np.mean(distances), np.max(distances)

def analyze_action(frames, device):
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device)
    
    # Resample to 16 frames for VideoMAE
    if len(frames) != 16:
        indices = np.linspace(0, len(frames)-1, 16, dtype=int)
        sampled = [frames[i] for i in indices]
    else:
        sampled = frames
        
    inputs = processor(list(sampled), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Kinetics-400 indices: 318(somersaulting), 355(tumbling), 147(gymnastics)
    target_indices = [318, 355, 147]
    action_score = sum([probs[0, i].item() for i in target_indices])
    
    pred_idx = logits.argmax(-1).item()
    pred_label = model.config.id2label[pred_idx]
    
    return action_score, pred_label, probs[0, pred_idx].item()

def analyze_disappearance(frames, obj_prompt, emp_prompt, device):
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    inputs = processor(text=[obj_prompt, emp_prompt], images=frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
        
    success = probs[-1, 1] > probs[-1, 0]
    return success, probs[:, 0], probs[:, 1]

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
    
    with open(report_path, "w") as f:
        f.write(f"# Batch Report: {args.task}\n")
        f.write(f"Date: {datetime.now()}\n\n")
        
        if args.task == "backflip":
            f.write("| File | Action Score | Top Class | Consistency (LPIPS/MSE) | Success | Thumb |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        else:
            f.write("| File | Empty Prob | Success | Thumb |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            
        for path in files:
            print(f"Processing {os.path.basename(path)}...")
            frames = load_frames(path)
            if not frames: continue
            
            thumb = generate_thumb(path, args.output_dir)
            thumb_md = f"![](./{thumb})" if thumb else ""
            
            if args.task == "backflip":
                score, label, prob = analyze_action(frames, device)
                avg_met, max_met = calculate_consistency(frames, device)
                
                # Thresholds: Action > 0.05 AND Consistency < 0.5 (LPIPS) or < 0.05 (MSE)
                is_success = (score > 0.05) and (max_met < 0.5 if LPIPS_AVAILABLE else max_met < 2.0)
                icon = "✅" if is_success else "❌"
                
                f.write(f"| {os.path.basename(path)} | {score:.3f} | {label} ({prob:.2f}) | {avg_met:.3f} | {icon} | {thumb_md} |\n")
            else:
                suc, obj, emp = analyze_disappearance(frames, args.obj_prompt, args.emp_prompt, device)
                icon = "✅" if suc else "❌"
                f.write(f"| {os.path.basename(path)} | {emp[-1]:.3f} | {icon} | {thumb_md} |\n")
            
            f.flush()
            
    print(f"Saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="evaluation/results")
    parser.add_argument("--task", default="backflip")
    parser.add_argument("--obj_prompt", default="object")
    parser.add_argument("--emp_prompt", default="empty")
    args = parser.parse_args()
    
    process_batch(args)


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
import traceback

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
    
    # Stratified sampling
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
    current = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        current += 1
        # Stop maximizing reading to avoid long loops if video is huge (unlikely here)
        if current > indices[-1]: break
        
    cap.release()
    
    # If we didn't get enough frames (e.g. short video), pad with last frame
    if len(frames) > 0 and len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
        
    return frames

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.clip_model = None
        self.videomae_model = None
        self.lpips_model = None
        
    def init_clip(self):
        if self.clip_model is None:
            print("Loading CLIP...")
            model_name = "openai/clip-vit-large-patch14"
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)

    def init_videomae(self):
        if self.videomae_model is None:
            print("Loading VideoMAE...")
            model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
            self.videomae_image_processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.videomae_model = VideoMAEForVideoClassification.from_pretrained(model_name).to(self.device)

    def init_lpips(self):
        if self.lpips_model is None and LPIPS_AVAILABLE:
            print("Loading LPIPS...")
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)

    def calculate_clip_score(self, frames, prompt):
        self.init_clip()
        inputs = self.clip_processor(text=[prompt], images=frames, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Normalize embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            
            # cosine similarity
            similarity = (image_embeds @ text_embeds.t()).squeeze()
            
        return similarity.mean().item()

    def calculate_lpips_score(self, frames):
        self.init_lpips()
        if not LPIPS_AVAILABLE:
            # MSE Fallback
            distances = []
            for i in range(len(frames)-1):
                arr0 = np.array(frames[i]).astype(float)/255.0
                arr1 = np.array(frames[i+1]).astype(float)/255.0
                mse = np.mean((arr0 - arr1)**2)
                distances.append(mse * 100.0) # Scale up
            return np.mean(distances)
            
        to_tensor = lambda x: torch.from_numpy(np.array(x)).permute(2,0,1).float()/127.5 - 1.0
        distances = []
        with torch.no_grad():
            for i in range(len(frames)-1):
                img0 = to_tensor(frames[i]).unsqueeze(0).to(self.device)
                img1 = to_tensor(frames[i+1]).unsqueeze(0).to(self.device)
                dist = self.lpips_model(img0, img1)
                distances.append(dist.item())
        return np.mean(distances)

    def calculate_videomae_score(self, frames, target_labels=['somersaulting', 'tumbling', 'gymnastics']):
        self.init_videomae()
        
        inputs = self.videomae_image_processor(list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.videomae_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
        # Get top predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_classes = [self.videomae_model.config.id2label[idx.item()] for idx in top5_indices[0]]
        top_class = top5_classes[0]
        top_score = top5_probs[0][0].item()

        # Check target class probability specifically
        target_score = 0.0
        for label_id, label_name in self.videomae_model.config.id2label.items():
            if any(target in label_name.lower() for target in target_labels):
                target_score += probs[0][label_id].item()
                
        return target_score, f"{top_class} ({top_score:.2f})"

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
    report_path = os.path.join(args.output_dir, f"full_report_{args.task}.md")
    
    print(f"Generating FULL report for {len(files)} videos...")
    evaluator = Evaluator(device)
    
    # Decide target labels for VideoMAE
    target_labels = ['somersaulting', 'tumbling'] if 'backflip' in args.task else ['walking', 'standing'] # Default
    if 'disappear' in args.task:
        target_labels = ['walking', 'standing'] # Disappearance is hard for action recognition

    with open(report_path, "w") as f:
        f.write(f"# Comprehensive Batch Report: {args.task}\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Prompt evaluated: '{args.prompt}'\n\n")
        
        f.write("| File | CLIP Score | LPIPS (Consistency) | VideoMAE (Target) | Top Class | Thumb |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            
        for path in files:
            try:
                print(f"Processing {os.path.basename(path)}...")
                frames = load_frames(path)
                if not frames: continue
                
                thumb = generate_thumb(path, args.output_dir)
                thumb_md = f"![](./{thumb})" if thumb else ""
                
                # 1. CLIP Score
                clip_score = evaluator.calculate_clip_score(frames, args.prompt)
                
                # 2. LPIPS
                lpips_score = evaluator.calculate_lpips_score(frames)
                
                # 3. VideoMAE
                mae_score, top_class_str = evaluator.calculate_videomae_score(frames, target_labels)
                
                f.write(f"| {os.path.basename(path)} | {clip_score:.4f} | {lpips_score:.4f} | {mae_score:.4f} | {top_class_str} | {thumb_md} |\n")
                f.flush()
            except Exception as e:
                print(f"Error processing {path}: {e}")
                traceback.print_exc()
            
    print(f"Saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="evaluation/results")
    parser.add_argument("--task", default="backflip")
    parser.add_argument("--prompt", default="A man performs a backflip") 
    args = parser.parse_args()
    
    process_batch(args)

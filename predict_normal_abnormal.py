"""
predict_normal_abnormal.py
--------------------------
Give it a video (about 30 s) and it outputs ONLY the probability that the video is
Normal and the probability that it is Abnormal.

Pipeline (identical to training):
  first N seconds of the video -> every 10th frame -> DenseNet121 (1024-d per frame)
  -> [mean ‖ max] pooling = one 2048-d vector -> trained binary classifiers
  -> P(normal), P(abnormal)   (average of the models that give probabilities)

Usage
  python predict_normal_abnormal.py --video clip.mp4
"""

import argparse
import json
import os
import sys

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROB_MODELS = ["LogisticRegression", "HistGradientBoosting", "RandomForest", "SVM_RBF", "MLP"]

PREPROCESS = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_densenet(device):
    try:
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    except AttributeError:
        m = models.densenet121(pretrained=True)
    m.classifier = nn.Identity()
    return m.to(device).eval()


def read_frames(path, seconds, every_n, frame_size):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or idx / fps > seconds:
            break
        if idx % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_size > 0:
                rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    return frames, idx / fps


@torch.no_grad()
def video_vector(path, densenet, device, seconds, every_n, frame_size):
    frames, used_s = read_frames(path, seconds, every_n, frame_size)
    if not frames:
        raise RuntimeError("no frames read")
    feats = []
    for i in range(0, len(frames), 64):
        batch = torch.stack([PREPROCESS(f) for f in frames[i:i + 64]]).to(device)
        feats.append(densenet(batch).cpu())
    f = torch.cat(feats).float()
    return torch.cat([f.mean(0), f.max(0)[0]]).numpy().reshape(1, -1), len(frames), used_s


def load_models(model_dir):
    out = {}
    for name in PROB_MODELS:
        p = os.path.join(model_dir, f"binary_{name}.joblib")
        if os.path.exists(p):
            out[name] = joblib.load(p)
    if not out:
        sys.exit(f"No binary_*.joblib models found in {model_dir}. Run 2_train.bat first.")
    return out


def p_abnormal(model, x):
    """Probability of class 1 (= Anomaly); class order is [0=Normal, 1=Anomaly]."""
    return float(model.predict_proba(x)[0][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--models", default="results/models")
    ap.add_argument("--seconds", type=float, default=30.0, help="use only the first N seconds")
    ap.add_argument("--every", type=int, default=10, help="keep every Nth frame")
    ap.add_argument("--frame-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    densenet = load_densenet(device)
    clfs = load_models(args.models)

    x, n_frames, used_s = video_vector(args.video, densenet, device, args.seconds, args.every, args.frame_size)
    per_model = {m: p_abnormal(c, x) for m, c in clfs.items()}
    pa = float(np.mean(list(per_model.values())))
    print(f"{os.path.basename(args.video)}  ({n_frames} frames, {min(used_s, args.seconds):.1f}s)")
    for m, v in per_model.items():
        print(f"   {m:22s} P(abnormal)={v:.3f}")
    print(f"P(normal)={1 - pa:.3f}   P(abnormal)={pa:.3f}   -> {'abnormal' if pa >= 0.5 else 'normal'}")


if __name__ == "__main__":
    main()

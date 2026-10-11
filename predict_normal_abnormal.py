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
  python predict_normal_abnormal.py --folder test_videos            # every video in a folder
  python predict_normal_abnormal.py --urls urls.txt                 # download links, then predict

urls.txt: one link per line. Optional label in front so accuracy can be computed:
  abnormal https://www.youtube.com/watch?v=XXXX
  normal   https://www.youtube.com/watch?v=YYYY
Videos in a folder are labelled from their file name: a name containing "abnormal" counts as
abnormal, a name containing "normal" as normal, anything else is unlabelled.

Results are printed and saved to predictions.csv.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
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


def label_from_name(name):
    n = name.lower()
    if "abnormal" in n or "anomaly" in n:
        return "abnormal"
    if "normal" in n:
        return "normal"
    return ""


def download_urls(url_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with open(url_file, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    for i, line in enumerate(lines, 1):
        parts = line.split()
        label = parts[0].lower() if len(parts) > 1 and parts[0].lower() in ("normal", "abnormal") else ""
        url = parts[-1]
        if not url.lower().startswith("http"):
            continue
        base = f"{label or 'video'}_{i:02d}"
        dest = os.path.join(out_dir, base + ".mp4")
        if not os.path.exists(dest):
            print(f"[{i}/{len(lines)}] downloading {url}")
            cmd = [sys.executable, "-m", "yt_dlp", "--force-overwrites", "--no-playlist",
                   "-f", "b[height<=480]/bv*[height<=480]/b/bv*", "-o", dest, url]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(dest):
                print("   download FAILED:", (r.stderr.strip().splitlines() or ["?"])[-1])
                continue
        paths.append(dest)
    return paths


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--folder")
    g.add_argument("--urls")
    ap.add_argument("--models", default="results/models")
    ap.add_argument("--seconds", type=float, default=30.0, help="use only the first N seconds")
    ap.add_argument("--every", type=int, default=10, help="keep every Nth frame")
    ap.add_argument("--frame-size", type=int, default=64)
    ap.add_argument("--download-dir", default="test_videos")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    if args.video:
        paths = [args.video]
    elif args.folder:
        paths = sorted(p for p in glob.glob(os.path.join(args.folder, "*")) if p.lower().endswith(VIDEO_EXT))
    else:
        paths = download_urls(args.urls, args.download_dir)
    if not paths:
        sys.exit("No videos to process.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    densenet = load_densenet(device)
    clfs = load_models(args.models)

    rows = []
    for p in paths:
        name = os.path.basename(p)
        try:
            x, n_frames, used_s = video_vector(p, densenet, device, args.seconds, args.every, args.frame_size)
        except Exception as e:
            print(f"{name}: skipped ({e})")
            continue
        per_model = {m: p_abnormal(c, x) for m, c in clfs.items()}
        pa = float(np.mean(list(per_model.values())))
        row = {"video": name, "seconds_used": round(min(used_s, args.seconds), 1), "frames": n_frames,
               "P_normal": round(1 - pa, 3), "P_abnormal": round(pa, 3),
               "prediction": "abnormal" if pa >= 0.5 else "normal",
               "true_label": label_from_name(name)}
        for m, v in per_model.items():
            row[f"P_abnormal_{m}"] = round(v, 3)
        rows.append(row)
        print(f"{name:28s} P(normal)={row['P_normal']:.3f}   P(abnormal)={row['P_abnormal']:.3f}"
              f"   -> {row['prediction']}" + (f"   (true: {row['true_label']})" if row["true_label"] else ""))

    if not rows:
        sys.exit("Nothing was predicted.")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lab = [r for r in rows if r["true_label"]]
    if lab:
        ok = sum(r["prediction"] == r["true_label"] for r in lab)
        print(f"\nAccuracy on labelled videos: {ok}/{len(lab)} = {ok / len(lab):.0%}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

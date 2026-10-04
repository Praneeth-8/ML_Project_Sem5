"""
test_youtube_video.py
---------------------
Classify a NEW video (e.g. downloaded from YouTube) with the models trained by
train_classifiers.py.

It reproduces the training feature pipeline exactly:
  video -> sample every Nth frame -> resize to 64x64 (UCF-Crime Kaggle frames are 64x64 PNGs)
        -> Resize(256) -> CenterCrop(224) -> ImageNet normalize
        -> DenseNet121 (classifier = Identity) -> 1024-dim per frame
        -> [mean(1024), max(1024)] = 2048-dim video vector
        -> every saved model predicts Normal/Anomaly and the 14-class label

Download a video first, for example:
    pip install yt-dlp
    yt-dlp -f "mp4[height<=480]" -o test_video.mp4 "<youtube url>"

Run:
    python test_youtube_video.py --video test_video.mp4 --models results/models
    python test_youtube_video.py --video test_video.mp4 --start 30 --end 75   # only a clip
    python test_youtube_video.py --video test_video.mp4 --window 10           # per-10s-window predictions

Requirements: torch, torchvision, opencv-python, scikit-learn, joblib, numpy, pillow
"""

import argparse
import glob
import json
import os

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_densenet(device):
    try:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    except AttributeError:          # older torchvision
        model = models.densenet121(pretrained=True)
    model.classifier = nn.Identity()
    return model.to(device).eval()


PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_frames(path, every_n, frame_size, start_s, end_s):
    """Returns list of (timestamp_seconds, PIL image) for every Nth frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {path}\n  fps={fps:.1f}  frames={total}  duration={total / fps:.1f}s")

    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if end_s is not None and t > end_s:
            break
        if t >= start_s and idx % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_size > 0:   # match the low-res training frames
                rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            frames.append((t, Image.fromarray(rgb)))
        idx += 1
    cap.release()
    print(f"  sampled {len(frames)} frames (every {every_n}th, resized to "
          f"{frame_size}x{frame_size})" if frame_size > 0 else f"  sampled {len(frames)} frames")
    return frames


@torch.no_grad()
def extract_frame_features(frames, model, device, batch_size=64):
    feats = []
    for i in range(0, len(frames), batch_size):
        batch = torch.stack([PREPROCESS(img) for _, img in frames[i:i + batch_size]]).to(device)
        feats.append(model(batch).cpu())
    return torch.cat(feats, dim=0)          # (n_frames, 1024)


def mean_max(frame_feats):
    """Same pooling as compile_all_videos_mean_max.py."""
    f = frame_feats.float()
    return torch.cat([f.mean(dim=0), f.max(dim=0)[0]]).numpy()   # (2048,)


def load_models(model_dir):
    out = {"binary": {}, "multiclass": {}}
    for p in sorted(glob.glob(os.path.join(model_dir, "*.joblib"))):
        task, name = os.path.basename(p)[:-7].split("_", 1)
        out[task][name] = joblib.load(p)
    return out


def predict_all(vec, models_by_task, meta):
    x = vec.reshape(1, -1)
    results = {}
    for task, labels in [("binary", meta["binary_labels"]), ("multiclass", meta["classes"])]:
        results[task] = {}
        for name, m in models_by_task[task].items():
            pred = int(m.predict(x)[0])
            conf = None
            if hasattr(m, "predict_proba"):
                conf = float(m.predict_proba(x)[0][pred])
            results[task][name] = (labels[pred], conf)
    return results


def print_results(results, header):
    print("\n" + "-" * 62 + f"\n{header}\n" + "-" * 62)
    for task in ["binary", "multiclass"]:
        print(f"[{task}]")
        votes = {}
        for name, (label, conf) in results[task].items():
            c = f"{conf:.2f}" if conf is not None else "  - "
            print(f"   {name:22s} -> {label:14s} (conf {c})")
            votes[label] = votes.get(label, 0) + 1
        winner = max(votes, key=votes.get)
        print(f"   {'MAJORITY VOTE':22s} -> {winner}  ({votes[winner]}/{len(results[task])} models)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--models", default="results/models")
    ap.add_argument("--every", type=int, default=10, help="keep every Nth frame (dataset used 10)")
    ap.add_argument("--frame-size", type=int, default=64,
                    help="resize frames to this square size before DenseNet (training frames were 64x64; 0 = keep original)")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--end", type=float, default=None, help="end time (s)")
    ap.add_argument("--window", type=float, default=0,
                    help="if >0, also predict on consecutive windows of this many seconds")
    args = ap.parse_args()

    with open(os.path.join(args.models, "meta.json")) as f:
        meta = json.load(f)
    models_by_task = load_models(args.models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    densenet = load_densenet(device)

    frames = read_frames(args.video, args.every, args.frame_size, args.start, args.end)
    if not frames:
        raise SystemExit("No frames sampled - check --start/--end.")
    feats = extract_frame_features(frames, densenet, device)
    print(f"  frame features: {tuple(feats.shape)}")

    # whole video (this is exactly how training videos were represented)
    print_results(predict_all(mean_max(feats), models_by_task, meta),
                  f"WHOLE VIDEO  ({frames[0][0]:.1f}s - {frames[-1][0]:.1f}s)")

    # optional: sliding windows, to see WHERE in a long video the anomaly is
    if args.window > 0:
        times = np.array([t for t, _ in frames])
        t0 = times[0]
        while t0 <= times[-1]:
            mask = (times >= t0) & (times < t0 + args.window)
            if mask.sum() >= 3:
                res = predict_all(mean_max(feats[torch.from_numpy(mask)]), models_by_task, meta)
                b = [r[0] for r in res["binary"].values()]
                m = [r[0] for r in res["multiclass"].values()]
                b_vote = max(set(b), key=b.count)
                m_vote = max(set(m), key=m.count)
                print(f"  window {t0:6.1f}-{t0 + args.window:6.1f}s : {b_vote:8s} "
                      f"({b.count(b_vote)}/{len(b)})  | {m_vote} ({m.count(m_vote)}/{len(m)})")
            t0 += args.window


if __name__ == "__main__":
    main()

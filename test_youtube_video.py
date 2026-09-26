"""
test_youtube_video.py
---------------------
Turn a NEW video (e.g. downloaded from YouTube) into the same 2048-dim vector
used for training, so it can be classified by the models from train_classifiers.py.

It reproduces the training feature pipeline exactly:
  video -> sample every Nth frame -> resize to 64x64 (UCF-Crime Kaggle frames are 64x64 PNGs)
        -> Resize(256) -> CenterCrop(224) -> ImageNet normalize
        -> DenseNet121 (classifier = Identity) -> 1024-dim per frame
        -> [mean(1024), max(1024)] = 2048-dim video vector

Download a video first, for example:
    pip install yt-dlp
    yt-dlp -f "mp4[height<=480]" -o test_video.mp4 "<youtube url>"

Run:
    python test_youtube_video.py --video test_video.mp4

Requirements: torch, torchvision, opencv-python, numpy, pillow
"""

import argparse

import cv2
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


def read_frames(path, every_n, frame_size):
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
        if idx % every_n == 0:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=int, default=10, help="keep every Nth frame (dataset used 10)")
    ap.add_argument("--frame-size", type=int, default=64,
                    help="resize frames to this square size before DenseNet (training frames were 64x64; 0 = keep original)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    densenet = load_densenet(device)

    frames = read_frames(args.video, args.every, args.frame_size)
    if not frames:
        raise SystemExit("No frames sampled.")
    feats = extract_frame_features(frames, densenet, device)
    print(f"  frame features: {tuple(feats.shape)}")

    vec = mean_max(feats)
    print(f"  video vector: {vec.shape}  (mean 1024 + max 1024)")


if __name__ == "__main__":
    main()

# Video Anomaly Classification – Final Phase

Every UCF-Crime training video is turned into **one 2048-dim vector**:
DenseNet121 frame features (1024) → **mean pool ‖ max pool** over all frames.
Data: 1610 videos, 14 classes (800 NormalVideos + 810 anomalies across 13 crime types).

## New in this phase
| Script | What it does |
|---|---|
| `train_classifiers.py` | Trains mid-term baselines (Logistic, LinearSVC, Perceptron) **and new models**: HistGradientBoosting, RBF-kernel SVM, MLP, Random Forest. Runs both tasks (binary Normal vs Anomaly, and 14-class), 5-fold CV + 80/20 hold-out, saves reports, confusion matrices and models. |
| `test_youtube_video.py` | Takes any `.mp4` (e.g. from YouTube), runs the **same** pipeline (every 10th frame → 64×64 → DenseNet121 → mean+max) and prints every model's prediction + majority vote. `--window 10` gives per-10-second predictions to locate the anomaly in long videos. |

## How to run
```bash
pip install scikit-learn pandas matplotlib joblib opencv-python torch torchvision yt-dlp

# 1. train + evaluate everything (creates results/ incl. results/models/)
python train_classifiers.py --data all_classes_dataset_2048.npz
#    faster: --no-cv     or skip slow ones: --skip HistGradientBoosting

# 2. download a test video and classify it
yt-dlp -f "mp4[height<=480]" -o test_video.mp4 "<youtube url>"
python test_youtube_video.py --video test_video.mp4 --models results/models --window 10
#    only a part of the video:  --start 30 --end 75
```

## Results
Hold-out = stratified 80/20 split (seed 42). CV = 5-fold stratified, macro-F1 mean ± std.

### Binary: Normal vs Anomaly (balanced, 800 vs 810)
| Model | Accuracy | Balanced acc | Macro F1 | ROC-AUC | CV macro F1 |
|---|---|---|---|---|---|
| Logistic Regression *(mid-term)* | 0.826 | 0.826 | 0.826 | 0.911 | 0.830 ± 0.013 |
| Linear SVC *(mid-term)* | 0.804 | 0.804 | 0.804 | 0.896 | 0.819 ± 0.014 |
| Perceptron *(mid-term)* | 0.820 | 0.820 | 0.820 | 0.896 | 0.811 ± 0.014 |
| **HistGradientBoosting** | 0.839 | 0.838 | 0.838 | 0.898 | 0.827 ± 0.020 |
| **SVM (RBF kernel)** | **0.860** | **0.860** | **0.860** | **0.943** | **0.864 ± 0.019** |
| **MLP (1×512)** | 0.814 | 0.814 | 0.814 | 0.907 | 0.845 ± 0.011 |
| Random Forest | 0.811 | 0.811 | 0.810 | 0.903 | 0.822 ± 0.016 |

### Multiclass: 14 classes (imbalanced: 800 Normal, 27–145 per crime)
| Model | Accuracy | Balanced acc | Macro F1 | CV macro F1 |
|---|---|---|---|---|
| Logistic Regression *(mid-term)* | 0.599 | **0.327** | **0.332** | **0.300 ± 0.027** |
| Linear SVC *(mid-term)* | 0.578 | 0.282 | 0.293 | 0.265 ± 0.019 |
| Perceptron *(mid-term)* | 0.519 | 0.229 | 0.216 | 0.231 ± 0.016 |
| **HistGradientBoosting** | 0.602 | 0.294 | 0.318 | 0.274 ± 0.048 |
| **SVM (RBF kernel)** | **0.618** | 0.265 | 0.274 | 0.279 ± 0.022 |
| **MLP (1×512)** | **0.618** | 0.284 | 0.289 | 0.279 ± 0.025 |
| Random Forest | 0.516 | 0.108 | 0.108 | 0.095 ± 0.013 |


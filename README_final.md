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

## Findings
1. **Normal vs Anomaly works well with one vector per video**: ~82–86 % accuracy. A non-linear model (RBF-SVM) adds ~3–4 points over the linear baselines (AUC 0.94 vs 0.91).
2. **Which crime it is stays hard**: macro-F1 ≈ 0.3 for every model. Accuracy (~60 %) looks better only because Normal is half the data. Most errors are crimes predicted as *Normal* (see `confusion_matrices/`). Mean/max pooling over a whole long video dilutes a crime that lasts a few seconds.
3. **Non-linear models barely help on 14 classes**: plain Logistic Regression has the best balanced accuracy / macro-F1. The limit is the representation and the small number of examples per crime (27–48 for several classes), not the classifier.
4. **HistGradientBoosting** is the best tree model (0.838 binary, 0.318 multiclass F1) but is the slowest to train. **Random Forest** collapses on the 14-class task: it predicts almost everything as Normal despite class weighting.
5. Classes that do best: Robbery, Stealing, Shoplifting, RoadAccidents (~0.4–0.5 F1). Explosion and Shooting (only 27–29 videos) are almost never recognised.

## Real-world test (YouTube videos)
Two 15-second clips downloaded from YouTube and run through `test_youtube_video.py` (every 10th frame, 64x64, DenseNet121, mean+max, all 7 models).

| Clip | Normal vs Anomaly | 14-class prediction |
|---|---|---|
| CCTV robbery (YouTube Short) | **Anomaly, 7/7 models** (LogReg 0.97, MLP 1.00, SVM-RBF 0.92, HGB 0.89). Both 10 s windows: Anomaly 7/7 | Split: Robbery 2, Burglary 2, Stealing 1, Normal 2 (no clear winner) |
| Normal CCTV footage | **Normal, 7/7 models** (LogReg 1.00, MLP 0.98, HGB 0.98, SVM-RBF 0.92). Both windows: Normal 7/7 | **NormalVideos, 7/7 models** |

This matches the validation results: on unseen real footage the binary decision is reliable, while the exact crime type is confused between similar "taking things" classes (Robbery / Burglary / Stealing).

## Notes / limitations
* Only the **Train** split features exist, so all scores come from splits of it. Report them as validation scores.
* The YouTube test resizes frames to 64×64 so they look like the training frames (the Kaggle UCF-Crime frames are 64×64 PNGs, every 10th frame). Use `--frame-size 0` to compare with full resolution.
* Trust the binary prediction more than the 14-class label, and test with short clips around the event (`--start/--end`). A long video is mostly normal footage, so the pooled vector looks normal.

## 10-video test (videos NOT from UCF-Crime)
Pipeline: `predict_normal_abnormal.py` takes a ~30 s video and outputs only P(normal) and P(abnormal) (mean of LogReg, HistGradientBoosting, RandomForest, SVM-RBF, MLP probabilities; abnormal if P>=0.5).
Test set: 5 abnormal + 5 normal YouTube clips (see `urls.txt`, `predictions.csv`, `predict_log.txt`).
Result: 10/10 correct. Closest calls: abnormal_04 (P(abnormal)=0.574) and normal_06 (P(normal)=0.596).
Caveat: only 10 clips, so this is a demo, not a statistically strong estimate; the CV accuracy (~0.83-0.86) is the reliable number.

"""
train_classifiers.py
--------------------
Trains classifiers on the 2048-dim (mean + max pooled DenseNet121)
video vectors produced by compile_all_videos_mean_max.py.

Two tasks:
  * binary     : NormalVideos (0) vs any anomaly (1)
  * multiclass : all 14 classes (13 anomaly types + NormalVideos)

Models:
  Mid-term baselines : Logistic Regression, Linear SVC, Perceptron

Usage:
  python train_classifiers.py --data all_classes_dataset_2048.npz
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
NORMAL_CLASS = "NormalVideos"


def build_models():
    """Every model is a pipeline so scaling is learned only on training data."""
    return {
        # ---- mid-term baselines ----
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")),
        "LinearSVC": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.01, max_iter=10000, class_weight="balanced")),
        "Perceptron": make_pipeline(
            StandardScaler(),
            Perceptron(max_iter=1000, class_weight="balanced", random_state=SEED)),
    }


def run_task(task, X, y, labels):
    print("\n" + "=" * 70)
    print(f"TASK: {task}   samples={len(y)}   classes={len(labels)}")
    print("=" * 70)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"  {name:22s} hold-out acc={acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="all_classes_dataset_2048.npz")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y, classes = d["X"].astype(np.float32), d["y"], list(d["classes"])
    print(f"Loaded {args.data}: X={X.shape}, {len(classes)} classes")

    normal_idx = classes.index(NORMAL_CLASS)
    y_bin = (y != normal_idx).astype(int)          # 1 = anomaly

    run_task("binary", X, y_bin, ["Normal", "Anomaly"])
    run_task("multiclass", X, y, classes)


if __name__ == "__main__":
    main()

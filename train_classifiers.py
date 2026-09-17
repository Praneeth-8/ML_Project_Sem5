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
  New                : HistGradientBoosting, Random Forest, RBF-kernel SVM, MLP

Usage:
  python train_classifiers.py --data all_classes_dataset_2048.npz
  python train_classifiers.py --skip HistGradientBoosting                   # HGB is the slowest
"""

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

SEED = 42
NORMAL_CLASS = "NormalVideos"
SKIP = set()


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
        # ---- new models ----
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                probability=True, random_state=SEED)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-2, max_iter=300,
                          early_stopping=True, random_state=SEED)),
    }


def run_task(task, X, y, labels):
    print("\n" + "=" * 70)
    print(f"TASK: {task}   samples={len(y)}   classes={len(labels)}")
    print("=" * 70)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    for name, model in build_models().items():
        if name in SKIP:
            continue
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"  {name:22s} hold-out acc={acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="all_classes_dataset_2048.npz")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="model names to skip, e.g. --skip HistGradientBoosting RandomForest")
    args = ap.parse_args()
    SKIP.update(args.skip)

    d = np.load(args.data)
    X, y, classes = d["X"].astype(np.float32), d["y"], list(d["classes"])
    print(f"Loaded {args.data}: X={X.shape}, {len(classes)} classes")

    normal_idx = classes.index(NORMAL_CLASS)
    y_bin = (y != normal_idx).astype(int)          # 1 = anomaly

    run_task("binary", X, y_bin, ["Normal", "Anomaly"])
    run_task("multiclass", X, y, classes)


if __name__ == "__main__":
    main()

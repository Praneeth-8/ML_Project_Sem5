"""
train_classifiers.py
--------------------
Trains and compares classifiers on the 2048-dim (mean + max pooled DenseNet121)
video vectors produced by compile_all_videos_mean_max.py.

Two tasks:
  * binary     : NormalVideos (0) vs any anomaly (1)
  * multiclass : all 14 classes (13 anomaly types + NormalVideos)

Models:
  Mid-term baselines : Logistic Regression, Linear SVC, Perceptron
  New                : HistGradientBoosting, Random Forest, RBF-kernel SVM, MLP

Evaluation:
  * 5-fold stratified cross-validation on the whole dataset (mean +- std)
  * one stratified 80/20 hold-out split -> classification report + confusion matrix
  * every model is then refit on ALL data and saved to models/ for test_youtube_video.py

Usage:
  python train_classifiers.py --data all_classes_dataset_2048.npz
  python train_classifiers.py --data all_classes_dataset_2048.npz --no-cv   # faster
  python train_classifiers.py --skip HistGradientBoosting                   # HGB is the slowest
"""

import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

SEED = 42
NORMAL_CLASS = "NormalVideos"
SKIP = set()


def build_models():
    """Every model is a pipeline so scaling is learned only on training folds."""
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


def scores_for_auc(model, X):
    """Continuous score for ROC-AUC (binary task only)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def plot_cm(cm, labels, title, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = max(5, 0.55 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_task(task, X, y, labels, out_dir, do_cv):
    print("\n" + "=" * 70)
    print(f"TASK: {task}   samples={len(y)}   classes={len(labels)}")
    print("=" * 70)
    os.makedirs(os.path.join(out_dir, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    rows = []
    for name, model in build_models().items():
        if name in SKIP:
            continue
        t0 = time.time()
        row = {"task": task, "model": name}

        # ---- 5-fold CV on all data ----
        if do_cv:
            scoring = {"acc": "accuracy", "bal_acc": "balanced_accuracy",
                       "f1_macro": "f1_macro"}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            for k in scoring:
                row[f"cv_{k}_mean"] = res[f"test_{k}"].mean()
                row[f"cv_{k}_std"] = res[f"test_{k}"].std()

        # ---- hold-out split ----
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        row["holdout_acc"] = accuracy_score(y_te, pred)
        row["holdout_bal_acc"] = balanced_accuracy_score(y_te, pred)
        row["holdout_f1_macro"] = f1_score(y_te, pred, average="macro")
        if task == "binary":
            row["holdout_roc_auc"] = roc_auc_score(y_te, scores_for_auc(model, X_te))

        rep = classification_report(y_te, pred, labels=range(len(labels)),
                                    target_names=labels, digits=3, zero_division=0)
        with open(os.path.join(out_dir, "reports", f"{task}_{name}.txt"), "w") as f:
            f.write(f"{task} / {name}  (stratified 80/20 hold-out, seed={SEED})\n\n{rep}")
        cm = confusion_matrix(y_te, pred, labels=range(len(labels)))
        plot_cm(cm, labels, f"{task} - {name} (hold-out)",
                os.path.join(out_dir, "confusion_matrices", f"{task}_{name}.png"))

        # ---- refit on ALL data and save for inference ----
        model.fit(X, y)
        joblib.dump(model, os.path.join(out_dir, "models", f"{task}_{name}.joblib"))

        row["time_s"] = round(time.time() - t0, 1)
        rows.append(row)
        msg = f"  {name:22s} hold-out acc={row['holdout_acc']:.3f} " \
              f"bal_acc={row['holdout_bal_acc']:.3f} f1={row['holdout_f1_macro']:.3f}"
        if do_cv:
            msg += f" | CV f1={row['cv_f1_macro_mean']:.3f}±{row['cv_f1_macro_std']:.3f}"
        print(msg + f"  ({row['time_s']}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="all_classes_dataset_2048.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-cv", action="store_true", help="skip 5-fold CV (faster)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="model names to skip, e.g. --skip HistGradientBoosting RandomForest")
    args = ap.parse_args()
    SKIP.update(args.skip)

    d = np.load(args.data)
    X, y, classes = d["X"].astype(np.float32), d["y"], list(d["classes"])
    os.makedirs(args.out, exist_ok=True)

    normal_idx = classes.index(NORMAL_CLASS)
    y_bin = (y != normal_idx).astype(int)          # 1 = anomaly

    rows = []
    rows += run_task("binary", X, y_bin, ["Normal", "Anomaly"], args.out, not args.no_cv)
    rows += run_task("multiclass", X, y, classes, args.out, not args.no_cv)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results_summary.csv"), index=False)
    with open(os.path.join(args.out, "models", "meta.json"), "w") as f:
        json.dump({"classes": classes, "binary_labels": ["Normal", "Anomaly"],
                   "normal_class": NORMAL_CLASS, "feature_dim": int(X.shape[1])}, f, indent=2)

    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df.drop(columns=[c for c in df.columns if c.endswith("_std")]).to_string(index=False))
    print(f"\nSaved: {args.out}/results_summary.csv, reports/, confusion_matrices/, models/")


if __name__ == "__main__":
    main()

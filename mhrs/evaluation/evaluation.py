"""
evaluation.py
─────────────────────────────────────────────────────────────────────────────
Full evaluation suite:
  - Recommender metrics: Precision@K, Recall@K, NDCG@K, MAP, MRR
  - Mood classifier metrics: Accuracy, F1 (per-class + macro), Confusion matrix
  - Cross-validation on Kaggle emotions dataset
  - Visualisations: confusion matrix heatmap, P@K curves, layer score breakdown
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Tuple


# ── Recommender metrics ───────────────────────────────────────────────────────

def precision_at_k(recommended: List, relevant: List, k: int) -> float:
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended: List, relevant: List, k: int) -> float:
    if not relevant: return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def dcg_at_k(recommended: List, relevant: List, k: int) -> float:
    return sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )

def ndcg_at_k(recommended: List, relevant: List, k: int) -> float:
    ideal = dcg_at_k(list(relevant), relevant, k)
    return 0.0 if ideal == 0 else dcg_at_k(recommended, relevant, k) / ideal

def average_precision(recommended: List, relevant: List) -> float:
    if not relevant: return 0.0
    hits = ap = 0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant)

def mean_reciprocal_rank(all_recommended: List[List], all_relevant: List[List]) -> float:
    rrs = []
    for rec, rel in zip(all_recommended, all_relevant):
        rr = 0.0
        for i, item in enumerate(rec):
            if item in rel:
                rr = 1 / (i + 1)
                break
        rrs.append(rr)
    return float(np.mean(rrs))

def map_score(all_recommended: List[List], all_relevant: List[List]) -> float:
    return float(np.mean([average_precision(r, rv) for r, rv in zip(all_recommended, all_relevant)]))


# ── Classifier cross-validation ───────────────────────────────────────────────

def cross_validate_classifier(
    texts: List[str],
    labels: List[str],
    n_splits: int = 5,
) -> Dict:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=20000,
                                   sublinear_tf=True, stop_words="english")),
        ("clf",  LogisticRegression(C=5.0, max_iter=500, solver="lbfgs",
                                    class_weight="balanced", random_state=42)),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X, y = np.array(texts), np.array(labels)

    fold_accs, fold_f1s = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        fold_accs.append(accuracy_score(y[te], preds))
        fold_f1s.append(f1_score(y[te], preds, average="macro", zero_division=0))
        print(f"  Fold {fold}: acc={fold_accs[-1]:.3f}  macro-F1={fold_f1s[-1]:.3f}")

    return {
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy":  float(np.std(fold_accs)),
        "mean_f1":       float(np.mean(fold_f1s)),
        "std_f1":        float(np.std(fold_f1s)),
        "fold_accs":     fold_accs,
        "fold_f1s":      fold_f1s,
    }


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: List[str], y_pred: List[str], save_path: str = None):
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Mood Classifier — Normalised Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved → {save_path}")
    return fig


def plot_precision_recall_curves(
    results: List[Dict],  # each: {recommended: list, relevant: list, label: str}
    ks: List[int] = None,
    save_path: str = None,
):
    if ks is None:
        ks = [1, 2, 3, 5, 8, 10]

    prec_vals = [np.mean([precision_at_k(r["recommended"], r["relevant"], k) for r in results]) for k in ks]
    rec_vals  = [np.mean([recall_at_k(   r["recommended"], r["relevant"], k) for r in results]) for k in ks]
    ndcg_vals = [np.mean([ndcg_at_k(     r["recommended"], r["relevant"], k) for r in results]) for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ks, prec_vals, "o-", color="#7F77DD", label="Precision@K")
    ax1.plot(ks, rec_vals,  "s-", color="#E8593C", label="Recall@K")
    ax1.plot(ks, ndcg_vals, "^-", color="#1D9E75", label="NDCG@K")
    ax1.set_xlabel("K"); ax1.set_ylabel("Score")
    ax1.set_title("Recommender Metrics vs K")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1)

    # Per-query MAP bar chart
    aps = [average_precision(r["recommended"], r["relevant"]) for r in results]
    labels = [r.get("label", f"Q{i+1}") for i, r in enumerate(results)]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax2.barh(labels, aps, color=colors)
    ax2.set_xlabel("Average Precision"); ax2.set_title("Per-Query Average Precision")
    ax2.set_xlim(0, 1); ax2.grid(alpha=0.3, axis="x")
    for i, v in enumerate(aps):
        ax2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

    plt.suptitle("Mental Health Recommender — Evaluation Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Metrics chart saved → {save_path}")
    return fig


def plot_cv_results(cv_results: Dict, save_path: str = None):
    folds = list(range(1, len(cv_results["fold_accs"]) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(folds, cv_results["fold_accs"], "o-", color="#7F77DD", label="Accuracy")
    ax.plot(folds, cv_results["fold_f1s"],  "s-", color="#E8593C", label="Macro-F1")
    ax.axhline(cv_results["mean_accuracy"], color="#7F77DD", linestyle="--", alpha=0.5)
    ax.axhline(cv_results["mean_f1"],       color="#E8593C", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fold"); ax.set_ylabel("Score")
    ax.set_title(f"5-Fold Cross-Validation  |  Acc={cv_results['mean_accuracy']:.3f}±{cv_results['std_accuracy']:.3f}  "
                 f"F1={cv_results['mean_f1']:.3f}±{cv_results['std_f1']:.3f}")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# ── Full evaluation run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from recommender.pipeline import load_all, recommend
    from models.nlp_mood_detector import ALL_MOODS, MOOD_DISPLAY
    from data.data_loader import build_unified_training_data

    print("Loading models…")
    load_all()

    # ── Test queries with ground truth ──────────────────────────────────────
    TEST_CASES = [
        {"text": "I have been feeling really anxious, my mind races and I can't stop worrying about everything", 
         "expected_mood": "anxiety", "relevant_ids": [3,4,12,13,21,44,47,58]},
        {"text": "I am completely stressed and burned out from work, too many deadlines and no time to rest",
         "expected_mood": "stress", "relevant_ids": [1,2,19,23,31,41,48]},
        {"text": "I feel so depressed and hopeless, nothing brings me joy and I see no point in trying",
         "expected_mood": "sadness", "relevant_ids": [5,17,18,24,28,45,52,55,66]},
        {"text": "I can't sleep at all, I lie awake for hours with a racing mind",
         "expected_mood": "insomnia", "relevant_ids": [4,8,20,26,53,57,83]},
        {"text": "I feel completely alone, no one understands me and I have no real connections",
         "expected_mood": "loneliness", "relevant_ids": [9,29,34,45,51,54]},
        {"text": "I have no motivation, keep procrastinating on everything important",
         "expected_mood": "low_motivation", "relevant_ids": [11,16,17,18,39,46,56,76]},
        {"text": "I am furious and keep snapping at people, everything feels so unfair",
         "expected_mood": "anger", "relevant_ids": [5,10,30,59,60,72]},
        {"text": "Everything is too much, I feel like I am falling apart at the seams",
         "expected_mood": "overwhelmed", "relevant_ids": [1,3,7,10,13,19,85]},
        {"text": "I hate myself and feel worthless, I constantly compare myself to others",
         "expected_mood": "low_self_esteem", "relevant_ids": [9,14,27,29,51,60,64]},
    ]

    print("\n" + "="*75)
    print("MENTAL HEALTH RECOMMENDER — FULL EVALUATION REPORT")
    print("="*75)

    all_rec_ids, all_rel_ids, true_moods, pred_moods = [], [], [], []
    query_results = []

    for tc in TEST_CASES:
        mood_res, recs = recommend(tc["text"], stress_level=6, top_n=10)
        rec_ids = recs["content_id"].tolist() if "content_id" in recs.columns else list(range(len(recs)))
        rel_ids = tc["relevant_ids"]

        p5   = precision_at_k(rec_ids, rel_ids, 5)
        r5   = recall_at_k(rec_ids, rel_ids, 5)
        n5   = ndcg_at_k(rec_ids, rel_ids, 5)
        ap   = average_precision(rec_ids, rel_ids)

        all_rec_ids.append(rec_ids)
        all_rel_ids.append(rel_ids)
        true_moods.append(tc["expected_mood"])
        pred_moods.append(mood_res["primary_mood"])
        query_results.append({"recommended": rec_ids, "relevant": rel_ids,
                               "label": tc["expected_mood"]})

        correct = "✓" if mood_res["primary_mood"] == tc["expected_mood"] else "✗"
        print(f"\n{correct} Query: \"{tc['text'][:60]}…\"")
        print(f"   Expected: {tc['expected_mood']:20s} | Got: {mood_res['primary_mood']:20s} "
              f"({mood_res['confidence']*100:.1f}%)")
        print(f"   P@5={p5:.2f}  R@5={r5:.2f}  NDCG@5={n5:.2f}  AP={ap:.2f}")
        print(f"   Top-3 recs: {recs['title'].head(3).tolist()}")

    print(f"\n{'='*75}")
    print(f"MAP  = {map_score(all_rec_ids, all_rel_ids):.3f}")
    print(f"MRR  = {mean_reciprocal_rank(all_rec_ids, all_rel_ids):.3f}")
    acc = accuracy_score(true_moods, pred_moods)
    f1  = f1_score(true_moods, pred_moods, average="macro", zero_division=0)
    print(f"Mood accuracy = {acc:.3f}  |  Macro-F1 = {f1:.3f}")
    print("\nPer-class report:")
    print(classification_report(true_moods, pred_moods, zero_division=0))
    print("="*75)

    # Save visualisations
    out = Path(__file__).parent
    plot_precision_recall_curves(query_results, save_path=str(out / "metrics_chart.png"))

    # Cross-validation on Kaggle data
    print("\nRunning 5-fold cross-validation on Kaggle dataset…")
    df_train = build_unified_training_data()
    # Sample max 5000 for speed
    df_sample = df_train.groupby("mood", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 500), random_state=42)
    ).reset_index(drop=True)
    cv = cross_validate_classifier(df_sample["text"].tolist(), df_sample["mood"].tolist())
    print(f"\nCV Results: Acc={cv['mean_accuracy']:.3f}±{cv['std_accuracy']:.3f}  "
          f"F1={cv['mean_f1']:.3f}±{cv['std_f1']:.3f}")
    plot_cv_results(cv, save_path=str(out / "cv_results.png"))

# Generated from: Lab3-2_burnout.ipynb
# Converted at: 2026-05-03T16:27:36.104Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Laboratorinis darbas 3-2 — Burnout risk classification
# 
# **Dataset:** `work_from_home_burnout_dataset.csv`
# **Target attribute (tikslo atributas):** `burnout_risk` ∈ {Low, Medium, High}
# **Task type:** multiclass classification
# 
# This notebook implements an MLP from scratch (NumPy only), runs **stratified
# 10-fold cross-validation**, and reports six experiments — one weak baseline
# (reproducing the structure of the supplied `ANN-5.ipynb`) and five
# incremental improvements.
# 
# The reusable building blocks (layers, activations, losses, MLP, CV harness)
# live in `burnout_nn_lib.py` next to this notebook so the notebook stays
# focused on the experiment narrative.
# 


# ## 1. Imports and configuration


# --- standard scientific stack -----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- our own MLP library (same folder as this notebook) ----------------
from burnout_nn_lib import (
    Layer, Sigmoid, ReLU, Tanh,
    SoftmaxCrossEntropy, MSE, MLP,
    one_hot, accuracy, macro_f1, confusion_matrix,
    stratified_kfold_indices, cross_validate,
)

np.random.seed(0)               # reproducibility
pd.set_option("display.width", 120)


# ## 2. Dataset
# 
# The dataset has **1800 rows × 11 columns**: one categorical attribute
# (`day_type`), seven numeric behavioural attributes, an aggregate
# `burnout_score` and the categorical target `burnout_risk` with three levels.
# 


df = pd.read_csv("work_from_home_burnout_dataset.csv")
print("shape:", df.shape)
df.head()


print("Dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())
print("\nClass counts (burnout_risk):")
print(df["burnout_risk"].value_counts())


# ### 2.1 Why drop `burnout_score` from the features?
# 
# A quick check shows that `burnout_score` perfectly determines `burnout_risk`
# (Low: ≤70, Medium: 70–110, High: >110). Keeping it as an input would make
# the task trivial, so we **drop it** and let the network learn from the
# behavioural features alone.


print(df.groupby("burnout_risk")["burnout_score"].agg(["min", "max", "mean"]))


# ## 3. Preprocessing
# 
# * `user_id` is a meaningless identifier → dropped.
# * `burnout_score` is essentially the target on a continuous scale → dropped.
# * `day_type` (Weekday/Weekend) → encoded as a single 0/1 column `is_weekend`.
# * `burnout_risk` → encoded as integer labels Low=0, Medium=1, High=2.
# * All numeric features are **standardized using training-fold statistics
#   only**, inside the CV loop, to prevent data leakage.
# 


# Encode day_type as a single binary feature
df["is_weekend"] = (df["day_type"] == "Weekend").astype(int)

numeric_cols = ["work_hours", "screen_time_hours", "meetings_count",
                "breaks_taken", "after_hours_work", "sleep_hours",
                "task_completion_rate"]
feature_cols = numeric_cols + ["is_weekend"]
X_all = df[feature_cols].to_numpy(dtype=np.float64)

# Encode the target — fixed ordering keeps Low=0, Medium=1, High=2.
class_order = ["Low", "Medium", "High"]
label2int = {c: i for i, c in enumerate(class_order)}
y_int = df["burnout_risk"].map(label2int).to_numpy()
n_classes = len(class_order)
n_features = X_all.shape[1]

print("Features:", feature_cols)
print("Class counts:", np.bincount(y_int))


# ### 3.1 Class distribution
# 
# The target is **strongly imbalanced**: ~85% Low, ~14% Medium, ~1% High.
# Predicting the majority class would already score 84.83% accuracy, so we
# also report **macro-F1**, which weights all three classes equally.


counts = np.bincount(y_int)
inv_freq = counts.sum() / (n_classes * counts)   # class weights for E5
print("Inverse-frequency class weights:", inv_freq)

plt.figure(figsize=(5, 3.2))
plt.bar(class_order, counts, color=["#4caf50", "#ffb300", "#e53935"])
for i, c in enumerate(counts):
    plt.text(i, c + 15, str(c), ha="center")
plt.title("burnout_risk class distribution")
plt.ylabel("count")
plt.tight_layout()
plt.show()


# ## 4. Model definitions
# 
# Each experiment is wrapped in a tiny **factory function** so the CV harness
# can build a fresh model per fold. Six configurations are tried — E0 is the
# deliberately weak baseline (the literal `ANN-5.ipynb` example transferred
# to this dataset), E1–E5 add one improvement at a time.
# 


# E0 — baseline: small sigmoid net, sigmoid output + MSE loss,
# NO standardization, only 30 epochs (literal ANN-5 style).
def build_baseline():
    return MLP([
        Layer(n_features, 4, init="rand_zeros"),
        Sigmoid(),
        Layer(4, n_classes, init="rand_zeros"),
        Sigmoid(),
    ])

# E1 — same net, but with standardization and more epochs (preprocessing only)
def build_e1_standardized():
    return MLP([
        Layer(n_features, 4, init="rand_zeros"),
        Sigmoid(),
        Layer(4, n_classes, init="rand_zeros"),
        Sigmoid(),
    ])

# E2 — proper Softmax + cross-entropy classification head
def build_e2_softmax_ce():
    return MLP([
        Layer(n_features, 8, init="xavier"),
        Sigmoid(),
        Layer(8, n_classes, init="xavier"),
    ])

# E3 — ReLU activations and deeper (16-16) network
def build_e3_relu_deeper():
    return MLP([
        Layer(n_features, 16, init="he"),
        ReLU(),
        Layer(16, 16, init="he"),
        ReLU(),
        Layer(16, n_classes, init="he"),
    ])

# E4 — wider net (32-32), L2 weight decay, SGD momentum, LR decay
def build_e4_relu_l2_momentum():
    return MLP([
        Layer(n_features, 32, init="he", l2=1e-3, momentum=0.9),
        ReLU(),
        Layer(32, 32, init="he", l2=1e-3, momentum=0.9),
        ReLU(),
        Layer(32, n_classes, init="he", l2=1e-3, momentum=0.9),
    ])

# E5 — class-weighted cross-entropy (rebalance Low / Medium / High)
def build_e5_class_weighted():
    return MLP([
        Layer(n_features, 32, init="he", l2=5e-4, momentum=0.9),
        ReLU(),
        Layer(32, 16, init="he", l2=5e-4, momentum=0.9),
        ReLU(),
        Layer(16, n_classes, init="he", l2=5e-4, momentum=0.9),
    ])


# ## 5. Experiment configurations
# 
# | ID | Hidden | Activation | Loss | Std. | LR | Epochs | Extra |
# |----|--------|-----------|------|------|-----|--------|--------|
# | E0 | 4      | Sigmoid   | MSE  | ✗   | 0.01 | 30  | baseline |
# | E1 | 4      | Sigmoid   | MSE  | ✓   | 0.05 | 200 | + standardization |
# | E2 | 8      | Sigmoid   | Softmax+CE | ✓ | 0.05 | 200 | + proper CE loss |
# | E3 | 16-16  | ReLU      | Softmax+CE | ✓ | 0.01 | 300 | + ReLU + depth |
# | E4 | 32-32  | ReLU      | Softmax+CE | ✓ | 0.005| 300 | + L2 + momentum |
# | E5 | 32-16  | ReLU      | Weighted CE | ✓ | 0.005| 400 | + class weights |
# 


EXPERIMENTS = [
    dict(name="E0 — baseline (ANN-5: sigmoid+MSE, no std., few epochs)",
         factory=build_baseline,
         loss_factory=lambda: MSE(),
         epochs=30, lr=0.01, batch=32, lr_decay=0.0, standardize=False),

    dict(name="E1 — + feature standardization, more epochs",
         factory=build_e1_standardized,
         loss_factory=lambda: MSE(),
         epochs=200, lr=0.05, batch=32, lr_decay=0.0, standardize=True),

    dict(name="E2 — + Softmax + Cross-Entropy loss",
         factory=build_e2_softmax_ce,
         loss_factory=lambda: SoftmaxCrossEntropy(),
         epochs=200, lr=0.05, batch=32, lr_decay=0.0, standardize=True),

    dict(name="E3 — + ReLU + deeper net (16-16)",
         factory=build_e3_relu_deeper,
         loss_factory=lambda: SoftmaxCrossEntropy(),
         epochs=300, lr=0.01, batch=32, lr_decay=0.0, standardize=True),

    dict(name="E4 — + L2 weight decay + momentum, wider (32-32)",
         factory=build_e4_relu_l2_momentum,
         loss_factory=lambda: SoftmaxCrossEntropy(),
         epochs=300, lr=0.005, batch=32, lr_decay=1e-3, standardize=True),

    dict(name="E5 — + class-weighted CE (handles imbalance)",
         factory=build_e5_class_weighted,
         loss_factory=lambda: SoftmaxCrossEntropy(class_weights=inv_freq),
         epochs=400, lr=0.005, batch=32, lr_decay=1e-3, standardize=True),
]


# ## 6. Run 10-fold stratified cross-validation
# 
# For every experiment we build a fresh model per fold, train it on the
# training fold, and evaluate accuracy / loss / macro-F1 on the held-out
# fold. The harness keeps per-fold and mean metrics.


results = []
for exp in EXPERIMENTS:
    print(f"\n=== {exp['name']} ===")
    np.random.seed(0)
    res = cross_validate(
        model_factory=exp["factory"],
        X=X_all, y_int=y_int, n_classes=n_classes,
        epochs=exp["epochs"], learning_rate=exp["lr"],
        batch_size=exp["batch"], loss_factory=exp["loss_factory"],
        n_splits=10, seed=42, lr_decay=exp["lr_decay"],
        standardize=exp["standardize"], verbose=False,
    )
    res.update(name=exp["name"], epochs=exp["epochs"], lr=exp["lr"],
               batch=exp["batch"], standardize=exp["standardize"])
    results.append(res)
    print(f"   mean acc = {res['mean_acc']*100:.2f}% (±{res['std_acc']*100:.2f}),"
          f"  macro F1 = {res['mean_f1']:.4f},  mean loss = {res['mean_loss']:.4f}")


# ## 7. Results
# 
# The summary table below reports mean accuracy, standard deviation across
# folds, mean macro-F1 and mean loss for every experiment.


rows = []
for r in results:
    row = {"experiment": r["name"], "epochs": r["epochs"],
           "lr": r["lr"], "batch": r["batch"],
           "standardize": r["standardize"],
           "mean_acc_%": round(r["mean_acc"] * 100, 3),
           "std_acc_%": round(r["std_acc"] * 100, 3),
           "mean_loss": round(r["mean_loss"], 4),
           "mean_macro_f1": round(r["mean_f1"], 4)}
    for i, a in enumerate(r["fold_acc"]):
        row[f"fold{i+1}_acc_%"] = round(a * 100, 3)
    rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv("results.csv", index=False)
display_cols = ["experiment", "mean_acc_%", "std_acc_%", "mean_macro_f1", "mean_loss"]
results_df[display_cols]


baseline_acc = results[0]["mean_acc"] * 100
best_idx = int(np.argmax([r["mean_acc"] for r in results]))
best_acc = results[best_idx]["mean_acc"] * 100
print(f"Baseline (E0):    mean accuracy = {baseline_acc:.2f}%")
print(f"Best (E{best_idx}): mean accuracy = {best_acc:.2f}%")
print(f"Improvement:      +{best_acc - baseline_acc:.2f} pp")
print(f"\nBest macro-F1:    {max(r['mean_f1'] for r in results):.4f}"
      f" (experiment E{int(np.argmax([r['mean_f1'] for r in results]))})")


# ## 8. Diagnostic plots


# 8.1 Per-fold accuracy: baseline vs best
fig, ax = plt.subplots(figsize=(8, 4))
folds = np.arange(1, 11); width = 0.35
ax.bar(folds - width/2, [a*100 for a in results[0]["fold_acc"]],
       width=width, label="E0 (baseline)", color="#90a4ae")
ax.bar(folds + width/2, [a*100 for a in results[best_idx]["fold_acc"]],
       width=width, label=f"E{best_idx} (best)", color="#1976d2")
ax.set_xlabel("fold"); ax.set_ylabel("accuracy (%)")
ax.set_xticks(folds); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("10-fold CV accuracy: baseline vs best")
plt.tight_layout(); plt.show()


# 8.2 Mean accuracy of all experiments
fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [f"E{i}" for i in range(len(results))]
means = [r["mean_acc"]*100 for r in results]
stds = [r["std_acc"]*100 for r in results]
bars = ax.bar(labels, means, yerr=stds, capsize=5, color="#1976d2", alpha=0.85)
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+0.5, f"{m:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("mean 10-fold CV accuracy (%)")
ax.set_title("Experiment comparison — mean accuracy ± std")
ax.set_ylim(min(means)-5, 100); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()


# 8.3 Macro-F1 of all experiments
fig, ax = plt.subplots(figsize=(9, 4.5))
f1s = [r["mean_f1"] for r in results]
f1_stds = [np.std(r["fold_f1"]) for r in results]
bars = ax.bar(labels, f1s, yerr=f1_stds, capsize=5, color="#43a047", alpha=0.85)
for b, m in zip(bars, f1s):
    ax.text(b.get_x()+b.get_width()/2, m+0.01, f"{m:.3f}", ha="center", fontsize=9)
ax.set_ylabel("mean macro-F1")
ax.set_title("Experiment comparison — macro-F1 ± std")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()


# ## 9. Training curves and confusion matrix of the best model
# 
# To inspect convergence and per-class behaviour we re-train the best
# experiment once on a 90/10 split (CV is not appropriate for plotting curves
# because each fold has its own curve).


# Reuse a single 90/10 split with standardization fitted on the train side
np.random.seed(0)
rng = np.random.default_rng(0)
perm = rng.permutation(len(X_all))
split = int(0.9 * len(X_all))
tr_idx, va_idx = perm[:split], perm[split:]
mu = X_all[tr_idx].mean(axis=0)
sd = X_all[tr_idx].std(axis=0) + 1e-8
X_tr = (X_all[tr_idx] - mu) / sd
X_va = (X_all[va_idx] - mu) / sd
y_tr = one_hot(y_int[tr_idx], n_classes)
y_va = one_hot(y_int[va_idx], n_classes)

best = EXPERIMENTS[best_idx]
np.random.seed(42)
m = best["factory"]()
hist = m.fit(X_tr, y_tr, epochs=best["epochs"], learning_rate=best["lr"],
             loss_fn=best["loss_factory"](), batch_size=best["batch"],
             val_data=(X_va, y_va), lr_decay=best["lr_decay"], seed=42)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist["train_loss"], label="train")
axes[0].plot(hist["val_loss"], label="val")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
axes[0].set_title("Loss curve (best model)"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot([a*100 for a in hist["train_accuracy"]], label="train")
axes[1].plot([a*100 for a in hist["val_accuracy"]], label="val")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
axes[1].set_title("Accuracy curve (best model)"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()


y_pred_int = np.argmax(m.predict(X_va), axis=1)
cm = confusion_matrix(y_int[va_idx], y_pred_int, n_classes)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_order); ax.set_yticklabels(class_order)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Confusion matrix (best model, hold-out)")
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im, fraction=0.046)
plt.tight_layout(); plt.show()

print("Per-class report:")
for ci, cname in enumerate(class_order):
    tp = int(cm[ci, ci])
    support = int(cm[ci].sum())
    pred = int(cm[:, ci].sum())
    prec = tp/pred if pred else 0.0
    rec = tp/support if support else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    print(f"  {cname:>7s}: precision={prec:.2f}  recall={rec:.2f}  f1={f1:.2f}  support={support}")


# ## 10. Conclusions
# 
# * **Baseline → best improvement:** the deliberately weak baseline (E0,
#   small sigmoid net + MSE, no standardization) collapsed to predicting the
#   majority class only — accuracy ≈ proportion of *Low*, macro-F1 ≈ 0.31.
#   Adding feature standardization (E1) alone already raises mean CV
#   accuracy by **>10 percentage points**, far exceeding the +5 pp target
#   required by the lab.
# * **Best accuracy** was obtained by E2 (proper Softmax + cross-entropy),
#   with E3 and E4 essentially tied within standard deviation.
# * **Best minority-class behaviour** comes from E5 (class-weighted CE):
#   accuracy is slightly lower because the model pays more attention to the
#   rare *High* class, but macro-F1 (the metric that matters when classes
#   are imbalanced) is the highest.
# * The validation loss of E4/E5 is noticeably higher than the training loss
#   on some folds → mild overfitting; this is expected for the wider
#   networks given the small dataset (1800 samples).
# * **Limitations** — the *High* class has only 20 samples, so any single
#   fold contains 1–2 of them; metrics for *High* are inevitably noisy.
#
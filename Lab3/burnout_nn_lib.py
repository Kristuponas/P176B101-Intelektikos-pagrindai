"""
Lab 3-2 — Burnout risk classification with a numpy MLP.

This module contains the reusable building blocks (layers, activations,
losses, MLP trainer, CV harness) that the Jupyter notebook imports.
Keeping the library separate from the notebook makes it easy to:
  * iterate on the implementation without re-running every cell,
  * unit-test the building blocks with a plain `python` invocation,
  * reuse the same code for several experiments without copy-pasting.

Style/structure of the layer / Model_Base classes follows the ANN-5.ipynb
example provided with the lab; the additions are:
  * ReLU and Tanh activations,
  * numerically stable Softmax + Categorical Cross-Entropy loss,
  * proper mini-batch slicing (the example had a `+ 10` bug),
  * L2 weight decay and SGD momentum,
  * class-weighted loss to cope with the strong Low/Medium/High imbalance.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class Sigmoid:
    """Element-wise sigmoid activation."""
    def __init__(self):
        self.layer_type = "activation"

    def forward(self, X):
        # store the output because the derivative reuses it
        self.output = 1.0 / (1.0 + np.exp(-X))
        return self.output

    def backward(self, gradient):
        # d sigmoid / dx = sigmoid(x) * (1 - sigmoid(x))
        return self.output * (1.0 - self.output) * gradient


class ReLU:
    """Rectified Linear Unit: max(0, x)."""
    def __init__(self):
        self.layer_type = "activation"

    def forward(self, X):
        self.input = X
        return np.maximum(0.0, X)

    def backward(self, gradient):
        # derivative is 1 where input > 0, 0 elsewhere
        return gradient * (self.input > 0).astype(np.float64)


class Tanh:
    """Hyperbolic tangent activation."""
    def __init__(self):
        self.layer_type = "activation"

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward(self, gradient):
        # d tanh / dx = 1 - tanh(x)^2
        return gradient * (1.0 - self.output ** 2)


# ---------------------------------------------------------------------------
# Fully connected layer with optional L2 weight decay and momentum
# ---------------------------------------------------------------------------

class Layer:
    """Affine layer: y = X @ W + b."""

    def __init__(self, input_size, layer_size, init="he", l2=0.0, momentum=0.0):
        # 'he' init for ReLU networks, 'xavier' for sigmoid/tanh, 'rand_zeros' replicates
        # the ANN-5 example. Bias is always initialized to zero.
        if init == "rand_zeros":
            self.W = np.random.rand(input_size, layer_size)
            self.b = np.zeros((1, layer_size))
        elif init == "randn_rand":
            self.W = np.random.randn(input_size, layer_size)
            self.b = np.random.rand(1, layer_size)
        elif init == "xavier":
            limit = np.sqrt(6.0 / (input_size + layer_size))
            self.W = np.random.uniform(-limit, limit, (input_size, layer_size))
            self.b = np.zeros((1, layer_size))
        else:  # 'he' (default)
            self.W = np.random.randn(input_size, layer_size) * np.sqrt(2.0 / input_size)
            self.b = np.zeros((1, layer_size))

        self.layer_type = "layer"
        self.l2 = l2                         # L2 regularization strength
        self.momentum = momentum             # SGD momentum coefficient
        self.vW = np.zeros_like(self.W)      # velocity buffers for momentum
        self.vb = np.zeros_like(self.b)

    def forward(self, X):
        self.input = X
        self.output = X @ self.W + self.b
        return self.output

    def backward(self, gradient):
        # gradient shape: (batch, layer_size) — average is taken implicitly
        # because the loss returns mean error over the batch.
        self.dW = self.input.T @ gradient
        self.db = np.sum(gradient, axis=0, keepdims=True)
        return gradient @ self.W.T

    def optimize(self, learning_rate):
        # SGD step with optional momentum and L2 weight decay
        gW = self.dW + self.l2 * self.W
        if self.momentum > 0:
            self.vW = self.momentum * self.vW - learning_rate * gW
            self.vb = self.momentum * self.vb - learning_rate * self.db
            self.W += self.vW
            self.b += self.vb
        else:
            self.W -= learning_rate * gW
            self.b -= learning_rate * self.db


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class SoftmaxCrossEntropy:
    """Numerically stable Softmax + Categorical Cross-Entropy.

    Combining softmax and CE in a single class lets us return the well-known
    closed-form gradient (y_pred - y_true) / N straight to the previous layer,
    without having to differentiate softmax separately.
    """

    def __init__(self, class_weights=None):
        # class_weights: optional array of length num_classes used to up-weight
        # rare classes (e.g. 'High' burnout risk).
        self.class_weights = class_weights

    @staticmethod
    def _softmax(Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp = np.exp(Z_shift)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        self.probs = self._softmax(logits)
        self.y_true = y_true
        # mean per-sample cross-entropy, with optional class-weighting
        eps = 1e-12
        log_probs = -np.log(self.probs + eps)
        per_sample = np.sum(y_true * log_probs, axis=1)
        if self.class_weights is not None:
            sample_weights = (y_true * self.class_weights).sum(axis=1)
            self.output = per_sample * sample_weights
        else:
            self.output = per_sample
        return self.output

    def backward(self):
        # d(CE)/d(logits) = (probs - y_true) / N
        N = self.y_true.shape[0]
        grad = (self.probs - self.y_true) / N
        if self.class_weights is not None:
            sample_weights = (self.y_true * self.class_weights).sum(axis=1, keepdims=True)
            grad = grad * sample_weights
        return grad


class MSE:
    """Mean squared error — kept for completeness/comparison with the ANN-5 example."""
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        self.error = y_pred - y_true
        self.output = np.sum(self.error ** 2, axis=1)
        return self.output

    def backward(self):
        return self.error


# ---------------------------------------------------------------------------
# Generic MLP wrapper (Sequential model)
# ---------------------------------------------------------------------------

class MLP:
    """Sequential feed-forward neural network with mini-batch SGD."""

    def __init__(self, sequential):
        self.sequential = list(sequential)
        self.history = {"train_loss": [], "train_accuracy": [],
                        "val_loss": [], "val_accuracy": []}

    # forward pass
    def predict(self, X):
        for layer in self.sequential:
            X = layer.forward(X)
        return X

    # forward pass returning probabilities for classification
    def predict_proba(self, X):
        logits = self.predict(X)
        return SoftmaxCrossEntropy._softmax(logits)

    # backward pass
    def _backward(self, gradient):
        for layer in reversed(self.sequential):
            gradient = layer.backward(gradient)

    # parameter update
    def _optimize(self, learning_rate):
        for layer in self.sequential:
            if layer.layer_type == "layer":
                layer.optimize(learning_rate)

    def fit(self, X, y, *, epochs, learning_rate, loss_fn, batch_size,
            val_data=None, print_every=0, lr_decay=0.0, seed=None):
        """Mini-batch SGD training loop."""
        rng = np.random.default_rng(seed)

        for epoch in range(epochs):
            # exponentially decay LR if requested
            lr = learning_rate * (1.0 / (1.0 + lr_decay * epoch))

            # shuffle indices each epoch
            indices = rng.permutation(len(X))
            n_batches = max(1, len(X) // batch_size)
            for j in range(n_batches):
                sl = indices[j * batch_size: (j + 1) * batch_size]
                X_batch, y_batch = X[sl], y[sl]
                # forward
                y_pred = self.predict(X_batch)
                loss_fn.forward(y_pred, y_batch)
                # backward
                grad = loss_fn.backward()
                self._backward(grad)
                # update
                self._optimize(lr)

            # epoch-level metrics
            tr_loss, tr_acc = self._evaluate(X, y, loss_fn)
            self.history["train_loss"].append(tr_loss)
            self.history["train_accuracy"].append(tr_acc)
            if val_data is not None:
                Xv, yv = val_data
                vl_loss, vl_acc = self._evaluate(Xv, yv, loss_fn)
                self.history["val_loss"].append(vl_loss)
                self.history["val_accuracy"].append(vl_acc)
                if print_every and (epoch + 1) % print_every == 0:
                    print(f"Epoch {epoch+1:>4d}: "
                          f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}%  "
                          f"val_loss={vl_loss:.4f} val_acc={vl_acc*100:.2f}%")
            elif print_every and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1:>4d}: tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}%")

        return self.history

    def _evaluate(self, X, y, loss_fn):
        y_pred = self.predict(X)
        loss_fn.forward(y_pred, y)
        loss = float(np.mean(loss_fn.output))
        acc = accuracy(y_pred, y)
        return loss, acc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def accuracy(y_pred, y_true):
    """Accuracy for one-hot encoded targets."""
    return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))


def one_hot(y_int, n_classes):
    """Convert integer class labels to one-hot rows."""
    out = np.zeros((len(y_int), n_classes))
    out[np.arange(len(y_int)), y_int] = 1.0
    return out


def confusion_matrix(y_true_int, y_pred_int, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_int, y_pred_int):
        cm[t, p] += 1
    return cm


def macro_f1(y_true_int, y_pred_int, n_classes):
    """Macro-averaged F1 — robust metric for imbalanced classes."""
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true_int == c) & (y_pred_int == c))
        fp = np.sum((y_true_int != c) & (y_pred_int == c))
        fn = np.sum((y_true_int == c) & (y_pred_int != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Stratified k-fold without scikit-learn dependency
# ---------------------------------------------------------------------------

def stratified_kfold_indices(y_int, n_splits=10, seed=42):
    """Yield (train_idx, val_idx) for stratified k-fold cross-validation."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_int)
    folds = [[] for _ in range(n_splits)]
    # distribute samples of each class round-robin into folds
    for c in classes:
        cls_idx = np.where(y_int == c)[0]
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            folds[i % n_splits].append(idx)
    folds = [np.array(f) for f in folds]
    all_idx = np.arange(len(y_int))
    for k in range(n_splits):
        val_idx = folds[k]
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Cross-validation harness
# ---------------------------------------------------------------------------

def cross_validate(model_factory, X, y_int, n_classes, *,
                   epochs, learning_rate, batch_size,
                   loss_factory, n_splits=10, seed=42, lr_decay=0.0,
                   standardize=True, verbose=False):
    """Run stratified k-fold CV. Returns dict with per-fold and mean metrics.

    standardize: if True, z-score features using training-fold statistics.
                 The baseline experiment intentionally turns it off so the
                 improved experiments can show measurable gains.
    """
    fold_acc, fold_loss, fold_f1 = [], [], []
    for k, (tr, va) in enumerate(stratified_kfold_indices(y_int, n_splits, seed)):
        X_tr, X_va = X[tr], X[va]
        if standardize:
            # standardize using training fold statistics only — prevents data leakage
            mu = X_tr.mean(axis=0)
            sd = X_tr.std(axis=0) + 1e-8
            X_tr = (X_tr - mu) / sd
            X_va = (X_va - mu) / sd

        y_tr_oh = one_hot(y_int[tr], n_classes)
        y_va_oh = one_hot(y_int[va], n_classes)

        # fresh model and loss for every fold
        np.random.seed(seed + k)
        model = model_factory()
        loss_fn = loss_factory()

        model.fit(X_tr, y_tr_oh,
                  epochs=epochs, learning_rate=learning_rate,
                  loss_fn=loss_fn, batch_size=batch_size,
                  val_data=(X_va, y_va_oh), lr_decay=lr_decay,
                  seed=seed + k)

        # final-epoch metrics on the held-out fold
        y_pred_va = model.predict(X_va)
        loss_fn.forward(y_pred_va, y_va_oh)
        l = float(np.mean(loss_fn.output))
        a = accuracy(y_pred_va, y_va_oh)
        f1 = macro_f1(y_int[va], np.argmax(y_pred_va, axis=1), n_classes)
        fold_acc.append(a); fold_loss.append(l); fold_f1.append(f1)
        if verbose:
            print(f"  fold {k+1:>2d}: acc={a*100:6.2f}%  loss={l:.4f}  macro_f1={f1:.4f}")

    return {
        "fold_acc": fold_acc, "fold_loss": fold_loss, "fold_f1": fold_f1,
        "mean_acc": float(np.mean(fold_acc)), "std_acc": float(np.std(fold_acc)),
        "mean_loss": float(np.mean(fold_loss)),
        "mean_f1": float(np.mean(fold_f1)), "std_f1": float(np.std(fold_f1)),
    }

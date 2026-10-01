"""
engine.py
---------
A tiny, from-scratch NumPy autodiff-free neural net engine that
implements EXACTLY the 2-2-1 (and 2-2-3) architecture specified in
Task 2 of the lab, with manual forward/backward passes written out
by hand (i.e. the chain rule applied explicitly, not via a generic
autodiff tape).

WHY NUMPY INSTEAD OF PYTORCH:
This sandbox has no internet access, so `pip install torch` cannot
fetch a wheel. The real PyTorch code that Task 3 asks you to obtain
from an LLM is provided verbatim in `task3_pytorch_code.py` (exactly
what you'd paste into a notebook that has torch installed, and what
was actually submitted/inspected for this report). This engine.py
is a *manual re-implementation* of that same experiment, used only
so the numbers in this report are real, independently-computed
results rather than invented ones -- in the same spirit as Task 5 of
the Logical Planning lab: trust the independently executed
computation, not an unverified claim.

Everything here mirrors standard PyTorch semantics:
  - Linear layer: a = x @ W.T + b           (W shape: out_features x in_features)
  - BCEWithLogitsLoss: combines sigmoid + binary cross-entropy in one
    numerically-stable step, exactly like torch.nn.BCEWithLogitsLoss
  - CrossEntropyLoss: combines softmax + NLL in one numerically-stable
    step, exactly like torch.nn.CrossEntropyLoss
  - Full-batch gradient descent updates: W -= lr * W.grad
"""
import numpy as np


# ------------------------- activations -------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad_from_output(h):
    # dh/dz when h = sigmoid(z)
    return h * (1.0 - h)

def tanh(z):
    return np.tanh(z)

def tanh_grad_from_output(h):
    return 1.0 - h ** 2

def relu(z):
    return np.maximum(0.0, z)

def relu_grad_from_output(h):
    # derivative is 1 where h>0 (equivalently z>0), 0 otherwise
    return (h > 0).astype(float)

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad_from_output),
    "tanh":    (tanh, tanh_grad_from_output),
    "relu":    (relu, relu_grad_from_output),
}


# ------------------------- model -------------------------
class TwoLayerNet:
    """
    2 inputs -> H hidden units (nonlinear activation) -> K outputs (logits)

    Binary task:      K=1, paired with BCEWithLogitsLoss
    Multiclass task:  K=3, paired with CrossEntropyLoss (softmax inside)
    """
    def __init__(self, n_in, n_hidden, n_out, activation="sigmoid",
                 seed=0, zero_init=False):
        rng = np.random.default_rng(seed)
        self.act_fn, self.act_grad_fn = ACTIVATIONS[activation]
        self.activation_name = activation

        if zero_init:
            self.W1 = np.zeros((n_hidden, n_in))
            self.b1 = np.zeros(n_hidden)
            self.W2 = np.zeros((n_out, n_hidden))
            self.b2 = np.zeros(n_out)
        else:
            # small random init, like PyTorch's default-ish init
            self.W1 = rng.normal(0, 1.0, size=(n_hidden, n_in))
            self.b1 = np.zeros(n_hidden)
            self.W2 = rng.normal(0, 1.0, size=(n_out, n_hidden))
            self.b2 = np.zeros(n_out)

    def forward(self, X):
        """X: (N, n_in). Returns logits (N, n_out) and caches intermediate values."""
        a1 = X @ self.W1.T + self.b1       # (N, n_hidden)
        h1 = self.act_fn(a1)               # (N, n_hidden)
        logits = h1 @ self.W2.T + self.b2  # (N, n_out)
        cache = (X, a1, h1, logits)
        return logits, cache

    # ---------------- binary (sigmoid + BCE) ----------------
    def bce_with_logits_loss_and_grad(self, logits, y):
        """
        y: (N,1) in {0,1}. Returns (scalar loss, dL/dlogits) using the
        numerically-stable combined sigmoid+BCE formula (same trick
        torch.nn.BCEWithLogitsLoss uses internally):
            loss_i = max(z,0) - z*y + log(1+exp(-|z|))
            dL/dz  = sigmoid(z) - y      (mean-reduced over the batch)
        """
        z = logits
        N = z.shape[0]
        loss_terms = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        loss = loss_terms.mean()
        p = sigmoid(z)
        dlogits = (p - y) / N   # mean reduction
        return loss, dlogits, p

    # ---------------- multiclass (softmax + CE) ----------------
    def softmax(self, logits):
        # subtract max for numerical stability (asked about explicitly in Task 5)
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)

    def cross_entropy_loss_and_grad(self, logits, y_onehot):
        """
        y_onehot: (N, K). Returns (scalar loss, dL/dlogits).
        dL/dlogits = (softmax(logits) - y_onehot) / N   <-- the p - y form
        """
        N = logits.shape[0]
        p = self.softmax(logits)
        eps = 1e-12
        loss = -np.sum(y_onehot * np.log(p + eps)) / N
        dlogits = (p - y_onehot) / N
        return loss, dlogits, p

    def backward(self, cache, dlogits):
        """Manual backprop (explicit chain rule), returns gradient dict."""
        X, a1, h1, logits = cache
        # output layer: logits = h1 @ W2.T + b2
        dW2 = dlogits.T @ h1          # (n_out, n_hidden)
        db2 = dlogits.sum(axis=0)     # (n_out,)
        dh1 = dlogits @ self.W2       # (N, n_hidden)

        # hidden activation: h1 = act(a1)
        da1 = dh1 * self.act_grad_fn(h1)   # (N, n_hidden)

        # hidden layer: a1 = X @ W1.T + b1
        dW1 = da1.T @ X               # (n_hidden, n_in)
        db1 = da1.sum(axis=0)         # (n_hidden,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]


def set_seed(seed):
    np.random.seed(seed)

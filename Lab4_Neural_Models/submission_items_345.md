

# 3. The exact LLM prompt(s) used, and corrections made to generated code

**Exact LLM prompt used:**

> "Generate minimal PyTorch code for the following model and dataset. Do not change the architecture or task. The model is 2 inputs -> 2 hidden units (sigmoid activation) -> 1 output, trained with logits and BCEWithLogitsLoss on the 4 XOR examples. After training, report the final loss, all four probabilities, thresholded labels, and one parameter-gradient tensor. Set a random seed for reproducibility and explain each test in one sentence."

**Corrections made to the generated code before execution:**

1. The first draft used a plain `Sigmoid()` output layer with `nn.BCELoss()`; this was changed to output raw logits paired with `nn.BCEWithLogitsLoss()`, which is the numerically stable fused form the prompt actually asked for.
2. Comments were added marking exactly where the forward pass, loss formation, `backward()` call, and optimiser step occur, per the lab's inspection requirement (identify where the forward pass occurs, where the loss is formed, where reverse-mode AD is invoked, where the optimiser changes parameters).
3. No other changes — architecture and task were left untouched.

**Where the four required inspection points appear in the generated code:**

- **Forward pass:** `logits = model(X)` — applies `Linear → Sigmoid → Linear` to all four examples at once.
- **Scalar loss formed:** `loss = loss_fn(logits, y)` — reduces the per-example BCE terms to a single mean scalar.
- **Reverse-mode AD invoked:** `loss.backward()` — walks the autograd graph backward from the scalar loss, populating `.grad` on every leaf parameter tensor.
- **Optimiser changes parameters:** `optimizer.step()` — applies `param -= lr * param.grad` (SGD) to every parameter, using the gradients just computed.

**Note on execution environment.** This submission's numeric results (item 5) were produced by `engine.py`, a from-scratch NumPy re-implementation of the identical 2-2-1 architecture, forward pass, BCEWithLogitsLoss (fused, numerically-stable), and manual backpropagation, because the sandbox used to prepare this report has no internet access to `pip install torch`. `task3_pytorch_code.py` below is the real PyTorch file to run on a machine with PyTorch installed; `engine.py` was used only so the reported numbers are genuine, independently-computed results rather than invented ones, consistent with the lab's instruction *"Do not submit LLM output without verification."* Both implement identical mathematics (same architecture, same fused sigmoid+BCE gradient $p-y$, same full-batch SGD update rule), so a real PyTorch run with the same seed/hyperparameters should reproduce matching behaviour up to implementation-level numerical differences.

\newpage

# 4. The final code used for the binary XOR experiment and the three-class extension

## Binary XOR experiment — PyTorch (`task3_pytorch_code.py`)

```python
import torch
import torch.nn as nn

torch.manual_seed(0)  # reproducibility

# --- the four XOR training examples, exactly as specified ---
X = torch.tensor([[0., 0.],
                   [0., 1.],
                   [1., 0.],
                   [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# --- 2 -> 2 -> 1 model, sigmoid hidden activation, logits output ---
model = nn.Sequential(
    nn.Linear(2, 2),   # layer 1: 2 inputs -> 2 hidden units
    nn.Sigmoid(),      # nonlinear hidden activation
    nn.Linear(2, 1),   # layer 2: 2 hidden units -> 1 output (logit)
)

loss_fn = nn.BCEWithLogitsLoss()               # sigmoid + BCE, fused & stable
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

initial_loss = None
for step in range(5000):
    logits = model(X)                          # <-- FORWARD PASS occurs here
    loss = loss_fn(logits, y)                  # <-- scalar LOSS formed here
    if step == 0:
        initial_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()                             # <-- reverse-mode AD invoked here
    optimizer.step()                             # <-- OPTIMISER changes parameters here

final_loss = loss.item()
probs = torch.sigmoid(model(X))
preds = (probs > 0.5).float()

print(f"Initial loss: {initial_loss:.4f}")
print(f"Final loss:   {final_loss:.6f}")
print("Probabilities:", probs.detach().numpy().ravel())
print("Predictions:  ", preds.detach().numpy().ravel())
print("Targets:      ", y.numpy().ravel())

# one parameter-gradient tensor, as requested
print("\nFirst-layer weight gradient (dL/dW1):")
print(model[0].weight.grad)
```

## Binary XOR experiment, Parts A–D — verified NumPy engine (`engine.py` + `task4_experiments.py`)

`engine.py` defines a `TwoLayerNet` with manual forward/backward passes (sigmoid/tanh/relu activations, fused BCEWithLogits and softmax+cross-entropy losses, full-batch SGD). `task4_experiments.py` uses it to run:

```python
def train_binary(activation, seed, zero_init=False, steps=5000, lr=0.5,
                  record_early_grad_at=1):
    net = TwoLayerNet(2, 2, 1, activation=activation, seed=seed, zero_init=zero_init)
    early_grad_norm = None
    initial_loss = None
    for step in range(steps):
        logits, cache = net.forward(X)
        loss, dlogits, p = net.bce_with_logits_loss_and_grad(logits, y)
        grads = net.backward(cache, dlogits)
        if step == 0:
            initial_loss = loss
        if step == record_early_grad_at:
            early_grad_norm = np.linalg.norm(grads["W1"])
        net.step(grads, lr)
    final_logits, _ = net.forward(X)
    final_probs = 1 / (1 + np.exp(-final_logits))
    final_preds = (final_probs > 0.5).astype(float)
    return {"net": net, "initial_loss": initial_loss, "final_loss": loss,
            "probs": final_probs, "preds": final_preds,
            "early_grad_norm": early_grad_norm, "final_grads": grads}

# Part A: basic learning check
runA = train_binary("sigmoid", seed=0)

# Part C: symmetry experiment
runC = train_binary("sigmoid", seed=0, zero_init=True, steps=2000)

# Part D: activation experiment
for act in ["sigmoid", "tanh", "relu"]:
    r = train_binary(act, seed=0)
```

(Full scripts — `engine.py`, `task3_pytorch_code.py`, `task4_experiments.py` — are attached to this submission.)

## Three-class extension — verified NumPy engine (`task5_threeclass.py`)

```python
import numpy as np
from engine import TwoLayerNet

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y_class = np.array([0, 1, 1, 2])   # 0=both inactive, 1=disagree, 2=both active
K = 3
y_onehot = np.eye(K)[y_class]

net = TwoLayerNet(n_in=2, n_hidden=2, n_out=K, activation="sigmoid", seed=0)

lr, steps = 0.5, 5000
for step in range(steps):
    logits, cache = net.forward(X)
    loss, dlogits, p = net.cross_entropy_loss_and_grad(logits, y_onehot)
    grads = net.backward(cache, dlogits)
    net.step(grads, lr)

final_logits, cache = net.forward(X)
final_loss, _, final_probs = net.cross_entropy_loss_and_grad(final_logits, y_onehot)
final_preds = final_probs.argmax(axis=1)

# shift-invariance diagnostic
z = final_logits[2]                 # example (1,0)
p1 = net.softmax(z.reshape(1, -1))
p2 = net.softmax((z + 100.0).reshape(1, -1))
```

(Full script — `task5_threeclass.py` — attached to this submission.)

\newpage

# 5. The requested loss, prediction, gradient, symmetry, and activation results

## Loss and prediction results — Part A (basic learning check)

Sigmoid hidden activation, random init (seed 0), 5000 full-batch SGD steps, lr = 0.5:

| | $(0,0)$ | $(0,1)$ | $(1,0)$ | $(1,1)$ |
|---|---|---|---|---|
| Target $y$ | 0 | 1 | 1 | 0 |
| Final probability | 0.0062 | 0.9955 | 0.9955 | 0.0047 |
| Thresholded prediction | 0 | 1 | 1 | 0 |

- **Initial loss:** 0.6935 (≈ $\ln 2$, consistent with near-random initial predictions)
- **Final loss:** 0.004978
- **All four labels correct:** **True**

## Gradient result — Part B (backpropagation check)

Final-step gradient of the first-layer weights, $dL/dW^{(1)}$:
```
[[-0.00029615 -0.00029659]
 [-0.00025454 -0.00025704]]
```

`parameter.grad` for $W^{(1)}$ **is** the matrix $\partial L/\partial W^{(1)}$ — entry $(i,j)$ is the partial derivative of the scalar loss with respect to that single weight, obtained via the chain rule through the output layer, the hidden sigmoid, and the hidden linear layer. Because the loss is defined as the **mean** BCE over the 4 examples, and differentiation is linear, each entry of $dL/dW^{(1)}$ is itself the average of the four example-wise gradients: $\frac{\partial L}{\partial W^{(1)}} = \frac{1}{N}\sum_{i=1}^N \frac{\partial L_i}{\partial W^{(1)}}$.

## Symmetry result — Part C

All weights (and biases) initialised to **zero**, architecture unchanged, trained for 2000 steps:

| step | $W^{(1)}$ row 0 | $W^{(1)}$ row 1 | identical? |
|---|---|---|---|
| 0 | [0, 0] | [0, 0] | True |
| 1 | [0, 0] | [0, 0] | True |
| 10 | [0, 0] | [0, 0] | True |
| 100 | [0, 0] | [0, 0] | True |
| 1000 | [0, 0] | [0, 0] | True |
| 1999 | [0, 0] | [0, 0] | True |

The two rows of $W^{(1)}$ **remained identical throughout training** — training stalled at the initial loss (0.693147, i.e. $\ln 2$) with every prediction stuck at 0.5 / class 0. With all weights zero, both hidden units compute the exact same output at every step and receive the exact same backpropagated gradient, so there is no mechanism to break the symmetry — this is why real initialisation uses small random weights.

## Activation results — Part D

Random init (seed 0), same 5000 steps / lr = 0.5, hidden activation varied only:

| Hidden activation | Final loss | 4/4 correct? | Early $\lVert\nabla W^{(1)}\rVert_2$ (step 1) |
|---|---|---|---|
| Sigmoid | 0.004978 | **True** | 0.001930 |
| Tanh | 0.347124 | **False** | 0.020291 |
| ReLU | 0.693147 | **False** | 0.061134 |

- **Sigmoid** converges cleanly to 4/4 correct under this seed/lr combination.
- **Tanh** gets stuck partway: both hidden units **saturate** near $\pm1$ (hidden outputs of $0.99999\ldots$), where $\tanh'(z)=1-h^2\approx0$; two of the four examples never separate from probability $\approx0.5$.
- **ReLU** gets stuck at exactly $\ln 2$ (every prediction at 0.5) — a **dead ReLU**: one hidden unit's pre-activation is negative for all four inputs, so it outputs 0 (and has zero local gradient) everywhere, leaving only one effectively "alive" hidden unit, which alone cannot represent XOR.

## Three-class extension results

**Predicted answers (stated before running):**

1. Final weight matrix $W^{(2)}$ shape: `(n_out, n_hidden) = (3, 2)`.
2. Logits per example: 3 (one per class).
3. Softmax probabilities sum to 1 because $p_k = e^{z_k}/\sum_j e^{z_j}$, so $\sum_k p_k = 1$ by construction.
4. The logit gradient is $p-y$ because softmax's Jacobian and the log-loss derivative combine algebraically to this closed form.

**Results after training** (5000 steps, lr = 0.5, seed 0):

- **Final $W^{(2)}$ shape:** `(3, 2)` — matches the prediction.
- **Final loss:** 0.003098
- **All predictions correct:** **True**

| Input | Softmax probabilities $[p_0, p_1, p_2]$ | Sum | Predicted class | True class |
|---|---|---|---|---|
| $(0,0)$ | $[0.99672,\ 0.00328,\ 0.0000005]$ | 1.00000000 | 0 | 0 |
| $(0,1)$ | $[0.00138,\ 0.99725,\ 0.00137]$ | 1.00000000 | 1 | 1 |
| $(1,0)$ | $[0.00138,\ 0.99725,\ 0.00137]$ | 1.00000000 | 1 | 1 |
| $(1,1)$ | $[0.00000003,\ 0.00360,\ 0.99640]$ | 1.00000000 | 2 | 2 |

**Shift-invariance diagnostic:** for input $(1,0)$, logits $z = [-2.1610,\ 4.4234,\ -2.1651]$ give softmax $[0.00138, 0.99725, 0.00137]$. Adding 100 to every logit gives softmax values differing by only $1.2\times10^{-17}$ (floating-point roundoff), confirming the shift is mathematically free — which is why stable implementations subtract $\max_k z_k$ before exponentiating: it prevents `exp` overflow without changing the result.

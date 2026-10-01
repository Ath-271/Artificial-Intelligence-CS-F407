import numpy as np
from engine import TwoLayerNet, set_seed

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([[0.],[1.],[1.],[0.]])

def train_binary(activation, seed, zero_init=False, steps=5000, lr=0.5, record_early_grad_at=1):
    net = TwoLayerNet(2, 2, 1, activation=activation, seed=seed, zero_init=zero_init)
    history = {"loss": [], "W1_rows": []}
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
        if step % 500 == 0 or step == steps - 1:
            history["loss"].append((step, loss))
        history["W1_rows"].append(net.W1.copy())
    final_logits, _ = net.forward(X)
    final_probs = 1/(1+np.exp(-final_logits))
    final_preds = (final_probs > 0.5).astype(float)
    return {
        "net": net, "initial_loss": initial_loss, "final_loss": loss,
        "probs": final_probs, "preds": final_preds,
        "early_grad_norm": early_grad_norm, "history": history,
        "final_grads": grads,
    }

print("="*70)
print("PART A — Basic learning check (sigmoid hidden, random init, seed=0)")
print("="*70)
runA = train_binary("sigmoid", seed=0)
print(f"Initial loss: {runA['initial_loss']:.4f}")
print(f"Final loss:   {runA['final_loss']:.6f}")
print("Probabilities:", runA["probs"].ravel())
print("Predictions:  ", runA["preds"].ravel())
print("Targets:      ", y.ravel())
all_correct = np.array_equal(runA["preds"], y)
print("All 4 labels correct:", all_correct)

print()
print("="*70)
print("PART B — Backpropagation check")
print("="*70)
print("dL/dW1 (first-layer weight gradient, final step):")
print(runA["final_grads"]["W1"])
print("""
Interpretation: parameter.grad for W1 IS the matrix dL/dW(1) -- each
entry (i,j) is dL/dW1[i,j], the partial derivative of the scalar loss
with respect to that single weight, obtained by the chain rule through
the output layer, the hidden activation, and the hidden layer. Because
the loss is defined as the MEAN BCE over the 4 examples, each entry of
dL/dW1 is itself the average of the 4 example-wise gradients
(1/N * sum_i dL_i/dW1), since differentiation is linear and the mean
is a linear combination of the per-example losses.
""")

print("="*70)
print("PART C — Symmetry experiment (all weights zero-initialised)")
print("="*70)
runC = train_binary("sigmoid", seed=0, zero_init=True, steps=2000)
rows_over_time = runC["history"]["W1_rows"]
checkpoints = [0, 1, 10, 100, 1000, len(rows_over_time)-1]
identical_flags = []
for cp in checkpoints:
    row0, row1 = rows_over_time[cp][0], rows_over_time[cp][1]
    identical = np.allclose(row0, row1)
    identical_flags.append(identical)
    print(f"  step {cp:5d}: W1 row0={row0}, row1={row1}, identical={identical}")
print(f"Rows remained identical throughout training: {all(identical_flags)}")
print(f"Final loss under zero init: {runC['final_loss']:.6f}  "
      f"(predictions: {runC['preds'].ravel()})")

print()
print("="*70)
print("PART D — Activation experiment (sigmoid vs tanh vs relu)")
print("="*70)
results_D = {}
for act in ["sigmoid", "tanh", "relu"]:
    r = train_binary(act, seed=0)
    all4 = np.array_equal(r["preds"], y)
    results_D[act] = (r["final_loss"], all4, r["early_grad_norm"])
    print(f"{act:8s} | final_loss={r['final_loss']:.6f} | 4/4 correct={all4} | "
          f"early ||grad W1||_2 (step1)={r['early_grad_norm']:.6f}")

print()
print("="*70)
print("PART D diagnostics — why tanh/relu underperform sigmoid here")
print("="*70)
for act in ["sigmoid", "tanh", "relu"]:
    r = train_binary(act, seed=0)
    print(f"\n--- {act} ---")
    print("probs:", r["probs"].ravel())
    logits, cache = r["net"].forward(X)
    X_, a1, h1, logits_ = cache
    print("pre-activations a1 (last step):\n", a1)
    print("hidden outputs h1 (last step):\n", h1)

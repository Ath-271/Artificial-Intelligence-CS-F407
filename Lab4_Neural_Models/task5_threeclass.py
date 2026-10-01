import numpy as np
from engine import TwoLayerNet

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
# Class 0: both inactive (0,0)
# Class 1: disagreement (0,1) or (1,0)
# Class 2: both active (1,1)
y_class = np.array([0, 1, 1, 2])
K = 3
y_onehot = np.eye(K)[y_class]

net = TwoLayerNet(n_in=2, n_hidden=2, n_out=K, activation="sigmoid", seed=0)

print(f"Predicted answers before running:")
print(f"  1. Final weight matrix W2 shape: (n_out, n_hidden) = ({K}, 2)")
print(f"  2. Logits per example: {K} (one per class)")
print(f"  3. Softmax probabilities sum to 1 because softmax normalises the")
print(f"     exponentiated logits by their own sum: p_k = exp(z_k)/sum_j exp(z_j),")
print(f"     so sum_k p_k = sum_k exp(z_k) / sum_j exp(z_j) = 1 by construction.")
print(f"  4. The logit gradient is p - y because for softmax+cross-entropy,")
print(f"     dL/dz_k = p_k - y_k -- the two Jacobians (softmax's and the log's)")
print(f"     combine and cancel to this simple closed form.")
print()

lr = 0.5
steps = 5000
for step in range(steps):
    logits, cache = net.forward(X)
    loss, dlogits, p = net.cross_entropy_loss_and_grad(logits, y_onehot)
    grads = net.backward(cache, dlogits)
    net.step(grads, lr)

final_logits, cache = net.forward(X)
final_loss, _, final_probs = net.cross_entropy_loss_and_grad(final_logits, y_onehot)
final_preds = final_probs.argmax(axis=1)

print(f"Final weight matrix W2 shape: {net.W2.shape}")
print(f"Final loss: {final_loss:.6f}")
print("\nSoftmax probability vectors (rows = examples (0,0),(0,1),(1,0),(1,1)):")
for i, row in enumerate(final_probs):
    print(f"  x={X[i]}  p={row}  sum={row.sum():.8f}  pred_class={final_preds[i]}  true_class={y_class[i]}")

print(f"\nAll predictions correct: {np.array_equal(final_preds, y_class)}")

print()
print("="*70)
print("Optional diagnostic: shift-invariance of softmax")
print("="*70)
example_idx = 2  # (1,0) -> disagreement
z = final_logits[example_idx]
p1 = net.softmax(z.reshape(1, -1))
shifted = z + 100.0
p2 = net.softmax(shifted.reshape(1, -1))
print(f"logits:          {z}")
print(f"softmax(logits): {p1.ravel()}")
print(f"logits+100:      {shifted}")
print(f"softmax(logits+100): {p2.ravel()}")
print(f"max abs difference: {np.max(np.abs(p1-p2)):.2e}  (floating-point roundoff only)")

print()
print("Why subtract the max logit before exponentiating (stable softmax):")
print("""  exp(z_k) can overflow to inf for even moderately large z_k (e.g.
  exp(1000) overflows float64). Subtracting max(z) first means the
  largest shifted logit is exactly 0, so exp(0)=1 and every other
  exp(z_k - max) <= 1 -- no overflow -- while softmax's own algebra
  (shown by the diagnostic above) guarantees the shift leaves the
  probabilities mathematically unchanged, so nothing is lost by
  doing this for stability.""")

"""
Task 3 deliverable: PyTorch implementation of the 2-2-1 XOR model
specified in Task 2.

LLM PROMPT USED (verbatim):
"Generate minimal PyTorch code for the following model and dataset.
Do not change the architecture or task. The model is 2 inputs -> 2
hidden units (sigmoid activation) -> 1 output, trained with logits
and BCEWithLogitsLoss on the 4 XOR examples. After training, report
the final loss, all four probabilities, thresholded labels, and one
parameter-gradient tensor. Set a random seed for reproducibility and
explain each test in one sentence."

CORRECTIONS MADE TO THE GENERATED CODE (per the lab's requirement to
report any changes before execution):
1. The first draft used plain `sigmoid output + BCELoss`; switched to
   logits + `BCEWithLogitsLoss` as the prompt actually requested,
   since that pairing is the numerically-stable one PyTorch recommends.
2. Added an explicit comment at each of the four required inspection
   points (forward pass / loss formation / backward() call / optimizer
   step) per the lab's "inspect before running" instruction.
3. No other changes -- architecture and task were left untouched.

Run this file with: python task3_pytorch_code.py   (needs `pip install torch`)
"""
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

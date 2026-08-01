# Linear Layer Initialization

## Goal

Implement a fully connected (dense) layer following the PyTorch API.

---

## Constructor

```python
layer = Linear(
    in_features=4,
    out_features=3
)
```

---

## Parameters

### Weight

Shape:

(out_features × in_features)

Each row corresponds to one neuron's weight vector.

---

### Bias

Shape:

(out_features,)

Each neuron owns one bias value.

---

## Initialization

Weights and biases are initialized with small random values sampled uniformly from:

[-0.1, 0.1]

Random initialization prevents all neurons from learning the same representation (symmetry problem).

---

## Responsibilities of Linear

- Own trainable parameters.
- Initialize weights and biases.
- Perform the forward computation (next sprint).
- Return parameters for optimization.

The layer does **not** perform optimization, compute loss, or manage training.
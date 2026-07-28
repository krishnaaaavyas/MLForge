# 07 - The Backward Pass

## Objective

The forward pass computes predictions.

The backward pass computes **how each trainable parameter should change** to reduce the loss.

This is the learning step of machine learning.

---

# The Learning Cycle

Every training iteration follows the same sequence:

```text
Input
    ↓
Forward Pass
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backward Pass
    ↓
Parameter Gradients
    ↓
Optimizer Step
    ↓
Updated Parameters
```

---

# Forward Pass

For Linear Regression,

[
\hat y = wx + b
]

Example:

```text
x = 5
w = 2
b = 1

Prediction = 11
```

The forward pass answers:

> "What does the model currently predict?"

---

# Loss

The loss compares the prediction with the true value.

Example:

```text
Prediction = 11

Actual = 15
```

The model made an error.

The loss measures **how large that error is**.

---

# Backward Pass

The backward pass answers a different question:

> "Which parameters caused the error, and how should they change?"

Instead of producing predictions, it produces **gradients**.

---

# What Is a Gradient?

A gradient measures the sensitivity of the loss to a parameter.

For a weight (w):

[
\frac{\partial L}{\partial w}
]

This tells us:

* Positive gradient → decrease the weight.
* Negative gradient → increase the weight.
* Larger magnitude → larger update.
* Smaller magnitude → smaller update.

---

# Chain Rule

For Linear Regression:

[
y = wx + b
]

Using the chain rule:

[
\frac{\partial L}{\partial w}
=============================

\frac{\partial L}{\partial y}
\times
\frac{\partial y}{\partial w}
]

Since

[
\frac{\partial y}{\partial w}=x
]

we obtain

[
\frac{\partial L}{\partial w}
=============================

\text{gradient} \times x
]

Similarly,

[
\frac{\partial y}{\partial b}=1
]

therefore

[
\frac{\partial L}{\partial b}
=============================

\text{gradient}
]

---

# Responsibilities

Each component has a single responsibility.

## Loss Function

Computes the gradient with respect to the model output.

```text
Prediction

↓

dL/dy
```

---

## Model

Converts output gradients into parameter gradients.

```text
dL/dy

↓

weight.grad

bias.grad
```

---

## Optimizer

Updates the parameters.

```text
parameter.value

↓

parameter.value - learning_rate × parameter.grad
```

---

# Why Store Gradients?

Each Parameter stores:

```python
value
grad
```

The optimizer only needs these two values.

This separation allows the optimizer to work with any model.

---

# Comparison with PyTorch

MLForge

```python
predictions = model.forward(X)

loss = loss_fn.forward(predictions, y)

gradients = loss_fn.backward(predictions, y)

model.backward(X, gradients)

optimizer.step(model.parameters())

optimizer.zero_grad(model.parameters())
```

PyTorch

```python
predictions = model(X)

loss = criterion(predictions, y)

loss.backward()

optimizer.step()

optimizer.zero_grad()
```

Although PyTorch hides many implementation details, the overall workflow is the same.

---

# Key Takeaways

* The forward pass computes predictions.
* The loss measures prediction error.
* The backward pass computes parameter gradients.
* The optimizer updates parameters using those gradients.
* Separating these responsibilities makes the framework modular and extensible.

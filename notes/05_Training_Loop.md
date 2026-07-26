# 05 - Training Loop

## Objective

A machine learning model learns by repeatedly making predictions, measuring its mistakes, and adjusting its parameters to reduce those mistakes.

This repeated process is called the **training loop**.

---

# The Complete Learning Pipeline

```
Dataset
    │
    ▼
Forward Pass
    │
Prediction
    │
Loss Function
    │
Gradient Calculation
    │
Optimizer
    │
Parameter Update
    │
Repeat
```

Every supervised machine learning algorithm follows this idea.

---

# Epoch

An **epoch** is one complete pass through the entire training dataset.

Example:

Dataset:

```
100 samples
```

Training for

```
50 epochs
```

means the model sees those 100 samples **50 times**.

More epochs generally improve learning until the model converges or begins to overfit.

---

# Forward Pass

The forward pass computes predictions using the current parameters.

For Linear Regression,

[
\hat y = wx+b
]

Example:

```
Weight = 2

Bias = 5

Input = 3
```

Prediction:

```
2×3+5=11
```

No learning happens during the forward pass.

It is only a prediction.

---

# Loss Calculation

After making predictions, we compare them with the actual values.

We use Mean Squared Error (MSE):

[
L=\frac1n\sum(y-\hat y)^2
]

The loss tells us how good or bad the current parameters are.

Lower loss indicates better predictions.

---

# Gradient Calculation

The gradient tells us how the loss changes with respect to each parameter.

For intuition,

```
Weight Gradient

≈

Error × Input
```

The actual formulas are

[
\frac{dL}{dw}
=============

\frac{2}{n}\sum(\hat y-y)x
]

[
\frac{dL}{db}
=============

\frac{2}{n}\sum(\hat y-y)
]

Weight gradients depend on both the prediction error and the input value.

Bias gradients depend only on the prediction error.

---

# Optimizer Step

The optimizer updates each parameter using its gradient.

Gradient Descent update rule:

[
w_{new}
=======

## w_{old}

\eta
\frac{dL}{dw}
]

where

* (w) = weight
* (\eta) = learning rate
* (\frac{dL}{dw}) = gradient

The optimizer always moves in the direction that reduces the loss.

---

# Complete Training Algorithm

```
Initialize weight and bias

Repeat:

    Forward Pass

    Compute Loss

    Compute Gradients

    Update Weight

    Update Bias

Until training finishes
```

This is the foundation of nearly every machine learning algorithm.

---

# Why Separate the Trainer?

The model should only know how to make predictions.

The trainer should know how to improve the model.

Separating these responsibilities follows the **Single Responsibility Principle (SRP)** and makes the framework easier to extend.

---

# Software Architecture

```
Model
    │
Forward Pass

Loss Function

Gradient Calculation

Optimizer

Trainer
```

Each component has exactly one responsibility.

---

# Comparison with Industry Libraries

## Our Framework

```python
trainer.fit(model, X, y)
```

Training is explicit so we understand every step.

---

## Scikit-learn

```python
model.fit(X, y)
```

The training loop is hidden from the user.

---

## PyTorch

```python
for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = model(X)

    loss = criterion(prediction, y)

    loss.backward()

    optimizer.step()
```

PyTorch computes gradients automatically using **Autograd**.

Our framework computes gradients manually so we understand the mathematics before using automation.

---

# Key Takeaways

* A training loop is the engine that enables learning.
* An epoch is one complete pass through the dataset.
* The forward pass only makes predictions.
* The loss function evaluates prediction quality.
* Gradients indicate how parameters should change.
* The optimizer updates parameters.
* Repeating this process gradually reduces the loss.

---

# My Understanding

A training loop is the heart of machine learning. It repeatedly predicts, measures error, computes gradients, updates parameters, and improves the model. Every modern machine learning framework follows this same pattern, differing mainly in how these steps are optimized and automated.

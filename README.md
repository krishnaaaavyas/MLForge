# MLForge

> A Machine Learning framework built completely from scratch in pure Python.

---

## Overview

MLForge is an educational machine learning framework whose goal is to rebuild the core ideas behind libraries like **PyTorch** and **scikit-learn** from first principles.

Instead of treating machine learning as a black box, every algorithm is implemented manually with extensive documentation and notes.

The project focuses equally on:

- Mathematics
- Machine Learning
- Software Engineering
- Clean Architecture

---

## Features

### Core

- Parameter
- Linear Layer

### Activations

- Identity
- Sigmoid

### Loss Functions

- Mean Squared Error
- Binary Cross Entropy

### Models

- Linear Regression
- Logistic Regression

### Optimizers

- Gradient Descent

### Training

- Generic Trainer

---

## Repository Structure

```text
mlforge/
│
├── activations/
├── core/
├── losses/
├── models/
├── optimizers/
├── training/
├── tests/
├── examples/
├── notes/
└── docs/
```

---

## Example

```python
from mlforge.models import LinearRegression
from mlforge.training import Trainer
from mlforge.losses import MeanSquaredError
from mlforge.optimizers import GradientDescent

model = LinearRegression()

trainer = Trainer(
    loss_fn=MeanSquaredError(),
    optimizer=GradientDescent(learning_rate=0.01),
    epochs=100
)

trainer.fit(model, X, y)
```

---

## Learning Notes

The repository includes detailed notes covering every concept implemented.

| Note | Topic |
|------|-------|
| 01 | Neuron |
| 02 | Parameters |
| 03 | Mean Squared Error |
| 04 | Gradient Descent |
| 05 | Linear Regression |
| 06 | Logistic Regression |

---

## Future Roadmap

- Vector Inputs
- Neural Networks
- Backpropagation
- Automatic Differentiation
- CNNs
- Transformers
- Attention
- LLM Components

---

## Why this project?

This project exists to bridge the gap between learning machine learning theory and understanding how modern frameworks actually work.

Every algorithm is first implemented manually before comparing it with industrial libraries such as PyTorch and scikit-learn.

---


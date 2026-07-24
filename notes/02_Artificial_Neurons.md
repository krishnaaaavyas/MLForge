# 02 - Artificial Neurons

## Why Artificial Neurons?

A single mathematical equation can model only simple relationships.

Artificial neurons are the basic building blocks of neural networks. Each neuron learns how important different inputs are and combines them into a single output.

---

# Components of a Neuron

### Inputs (Features)

The information provided to the neuron.

Examples:

* Hours Studied
* Attendance
* Sleep
* Age

Inputs change for every prediction.

---

### Weights

Each input has an associated weight.

A weight represents the importance of that input.

Higher positive weight:

* Stronger positive influence.

Negative weight:

* Decreases the output.

Weights are learned during training.

---

### Bias

Bias shifts the entire prediction.

Think of it as adjusting the decision threshold.

Bias is also learned during training.

---

### Output Score

The neuron combines all weighted inputs.

For one feature:

[
y = wx+b
]

For multiple features:

[
y=w_1x_1+w_2x_2+\cdots+w_nx_n+b
]

This output is often called the **score** or **logit**.

---

# Decision Boundary

The neuron predicts based on the score.

Example:

```text
Score > 0

↓

PASS
```

```text
Score ≤ 0

↓

FAIL
```

The point where the score equals zero is called the **decision boundary**.

---

# Why One Neuron Isn't Enough

A single neuron creates only one linear decision boundary.

Many real-world problems require multiple conditions.

Example:

```
Pass if

Hours > 5

AND

Attendance > 75%
```

One neuron cannot perfectly model many such problems.

Instead, we combine multiple neurons into layers.

---

# Designing a Neuron (Software Engineering)

A Neuron object should permanently store:

* Weights
* Bias

These are the learned parameters.

Temporary values should NOT be stored:

* Inputs
* Prediction
* Loss

These change for every example and are computed when needed.

---

# Python Implementation

Current implementation:

```python
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, inputs):
        score = 0

        for weight, value in zip(self.weights, inputs):
            score += weight * value

        score += self.bias

        return score
```

---

# New Python Concepts

### Class

Blueprint for creating objects.

---

### Object

An instance created from a class.

---

### self

Refers to the current object.

---

### Constructor (`__init__`)

Initializes an object when it is created.

---

### Method

A function that belongs to a class.

---

### `zip()`

Iterates through two lists together.

Example:

```python
weights = [2,5]
inputs = [6,0.8]

for w,x in zip(weights,inputs):
    print(w,x)
```

Output:

```
2 6
5 0.8
```

---

# Key Takeaways

* A neuron is a tiny computational unit.
* Inputs are multiplied by weights.
* Bias shifts the output.
* Weights and bias are learned.
* A neuron should store only learned parameters.
* A reusable neuron should support any number of input features.

---

## My Understanding

A neuron is a reusable software component and a mathematical function. It receives multiple inputs, weighs their importance, adds a bias, and produces a score. During training, the weights and bias are adjusted so that the score becomes more accurate over time.

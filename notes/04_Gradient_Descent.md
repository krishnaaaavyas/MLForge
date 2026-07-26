# 04 - Gradient Descent

## The Problem

A model can make predictions and calculate its loss, but it still doesn't know how to improve itself.

We need an algorithm that updates the model's parameters (weights and biases) so that the loss becomes smaller.

This algorithm is called **Gradient Descent**.

---

# Intuition

Imagine standing on a mountain while blindfolded.

Your goal is to reach the lowest point.

To decide which direction to move, you take a very small step and observe whether you go uphill or downhill.

Gradient Descent follows the same idea.

It measures how the loss changes with respect to a parameter and then moves in the direction that decreases the loss.

---

# Gradient

The gradient is the derivative (slope) of the loss function with respect to a parameter.

[
\frac{dL}{dw}
]

It answers the question:

> If I change the weight slightly, how much does the loss change?

A positive gradient means increasing the weight increases the loss.

A negative gradient means increasing the weight decreases the loss.

---

# Gradient Descent Update Rule

[
w_{new}=w-\eta\frac{dL}{dw}
]

Where:

* (w) = current weight
* (\eta) = learning rate
* (\frac{dL}{dw}) = gradient

The minus sign ensures we move in the direction that reduces the loss.

---

# Learning Rate

The learning rate determines the size of each update.

Small learning rate:

* Stable
* Slow convergence

Large learning rate:

* Faster learning
* Can overshoot the minimum and become unstable

Choosing an appropriate learning rate is one of the most important parts of training a model.

---

# Numerical vs Analytical Gradient

## Numerical Gradient

Estimate the slope using two nearby points.

[
\frac{L(w+\varepsilon)-L(w)}{\varepsilon}
]

Advantages:

* Easy to understand
* Useful for gradient checking

Disadvantages:

* Slow
* Approximate

---

## Analytical Gradient

Differentiate the loss function mathematically.

Advantages:

* Exact
* Fast
* Used in modern machine learning

Backpropagation computes analytical gradients efficiently.

---

# Software Design

The optimizer should only update parameters.

It should not know anything about:

* datasets
* neurons
* predictions
* loss functions

This follows the Single Responsibility Principle.

---

# Python Implementation

```python
class GradientDescent:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weight, gradient):
        return weight - self.learning_rate * gradient
```

---

# Key Takeaways

* Gradient is the slope of the loss function.
* Gradient Descent updates parameters to reduce loss.
* The learning rate controls the step size.
* Numerical gradients are useful for understanding and debugging.
* Analytical gradients are used in practical machine learning.
* Optimizers should focus only on updating parameters.

---

# My Understanding

Gradient Descent is an optimization algorithm that repeatedly adjusts model parameters in the direction that decreases the loss. The gradient tells us which direction to move, and the learning rate determines how large each step should be.

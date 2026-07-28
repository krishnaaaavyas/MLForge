# Logistic Regression Training

## Goal

Teach a model to separate two classes.

Pipeline

Input
↓

Linear Layer

↓

Sigmoid

↓

Probability

↓

Binary Cross Entropy

↓

Gradient

↓

Weight Update

↓

Repeat

---

The combination of Sigmoid and Binary Cross Entropy is mathematically elegant because their derivatives simplify significantly during optimization. In many derivations, the gradient reaching the linear output becomes proportional to (prediction - target), making training both stable and efficient.

Training follows the same cycle:

1. Forward pass
2. Compute loss
3. Compute gradients
4. Backward pass
5. Optimizer step
6. Zero gradients

This is the same high-level process used by modern deep learning libraries such as PyTorch.
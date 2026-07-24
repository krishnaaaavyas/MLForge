# ML From Scratch - Architecture

## Vision

This project is an educational machine learning framework built entirely from scratch.

Its goals are:

1. Understand machine learning algorithms from first principles.
2. Build reusable software components instead of one-off scripts.
3. Learn software engineering practices alongside machine learning.
4. Provide mathematical explanations, implementation details, and practical examples for every algorithm.

---

## Design Philosophy

Every module should have one responsibility.

Examples:

* Neuron → computes predictions
* Loss Function → evaluates predictions
* Optimizer → updates parameters
* Model → combines components into a trainable algorithm

This follows the **Single Responsibility Principle (SRP)**.

---

## Framework Layers

```
Examples
    │
Models
    │
Optimizers
    │
Loss Functions
    │
Core Components
```

Each layer depends only on the layer below it.

---

## Development Workflow

Every feature follows the same engineering process:

1. Theory
2. Mathematical derivation
3. Software design
4. Implementation
5. Testing
6. Documentation
7. Git commit
8. Merge into main

This ensures the repository grows in a clean and maintainable way.

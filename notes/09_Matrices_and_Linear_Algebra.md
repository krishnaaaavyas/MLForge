# Note 09: Matrices and Batch Processing in ML

## 1. The Intuition
Single-neuron dot products ($w \cdot x + b$) handle one signal at a time.
Real machine learning models evaluate multiple outputs (neurons) over multiple inputs (samples) simultaneously.

A **Weight Matrix** $W$ stacks $M$ neuron weight vectors vertically:
- **Rows:** Represent individual neurons ($M$ neurons).
- **Columns:** Represent incoming features ($N$ features).

## 2. Matrix-Vector Multiplication
For $M$ neurons processing a single sample $x$ of shape $(N \times 1)$:
$$Y = Wx + b$$

Where:
- $W$: shape $(M \times N)$
- $x$: shape $(N \times 1)$
- $Y$: shape $(M \times 1)$

Each output element $y_i$ is computed as:
$$y_i = \text{Row}_i(W) \cdot x + b_i$$

## 3. Shape Multiplication Rule
For $A(M \times N)$ and $B(N \times P)$:
1. Inner dimensions MUST match ($N == N$).
2. Output shape will be outer dimensions ($M \times P$).

## 4. Hardware Optimization
GPUs perform matrix operations in parallel block operations rather than sequential CPU `for` loops. Expressing neural network layers as matrix multiplications allows massive parallel execution.
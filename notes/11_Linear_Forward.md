# Linear Layer Forward Pass

## Equation

\[
Y = XW^T + b
\]

---

## Shapes

Input:

(batch_size × in_features)

Weight:

(out_features × in_features)

Bias:

(out_features,)

Output:

(batch_size × out_features)

---

## Why transpose the weight matrix?

Each row of the weight matrix belongs to one neuron.

Transposing converts neuron weight vectors into columns, allowing every sample in the batch to compute a dot product with every neuron simultaneously using matrix multiplication.

---

## Bias Broadcasting

The bias vector contains one bias value per neuron.

During the forward pass, the same bias vector is added to every sample in the batch.

---

## Responsibilities of Linear.forward()

1. Validate input shape.
2. Perform matrix multiplication.
3. Add bias.
4. Return output matrix.

The layer does **not** compute loss, update parameters, or perform optimization.
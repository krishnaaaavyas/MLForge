# 03 - Loss Functions (Mean Squared Error)

## Why Do We Need a Loss Function?

A machine learning model can make predictions, but it has no idea whether those predictions are good or bad.

A **loss function** measures how far the predictions are from the actual values.

Without a loss function, a model cannot learn because it has no way to evaluate its performance.

---

# Error vs Loss

## Error

Error measures the difference for a **single prediction**.

[
\text{Error} = \text{Actual} - \text{Prediction}
]

Example:

```text
Prediction = 60

Actual = 70

Error = 10
```

---

## Loss

Loss summarizes the errors across an entire dataset.

Instead of asking,

> "How wrong was one prediction?"

Loss asks,

> "How wrong is the model overall?"

---

# Why Not Simply Average the Errors?

Example:

```text
Prediction = 60
Actual = 70
Error = +10

Prediction = 80
Actual = 70
Error = -10
```

Average:

```text
(+10 + -10)/2 = 0
```

The model appears perfect even though both predictions are wrong.

Positive and negative errors cancel each other.

---

# Why Square the Errors?

Squaring removes the sign.

```text
(+10)² = 100

(-10)² = 100
```

Benefits:

* No cancellation
* Larger mistakes are penalized much more heavily
* Smooth mathematical function that is easy to optimize

---

# Mean Squared Error (MSE)

Steps:

1. Compute the error for every prediction.
2. Square each error.
3. Sum all squared errors.
4. Divide by the number of samples.

---

# Example

Predictions:

```text
[60, 40, 90]
```

Actuals:

```text
[70, 42, 85]
```

Errors:

```text
10
2
-5
```

Squared Errors:

```text
100
4
25
```

Total:

```text
129
```

Mean:

```text
129 / 3 = 43
```

Therefore,

```text
MSE = 43
```

---

# Python Implementation

```python
def calculate_mse(predictions, actuals):

    if len(predictions) != len(actuals):
        raise ValueError("Lists must have the same length.")

    squared_error_sum = 0

    for predicted, actual in zip(predictions, actuals):

        error = actual - predicted

        squared_error_sum += error ** 2

    return squared_error_sum / len(predictions)
```

---

# Engineering Decisions

## Why separate `mse.py`?

Instead of one large `loss.py`, every loss function gets its own module.

Future additions:

* mse.py
* mae.py
* huber.py
* cross_entropy.py

This follows the **Open/Closed Principle**.

---

## Why doesn't the neuron calculate loss?

The neuron should only make predictions.

The loss function should only evaluate predictions.

Each module has a single responsibility, making the framework easier to test, maintain, and extend.

---

## Why validate input lengths?

Predictions and actual values must correspond one-to-one.

If the lengths differ, continuing the calculation would produce incorrect results.

Professional software should fail early with a clear error message.

---

# Key Takeaways

* Error measures one prediction.
* Loss measures the whole dataset.
* Squaring prevents positive and negative errors from cancelling.
* Mean Squared Error heavily penalizes large mistakes.
* Lower MSE indicates a better model.
* Loss functions evaluate models but do not make predictions.

---

# My Understanding

A loss function acts as the model's judge. It compares predictions with actual values and produces a single number representing overall performance. During training, the objective is to minimize this loss by adjusting the model's parameters.

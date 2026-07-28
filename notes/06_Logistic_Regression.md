# 06 - Logistic Regression & Sigmoid Activation

## Objective

Linear Regression predicts continuous numerical values.

However, many real-world problems require predicting **categories** rather than numbers.

Examples:

* Spam or Not Spam
* Diabetes or No Diabetes
* Fraud or Not Fraud
* Pass or Fail

These are called **binary classification** problems.

Logistic Regression is one of the most fundamental algorithms used to solve them.

---

# Regression vs Classification

## Regression

Predicts continuous numerical values.

Examples:

```text
House Price

₹ 52,30,000
```

```text
Temperature

32.4°C
```

```text
Salary

₹ 8,75,000
```

Output can be any real number.

---

## Classification

Predicts classes.

Examples:

```text
Spam

Not Spam
```

```text
Diabetic

Not Diabetic
```

```text
Pass

Fail
```

Instead of predicting the class directly, Logistic Regression predicts the **probability** of belonging to the positive class.

---

# Why Linear Regression Cannot Be Used

Linear Regression predicts

[
\hat y = wx+b
]

Possible outputs:

```text
-12

3

98

1200
```

These values cannot be interpreted as probabilities.

A probability must always lie between

```text
0

and

1
```

Therefore we need a function that converts any real number into this range.

---

# Sigmoid Activation

The sigmoid function is

[
\sigma(x)=\frac1{1+e^{-x}}
]

Although the formula appears complex, its purpose is simple:

Convert any real number into a probability.

Examples:

| Input | Output |
| ----: | -----: |
|  -100 |     ≈0 |
|   -10 |     ≈0 |
|     0 |    0.5 |
|    10 |     ≈1 |
|   100 |     ≈1 |

Properties:

* Output always lies between 0 and 1.
* Very negative values become close to 0.
* Very positive values become close to 1.
* An input of 0 always maps to 0.5.

---

# Logistic Regression Pipeline

The model performs two steps.

Step 1

Compute the linear output

[
z=wx+b
]

Step 2

Apply sigmoid

[
p=\sigma(z)
]

where

* (z) is the linear score.
* (p) is the predicted probability.

---

# Making Predictions

The model outputs probabilities.

Example:

```text
0.91
```

means

```text
91% probability
```

To convert probabilities into classes, a threshold is used.

Common threshold:

```text
0.5
```

Decision rule:

```text
Probability ≥ 0.5

↓

Positive Class (1)
```

```text
Probability < 0.5

↓

Negative Class (0)
```

This threshold can be changed depending on the application.

---

# Why Confidence Matters

Consider two predictions for a patient who actually has diabetes.

Prediction A

```text
0.99
```

Prediction B

```text
0.55
```

Both predict the correct class.

However,

Prediction A is much more confident.

Machine learning models should not only be correct but should also assign appropriate confidence to their predictions.

---

# Binary Cross-Entropy Motivation

Suppose the true label is

```text
1
```

Desired behaviour:

| Prediction |        Penalty |
| ---------: | -------------: |
|       0.99 |           Tiny |
|       0.90 |          Small |
|       0.70 |       Moderate |
|       0.50 |         Medium |
|       0.20 |           High |
|       0.05 |      Very High |
|       0.01 | Extremely High |

Notice that the penalty should not increase linearly.

Confidently wrong predictions should receive much larger penalties.

This motivates the use of **Binary Cross-Entropy (BCE)** instead of Mean Squared Error.

BCE will be implemented in the next milestone.

---

# Software Architecture

```text
Input

↓

Neuron

↓

Linear Output (z)

↓

Sigmoid

↓

Probability

↓

Threshold

↓

Prediction
```

---

# Comparison with Industry Libraries

## Our Framework

```python
probabilities = model.predict_proba(X)

predictions = model.predict(X)
```

Training will be added after Binary Cross-Entropy is implemented.

---

## Scikit-learn

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X, y)

model.predict(X)

model.predict_proba(X)
```

Scikit-learn hides the mathematics and optimization.

Our framework exposes each component so we understand how the algorithm works internally.

---

# Key Takeaways

* Logistic Regression is a classification algorithm.
* It predicts probabilities rather than arbitrary numbers.
* Sigmoid converts any real number into the interval [0, 1].
* A threshold converts probabilities into class labels.
* Confidently wrong predictions should receive much larger penalties than slightly wrong predictions.
* Binary Cross-Entropy is designed to provide this behaviour and will be implemented next.

---

# My Understanding

Logistic Regression extends Linear Regression by adding a sigmoid activation that transforms the linear output into a probability. Instead of predicting arbitrary numbers, it predicts the likelihood of belonging to a class. The final class is obtained by applying a threshold to the predicted probability. Because classification requires confidence-aware learning, Binary Cross-Entropy is preferred over Mean Squared Error for training.

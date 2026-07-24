# 01 - Machine Learning Fundamentals

## What is Artificial Intelligence?

Artificial Intelligence (AI) is the broad field of creating systems that perform tasks requiring human intelligence such as reasoning, learning, planning, language understanding, and perception.

Examples:

* ChatGPT
* Self-driving cars
* Face recognition
* Recommendation systems

---

## What is Machine Learning?

Machine Learning (ML) is a subset of AI.

Instead of explicitly programming rules, we allow a computer to learn patterns from data.

Traditional Programming:

```
Rules + Data
      ↓
   Output
```

Machine Learning:

```
Data + Correct Answers
          ↓
     Learning Algorithm
          ↓
        Model
          ↓
      Predictions
```

---

## Types of Machine Learning

### 1. Supervised Learning

The dataset contains both inputs (features) and correct outputs (targets).

Goal:

Learn the relationship between inputs and outputs.

Examples:

* Predict house prices
* Predict diabetes risk
* Email spam detection

---

### 2. Unsupervised Learning

No target labels are provided.

Goal:

Discover hidden patterns.

Examples:

* Customer segmentation
* Fraud detection
* Topic clustering

---

### 3. Reinforcement Learning

The model learns through rewards and penalties while interacting with an environment.

Examples:

* Chess engines
* Robotics
* Game-playing AI

---

## Dataset

A dataset is a collection of training examples.

Each example contains:

### Features (Inputs)

Information used to make predictions.

Examples:

* Age
* Weight
* Height
* Hours Studied
* Attendance

---

### Target (Output)

The value we want to predict.

Examples:

* Marks
* House Price
* Pass/Fail
* Diabetes Risk

---

## Train/Test Split

Training Data

Used to learn patterns.

Testing Data

Used to evaluate how well the model performs on unseen data.

The model should never memorize the training data.

It should learn general patterns.

---

## Model

A model is the mathematical relationship learned from the training data.

Example:

```
Marks = 10 × Hours + 5
```

The numbers are learned from data—not manually chosen.

---

## Prediction

The model uses learned parameters to make predictions on new data.

Prediction is simply applying the learned model to unseen inputs.

---

## Error

```
Error = Prediction − Actual
```

Error tells us how wrong one prediction is.

---

## Loss

Loss summarizes the total error across many examples.

The most common regression loss:

Mean Squared Error (MSE)

Steps:

1. Calculate every error.
2. Square each error.
3. Add them together.
4. Divide by the number of examples.

Goal:

Minimize the loss.

---

## Gradient Descent (Intuition)

Training follows this cycle:

```
Predict
   ↓
Measure Loss
   ↓
Adjust Parameters
   ↓
Predict Again
```

Repeat until the loss is sufficiently small.

---

## Learning Rate

Learning rate controls how much parameters change during each update.

Small learning rate:

* Stable
* Slow

Large learning rate:

* Fast
* Can overshoot the optimum

---

## Key Engineering Principles

* A model is only as good as its data.
* Better features usually improve predictions.
* Lower loss generally means a better model.
* Always evaluate on unseen data.
* Never optimize using only one example.

---

## My Understanding

Machine Learning is the process of learning mathematical relationships from data instead of manually writing rules. A model learns parameters that minimize loss, allowing it to make useful predictions on new, unseen examples.

---

## Common Mistakes

* Thinking the model memorizes data.
* Confusing error with loss.
* Ignoring the quality of the dataset.
* Evaluating only on training data.

---

## Progress Checklist

✅ AI vs ML

✅ Supervised Learning

✅ Unsupervised Learning

✅ Reinforcement Learning

✅ Features

✅ Targets

✅ Training Data

✅ Testing Data

✅ Models

✅ Prediction

✅ Error

✅ Loss (MSE)

✅ Gradient Descent (Intuition)

✅ Learning Rate

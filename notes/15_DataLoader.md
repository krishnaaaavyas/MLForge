# DataLoader

## Purpose

A DataLoader groups individual samples from a Dataset into batches.

---

## Why batching?

Training one sample at a time is inefficient.

Batching allows:

- Better CPU/GPU utilization.
- Faster matrix operations.
- Stable gradient estimates.

---

## Responsibilities

- Iterate through a Dataset.
- Return batches of samples.
- Stop automatically at the end of the dataset.

---

## Python Iterator Protocol

A DataLoader implements:

- `__iter__()` — starts iteration.
- `__next__()` — returns the next batch.

When no batches remain, `StopIteration` ends the loop automatically.

---

## Design Principle

Dataset stores data.

DataLoader controls iteration and batching.

Each class has a single responsibility.
# Dataset

## Purpose

A Dataset provides access to training samples.

It answers only two questions:

1. How many samples exist?
2. Give me sample i.

---

## Interface

Every Dataset implements:

- `__len__()`
- `__getitem__()`

---

## Why?

The training loop should not care where the data comes from.

Today:

- TensorDataset

Tomorrow:

- CSVDataset
- ImageDataset
- HealthGuardDataset
- LASIDataset

The training code remains unchanged.

---

## Design Principle

Program to an interface, not to an implementation.

The Dataset defines a common interface that every concrete dataset follows.
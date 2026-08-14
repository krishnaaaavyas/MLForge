# Module System

## Goal

Provide a common base class for all trainable components.

---

## Responsibilities

- Register Parameters automatically.
- Register child Modules automatically.
- Return parameters recursively.
- Provide a common forward() interface.
- Allow modules to be called like functions.

---

## Automatic Registration

Python's `__setattr__` intercepts every attribute assignment.

When a `Parameter` is assigned, it is stored in `_parameters`.

When a child `Module` is assigned, it is stored in `_modules`.

This removes the need for manually implementing `parameters()` in every layer.

---

## Benefits

- Less repetitive code.
- Recursive parameter discovery.
- Similar architecture to `torch.nn.Module`.
- Easier to build complex neural networks.
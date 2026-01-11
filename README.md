# abdou_torch ⭐⭐⭐
This is a minimalistic deep learning framework implemented purely in Python and NumPy.

Inspired by the mechanics of modern libraries like PyTorch and Andrej Karpathy's micrograd, this repository aims to replicate the core components required for training neural networks, specifically focusing on automatic differentiation (autograd).

## 🛠️Core Design Principles

### 1. The Tensor Class 🧊

The core data structure is the `Tensor`. It is a wrapper around a NumPy array that is responsible for:

* **Numerical Data**: Stored in `.data`.
* **Gradients**: Stored in `.grad`, representing the derivative of the loss with respect to the tensor.
* **Graph Tracking**: Maintaining the computational history via its parents (`._prev`) and the operation that created it (`._op`).

### 2. Backpropagation via the Chain Rule 🔄

The `.backward()` method is the heart of the project. It uses a **topological sort** to process the computation graph in reverse order. This ensures that gradients are correctly accumulated and propagated through every node based on the chain rule:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x}$$

### 3. Operator overloading ➕

To make model building intuitive, essential mathematical operations (`+`, `*`, `**`, `@`, `/`) and functional layers (`tanh`, `relu`, `sum`) are implemented using Python's operator overloading (`__add__`, `__mul__`, etc.).

### 4. Neural Network Modules 🧠

The framework provides an object-oriented API for building architectures:
* **Module**: A base class for parameter management and state tracking.
* **Linear**: Implements standard affine transformations ($W \cdot x + b$).
* **MLP**: A Multi-Layer Perceptron class that stacks linear layers and non-linearities.

### 5. Custom WAdam Optimizer 🎯

This project features a custom implementation of the **WAdam** optimizer. It combines the adaptive learning rates and momentum of Adam with **Decoupled Weight Decay**, providing superior regularization and more stable parameter updates than standard Adam.

## 🚀 Usage Example: Training on MNIST and make_blobs

This framework is powerful enough to achieve **+97.8%** on real datasets. Here is the workflow used to train on MNIST:

```python
from abdou_torch import Tensor, MLP, WAdam
import numpy as np

# 1. Initialize a Deep MLP (784 -> 64 -> 64 -> 64 -> 10)
model = MLP(784, [64, 64, 64, 10])
optimizer = WAdam(model.parameters(), lr=0.0004, weight_decay=1e-4)

# 2. Training Loop (Vectorized)
for epoch in range(4):
    for batch_x, batch_y in train_loader:
        # Wrap numpy data in Tensors
        inputs = Tensor(batch_x)
        targets = Tensor(batch_y, requires_grad=False)

        # Forward pass & MSE Loss
        preds = model(inputs)
        loss = ((preds - targets)**2).sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

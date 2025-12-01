# abdou_torch
This is a minimalistic deep learning framework implemented purely in Python using NumPy for array operations.

Inspired by the mechanics of modern libraries like PyTorch and Andrej Karpathy's micrograd, this repository aims to replicate the core components required for training neural networks, specifically focusing on automatic differentiation (autograd) and fundamental building blocks.

**Core Design Principles**

1. The Tensor Class

The core data structure is the Tensor. It is a wrapper around a NumPy array that is responsible for:

Holding the numerical data (.data).

Storing the derivative (.grad).

Tracking the computational history via its parents (._prev) and the operation that created it (._op).

2. Backpropagation via the Chain Rule

The backward() method is the heart of the project. It uses a topological sort to process the computation graph in reverse order, ensuring that gradients are correctly accumulated and propagated through every node (tensor) based on the chain rule.

3. Operator Overloading

Essential mathematical operations (+, *, **, @, /) and functional layers (tanh, relu, sum) are implemented using Python's operator overloading (__add__, __mul__, etc.).

4. Neural Network Modules

The framework includes simple object-oriented components:

Module: A base class for parameter management.

Linear: Implements the standard affine transformation ($W \cdot x + b$).

MLP: Stacks linear layers and non-linearities to create a full Multi-Layer Perceptron.

5. Custom WAdam Optimizer

This project features a custom implementation of the WAdam optimizer, which combines the adaptive learning rates and momentum of Adam with Decoupled Weight Decay for regularization, providing a robust update rule for parameter optimization.


**Demo: Training a simple MLP**

I made a demo for the make_blobs classification and the MNIST dataset.


**Educational Value :**

This framework serves as an excellent educational tool for anyone looking to solidify their understanding of:

- How computation graphs are built and traversed.
- The exact mechanics of reverse-mode automatic differentiation.
- The role of momentum and adaptive learning rates in optimizers.

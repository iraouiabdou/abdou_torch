# abdou_torch

A tiny autograd engine and neural network library written from scratch in ~300 lines of Python.

No PyTorch. No TensorFlow. Just NumPy — with automatic GPU acceleration via CuPy when a GPU is available.

It trains a 5-layer MLP on MNIST to **98.28% test accuracy**.

---

## Why

Backpropagation is easy to *use* and surprisingly easy to get *wrong* when you write it yourself. This project is a from-first-principles implementation of reverse-mode automatic differentiation, complete with broadcasting-aware gradients, batched matmul, a numerically stable cross-entropy, and a working AdamW optimizer — the pieces you actually need to train a real network on a real dataset.

The API deliberately mirrors PyTorch, so if you know `torch`, you already know this.

## Features

- **Reverse-mode autodiff** over a dynamically built computation graph, with topological-sort backward pass
- **Transparent GPU support** — imports CuPy if present, falls back to NumPy silently
- **Correct broadcasting gradients** via an `unbroadcast` helper that sums gradients back down to the input shape
- **Full `@` support** — handles 1D×1D, 1D×2D, 2D×1D, and batched N-D matmul, each with its own gradient path
- **Numerically stable cross-entropy** (max-subtraction + logsumexp) with a fused softmax gradient
- **`Module` / `Linear` / `MLP`** building blocks with He initialization
- **AdamW** with bias correction and decoupled weight decay

## Install

There's nothing to install — the library is a single file with one dependency.

```bash
pip install numpy
pip install cupy-cuda12x   # optional, for GPU
```

```bash
wget https://raw.githubusercontent.com/iraouiabdou/abdou_torch/main/abdou_torch.py
```

## Quickstart

```python
from abdou_torch import Tensor, MLP, AdamW, xp

model = MLP(784, [1024, 512, 256, 128, 10])
optim = AdamW(model.parameters(), lr=1e-3)

x = Tensor(x_batch, True)          # (batch, 784)
logits = model(x)                  # (batch, 10)
loss = logits.crossEntropyLoss(y_batch)

model.zero_grad()
loss.backward()
optim.step()
```

Or use the `Tensor` primitives directly:

```python
a = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = Tensor([[5.], [6.]], requires_grad=True)   # broadcasts

c = (a * b + 2).leaky_relu().sum()
c.backward()

print(a.grad, b.grad)
```

## API

### `Tensor`

| Method | Notes |
| --- | --- |
| `+ - * / **` | `**` takes int/float exponents only |
| `@` | 1D, 2D and batched N-D, with broadcasting |
| `.sum(axes=None, keepdims=False)` | gradient re-expands collapsed axes |
| `.leaky_relu(leak=0.01)` | |
| `.crossEntropyLoss(targets)` | `targets` = integer class indices, shape `(batch,)` |
| `.backward()` | builds the topo order and propagates from this node |

Gradients **accumulate** — call `model.zero_grad()` between steps.

### `Module`

Subclass and implement `parameters()`. `zero_grad()` comes for free.

- `Linear(nin, nout)` — He-initialized weights, zero bias
- `MLP(nin, [h1, h2, ..., nout])` — leaky-ReLU between layers, linear output head
- `AdamW(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)`

## Files

| File | |
| --- | --- |
| `abdou_torch.py` | The whole library — engine, layers, optimizer |
| `mnist_demo.ipynb` | End-to-end MNIST training + evaluation (Colab-ready, GPU) |

## MNIST results

`mnist_demo.ipynb` trains `MLP(784, [1024, 512, 256, 128, 10])` for 50 epochs, batch size 2048, AdamW at default settings.

| | |
| --- | --- |
| Train loss (epoch 1) | 0.6062 |
| Train loss (epoch 50) | ~0.0000 |
| **Test accuracy** | **98.28%** |

The model fully memorizes the training set well before epoch 50 — there's no dropout, augmentation, or LR schedule here. Test accuracy is what a plain MLP of this size should get on MNIST, which is the point: the gradients are correct.

## Design notes

**Broadcasting.** Every elementwise op computes its gradient at the *output* shape, then `unbroadcast` figures out which axes were stretched and sums over them before reshaping back. This is the part that most from-scratch autograds quietly get wrong.

**Matmul.** `@` branches on whether either operand is 1D, because NumPy's matmul silently promotes 1D operands to matrices and then strips the extra dimension back off. The backward pass has to undo that promotion by hand: `expand_dims` where NumPy inserted a dimension, `squeeze` where it removed one.

**Cross-entropy.** Softmax and NLL are fused into one node, so the backward pass is just `probs - onehot` divided by batch size, instead of differentiating through `exp` and `log` separately.

**GPU.** The library never references `numpy` or `cupy` directly in its ops — everything goes through `xp`, bound once at import. Export it (`from abdou_torch import xp`) and use it for your data too, so arrays land on the right device.

## Limitations

Honest list, since this is a learning project:

- Only the ops listed above — no conv, no norm layers, no attention
- No `no_grad()` context; inference still builds a graph
- Gradients are allocated eagerly on every tensor, even when `requires_grad=False`
- `backward()` recurses, so very deep graphs can hit the recursion limit
- `crossEntropyLoss` mutates its `probs` buffer in place, and assumes 2D logits

## Credits

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) — extended from scalars to n-dimensional arrays, GPU, and a real optimizer.

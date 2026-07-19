
from __future__ import annotations 
import numpy as np

try:
  import cupy as xp
  has_gpu = True
  print("GPU detected")
except:
  xp = np
  has_gpu = False
  print("GPU not detected - using CPU")

class Tensor():
  def __init__(self, data, requires_grad = False, _children: tuple[Tensor, ...]=(), _op="", dtype = xp.float32):
    if isinstance(data, (int, float, np.ndarray, xp.ndarray, list, xp.number, np.number)):
      self.data = xp.array(data, dtype = dtype)
    else:
      raise TypeError(f"{type(data)} is not supported")
    self.dtype = dtype
    self.shape = self.data.shape
    self.requires_grad = requires_grad
    self._op = _op
    self._prev = set(_children)
    self.grad = xp.zeros_like(self.data, dtype=dtype)
    self._backward = lambda: None

  def __repr__(self):
    return f"Tensor({self.data})"

  def sum(self, axes: tuple[int,...] | None = None, keepdims: bool = False):
    out = Tensor(xp.sum(self.data, axis= axes, keepdims = keepdims),
                 self.requires_grad, (self,), "sum")
    def _backward():
      if self.requires_grad:
        grad = out.grad
        # in case of for example : sum over axis 1 the tensor with shape (2,3,4)
        # if we don't keepdims it will give us a (2,4) tensor, broadcasting can't
        # work here so we need to expand dims to get (2,1,4) and then we can
        # broadcast on the axis 1
        if axes is not None and not keepdims:
          grad = xp.expand_dims(grad, axes)
        self.grad += grad
    out._backward = _backward
    return out

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    rq_grad = self.requires_grad or other.requires_grad
    out = Tensor(self.data + other.data, rq_grad, (self, other), "+")
    def _backward():
      if self.requires_grad:
        self.grad += out.grad.sum(unbroadcast(self.shape, out.shape), keepdims=True).reshape(self.shape)
      if other.requires_grad:
        other.grad += out.grad.sum(unbroadcast(other.shape, out.shape), keepdims=True).reshape(other.shape)
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    rq_grad = self.requires_grad or other.requires_grad
    out = Tensor(self.data * other.data, rq_grad, (self, other), "*")
    def _backward():
      if self.requires_grad:
        self.grad += (other.data * out.grad).sum(unbroadcast(self.shape, out.shape), keepdims=True).reshape(self.shape)
      if other.requires_grad:
        other.grad += (self.data * out.grad).sum(unbroadcast(other.shape, out.shape), keepdims=True).reshape(other.shape)
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), f"{type(other)} is not" \
                                             "supported only int/float"

    out = Tensor(self.data ** other, self.requires_grad, (self,), "^")
    def _backward():
      if self.requires_grad:
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out

  def leaky_relu(self, leak: float = 0.01):
    out = Tensor(xp.where(self.data>0, self.data, leak*self.data),
                 self.requires_grad, (self,), "leaky_relu")
    def _backward():
      if self.requires_grad:
        self.grad += xp.where(out.data>0, 1, leak)  * out.grad
    out._backward = _backward
    return out

  def __matmul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    rq_grad = self.requires_grad or other.requires_grad
    out = Tensor(self.data @ other.data, rq_grad, (self, other), "@")

    def _backward():
      l_1D = len(self.shape) == 1
      r_1D = len(other.shape) == 1

      if l_1D and r_1D :
        if self.requires_grad:
          self.grad += out.grad * other.data
        if other.requires_grad:
          other.grad += out.grad * self.data
      elif r_1D:
        if self.requires_grad:
          self.grad += xp.expand_dims(out.grad, -1) @ xp.expand_dims(other.data, 0)
        if other.requires_grad:
          grad_other = xp.swapaxes(self.data, -1, -2) @ xp.expand_dims(out.grad, -1)
          grad_other = xp.squeeze(grad_other, -1)
          axes = unbroadcast(other.shape, grad_other.shape)
          if axes:
              grad_other = grad_other.sum(axes, keepdims=True).reshape(other.shape)
          other.grad += grad_other
      elif l_1D:
        if self.requires_grad:
            grad_self = other.data @ xp.expand_dims(out.grad, -1)
            grad_self = xp.squeeze(grad_self, -1)
            axes = unbroadcast(self.shape, grad_self.shape)
            if axes:
                grad_self = grad_self.sum(axes, keepdims=True).reshape(self.shape)
            self.grad += grad_self
        if other.requires_grad:
            other.grad += xp.expand_dims(self.data, -1) @ xp.expand_dims(out.grad, -2)
      else:
        if self.requires_grad:
          grad_self = out.grad @ xp.swapaxes(other.data, -1, -2)
          axes = unbroadcast(self.shape, grad_self.shape)
          if axes:
            grad_self = grad_self.sum(axes, keepdims=True).reshape(self.shape)
          self.grad += grad_self
        if other.requires_grad:
          grad_other = xp.swapaxes(self.data, -1, -2) @ out.grad
          axes = unbroadcast(other.shape, grad_other.shape)
          if axes:
            grad_other = grad_other.sum(axes, keepdims=True).reshape(other.shape)
          other.grad += grad_other
    out._backward = _backward
    return out

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __truediv__(self, other): # self / other
          # Division is treated as self * (other**-1)
    return self * other**-1

  def __rmul__(self, other): # other * self
    return self * other

  def __radd__(self, other): # other + self
    return self + other

  def __rsub__(self, other): # other - self
    return other + (-self)

  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = xp.ones_like(self.data, dtype=self.dtype)
    for node in reversed(topo):
      node._backward()

  def crossEntropyLoss(self, targets):
    """
    logits: Tensor of shape (batch_size, num_classes)
    targets: array of shape (batch_size,) containing integer class indices
    """
    # for numerical stability substract the max element, it changes nothing fo the end result
    logits = self.data
    shifted_logits = logits - xp.max(logits, axis = 1, keepdims = True)
    batch_size = logits.shape[0]

    logsumexp = xp.log(xp.sum(xp.exp(shifted_logits), axis=1))
    losses = logsumexp - shifted_logits[xp.arange(batch_size), targets]
    data_loss = xp.mean(losses)
    out = Tensor(data_loss, self.requires_grad, (self,), "cross_entropy")

    def _backward():
      if self.requires_grad:
        exp_shifted = xp.exp(shifted_logits)
        probs = exp_shifted / xp.sum(exp_shifted, axis=1, keepdims=True)

        dlogits = probs
        dlogits[xp.arange(batch_size), targets] -= 1.0
        dlogits /= batch_size
        self.grad += out.grad * dlogits
    out._backward = _backward
    return out

def unbroadcast(in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> tuple[int, ...]:
  if len(in_shape) > len(out_shape):
    raise ValueError(f"Unbroadcasting is impossible between the input " \
                     f"{in_shape} and the output {out_shape}")
  added_dims = len(out_shape) - len(in_shape)
  axes = []
  for i in range(added_dims):
    axes.append(i)
  for i in range(len(in_shape)):
    if out_shape[i + added_dims] != in_shape[i]:
      if in_shape[i] == 1:
        axes.append(i+added_dims)
      else:
        raise ValueError(f"Unbroadcasting is impossible between the input " \
                         f"{in_shape} and the output {out_shape}")
  return tuple(axes)


class Module():
  def zero_grad(self):
    for p in self.parameters():
      p.grad = xp.zeros_like(p.grad)
  def parameters(self):
    return []


class Linear(Module):
  def __init__(self, nin, nout):
    self.b = Tensor(xp.zeros(nout,), True)
    self.w = Tensor(xp.random.normal(0.0, xp.sqrt(2/nin), (nin,nout)), True)

  def __call__(self, x):
    return x @ self.w + self.b

  def parameters(self):
    return [self.w, self.b]

class MLP(Module):
  def __init__(self, nin: int, nouts: list[int]):
    sz = [nin] + nouts
    self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
      x = x.leaky_relu() if layer is not self.layers[-1] else x
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]


class AdamW(Module):
  def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
    self.parameters = parameters
    self.alpha = lr
    self.beta1, self.beta2 = betas
    self.lambda_wd = weight_decay
    self.eps = eps
    self.t = 0

    self.m = [xp.zeros_like(p.data, dtype=float) for p in parameters]
    self.v = [xp.zeros_like(p.data, dtype=float) for p in parameters]

  def step(self):

      self.t += 1
      bc1 = 1 - self.beta1 ** self.t
      bc2 = 1 - self.beta2 ** self.t

      for i, p in enumerate(self.parameters):
        g = p.grad
        p.data -= self.alpha * self.lambda_wd * p.data

        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

        denom = xp.sqrt(self.v[i] / bc2) + self.eps
        p.data -= (self.alpha / bc1) * (self.m[i] / denom)




import numpy as np
import random

class Tensor:
    def __init__(self, data, requires_grad = True, _children=(), _op=""):
        if isinstance(data, (int, float, list)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError("Invalid data type")

        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"tensor({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, True, (self, other), '+')

        def _backward():
            def get_grad_to_add(child_data_shape, incoming_grad):
                if child_data_shape != incoming_grad.shape:
                    return np.sum(incoming_grad, axis=0, keepdims=True)
                return incoming_grad

            if self.requires_grad:
                self.grad += get_grad_to_add(self.data.shape, out.grad)

            if other.requires_grad:
                other.grad += get_grad_to_add(other.data.shape, out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, True, (self, other), '*')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other.data
            if other.requires_grad:
                other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, True, (self,), f'**{other}')

        def _backward():
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, True, (self, ), 'tanh')
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), True, (self,), "sum")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.where(self.data > 0, self.data, 0.01*self.data), True, (self,), 'ReLU')

        def _backward():
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, 0.01) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        if self.shape[-1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        out = Tensor (np.matmul(self.data, other.data), True, (self, other), "matmul")
        def _backward():
            if self.requires_grad:
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward
        return out

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
        self.grad[...] = 1.0
        for node in reversed(topo):
            node._backward()


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad[...] = 0

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, nin, nout):
        std = np.sqrt(2. / nin)
        self.w = Tensor([[random.gauss(0, std) for _ in range(nout)] for _ in range(nin)])
        self.b = Tensor(np.zeros((1, nout)))

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for linear in self.layers:
            x = linear(x)
            x = x.relu() if linear != self.layers[-1] else x
        return x

    def parameters(self):
        return [p for linear in self.layers for p in linear.parameters()]

class WAdam(Module):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        self.parameters = parameters
        self.alpha = lr
        self.beta1, self.beta2 = betas
        self.lambda_wd = weight_decay
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.data, dtype=float) for p in parameters]
        self.v = [np.zeros_like(p.data, dtype=float) for p in parameters]

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None or np.all(p.grad == 0):
                continue

            g_t = p.grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g_t ** 2)

            beta1_t = self.beta1 ** self.t
            beta2_t = self.beta2 ** self.t

            m_hat = self.m[i] / (1 - beta1_t)
            v_hat = self.v[i] / (1 - beta2_t)


            adaptive_term = m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= self.alpha * adaptive_term

            p.data -= self.alpha * self.lambda_wd * p.data

    def zero_grad(self):
        for p in self.parameters:
            p.grad[...] = 0




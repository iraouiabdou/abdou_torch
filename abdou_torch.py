import numpy as np
import random

class Tensor:
    def __init__(self, data, requires_grad = True, _children=(), _op=""):
        # Convert different input types into a NumPy float64 array for numerical stability
        if isinstance(data, (int, float, list)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError("Invalid data type")
        # Shape tracking for matrix operations and broadcasting logic
        self.shape = self.data.shape
        # Initialize gradient with zeros, same shape as data. Gradients accumulate here during backprop
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        # Internal variables to track the graph structure (children nodes and the operation that created this one)
        self._prev = set(_children)
        self._op = _op
        # Placeholder for the function that will calculate the local gradient and propagate it backwards
        self._backward = lambda: None
        # Flag to determine if we should compute gradients for this specific tensor (e.g., False for inputs)
        self.requires_grad = requires_grad

    def __repr__(self):
        # String representation for debugging tensor values
        return f"tensor({self.data})"

    def __add__(self, other):
        # Ensure 'other' is a Tensor
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, True, (self, other), '+')

        def _backward():
            # Helper to handle NumPy broadcasting: if shapes don't match, we must sum gradients across axes
            # This is because a broadcasted value contributed to multiple elements in the output
            def get_grad_to_add(child_data_shape, incoming_grad):
                if child_data_shape != incoming_grad.shape:
                    # Summing across the broadcasted dimension to restore original shape
                    return np.sum(incoming_grad, axis=0, keepdims=True)
                return incoming_grad

            # The derivative of (a + b) with respect to a is 1.0, and with respect to b is 1.0
            # By chain rule: child.grad += output.grad * 1.0
            if self.requires_grad:
                self.grad += get_grad_to_add(self.data.shape, out.grad)
            if other.requires_grad:
                other.grad += get_grad_to_add(other.data.shape, out.grad)
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        # Element-wise multiplication
        out = Tensor(self.data * other.data, True, (self, other), '*')

        def _backward():
            # Product Rule: d/dx (f*g) = f'g + fg'
            # Here, the local gradient for 'self' is 'other.data' and vice-versa
            if self.requires_grad:
                self.grad += out.grad * other.data
            if other.requires_grad:
                other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        # Power rule implementation, restricting 'other' to scalars for simplicity
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, True, (self,), f'**{other}')

        def _backward():
            # Power Rule: d/dx (x^n) = n * x^(n-1)
            # Multiply by out.grad to chain the gradients from further down the graph
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad
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

    def tanh(self):
        # Hyperbolic tangent activation: maps values to range (-1, 1)
        t = np.tanh(self.data)
        out = Tensor(t, True, (self, ), 'tanh')
        def _backward():
            if self.requires_grad:
                # Derivative of tanh(x) is (1 - tanh(x)^2)
                self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        # Reduces tensor to a single scalar (0-d array)
        out = Tensor(np.sum(self.data), True, (self,), "sum")
        def _backward():
            if self.requires_grad:
                # The gradient of a sum is 1.0 for every input element
                # We broadcast the scalar out.grad back to the input shape
                self.grad += out.grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def relu(self):
        # Leaky ReLU implementation (alpha=0.01) to prevent "dead neurons"
        out = Tensor(np.where(self.data > 0, self.data, 0.01*self.data), True, (self,), 'ReLU')

        def _backward():
            if self.requires_grad:
                # Local derivative is 1 for positive values, and 0.01 for negative ones
                self.grad += np.where(self.data > 0, 1, 0.01) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        # Standard matrix multiplication (Dot product for 2D tensors)
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        # Check for dimension compatibility
        if self.shape[-1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        out = Tensor (np.matmul(self.data, other.data), True, (self, other), "matmul")
        def _backward():
            # Chain rule for matrix multiplication:
            # If Y = X @ W, then dL/dX = dL/dY @ W.T and dL/dW = X.T @ dL/dY
            if self.requires_grad:
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward
        return out

    def backward(self):
        # Build a topological ordering of the graph so we process nodes in the correct order
        # No node is processed before all its "dependents" (nodes it flows into) are processed
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

# Base class for all neural network modules, similar to torch.nn.Module
class Module:
    def zero_grad(self):
        # Resets all parameter gradients to zero, essential before a new backward pass
        for p in self.parameters():
            p.grad[...] = 0

    def parameters(self):
        # To be overridden by subclasses to return a list of Tensors (weights/biases)
        return []

class Linear(Module):
    def __init__(self, nin, nout):
        # He initialization: standard for ReLU networks to keep variance consistent
        std = np.sqrt(2. / nin)
        # Weights are initialized from a Gaussian distribution
        self.w = Tensor([[random.gauss(0, std) for _ in range(nout)] for _ in range(nin)])
        # Biases are initialized as zeros
        self.b = Tensor(np.zeros((1, nout)))

    def __call__(self, x):
        # The classic linear transformation: y = xW + b
        return x @ self.w + self.b

    def parameters(self):
        # Expose weights and biases for the optimizer to find
        return [self.w, self.b]

class MLP(Module):
    def __init__(self, nin, nouts):
        # Define the layer structure (input size followed by list of output sizes)
        sz = [nin] + nouts
        # List comprehension to instantiate multiple Linear layers
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # Sequential forward pass through each layer
        for linear in self.layers:
            x = linear(x)
            # Apply ReLU activation to all layers except the last output layer
            x = x.relu() if linear != self.layers[-1] else x
        return x

    def parameters(self):
        # Flatten the parameters of all sub-layers into a single list
        return [p for linear in self.layers for p in linear.parameters()]

# Implementation of the AdamW optimizer (Adam with Decoupled Weight Decay)
class WAdam(Module):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        self.parameters = parameters
        self.alpha = lr # Learning rate
        self.beta1, self.beta2 = betas # Decay rates for first and second moments
        self.lambda_wd = weight_decay # Weight decay coefficient
        self.eps = eps # Small constant for numerical stability (avoid divide by zero)
        self.t = 0 # Timestep counter

        # m: First moment vector (Moving average of gradients - "momentum")
        # v: Second moment vector (Moving average of squared gradients - "RMSProp logic")
        self.m = [np.zeros_like(p.data, dtype=float) for p in parameters]
        self.v = [np.zeros_like(p.data, dtype=float) for p in parameters]

    def step(self):
        self.t += 1 # Increment timestep for bias correction

        for i, p in enumerate(self.parameters):
            # Skip parameters that didn't receive a gradient calculation
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
        # Clear gradients in the optimizer to prepare for the next batch
        for p in self.parameters:
            p.grad[...] = 0




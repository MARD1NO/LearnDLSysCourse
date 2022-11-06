"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, nonlinearity="relu", dtype=dtype))
        self.use_bias = bias
        if bias: 
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, nonlinearity="relu", dtype=dtype), (1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        out_shape_len = len(out.shape)
        broadcast_shape = []
        for i in range(out_shape_len): 
            broadcast_shape.append(out.shape[i])
        if self.use_bias: 
            broadcast_bias = ops.broadcast_to(self.bias, shape=tuple(broadcast_shape))
            out = ops.add(out, broadcast_bias)
        return out 
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules: 
            x = module(x)
        return x 
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0]
        class_num = logits.shape[1]

        one_hot_label = init.one_hot(class_num, y)
        zi = logits * one_hot_label
        zi = ops.summation(zi, axes=1)

        softmax = ops.LogSumExp(axes=1)(logits)
        loss= softmax-zi
        # reduce mean 
        loss = ops.summation(loss) / batch_size
        return loss 
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        if self.training: 
            mean_val = ops.divide_scalar(ops.summation(x, 0), batch_size)
            broadcast_mean_val = ops.broadcast_to(ops.reshape(mean_val, (1, -1)), x.shape)
            var_val = ops.divide_scalar(ops.summation(ops.power_scalar(x - broadcast_mean_val, 2), 0), batch_size)
            broadcast_var_val = ops.broadcast_to(ops.reshape(var_val, (1, -1)), x.shape)
            broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
            broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
            out = broadcast_weight * (x - broadcast_mean_val) / ops.power_scalar(broadcast_var_val + self.eps, 0.5) + broadcast_bias
            # Update Running Mean and Variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_val 
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_val 
        else: 
            broadcast_mean_val = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape)
            broadcast_var_val = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape)
            broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
            broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
            out = broadcast_weight * (x - broadcast_mean_val) / ops.power_scalar(broadcast_var_val + self.eps, 0.5) + broadcast_bias
        return out
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean_val = ops.divide_scalar(ops.summation(x, 1), self.dim)
        broadcast_mean_val = ops.broadcast_to(ops.reshape(mean_val, (-1, 1)), x.shape)
        var_val = ops.divide_scalar(ops.summation(ops.power_scalar(x - broadcast_mean_val, 2), 1), self.dim)
        broadcast_var_val = ops.broadcast_to(ops.reshape(var_val, (-1, 1)), x.shape)
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        out = broadcast_weight * (x - broadcast_mean_val) / ops.power_scalar(broadcast_var_val + self.eps, 0.5) + broadcast_bias
        return out 
        ### END YOUR SOLUTION

class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        random_mask = init.randb(*x.shape, p=self.p) / (1.0 - self.p)
        return x * random_mask
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.EWiseAdd()(x,self.fn(x))
        ### END YOUR SOLUTION


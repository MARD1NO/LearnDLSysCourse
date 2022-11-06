"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params: 
            grad = self.u.get(param, 0) * self.momentum + (1 - self.momentum) * (param.grad + self.weight_decay * param.data)
            grad = ndl.Tensor(grad, dtype=param.dtype)
            self.u[param] = grad 
            param.data -= self.lr * grad 
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params: 
            grad_with_wd = param.grad + self.weight_decay * param.data
            new_m = self.m.get(param, 0) * self.beta1 + (1 - self.beta1) * grad_with_wd
            new_v = self.v.get(param, 0) * self.beta2 + (1 - self.beta2) * grad_with_wd * grad_with_wd
            self.m[param] = new_m
            self.v[param] = new_v
            m_with_bias_corr = new_m / (1 - self.beta1 ** self.t)
            v_with_bias_corr = new_v / (1 - self.beta2 ** self.t)

            update = self.lr * (m_with_bias_corr) / (v_with_bias_corr ** 0.5 + self.eps)
            update = ndl.Tensor(update, dtype=param.dtype)
            param.data -= update
        ### END YOUR SOLUTION

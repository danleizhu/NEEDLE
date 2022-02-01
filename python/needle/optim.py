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
        self.delta = {}
        self.weight_decay = weight_decay

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped

    def step(self):
        for p in self.params:
            if p not in self.delta:
                self.delta[p] = 0.0
            if p.grad is not None and p.requires_grad:
                velocity = self.momentum * self.delta[p] + p.grad.data + self.weight_decay * p.data
                p.data = p.data - self.lr * velocity
                self.delta[p] = velocity


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        bias_correction=True,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            if p not in self.m:
                self.m[p] = 0.0
            if p not in self.v:
                self.v[p] = 0.0
            if p.grad is not None and p.requires_grad:
                m = self.beta1 * self.m[p] + (1 - self.beta1) * (p.grad.data + self.weight_decay * p.data)
                v = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad.data + self.weight_decay * p.data) ** 2
                self.m[p] = m
                self.v[p] = v
                if self.bias_correction:
                    m = m / (1 - self.beta1 ** self.t)
                    v = v / (1 - self.beta2 ** self.t)
                p.data = p.data - self.lr * (m / (v ** 0.5 + self.eps))

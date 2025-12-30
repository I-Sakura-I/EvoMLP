import numpy as np
from .layers import Layer
class ReLU(Layer):
    def forward(self, x, training=True): self.x = x; return np.maximum(0, x)
    def backward(self, grad): return grad * (self.x > 0)
class Linear(Layer):
    def forward(self, x, training=True): return x
    def backward(self, grad): return grad
class PReLU(Layer):
    def __init__(self, init=0.25):
        self.alpha = init
        self.grad_alpha = 0
        self.x = None
    def forward(self, x, training=True):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)
    def backward(self, grad):
        self.grad_alpha = np.sum(np.where(self.x > 0, 0, self.x) * grad)
        return np.where(self.x > 0, 1, self.alpha) * grad
    def update_params(self, lr):
        self.alpha -= lr * self.grad_alpha

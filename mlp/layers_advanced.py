import numpy as np
from .layers import Layer, Dense

class ResidualBlock(Layer):
    def __init__(self, dim, activation_cls):
        self.dense1 = Dense(dim, dim)
        self.act = activation_cls()
        self.dense2 = Dense(dim, dim)
        
        # [小技巧] 将第二个全连接层初始化为非常小
        # 这样初始状态下 ResidualBlock 几乎就是恒等映射 Identity，极其稳定
        self.dense2.weights *= 0.01 
        self.input_cache = None

    def forward(self, x, training=True):
        self.input_cache = x
        out = self.dense1.forward(x, training)
        out = self.act.forward(out, training)
        out = self.dense2.forward(out, training)
        
        # 乘以 5 可以显著提高深层网络的收敛速度
        return x + 5 * out

    def backward(self, grad):
        # 对应前向传播，反向传播时梯度也要乘以 5
        grad_path = self.dense2.backward(grad)
        grad_path = self.act.backward(grad_path)
        grad_path = self.dense1.backward(grad_path)
        
        # Identity path + Residual path (scaled)
        return grad + 5 * grad_path

    def update_params(self, lr):
        self.dense1.update_params(lr)
        self.dense2.update_params(lr)

class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    def forward(self, x, training=True):
        if not training: return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        return x * self.mask
    def backward(self, grad): return grad * self.mask

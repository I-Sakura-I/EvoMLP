import numpy as np

class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def update_params(self, lr): pass

class Dense(Layer):
    def __init__(self, n_in, n_out, weight_initializer='he'):
        # 初始权重稍微小一点点，防止初始输出过大
        self.weights = np.random.randn(n_in, n_out) * (np.sqrt(2/n_in) if weight_initializer=='he' else 0.01)
        self.biases = np.zeros((1, n_out))
        self.x = None
        self.dw, self.db = None, None

    def forward(self, x, training=True):
        self.x = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad):
        self.dw = np.dot(self.x.T, grad) / self.x.shape[0]
        self.db = np.mean(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)

    def update_params(self, lr):
        # === [修复关键点 1] 梯度裁剪 ===
        # 将梯度强制限制在 [-1.0, 1.0] 范围内
        # 这样即使梯度计算出来是 10000，也只更新 1.0，保证稳定
        self.dw = np.clip(self.dw, -1.0, 1.0)
        self.db = np.clip(self.db, -1.0, 1.0)
        
        self.weights -= lr * self.dw
        self.biases -= lr * self.db

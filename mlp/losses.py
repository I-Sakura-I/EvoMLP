import numpy as np
class MSE:
    def forward(self, pred, true): return np.mean((pred - true.reshape(pred.shape))**2)
    def backward(self, pred, true): return 2*(pred - true.reshape(pred.shape))/pred.shape[0]

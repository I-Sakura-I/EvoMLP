import numpy as np
class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None 
    def add_layer(self, layer): self.layers.append(layer)
    def set_loss(self, loss): self.loss = loss
    def set_optimizer(self, opt): self.optimizer = opt
    def predict(self, x):
        out = x
        for l in self.layers: out = l.forward(out, training=False)
        return out
    def train(self, X, y, epochs, batch_size, validation_data=None):
        history = {'train_loss': [], 'val_loss': []}
        n = X.shape[0]
        for ep in range(epochs):
            indices = np.random.permutation(n)
            X, y = X[indices], y[indices]
            ep_loss = 0
            for i in range(0, n, batch_size):
                X_b, y_b = X[i:i+batch_size], y[i:i+batch_size]
                # Forward
                out = X_b
                for l in self.layers: out = l.forward(out, training=True)
                # Loss
                ep_loss += self.loss.forward(out, y_b)
                # Backward
                grad = self.loss.backward(out, y_b)
                for l in reversed(self.layers): grad = l.backward(grad)
                # Update
                for l in self.layers: 
                    l.update_params(self.optimizer.lr)
            
            history['train_loss'].append(ep_loss / (n//batch_size))
            if validation_data:
                X_v, y_v = validation_data
                out_v = self.predict(X_v)
                history['val_loss'].append(self.loss.forward(out_v, y_v))
            
            if ep % 20 == 0: print(f"Epoch {ep}: Loss {history['train_loss'][-1]:.4f}")
        return history

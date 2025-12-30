import numpy as np
from sklearn.datasets import make_circles
def make_circles_data(n_samples=1000, noise=0.1, factor=0.3):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    # X = (X - X.mean(0)) / X.std(0) # Standardize if needed
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]

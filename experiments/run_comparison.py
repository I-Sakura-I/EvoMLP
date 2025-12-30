import sys, os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from mlp.datasets import make_circles_data
from mlp.model import MLP
from mlp.layers import Dense
from mlp.layers_advanced import ResidualBlock
from mlp.activations import ReLU, PReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam

def build_model(kind='mlp', depth=5, dim=32):
    m = MLP()
    m.add_layer(Dense(2, dim))
    m.add_layer(ReLU() if kind=='mlp' else PReLU())
    for _ in range(depth):
        if kind == 'mlp':
            m.add_layer(Dense(dim, dim))
            m.add_layer(ReLU())
        else:
            m.add_layer(ResidualBlock(dim, PReLU))
    m.add_layer(Dense(dim, 1))
    m.add_layer(Linear())
    return m

def main():
    # 增加噪声让任务更难，更能体现深层网络优势
    X_train, y_train, X_test, y_test = make_circles_data(1000, 0.1, 0.3)
    
    print("1. Training Standard MLP (Deep)...")
    mlp = build_model('mlp', depth=8)
    mlp.set_loss(MSE())
    mlp.set_optimizer(Adam(0.005)) # 普通 MLP 保持原学习率
    h1 = mlp.train(X_train, y_train, 150, 32, (X_test, y_test))
    
    print("2. Training EvoMLP (ResNet)...")
    evo = build_model('evo', depth=8)
    evo.set_loss(MSE())
    evo.set_optimizer(Adam(0.002)) 
    h2 = evo.train(X_train, y_train, 150, 32, (X_test, y_test))
    
    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(h1['val_loss'], label='Standard MLP')
    plt.plot(h2['val_loss'], label='EvoMLP (ResNet)')
    plt.title('Validation Loss: Deep MLP vs EvoMLP')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/loss_curve.png')
    print("Done. Results saved to results/loss_curve.png")

if __name__ == "__main__": main()

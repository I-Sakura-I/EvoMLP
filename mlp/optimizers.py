import numpy as np
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1, self.beta2 = beta1, beta2
        self.m, self.v = {}, {}
        self.t = 0
        self.eps = 1e-8
    def update(self, layer): 
        pass 

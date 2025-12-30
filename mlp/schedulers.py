class StepLR:
    def __init__(self, optimizer, step_size=50, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            print(f"LR decayed to {self.optimizer.lr}")

import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt


class NeuralBursterAS():
    def __init__(self, dt, tau, alpha, beta, lmbd):
        self.v = 0.1 * np.random.rand()
        self.h = np.random.rand()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.lmbd = lmbd
        self.dt = dt
        self.v_history = deque()
        self.h_history = deque()
        pass

    def rhs(self):
        rhs_v = self.alpha * self.v - self.beta * self.v ** 3 - self.h + 4
        rhs_h = (self.lmbd * self.v + 7 - self.h) / self.tau
        return rhs_v, rhs_h

    def step(self):
        rhs_vals = self.rhs()
        new_v = self.v + self.dt * rhs_vals[0]
        new_h = self.h + self.dt * rhs_vals[1]
        self.v = new_v
        self.h = new_h
        self.v_history.append(deepcopy(new_v))
        self.h_history.append(deepcopy(new_h))
        return None

    def run(self, T_steps):
        for i in range(T_steps):
            self.step()
        return None

    def plot_history(self):
        fig = plt.figure(figsize=(10, 4))
        v = np.array(self.v_history)
        h = np.array(self.h_history)
        t = self.dt * np.arange(len(v))
        plt.plot(t, v, linewidth=3, color='r', label='v')
        plt.plot(t, h, linewidth=3, color='b', label='h')
        plt.xlabel("t, ms", fontsize = 24)
        plt.ylabel("v, units", fontsize = 24)
        plt.grid(True)
        plt.legend(fontsize = 24, loc = 1)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        return fig

if __name__ == '__main__':
        dt = 0.1
        tau = 5000
        alpha = 0.1
        beta = 0.001
        lmbd = 1
        T_steps = int(10000 / dt)
        burster = NeuralBursterAS(dt, tau, alpha, beta, lmbd)
        burster.run(T_steps)
        fig = burster.plot_history()
        plt.show()

import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from copy import deepcopy
import os

class NeuralNode():
    def __init__(self, dt, a, b, c, d):
        self.h = -15
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.history_h = deque()

    def rhs(self, input):
        rhs_h = self.a * self.h**3 + self.b*self.h**2 + self.c*self.h + self.d + input
        return rhs_h

    def step(self, input):
        rhs_h = self.rhs(input)
        h_next = self.h + self.dt * rhs_h
        self.h = h_next
        self.history_h.append(deepcopy(self.h))
        return None

    def run(self, T, input):
        for i in range(int(T/self.dt)):
            self.step(input)

        return None

    def reset_history(self):
        self.history_h = deque()
        return None


if __name__ == '__main__':
    a = -0.01
    b = 0
    c = 0.72
    d = -2.4
    dt = 0.1
    neural_population = NeuralNode(dt, a,b,c,d)
    T1 = 100
    dT = 1
    stim = 20
    input = 0
    neural_population.run(T1, input)
    neural_population.reset_history()
    neural_population.run(T1, input)
    input = stim
    neural_population.run(dT, input)
    input = 0
    neural_population.run(T1, input)

    h_array = np.array(neural_population.history_h)
    input_array = np.hstack([np.zeros(int(T1/dt)), stim * np.ones(int(dT/dt)), np.zeros(int(T1/dt))])
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), gridspec_kw={'height_ratios': [4, 1]})
    axes[0].plot(h_array, color = 'b', linewidth = 2, label = "Neural node response")
    axes[0].axis('off')
    axes[1].plot(input_array, color = 'r', linewidth = 2, label = "input stimulus")
    axes[1].axis('off')
    plt.subplots_adjust(hspace = 0)
    axes[0].axis('off')
    axes[0].legend(fontsize=24)
    axes[1].legend(fontsize=24)
    plt.savefig(os.path.join("../", 'img', 'plateau_potentials.png'))
    plt.show()


    #phase plane
    x = np.linspace(-12, 9, 1000)
    f = a * x**3 + b*x**2 + c*x + d
    # plt.scatter(h_array, np.zeros_like(h_array), )
    plt.plot(x, f, linewidth = 2, color = 'skyblue', label="rhs(h)")
    plt.plot(x, np.zeros_like(x), linewidth = 2, color = 'orange')
    # plt.grid(True)
    plt.legend(fontsize = 16)
    plt.xlabel("h", fontsize=16)
    # plt.ylabel("\frac{dh}{dt}", fontsize=16)
    # plt.axis('off')
    # for ax in axes:
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join("../", 'img', 'plateau_potentials_phase_plane.png'))
    plt.show()



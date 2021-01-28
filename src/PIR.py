import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from copy import deepcopy
import os

class PIR_node():
    def __init__(self, dt, a, b, c, d, alpha, beta):
        self.h = -10
        self.u = np.random.rand()
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.history_h = deque()
        self.history_u = deque()

    def rhs(self, input):
        rhs_h = (self.a * self.h**3 + self.b*self.h**2 + self.c*self.h + d - self.u + input)
        rhs_u = self.alpha * (self.beta * self.h - self.u)
        return rhs_h, rhs_u

    def step(self, input):
        rhs_h, rhs_u = self.rhs(input)
        h_next = self.h + self.dt * rhs_h
        u_next = self.u + self.dt * rhs_u
        self.h = h_next
        self.u = u_next
        self.history_h.append(deepcopy(self.h))
        self.history_u.append(deepcopy(self.u))
        return None

    def run(self, T, input):
        for i in range(int(T/self.dt)):
            self.step(input)
        return None

    def reset_history(self):
        self.history_h = deque()
        self.history_u = deque()
        return None


if __name__ == '__main__':
    a = -0.05
    b = 0
    c = 3.5
    d = -2.5
    alpha = 0.05
    beta = 2.5
    dt = 0.1
    neural_population = PIR_node(dt, a, b, c, d, alpha, beta)

    T1 = 100
    dT = 5
    inhibition = -5
    input = 0
    neural_population.run(T1, input)
    neural_population.reset_history()
    neural_population.run(T1, input)
    input = inhibition
    neural_population.run(dT, input)
    input = 0
    neural_population.run(T1, input)

    h_array = np.array(neural_population.history_h)
    u_array = np.array(neural_population.history_u)
    input_array = np.hstack([np.zeros(int(T1 / dt)), inhibition * np.ones(int(dT / dt)), np.zeros(int(T1 / dt))])
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), gridspec_kw={'height_ratios': [4, 1]})
    axes[0].plot(h_array, color='b', linewidth=2, label="Neural node response")
    axes[0].axis('off')
    axes[1].plot(input_array, color='r', linewidth=2, label="input stimulus")
    axes[1].axis('off')
    plt.subplots_adjust(hspace=0)
    axes[0].legend(fontsize=24)
    axes[1].legend(fontsize=24)
    plt.savefig(os.path.join("../", 'img', 'PIR.png'))
    plt.show()

    # plotting the phase plane
    x = np.linspace(-10, 10, 1000)
    u_null_h = a * x**3 + b*x**2 + c*x + d
    u_null_h_inh = a * x**3 + b*x**2 + c*x + d + inhibition
    u_null_u = beta * x

    plt.plot(x, u_null_h, linewidth = 2, color = 'skyblue', label="h-nullcline")
    plt.plot(x, u_null_h_inh, linewidth = 2, color = 'skyblue', linestyle = '--', label="h-nullcline after inhibition")
    plt.plot(x, u_null_u, linewidth = 2, color = 'orange', label="u-nullcline")
    plt.plot(h_array[::5], u_array[::5],  color = 'k', alpha = 0.5)
    # plt.grid(True)
    plt.legend(fontsize = 16)
    plt.xlabel("h", fontsize=16)
    plt.ylabel("u", fontsize=16)
    # plt.axis('off')
    plt.savefig(os.path.join("../", 'img', 'PIR_phase_plane.png'))
    plt.show()

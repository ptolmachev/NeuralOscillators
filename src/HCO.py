import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt

def firing_rate(v):
    return 1.0 / (1 + np.exp(-0.3 * v))

class HCO():
    def __init__(self, dt, weights, drives, tau):
        self.dt = dt
        self.state = np.hstack([np.random.rand(2), 0.2 + 0.2 * np.random.rand(2)])
        self.state_history = deque()
        self.drives = drives
        self.W = weights
        self.tau = tau

    def rhs(self):
        v1, v2, m1, m2 = self.state
        rhs_v_1 = (-0.01 * v1 - m1 + self.drives[0] - self.W[1] * firing_rate(v2))
        rhs_v_2 = (-0.01 * v2 - m2 + self.drives[1] - self.W[0] * firing_rate(v1))
        rhs_m_1 = (firing_rate(v1) - m1) / self.tau
        rhs_m_2 = (firing_rate(v2) - m2) / self.tau
        return np.array([rhs_v_1, rhs_v_2, rhs_m_1, rhs_m_2])

    def get_next_state(self):
        state = self.state + self.dt * self.rhs()
        return state

    def get_state_history(self):
        return np.array(self.state_history)

    def update_history(self):
        self.state_history.append(deepcopy(self.state))
        return None

    def step(self):
        self.state += self.dt * self.rhs()
        self.update_history()
        return None

    def run(self, T_steps):
        for i in range(T_steps):
            self.step()
        return None

    def plot_history(self):
        hist_data = np.array(self.state_history)
        v1 = hist_data[:, 0]
        v2 = hist_data[:, 1]
        m1 = hist_data[:, 2]
        m2 = hist_data[:, 3]
        fr1 = firing_rate(v1)
        fr2 = firing_rate(v2)
        t = np.arange(len(fr1)) * self.dt
        fig, axes = plt.subplots(2, 1, figsize = (10,4))
        axes[0].plot(t, fr1, color='r', linewidth = 3, label = 'firing rate, right neuron')
        axes[0].plot(t, m1, color='b', linewidth=3, label='adaptation variable')
        axes[0].legend(fontsize = 24, loc = 1)
        axes[0].set_ylim([-0.01,1.1])
        axes[0].grid(True)
        # axes[0].set_ylabel("Firing rate", fontsize = 24)

        axes[1].plot(t, fr2, color='r', linewidth=3, label='firing rate, left neuron')
        axes[1].plot(t, m2, color='b', linewidth=3, label='adaptation variable')
        axes[1].legend(fontsize = 24, loc = 1)
        axes[1].set_ylim([-0.01, 1.1])
        axes[1].grid(True)
        # axes[1].set_ylabel("Firing rate", fontsize = 24)
        axes[1].set_xlabel("t, ms", fontsize = 24)
        plt.subplots_adjust(wspace=0, hspace=0)


        plt.suptitle("Half Centre Oscillator", fontsize = 24)
        plt.show(block=True)
        return None


if __name__ == '__main__':
    dt = 1
    weights = np.array([0.8, 0.6])
    drives = 0.3 * np.ones(2)
    tau = 5000
    T_steps = int(30000/dt)
    hco = HCO(dt, weights, drives, tau)
    hco.run(T_steps)
    hco.plot_history()

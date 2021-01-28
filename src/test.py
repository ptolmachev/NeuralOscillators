import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt

def firing_rate(v):
    return 1.0 / (1 + np.exp(-v))


class TEST_oscillator():
    def __init__(self, dt, alpha, weights_e, weights_i, drives, tau):
        self.dt = dt
        self.alpha = alpha
        self.state = np.hstack([np.random.rand(6)])
        self.state_history = deque()
        self.drives = drives
        self.W_e = weights_e
        self.W_i = weights_i
        self.tau = tau
        self.past_vals = np.zeros(2)

    def rhs(self):
        v1_i, v2_i, v1_e, v2_e, m1, m2 = self.state
        v1_e_past = self.past_vals[0]
        v2_e_past = self.past_vals[1]
        rhs_v1_i = (-self.alpha * v1_i + self.drives[0]
                    - m1 # adaptation of firing rate
                    + self.W_e[0] * firing_rate(v1_e_past) # excitation from the support
                    - self.W_i[1] * firing_rate(v2_i)) # inhibition from the opposing neuron
        rhs_v2_i = (-self.alpha * v2_i + self.drives[1] - m2
                    + self.W_e[1] * firing_rate(v2_e_past)
                    - self.W_i[0] * firing_rate(v1_i))
        rhs_v1_e = (-self.alpha * v1_e + self.drives[0]
                    - self.W_i[1] * firing_rate(v2_i))
        rhs_v2_e = (-self.alpha * v2_e + self.drives[1]
                    - self.W_i[0] * firing_rate(v1_i))
        rhs_m1 = (firing_rate(v1_i) - m1) / self.tau
        rhs_m2 = (firing_rate(v2_i) - m2) / self.tau
        return np.array([rhs_v1_i, rhs_v2_i, rhs_v1_e, rhs_v2_e, rhs_m1, rhs_m2])

    def get_next_state(self):
        state = self.state + self.dt * self.rhs()
        return state

    def get_state_history(self):
        return np.array(self.state_history)

    def update_history(self):
        self.state_history.append(deepcopy(self.state))
        self.past_vals = self.past_vals * np.exp(-self.dt/tau) + (deepcopy(self.state[2:4]))
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
        v1_i = hist_data[:, 0]
        v2_i = hist_data[:, 1]
        v1_e = hist_data[:, 2]
        v2_e = hist_data[:, 3]
        m1 = hist_data[:, 4]
        m2 = hist_data[:, 5]
        fr1 = firing_rate(v1_i)
        fr2 = firing_rate(v2_i)
        t = np.arange(len(v1_i)) * self.dt
        fig, axes = plt.subplots(4, 1, figsize = (10, 4))
        axes[0].plot(t, fr1, color='r', linewidth=3, label='fr, right neuron')
        # axes[0].plot(t, v1_i, color='r', linewidth = 3, label = 'fr, right neuron')
        # axes[0].plot(t, m1, color='b', linewidth=3, label='m, right neuron')
        axes[0].legend(fontsize = 24, loc = 1)
        # axes[0].set_ylim([-0.01,1.1])
        axes[0].grid(True)
        axes[0].set_ylabel("fr", fontsize = 24)

        axes[1].plot(t, fr2, color='r', linewidth=3, label='fr, left neuron')
        # axes[1].plot(t, v2_i, color='r', linewidth=3, label='fr, left neuron')
        # axes[1].plot(t, m2, color='b', linewidth=3, label='m, right neuron')
        axes[1].legend(fontsize = 24, loc = 1)
        # axes[1].set_ylim([-0.01, 1.1])
        axes[1].grid(True)
        axes[1].set_ylabel("fr", fontsize = 24)
        axes[1].set_xlabel("t, ms", fontsize = 24)
        plt.subplots_adjust(wspace=0, hspace=0)

        axes[2].plot(t, v1_e, color='r', linewidth=3, label='v e, right neuron')
        axes[2].legend(fontsize = 24, loc = 1)
        # axes[1].set_ylim([-0.01, 1.1])
        axes[2].grid(True)
        axes[2].set_ylabel("v", fontsize = 24)
        axes[2].set_xlabel("t, ms", fontsize = 24)
        plt.subplots_adjust(wspace=0, hspace=0)

        axes[3].plot(t, v2_e, color='r', linewidth=3, label='v e, left neuron')
        axes[3].legend(fontsize = 24, loc = 1)
        # axes[1].set_ylim([-0.01, 1.1])
        axes[3].grid(True)
        axes[3].set_ylabel("v", fontsize = 24)
        axes[3].set_xlabel("t, ms", fontsize = 24)
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.suptitle("Neural oscillations on slow mixed synaptic transmision", fontsize = 24)
        plt.show(block=True)
        return None


if __name__ == '__main__':
    dt = 1
    weights_i =  np.array([0.4, 0.2])
    weights_e = np.array([0.3, 0.3])
    drives = 0.3 * np.ones(2)
    tau = 20000
    alpha = 0.01

    T_steps = int(200000/dt)
    sso = TEST_oscillator(dt, alpha, weights_e, weights_i, drives, tau)
    sso.run(T_steps)
    sso.plot_history()

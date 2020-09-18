import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt

def firing_rate(v):
    return 1.0 / (1 + np.exp(-v))

class SlowSynapticOscillator():
    def __init__(self, dt, alpha, weights, drives, tau):
        self.dt = dt
        self.alpha = alpha
        self.state = np.hstack([np.random.rand(2)])
        self.state_history = deque()
        self.drives = drives
        self.W = weights
        self.tau = tau
        self.past_vals = deque(maxlen=int(self.tau/self.dt))
        #self.past_vals.append(0*np.array([100, 100]))
        self.past_vals.append(np.random.randn(2))

    def rhs(self):
        v1, v2 = self.state
        v1_past, v2_past = self.past_vals[0]
        rhs_v_1 = (-self.alpha * v1 + self.drives[0] + self.W[1] * firing_rate(v2_past))
        rhs_v_2 = (-self.alpha * v2 + self.drives[1] + self.W[0] * firing_rate(v1_past))
        return np.array([rhs_v_1, rhs_v_2])

    def get_next_state(self):
        state = self.state + self.dt * self.rhs()
        return state

    def get_state_history(self):
        return np.array(self.state_history)

    def update_history(self):
        self.state_history.append(deepcopy(self.state))
        self.past_vals.append(deepcopy(self.state))
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
        fr1 = firing_rate(v1)
        fr2 = firing_rate(v2)
        t = np.arange(len(v1)) * self.dt
        fig, axes = plt.subplots(2, 1, figsize = (10, 4))
        axes[0].plot(t, v1, color='r', linewidth = 3, label = 'v, right neuron')
        axes[0].legend(fontsize = 24, loc = 1)
        # axes[0].set_ylim([-0.01,1.1])
        axes[0].grid(True)
        axes[0].set_ylabel("v", fontsize = 24)

        axes[1].plot(t, v2, color='r', linewidth=3, label='v, left neuron')
        axes[1].legend(fontsize = 24, loc = 1)
        # axes[1].set_ylim([-0.01, 1.1])
        axes[1].grid(True)
        axes[1].set_ylabel("v", fontsize = 24)
        axes[1].set_xlabel("t, ms", fontsize = 24)
        plt.subplots_adjust(wspace=0, hspace=0)


        plt.suptitle("Neural oscillations on slow mixed synaptic transmision", fontsize = 24)
        plt.show(block=True)
        return None


if __name__ == '__main__':
    # oscillations on excitatory synapses
    # dt = 0.1
    # weights = np.array([0.6, 0.6])
    # drives = -0.3 * np.ones(2)
    # tau = 500
    # alpha = 0.1

    # excitatory and inhibitory synaptic coupling
    dt = 0.1
    weights = np.array([-0.6, 0.6])
    drives = np.array([-0.3, 0.3])
    tau = 500
    alpha = 0.1

    # # inhibitory synaptic coupling
    # dt = 0.1
    # weights = np.array([-0.6, -0.6])
    # drives = np.array([0.3, 0.3])
    # tau = 500
    # alpha = 0.1

    T_steps = int(10000/dt)
    sso = SlowSynapticOscillator(dt, alpha, weights, drives, tau)
    sso.run(T_steps)
    sso.plot_history()

'''
A script containing a simple RNN description:
equations:
dh/dt = -h W sigma(h) + b

and tools to achieve oscillatory behaviour in such a system
'''
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import deque
from scipy.optimize import fsolve
from scipy.linalg import eig
import numdifftools as nd
import warnings
import os
warnings.filterwarnings("ignore")

# squashing state function of a neural activity ( sigma(h) )
def s(lmbd, h):
    return (2 / np.pi) * np.arctan(lmbd * h)

# derivative of sigma(h)
def der_s(lmbd, h):
    return (2 / np.pi) * lmbd * (1 / (1 + (lmbd * h) ** 2))

# function which calculates the fixed points of the described above dynamics
def calc_equilibria(lmbd, W, b):
    # a function which needs to be solved
    # h = W sigma(h) + b
    def func(h):
        return -h + W @ s(lmbd, h) + b

    fps = []
    fp_hashes = []
    #make sure you find all the fixed points
    for i in range(101):
        fp = fsolve(func, 100 * np.random.rand(len(b)), xtol=1e-18)
        fp_rounded = np.round(fp, 5)
        fp_hash = hash(np.sum(fp_rounded)**2)
        if fp_hash in fp_hashes:
            pass
        else:
            fp_hashes.append(fp_hash)
            fps.append(fp)
    return fps

# def calculate_jacobian(h_star, lmbd, W):
#     N = len(h_star)
#     return -np.identity(N) + W * der_s(lmbd, h_star)

# calculates jacobian of the rhs of the dynamics around a specific point
def calculate_Jacobian(h_star, lmbd, W, b):

    def func(h):
        return -h + W @ s(lmbd, h) + b

    return nd.Jacobian(func)(h_star)

# Neural network class
class RNN():
    def __init__(self, dt, lmbd, W, b):
        self.lmbd = lmbd
        self.dt = dt
        self.W = W
        self.b = b
        #number of neurons-nodes
        self.N = len(self.b)
        self.h = 10 * np.random.randn(self.N)

        self.t = 0
        self.h_history = deque()
        # self.h_history.append(deepcopy(self.h))

    #state function
    def state(self, h):
        return s(self.lmbd, h)

    def rhs(self):
        return -self.h + self.W @ self.state(self.h) + self.b

    def step(self):
        self.h = self.h + self.dt * self.rhs()
        self.t += self.dt
        return None

    def update_history(self):
        self.h_history.append(deepcopy(self.h))
        return None

    def run(self, T):
        N_steps = int(np.ceil(T/ self.dt))
        for i in (range(N_steps)):
            self.step()
            self.update_history()
        return None

    def plot_history(self):
        fig, ax = plt.subplots(self.N, 1, figsize=(15, 5))
        h_array = np.array(self.h_history)
        t_array = np.arange(h_array.shape[0]) * self.dt
        for i in range(self.N):
            ax[i].plot(t_array, h_array[:, i], linewidth=2, color='k')
            if (i == self.N//2):
                ax[i].set_ylabel(f'h', fontsize=24, rotation=0)
        ax[-1].set_xlabel('t', fontsize=24)
        plt.subplots_adjust(hspace=0)
        plt.suptitle(f"Trajectory of a neural network, N={self.N}, lmbd={self.lmbd}", fontsize=24)
        return fig, ax

if __name__ == '__main__':
    N = 8
    lmbd = 0.5
    dt = 0.1
    stable = 1
    # runs till it gets the system which produces oscillations
    while stable > 0:
        W = 10 * np.random.randn(N, N)
        np.fill_diagonal(W, 0)
        b = np.random.randn(N)

        # stability analysis
        fps = calc_equilibria(lmbd, W, b)
        print(f'number of fixed points: {len(fps)}')
        stable = 0
        unstable = 0
        for fp in fps:
            jac = calculate_Jacobian(fp, lmbd, W, b)
            res = eig(jac)
            largest_real_part = np.max(np.real(res[0]))
            if largest_real_part < 0:
                stable += 1
            else:
                unstable += 1

    rnn = RNN(dt, lmbd, W, b)
    T = 100
    rnn.run(T)
    fig, ax = rnn.plot_history()
    plt.savefig(os.path.join("../", 'img', 'multi_dim_oscillations.png'))
    plt.show(block=True)






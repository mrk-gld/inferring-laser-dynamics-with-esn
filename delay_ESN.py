import numpy as np
from scipy.sparse import random


def connectivity_matrix(networkSize, seed=0):
    W = np.array(random(networkSize, networkSize,
                        0.01, random_state=seed).A)
    for i in range(networkSize):
        for j in range(networkSize):
            if W[i, j] != 0:
                if W[i, j] > 0.5:
                    W[i, j] = W[i, j] * 2
                elif W[i, j] < 0.5:
                    W[i, j] = - W[i, j] * 2
    return W


class delay_ESN:

    def __init__(self, n_hid, n_inp, n_delay, seed=0):
        self.n_inp = n_inp
        self.W_in = np.random.random((n_hid, n_inp)) * 2.0 - 1.0
        self.W_back = 0.1 *(np.random.random(n_hid) * 2.0 - 1.0)
        self.W1 = connectivity_matrix(n_hid, seed=seed)
        self.W2 = connectivity_matrix(n_hid, seed=1)
        self.n_delay = n_delay
        self.x = np.random.rand(n_hid)
        self.queue = []
        self.training_noise = 1e-6

        for i in range(n_delay):
            self.queue.append(np.random.rand(n_hid))

    def set_params(self, params):

        self.alpha = params[0]
        self.rho1 = params[1]
        self.rho2 = params[2]
        #beta = params[2]
        self.beta = 1-self.alpha
        self.gamma = params[3]

        eigvals, eigvecs = np.linalg.eig(self.W1)
        max_eigvals = np.max(eigvals)
        self.W1 = self.W1 * self.rho1 / max_eigvals
        self.W1 = np.real(self.W1)

        eigvals, eigvecs = np.linalg.eig(self.W2)
        max_eigvals = np.max(eigvals)
        self.W2 = self.W2 * self.rho2 / max_eigvals
        self.W2 = np.real(self.W2)

    def reset_delay(self, delay):
        self.n_delay = delay
        while self.n_delay < len(self.queue):
            self.queue.pop(0)

    def eval_esn_withNoise(self, s):
        s_noise = s + self.training_noise * np.random.normal(0, 1, self.n_inp)
        self.eval_esn(s_noise)
        return self.x

    def eval_esn(self, s):
        xD = self.queue[0]
        nonlinearity = np.tanh(np.matmul(self.W1, self.x) + np.matmul(
            self.W2, xD) + self.gamma * np.matmul(self.W_in, s) + self.W_back * 0.2)
        self.x = self.alpha * self.x + self.beta * nonlinearity

        self.queue.append(self.x)
        if len(self.queue) > self.n_delay:
            self.queue.pop(0)

        return self.x

    def eval_esn_param_withNoise(self, s, b):
        s_noise = s + self.training_noise * np.random.normal(0, 1, self.n_inp)
        self.eval_esn_param(s_noise, b)
        return self.x

    def eval_esn_param(self, s, b):
        xD = self.queue[0]
        nonlinearity = np.tanh(np.matmul(self.W1, self.x) + np.matmul(
            self.W2, xD) + self.gamma * np.matmul(self.W_in, s) + self.W_back * (0.2+b))
        self.x = self.alpha * self.x + self.beta * nonlinearity

        self.queue.append(self.x)
        if len(self.queue) > self.n_delay:
            self.queue.pop(0)

        return self.x

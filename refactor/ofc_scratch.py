import numpy as np
import matplotlib.pyplot as plt


class OFC:
    ASSOCIATION_RANGE = np.linspace(0, 1, 21)

    def __init__(self):
        self.contexts = {}
        self.ctx = None
        self.prior = []

    def get_v(self):
        if (len(self.prior) == 0):
            return np.array([0.5, 0.5])
        else:
            idx_MAPs = np.where(self.prior == max(self.prior))[0]
            idx_MAP = np.random.choice(idx_MAPs)
            v1 = self.ASSOCIATION_RANGE[idx_MAP]
            v2 = 1 - v1
            return np.array([v1, v2])

    def set_context(self, ctx):
        self.contexts[self.ctx] = self.prior

        if (ctx in self.contexts):
            self.prior = self.contexts[ctx]
        else:
            n = len(self.ASSOCIATION_RANGE)
            self.prior = np.ones(n) / n  # Assume a uniform prior
        self.ctx = ctx

    def update_v(self, stimulus, choice, target):
        trial_type = "MATCH" if (stimulus == target) else "NON-MATCH"

        likelihood = list(map(lambda x:
                              x if trial_type == "MATCH" else (1-x), self.ASSOCIATION_RANGE))
        posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        self.prior = posterior


ofc = OFC()

ofc.set_context("0.7")
n_trials = 100

plt.plot(ofc.ASSOCIATION_RANGE, ofc.prior)
for i in range(n_trials):
    [v1, v2] = ofc.get_v()
    print("v1:", v1, "v2:", v2)

    stimulus = "UP" if np.random.rand() < 0.5 else "DOWN"  # 1 is UP, 0 is DOWN

    if (np.random.rand() < v1):
        choice = stimulus
    else:
        choice = "DOWN" if stimulus == "UP" else "UP"

    if (np.random.rand() < 0.7):
        target = stimulus
    else:
        target = "DOWN" if stimulus == "UP" else "UP"

    print('stimulus:', stimulus, 'choice:', choice, 'target:', target)
    ofc.update_v(stimulus, choice, target)

    plt.cla()
    plt.plot(ofc.ASSOCIATION_RANGE, ofc.prior)
    plt.pause(0.0001)


plt.show()

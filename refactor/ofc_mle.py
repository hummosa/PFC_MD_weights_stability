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
        trial_type = "MATCH" if (stimulus == target).all() else "NON-MATCH"

        likelihood = list(map(lambda x:
                              x if trial_type == "MATCH" else (1-x), self.ASSOCIATION_RANGE))
        posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        self.prior = posterior

class OFC_dumb:
    ASSOCIATION_RANGE = np.linspace(0, 1, 2)

    def __init__(self, horizon):
        self.contexts = {}
        self.ctx = None
        self.prior = np.array([0.5, 0.5])
        self.horizon = horizon
        self.trial_history = [np.array([0.5,0.5])] *2

    def get_v(self):
        return (np.array(self.prior))

    def set_context(self, ctx):
        self.prior = np.array([0.5, 0.5])

    def update_v(self, stimulus, choice, target):
        trial_type = "MATCH" if (stimulus == target).all() else "NON-MATCH"
        self.trial_history.append(trial_type)
        if len(self.trial_history) > self.horizon: self.trial_history = self.trial_history[-self.horizon:]

        likelihood = list(map(lambda trial_type:
                              np.array([0.45, 0.55]) if trial_type == "MATCH" else np.array([0.55, 0.45]), self.trial_history))
                            #   np.array([0.55, 0.45]) if trial_type == "MATCH" else np.array([0.45, 0.55]), self.trial_history))
        likelihood = np.prod(np.array(likelihood), axis=0)
        posterior = (likelihood * np.array([0.5, 0.5])) / np.sum(likelihood * np.array([0.5, 0.5]))
        # posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        # print(self.trial_history, posterior)
        
        self.prior = posterior


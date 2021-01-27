import numpy as np
import matplotlib.pyplot as plt


class OFC:
    ASSOCIATION_RANGE_N = 41
    ASSOCIATION_RANGE = np.linspace(0, 1, ASSOCIATION_RANGE_N)
    # NOTE: STD_THRESH should be adjusted based on association level
    # For association levels 0.9 and 0.1 we use a threshold value of 2
    # because we expect tight distributions
    STD_THRESH = 1.5
    STD_WINDOW_SZ = 5
    # NOTE: The error alpha chosen is 2 because the expected error is 0.1 with
    # association levels 0.9 and 0.1 when one is using the maximizing strategy.
    # So with the window size of 5, expected error becomes 0.5
    # If more than 1 errors occurs in the window, we switch
    # NOTE: Sabrina -- Alpha may need to be changed when I integrate into
    # the complete model
    ERR_ALPHA = 6
    ERR_WINDOW_SZ = 8

    def __init__(self):
        n = self.ASSOCIATION_RANGE_N
        self.prior = np.ones(n) / n  # Assume a uniform prior
        self.contexts = {}

        self.has_dist_convgered = False
        self.std_history = []

        self.trial_err_history = []
        self.trial_type_history = []

    def get_v(self):
        if (len(self.prior) == 0):
            return np.array([0.5, 0.5])
        else:
            idx_MAPs = np.where(self.prior == max(self.prior))[0]
            idx_MAP = np.random.choice(idx_MAPs)
            v1 = self.ASSOCIATION_RANGE[idx_MAP]
            v2 = 1 - v1
            return np.array([v1, v2])

    def switch_context(self):
        [v1, v2] = self.get_v()
        old_ctx = np.round(v1, 1)
        self.contexts[str(old_ctx)] = self.prior

        new_ctx = np.round(1 - np.mean(self.trial_err_history), 1)

        if new_ctx in self.contexts:
            self.prior = self.contexts[str(new_ctx)]
            self.has_dist_convgered = True
            self.std_history = []
        else:
            # NOTE: Prior 1 -- binominal
            # n = len(self.ASSOCIATION_RANGE)
            # p = ctx
            # self.prior = np.array([binom.pmf(k, n, p) for k in range(n)])

            # NOTE: Prior 2 -- uniform
            n = len(self.ASSOCIATION_RANGE)
            self.prior = np.ones(n) / n  # Assume a uniform prior
            for trial_type in self.trial_type_history:
                self.prior = self.compute_posterior(trial_type)
            self.has_dist_convgered = False
            self.std_history = []

    def compute_std(self):
        N = self.ASSOCIATION_RANGE_N * 1.
        mean = sum([(x/N) * p for x, p in enumerate(self.prior)])
        std = sum([p * ((x/N) - mean)**2
                   for x, p in enumerate(self.prior)]) ** 0.5
        return std

    def compute_trial_err(self, stimulus, choice, target):
        if (stimulus == target).all():
            trial_err = 1 if (stimulus != choice).any() else 0
        elif (stimulus != target).any():
            trial_err = 1 if (stimulus == choice).all() else 0
        return trial_err

    def compute_posterior(self, trial_type):
        likelihood = list(map(lambda x:
                              x if trial_type == "MATCH" else (1-x), self.ASSOCIATION_RANGE))
        posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        return posterior

    def update_v(self, stimulus, choice, target):
        # Waiting until distribution has converged...
        if not self.has_dist_convgered:
            if (len(self.std_history) >= self.STD_WINDOW_SZ and np.mean(self.std_history) < self.STD_THRESH):
                self.has_dist_convgered = True

    def __init__(self, config):
        self.config = config
        self.contexts = {}
        self.ctx = None
        self.prior = np.array([0.5, 0.5])
        self.horizon = config.horizon
        self.trial_history = [np.array([0.5, 0.5])] * 2

        self.follow = 'behavioral_context'  # 'association_levels'
        if self.follow == 'association_levels':
            contexts = 5
            self.association_levels_ids = {
                '90': 0, '70': 1, '50': 2, '30': 3, '10': 4}
        elif self.follow == 'behavioral_context':
            contexts = 2
            self.match_association_levels = {'90', '70', '50'}

        self.baseline_err = np.zeros(shape=contexts)
        self.Q_values = [0., 0.]

    def get_v(self):
        return (np.array(self.prior))

        (v1, v2) = self.get_v()
         expected_err = min(v1, v2)  # TODO not self documenting

          # win_size = min(3 + np.round(expected_err * 10, 0),
          #                self.ERR_WINDOW_SZ)
          win_size = self.ERR_WINDOW_SZ

           if len(self.trial_err_history) < win_size:
                return

            if len(self.trial_err_history) > win_size:
                self.trial_err_history.pop(0)
                trial_type = self.trial_type_history.pop(0)
                self.prior = self.compute_posterior(trial_type)

    def get_cid(self, association_level):
        if self.config.follow == 'association_levels':
            cid = self.association_levels_ids[association_level]
        elif self.follow == 'behavioral_context':
            if association_level in self.match_association_levels:
                cid = 0  # Match context
            else:
                cid = 1  # Non-Match context
        return (cid)

    def get_trial_err(self, errors, association_level):
        # error calc
        cid = self.get_cid(association_level)
        if self.config.response_delay:
            response_start = self.config.cuesteps + self.config.response_delay
            errorEnd = np.mean(errors[response_start:]*errors[response_start:])
        else:
            errorEnd = np.mean(errors*errors)  # errors is [tsteps x Nout]

        all_contexts_err = np.array(
            [errorEnd, 1-errorEnd]) if cid == 0. else np.array([errorEnd-1, errorEnd])
        return (errorEnd, all_contexts_err)

    def update_baseline_err(self, trial_err):

        self.baseline_err = (1.0 - self.config.decayErrorPerTrial) * self.baseline_err + \
            self.config.decayErrorPerTrial * trial_err

        return self.baseline_err

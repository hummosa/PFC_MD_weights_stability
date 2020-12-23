import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


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

        print("Switching from old ctx to new ctx", old_ctx, new_ctx)

        if new_ctx in self.contexts:
            self.prior = self.contexts[str(new_ctx)]
            self.has_dist_convgered = True
            self.std_history = []
            print("Pulling new ctx from cache...")
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
            print("Initializnig a new ctx...")

    def compute_std(self):
        N = self.ASSOCIATION_RANGE_N * 1.
        mean = sum([(x/N) * p for x, p in enumerate(self.prior)])
        std = sum([p * ((x/N) - mean)**2
                   for x, p in enumerate(self.prior)]) ** 0.5

        # print('mean std', mean, std)
        return std

    def compute_trial_err(self, stimulus, choice, target):
        if (stimulus == target):
            trial_err = 1 if (stimulus != choice) else 0
        elif (stimulus != target):
            trial_err = 1 if (stimulus == choice) else 0
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

            trial_type = "MATCH" if (stimulus == target) else "NON-MATCH"
            self.prior = self.compute_posterior(trial_type)

            self.std_history.append(self.compute_std())
            if len(self.std_history) > self.STD_WINDOW_SZ:
                self.std_history.pop(0)
        # Check for changes and switch
        else:
            trial_err = self.compute_trial_err(stimulus, choice, target)
            self.trial_err_history.append(trial_err)
            trial_type = "MATCH" if (stimulus == target) else "NON-MATCH"
            # print(trial_type, stimulus, choice, trial_err)
            self.trial_type_history.append((trial_type))

            (v1, v2) = self.get_v()
            expected_err = min(v1, v2)  # TODO not self documenting

            # win_size = min(3 + np.round(expected_err * 10, 0),
            #                self.ERR_WINDOW_SZ)
            win_size = self.ERR_WINDOW_SZ
            print(expected_err, win_size)

            if len(self.trial_err_history) < win_size:
                return

            if len(self.trial_err_history) > win_size:
                self.trial_err_history.pop(0)
                trial_type = self.trial_type_history.pop(0)
                self.prior = self.compute_posterior(trial_type)

            # print(v1, v2, np.mean(self.trial_err_history),
            #   expected_err, self.ERR_ALPHA * expected_err, self.trial_err_history)
            err_threshold = self.ERR_ALPHA * expected_err
            if np.mean(self.trial_err_history) > err_threshold:
                return "SWITCH"


ofc = OFC()
ctx = 0.1

data = np.load("../sabrina_tests/cues_targets.npy", allow_pickle=True)
stimuli = data.item().get("cue")
targets = data.item().get("target")

fig_v1 = []

for i in range(len(stimuli)):
    if i == 0:
        ofc.switch_context()
        continue

    stimulus = "UP" if (stimuli[i, :] == [1, 0]).all() else "DOWN"
    target = "UP" if (targets[i, :] == [1, 0]).all() else "DOWN"

    if (i % 500 == 0):
        ctx = 0.9 if ctx == 0.1 else 0.1
    #     ofc.set_context(ctx)

    [v1, v2] = ofc.get_v()

    fig_v1.append(v1)

    if (v1 > v2):
        choice = stimulus
        signal = ofc.update_v(stimulus, choice, target)
    else:
        choice = "DOWN" if stimulus == "UP" else "UP"
        signal = ofc.update_v(stimulus, choice, target)

    if signal == "SWITCH":
        print(i, "SWITCH ---------")
        ofc.switch_context()

    # plt.cla()
    # plt.plot(ofc.ASSOCIATION_RANGE, ofc.prior)
    # plt.pause(0.0001)

plt.axvline(x=500, color='k', linestyle=':')
plt.axvline(x=1000, color='k', linestyle=':')
plt.axvline(x=1500, color='k', linestyle=':')
plt.axvline(x=2000, color='k', linestyle=':')
plt.axvline(x=2500, color='k', linestyle=':')
plt.plot(fig_v1, 'b', label='With Caching')
plt.xlabel('Trial')
plt.ylabel('V1')
plt.legend(loc='best')
plt.title('OFC Maximum Likelihood Prediction of V1\n(With Uniform Prior)')
plt.show()

# ofc.set_context("0.9")
# n_trials = 500
# k = 5
# err = 0

# plt.plot(ofc.ASSOCIATION_RANGE, ofc.prior)
# for i in range(n_trials):
#     [v1, v2] = ofc.get_v()

#     stimulus = "UP" if np.random.rand() < 0.5 else "DOWN"  # 1 is UP, 0 is DOWN

#     if (v1 > v2):
#         choice = stimulus
#     else:
#         choice = "DOWN" if stimulus == "UP" else "UP"

#     if (np.random.rand() < 0.9):
#         target = stimulus
#     else:
#         target = "DOWN" if stimulus == "UP" else "UP"

#     ofc.update_v(stimulus, choice, target)

#     if (i > 10):
#         if (stimulus == target):
#             trial_err = 1 if (stimulus != choice) else 0
#         elif (stimulus != target):
#             trial_err = 1 if (stimulus == choice) else 0
#         err = (err * (k-1) / k) + (trial_err * (1/k))

#     print(i, err, stimulus, target, choice)

#     plt.cla()
#     plt.plot(ofc.ASSOCIATION_RANGE, ofc.prior)
#     plt.pause(0.0001)

# plt.show()

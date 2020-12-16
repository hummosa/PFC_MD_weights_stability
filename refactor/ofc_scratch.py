import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


class OFC:
    ASSOCIATION_RANGE = np.linspace(0, 1, 41)

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
        self.contexts[str(self.ctx)] = self.prior

        if (str(ctx) in self.contexts):
            self.prior = self.contexts[str(ctx)]
        else:
            # NOTE: Prior 1 -- binominal
            n = len(self.ASSOCIATION_RANGE)
            p = ctx
            self.prior = np.array([binom.pmf(k, n, p) for k in range(n)])

            # NOTE: Prior 2 -- uniform
            # n = len(self.ASSOCIATION_RANGE)
            # self.prior = np.ones(n) / n  # Assume a uniform prior
        self.ctx = ctx

    def update_v(self, stimulus, choice, target):
        trial_type = "MATCH" if (stimulus == target) else "NON-MATCH"

        likelihood = list(map(lambda x:
                              x if trial_type == "MATCH" else (1-x), self.ASSOCIATION_RANGE))
        posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        self.prior = posterior


# class OFC_change_point:
#     HORIZON = 20

#     def __init__(self):
#         self.horizon = np.array([])
#         self.v = np.array([0.5, 0.5])

#     def horizon_push(self, trial_type):
#         self.horizon.push(trial_type)
#         if len(self.horizon) > self.HORIZON:
#             self.horizon.pop(0)

#     def find_change_point():
#         # NOTE: Should we do a statistical analysis to see how many trials we
#         # need to get a good estimate? Or maybe this is person dependent?

#     def get_v(self):
#         return self.v

#     def set_context(self, ctx):
#         return None

#     def update_v(self, stimulus, choice, target):
#         trial_type = "MATCH" if (stimulus == target) else "NON-MATCH"
#         self.horizon_push(trial_type)

#         if len(self.horizon) < self.HORIZON:
#             m = map(lambda x: 1 if x == "MATCH" else 0, self.horizon)
#             r = reduce(lambda x, y: x + y, m, 0)
#             self.v = np.array([r, 1-r])k
#         else
#         self.find_change_point()


ofc = OFC()
ctx = 0.1

data = np.load("../sabrina_tests/cues_targets.npy", allow_pickle=True)
stimuli = data.item().get("cue")
targets = data.item().get("target")

fig_v1 = []

for i in range(len(stimuli)):
    if i == 0:
        ofc.set_context(ctx)
        continue

    stimulus = "UP" if (stimuli[i, :] == [1, 0]).all() else "DOWN"
    target = "UP" if (targets[i, :] == [1, 0]).all() else "DOWN"

    if (i % 500 == 0):
        ctx = 0.9 if ctx == 0.1 else 0.1
        ofc.set_context(ctx)

    [v1, v2] = ofc.get_v()

    fig_v1.append(v1)

    if (v1 > v2):
        choice = stimulus
    else:
        choice = "DOWN" if stimulus == "UP" else "UP"

    # print(i, stimuli[i, :], targets[i, :], stimulus, target, choice)

    ofc.update_v(stimulus, choice, target)

    # plt.cla()
    # plt.plot(ofc.ASSOCIATION_RANGE, ofc.prior)
    # plt.pause(0.0001)

plt.plot(fig_v1)
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

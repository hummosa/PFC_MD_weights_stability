import numpy as np
import random
from ofc_trailtype import OFC
import matplotlib.pyplot as plt

ASSOCIATION_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.1]
N_BLOCKS = 10
BLOCK_SIZE = 500
N_EXAMPLES = 5
PARAMS = [(0.15, 20), (0.2, 20), (0.25, 20), (0.3, 20)]


def generate_session(association_levels, n_blocks, block_size):
    trials = []
    for n in range(n_blocks):
        a_level = random.choice(association_levels)
        trials = np.concatenate((trials, np.repeat([a_level], block_size)))

    return trials


def run_session(trial_types, t, h):
    v1_history = []

    ofc = OFC(t, h)
    for a_level in trial_types:
        stimulus = np.array([1., 0.] if random.random() < 0.5 else [0., 1.])
        target = np.array(stimulus if random.random() <
                          a_level else abs(stimulus - 1))

        [v1, v2] = ofc.get_v()
        v1_history.append(v1)

        if (v1 > v2):
            choice = stimulus
            signal = ofc.update_v(stimulus, choice, target)
        else:
            choice = abs(stimulus - 1)
            signal = ofc.update_v(stimulus, choice, target)

        if signal == "SWITCH":
            ofc.switch_context()

    return v1_history


for n in range(N_EXAMPLES):
    trial_types = generate_session(ASSOCIATION_LEVELS, N_BLOCKS, BLOCK_SIZE)

    fig, axs = plt.subplots(4)
    for i, (t, h) in enumerate(PARAMS):
        v1 = run_session(trial_types, t, h)
        axs[i].plot(trial_types, 'k')
        axs[i].plot(v1)
        axs[i].set_title(f't={t}, h={h}')
    fig.suptitle(f'Association Levles = {ASSOCIATION_LEVELS}')
    plt.show()

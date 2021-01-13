import numpy as np
import random
import matplotlib.pyplot as plt
from ofc_error import OFC as OFC_Error
from ofc_trailtype import OFC as OFC_Trial
from ofc_mle import OFC as OFC_MLE

ASSOCIATION_LEVELS = [0.9, 0.1]
N_BLOCKS = 10
BLOCK_SIZE = 500

ofc = OFC_Trial(0.3, 20)

# data = np.load("../sabrina_tests/cues_targets.npy", allow_pickle=True)
# stimuli = data.item().get("cue")
# targets = data.item().get("target")

fig_v1 = []
fig_switches = []
fig_avg = []
fig_levels = []

for n in range(N_BLOCKS):
    a_level = random.choice(ASSOCIATION_LEVELS)

    for i in range(BLOCK_SIZE):  # range(len(stimuli)):
        if i == 0:
            continue

        stimulus = np.array([1., 0.] if random.random() < 0.5 else [0., 1.])
        target = np.array(stimulus if random.random() <
                          a_level else abs(stimulus - 1))

        [v1, v2] = ofc.get_v()

        fig_v1.append(v1)

        if (v1 > v2):
            choice = stimulus
            signal = ofc.update_v(stimulus, choice, target)
        else:
            choice = abs(stimulus - 1)
            signal = ofc.update_v(stimulus, choice, target)

        if signal == "SWITCH":
            print(i, "SWITCH ---------")
            ofc.switch_context()
            fig_switches.append(n * BLOCK_SIZE + i)

        fig_levels.append(a_level)

plt.axvline(x=500, color='k', linestyle=':')
plt.axvline(x=1000, color='k', linestyle=':')
plt.axvline(x=1500, color='k', linestyle=':')
plt.axvline(x=2000, color='k', linestyle=':')
plt.axvline(x=2500, color='k', linestyle=':')
for i in fig_switches:
    plt.axvline(x=i, color='r', linestyle=':')
plt.plot(fig_v1, 'b')
plt.plot(fig_levels, 'k')
plt.xlabel('Trial')
plt.ylabel('V1')
plt.title('OFC Maximum Likelihood Prediction of V1\n(With Uniform Prior)')
plt.show()

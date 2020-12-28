import numpy as np
import matplotlib.pyplot as plt
from ofc_error import OFC as OFC_Error
from ofc_trailtype import OFC as OFC_Trial
from ofc_mle import OFC as OFC_MLE

ofc = OFC_Error()
# ofc = OFC_Trial()

data = np.load("../sabrina_tests/cues_targets.npy", allow_pickle=True)
stimuli = data.item().get("cue")
targets = data.item().get("target")

fig_v1 = []
fig_switches = []
fig_avg = []

for i in range(1100):  # range(len(stimuli)):
    if i == 0:
        continue

    stimulus = stimuli[i, :]
    target = targets[i, :]

    [v1, v2] = ofc.get_v()

    fig_v1.append(v1)

    if (v1 > v2):
        choice = stimulus
        signal = ofc.update_v(stimulus, choice, target)
    else:
        choice = np.array([0, 1]) if (stimulus == [1, 0]).all() else stimulus
        signal = ofc.update_v(stimulus, choice, target)

    if signal == "SWITCH":
        print(i, "SWITCH ---------")
        ofc.switch_context()
        fig_switches.append(i)

plt.axvline(x=500, color='k', linestyle=':')
plt.axvline(x=1000, color='k', linestyle=':')
plt.axvline(x=1500, color='k', linestyle=':')
plt.axvline(x=2000, color='k', linestyle=':')
plt.axvline(x=2500, color='k', linestyle=':')
for i in fig_switches:
    plt.axvline(x=i, color='r', linestyle=':')
plt.plot(fig_v1, 'b')
plt.xlabel('Trial')
plt.ylabel('V1')
plt.title('OFC Maximum Likelihood Prediction of V1\n(With Uniform Prior)')
plt.show()

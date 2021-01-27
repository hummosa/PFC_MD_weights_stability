import numpy as np
import random
import matplotlib.pyplot as plt
from ofc_error import OFC as OFC_Error
from ofc_trailtype import OFC as OFC_Trial
from ofc_mle import OFC as OFC_MLE

ASSOCIATION_LEVELS = [0.9, 0.1, 0.7, 0.3, 0.5]
N_BLOCKS = 10
N_INITS = 200
BLOCK_SIZE = 500

THRESHOLDS = [0.1, 0.2, 0.3, 0.4]
HORIZONS = [5, 10, 15, 20, 25]


def test(ofc, blocks):
    mse = []
    n_switches = 0
    for n in range(len(blocks)):
        a_level = random.choice(ASSOCIATION_LEVELS)

        for i in range(BLOCK_SIZE):  # range(len(stimuli)):
            if i == 0:
                continue

            stimulus = np.array([1., 0.] if random.random()
                                < 0.5 else [0., 1.])
            target = np.array(stimulus if random.random() <
                              a_level else abs(stimulus - 1))

            [v1, v2] = ofc.get_v()
            mse.append(abs(v1 - a_level)**2)

            if (v1 > v2):
                choice = stimulus
                signal = ofc.update_v(stimulus, choice, target)
            else:
                choice = np.array([0, 1]) if (
                    stimulus == [1, 0]).all() else stimulus
                signal = ofc.update_v(stimulus, choice, target)

            if signal == "SWITCH":
                ofc.switch_context()
                n_switches += 1
    return (sum(mse) / len(mse), n_switches)


errors = {}
switches = {}

for x in range(N_INITS):
    blocks = [random.choice(ASSOCIATION_LEVELS) for x in range(N_BLOCKS)]
    print("Running itteration " + str(x) + "...")
    for t in THRESHOLDS:
        for h in HORIZONS:
            ofc = OFC_Trial(t, h)
            (error, n_switches) = test(ofc, blocks)
            key = f't={t}, h={h}'
            if key in errors:
                errors[key].append(error)
                switches[key].append(n_switches)
            else:
                errors[key] = [error]
                switches[key] = [n_switches]


def plot_bars(ax, d):
    stds = [np.std(x) / np.sqrt(len(x)) for x in d.values()]
    means = [np.mean(x) for x in d.values()]
    ax.bar(list(d.keys()), means, yerr=stds, zorder=1)
    for i, key in enumerate(d):
        ys = d[key]
        xs = [i] * len(ys)
        ax.scatter(xs, ys, s=5, c='k', marker='*', zorder=2)
        ax.text(i + 0.05, means[i] + 0.2, str(np.round(means[i], 0)), c='b')
    ax.set_xticklabels(list(d.keys()), rotation=45)


# for key in switches:
#     switches[key] = np.log(switches[key])
fig, axs = plt.subplots(2)
plot_bars(axs[0], errors)
axs[0].set_ylabel('MSE of V1 to association level per session')
plt.setp(axs[0].get_xticklabels(), visible=False)
plot_bars(axs[1], switches)
axs[1].set_ylabel('Number of interal switches per session')
fig.suptitle(f'''OFC horizon and threshold pair analysis --  
    {N_INITS} randomly initalized sessions of {N_BLOCKS} blocks 
    with {BLOCK_SIZE} trials per block''')
plt.show()

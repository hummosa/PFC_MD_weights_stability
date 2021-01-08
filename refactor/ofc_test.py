import numpy as np
import random
import matplotlib.pyplot as plt
from ofc_error import OFC as OFC_Error
from ofc_trailtype import OFC as OFC_Trial
from ofc_mle import OFC as OFC_MLE

ASSOCIATION_LEVELS = [0.9, 0.1, 0.7, 0.3, 0.5]
N_BLOCKS = 10
BLOCK_SIZE = 500

THRESHOLDS = [0.1, 0.2, 0.3, 0.4]
HORIZONS = [5, 10, 15, 20, 25]


def test(ofc, blocks):
    mse = []
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
    return mse


errors = {}

for x in range(5):
    blocks = [random.choice(ASSOCIATION_LEVELS) for x in range(10)]
    print(blocks)
    for t in THRESHOLDS:
        for h in HORIZONS:
            print(t, h)
            ofc = OFC_Trial(t, h)
            error = sum(test(ofc, blocks))
            key = f'{t}_{h}'
            if key in errors:
                errors[key] = errors[key] + error
            else:
                errors[key] = error

print(errors)

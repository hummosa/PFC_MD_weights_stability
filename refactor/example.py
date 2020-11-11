import numpy as np
import random
import matplotlib.pyplot as plt

from network import Network, Simulation
import models
import utils

# Network construction

pfc = models.PFC({}, 'PFC')
ofc = models.OFC({}, 'OFC')
output = models.Output({}, 'Output')

W_in = np.array([[1, 0], [0, 1]])
W_ofc_pfc = np.array([[1, 0], [0, 1]])
W_pfc_output = np.array([[1, 0], [0, 1]])


def update_W_ofc_pfc(step, state, W, ofc, pfc):
    '''
    step: ('STEP', timestep) | ('TRIAL_END', global_signals)
    '''
    return (state, W)


def update_W_pfc_output(step, state, W, ofc, pfc):
    return (state, W)


def compute_global_signals(network, expected_output, plasticity):
    output = network.models['Output'].neurons
    error = output - expected_output
    return {"error": error}


network = Network(compute_global_signals)
network.define_inputs(W_in, pfc)
network.connect(ofc, pfc, W_ofc_pfc, update_W_ofc_pfc, {})
network.connect(pfc, output, W_pfc_output, update_W_pfc_output, {})

# Simulation


dat_inp = []
dat_output = []
dat_output_expected = []
dat_OFC = []
dat_PFC = []


def cb(trial_num, step_name, timestep, inp, expected_output, network):
    if step_name == 'CUE' and timestep == 1:
        dat_inp.append(inp)

    if step_name == 'TRIAL_END':
        dat_output_expected.append(expected_output)
        for model in network.models.values():
            if model.name == 'OFC':
                dat_OFC.append(model.neurons)
            elif model.name == 'PFC':
                dat_PFC.append(model.neurons)
            elif model.name == 'Output':
                dat_output.append(model.neurons)


def get_input(trial_num, step_name, timestep, prev_inp):
    # [UP, DOWN]
    if step_name == 'CUE' and timestep == 1:
        return np.array([1., 0.]) if random.random() < 0.5 else np.array([0., 1.])
    else:
        return prev_inp


def get_output(trial_num, inps_arr):
    inp = inps_arr[0]
    if trial_num <= 25:  # Match context
        return inp if random.random() < 0.7 else (1 - inp)
    elif trial_num <= 50:
        return 1 - inp if random.random() < 0.7 else (1 - inp)
    else:
        return inp if random.random() < 0.7 else (1 - inp)


N = 75
simulation = Simulation(network)
trial_setup = [("CUE", 5, False)]
simulation.run_trials(trial_setup, get_input, get_output, N, cb)

# Plot results
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
# Plot input
dat_inp = np.array(dat_inp)
inp_y = dat_inp[:, 1] * 4
ax1.plot(np.arange(len(inp_y)), inp_y, 'ko', label='Input')
dat_output_expected = np.array(dat_output_expected)
dat_output = np.array(dat_output)
for i in range(0, len(dat_output)):
    icon = 'go'
    if not np.array_equal(dat_output[i], dat_output_expected[i]):
        icon = 'gx'
    ax1.plot(i, .3 + 4 * dat_output[i, 1], icon)
ax1.legend()
ax1.set_title('Behavior')
# Plot OFC
dat_OFC = np.array(dat_OFC)
ax2.plot(dat_OFC[:, 0], label='Favor match')
ax2.plot(dat_OFC[:, 1], label='Favor nonmatch')
ax2.legend()
ax2.set_title('OFC')
# Plot PFC
dat_PFC = np.array(dat_PFC)
ax3.plot(dat_PFC[:, 0], label='Chose up')
ax3.plot(dat_PFC[:, 1], label='Chose down')
ax3.legend()
ax3.set_title('PFC')
plt.show()

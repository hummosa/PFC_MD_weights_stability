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

W_in = np.array([[0, 0], [0, 0]])
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
    print('output', output, 'expected', expected_output)
    return {"error": error}


network = Network(compute_global_signals)
network.define_inputs(W_in, pfc)
network.connect(ofc, pfc, W_ofc_pfc, update_W_ofc_pfc, {})
network.connect(pfc, output, W_pfc_output, update_W_pfc_output, {})

# Simulation


dat_inp = []
dat_OFC = []
dat_PFC = []
dat_output = []


def cb(trial_num, step_name, timestep, inp, network):
    if step_name == 'CUE' and timestep == 1:
        dat_inp.append(inp)

    if step_name == 'TRIAL_END':
        for model in network.models.values():
            if model.name == 'OFC':
                dat_OFC.append(model.neurons)
            elif model.name == 'PFC':
                dat_PFC.append(model.neurons)
            elif model.name == 'Output':
                dat_output.append(model.neurons)


def get_input(trial_num, step_name, timestep):
    # [UP, DOWN]
    if step_name == 'CUE':
        return np.array([1., 0.]) if random.random() < 0.7 else np.array([0., 1.])
    else:
        return [0, 0]


def get_output(trial_num, inp):
    if trial_num <= 50:  # Match context
        return inp
    else:
        return 1 - inp


simulation = Simulation(network)
trial_setup = [("CUE", 5, False)]
simulation.run_trials(trial_setup, get_input, get_output, 100, cb)

# Plot results
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
# Plot input
dat_inp = np.array(dat_inp)
match = dat_inp[0:50, :]
nonmatch = dat_inp[50:100, :]
x_up_match = np.nonzero(match[:, 0])[0]
x_down_match = np.nonzero(match[:, 1])[0]
ax1.plot(x_up_match, np.zeros(len(x_up_match)), 'g^')
ax1.plot(x_down_match, np.zeros(len(x_down_match)), 'gv')
x_up_nonmatch = np.nonzero(nonmatch[:, 0])[0] + 50
x_down_nonmatch = np.nonzero(nonmatch[:, 1])[0] + 50
ax1.plot(x_up_nonmatch, np.zeros(len(x_up_nonmatch)), 'r^')
ax1.plot(x_down_nonmatch, np.zeros(len(x_down_nonmatch)), 'rv')
ax1.set_title('Input')
# Plot output
dat_output = np.array(dat_output)
output_up = np.nonzero(dat_output[:, 0])[0]
output_down = np.nonzero(dat_output[:, 1])[0]
ax2.plot(output_up, np.zeros(len(output_up)), 'k^')
ax2.plot(output_down, np.zeros(len(output_down)), 'kv')
ax2.set_title('Output')
# Plot OFC
dat_OFC = np.array(dat_OFC)
ax3.plot(dat_OFC)
ax3.set_title('OFC')
# Plot PFC
dat_PFC = np.array(dat_PFC)
ax4.plot(dat_PFC)
ax4.set_title('PFC')
plt.show()

import numpy as np
import random
import matplotlib.pyplot as plt

from network import Network, Simulation
import models
import utils

# Network construction

pfc = models.PFC({})
ofc = models.OFC({})
output = models.Output({})

W_in = np.array([[0, 0], [0, 0]])
W_ofc_pfc = np.array([[1, 0], [0, 1]])
W_pfc_output = np.array([1, 1])


def update_W_ofc_pfc(step, state, W, ofc, pfc):
    return (state, W)


def update_W_pfc_output(step, state, W, ofc, pfc):
    return (state, W)


network = Network()
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
        for model in network.models:
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
    if trial_num < 50:  # Match context
        return inp
    else:
        return 1 - inp


simulation = Simulation(network)
trial_setup = [("CUE", 5, False)]
simulation.run_trials(trial_setup, get_input, get_output, 100, cb)

# TODO finish ploting of data
dat_inp = np.array(dat_inp)
plt.plot(dat_inp)
plt.show()

# Plan:
# [] finish plot of data
# [] global signal - TRIAL_END output global signals as a dictionary?
# [] OFC as running average
# [] run simple simulation w/ plots

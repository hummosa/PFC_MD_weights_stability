import numpy as np
import random

from network import Network, Simulation
import models
import utils

# Network construction

md = models.MD({})
pfc = models.PFC({})

inputs = np.array([1])
W_in = np.array([1])
W_pfc_md = np.array([1])
W_md_pfc = np.array([1])


def update_W_pfc_md(step, state, W, pfc, md):
    action, time = step
    if action == 'STEP':
        W_new = W * 2
        x_new = state['x'] + 1
        return ({**state, 'x': x_new}, W_new)
    elif action == 'TRIAL_END':
        return (state, W)


network = Network()
network.define_inputs(W_in, pfc)
network.connect(pfc, md, W_pfc_md, update_W_pfc_md, {'x': 0})

# Simulation


def cb(trial_num, step_name, timestep, network):
    print(trial_num, step_name, timestep)
    # print(network.parents_conn)
    # print(network.children_conn)


def get_input(trial_num, step_name, timestep):
    return 1 if random.random() < 0.5 else 0


simulation = Simulation(network)
trial_setup = [("CUE", 2, True), ("DLEAY", 2, False), ("RESP", 2, False)]
simulation.run_trials(trial_setup, get_input, 2, cb)


utils.save_network('test.txt', network)
loaded_net = utils.load_network('test.txt')

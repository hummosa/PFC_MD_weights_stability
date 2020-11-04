import numpy as np
import random

from network import Network, Simulation
import models
import utils


md = models.MD({})
pfc = models.PFC({})

inputs = np.array([1])
W_in = np.array([1])
W_pfc_md = np.array([1])
W_md_pfc = np.array([1])


def update_W_md_pfc(md, pfc, W, tstep):
    k, t = tstep
    if k == 'STEP':
        W_new = W * 2
        print('STEP: MD->PFC W', W, W_new)
        return W_new
    elif k == 'TRIAL_END':
        print('TRIAL_END: MD->PFC W', W)
        return W


def update_W_pfc_md(pfc, md, W, tstep):
    k, t = tstep
    if k == 'STEP':
        W_new = W * 2
        print('STEP: PFC->MD W', W, W_new)
        return W_new
    elif k == 'TRIAL_END':
        print('TRIAL_END: PFC->MD W', W)
        return W


network = Network()
network.define_inputs(len(inputs), W_in, pfc)
network.connect(pfc, md, W_pfc_md, update_W_pfc_md)
network.connect(md, pfc, W_md_pfc, update_W_md_pfc)


def cb(trial_num, step_name, timestep, network):
    print(trial_num, step_name, timestep)


def get_input(trial_num, step_name, timestep):
    return 1 if random.random() < 0.5 else 0


simulation = Simulation(network)
trial_setup = [("CUE", 2, False), ("DLEAY", 2, False), ("RESP", 2, False)]
simulation.run_trials(trial_setup, get_input, 1, cb)

utils.save_network('test.txt', network)
loaded_net = utils.load_network('test.txt')
print('----------------------------------------')
print('loaded network', loaded_net)

simulation = Simulation(loaded_net)
trial_setup = [("CUE", 1, False)]
simulation.run_trials(trial_setup, get_input, 1, cb)

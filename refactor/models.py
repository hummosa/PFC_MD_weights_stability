import uuid
import numpy as np
import random


class Model:

    def __init__(self, config, name_uid):
        self.name = ''
        self.neurons = []
        self.weights = []
        self.config = None

    def step(self, xWs, plasticity):
        '''
        xWs: is an input array of [x1W1, x2W2, ...]
        returns an array of firing rates
        '''

    def trial_end(self, global_signals, plasticity):
        '''
        xWs: is an input array of [x1W1, x2W2, ...]
        returns an array of firing rates
        '''


class MD(Model):
    def __init__(self, config, name_uid):
        self.name = name_uid
        self.config = config
        # self.neurons = [1]
        self.neurons = np.zeros(shape=(config.Nmd))

    def step(self, xWs, plasticity):
        self.neurons = xWs
        return self.neurons

    def trial_end(self, global_signals, plasticity):
        return self.neurons


class PFC(Model):
    def __init__(self, config, name_uid):
        self.name = name_uid
        self.config = config
        # self.neurons = np.array([0., 0.])
        self.neurons = np.zeros(shape=(config.Npfc))

    def step(self, xWs, plasticity):
        inp = np.array([1, 0]) if xWs[0] > 1 else np.array([0, 1])
        ofc_inp = xWs - inp
        if ofc_inp[0] > ofc_inp[1]:  # Match
            self.neurons = inp
        else:  # Nonmatch
            self.neurons = abs(inp - 1)
        return self.neurons

    def trial_end(self, global_signals, plasticity):
        return self.neurons


class Output(Model):

    def __init__(self, config, name_uid):
        self.name = name_uid
        self.config = config
        # self.neurons = np.zeros(shape=(config.Npfc))
        self.neurons = np.array([0, 0])

    def step(self, xWs, plasticity):
        self.neurons = xWs
        return self.neurons

    def trial_end(self, global_signals, plasticity):
        return self.neurons


class OFC(Model):
    def __init__(self, config, name_uid):
        self.name = name_uid
        self.config = config
        self.neurons = np.array([.7, .3])
        self.n = 0

    def step(self, xWs, plasticity):
        # NOTE: If a model has no input,
        #   the step() function will not be called
        return self.neurons

    def trial_end(self, global_signals, plasticity):
        self.n += 1
        new_neurons = 0.65 * self.neurons + 0.35 * global_signals['error']
        new_neurons = new_neurons / np.linalg.norm(new_neurons, 1)

        idx = np.where(new_neurons == 1)
        if len(idx) > 0:
            new_neurons[idx[0]] = 0.9
            new_neurons[abs(idx[0]-1)] = 0.1

        self.neurons = new_neurons
        return self.neurons

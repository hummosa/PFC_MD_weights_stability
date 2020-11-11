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
        self.neurons = [1]  # np.zeros(shape=(config.Nmd))

    def step(self, xWs, plasticity):
        self.neurons = xWs
        print('MD', self.neurons)
        return self.neurons

    def trial_end(self, global_signals, plasticity):
        print('MD trial end called')
        return self.neurons


class PFC(Model):
    def __init__(self, config, name_uid):
        self.name = name_uid
        self.config = config
        self.neurons = np.zeros(shape=(2, 0))

    def step(self, xWs, plasticity):
        if xWs[0] > xWs[1]:
            self.neurons = np.array([1, 0])
        else:
            self.neurons = np.array([0, 1])
        return self.neurons

    def trial_end(self, global_signals, plasticity):
        return self.neurons


class Output(Model):

    def __init__(self, config, name_uid):
        self.name = name_uid
        self.config = config
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
        self.neurons = np.array([0, 0])

    def step(self, xWs, plasticity):
        # NOTE: If a model has no input,
        #   the step() function will not be called
        return self.neurons

    def trial_end(self, global_signals, plasticity):
        # TODO compute reward prediction error and averaging
        self.neurons = 0.9 * self.neurons - 0.1 * global_signals['error']
        print('error', global_signals['error'], 'ofc', self.neurons)
        return self.neurons

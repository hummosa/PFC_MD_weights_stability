import uuid
import numpy as np


class Model:
    name = ''

    def __init__(self, config):
        self.uid = uuid.uuid1()
        # TODO self.neurons can't be empty
        # it needs to be an initial network state from config
        self.neurons = []

    def step(self, xWs):
        '''
        xWs: is an input array of [x1W1, x2W2, ...]
        returns an array of firing rates
        '''

    def trial_end(self, xWs):
        '''
        xWs: is an input array of [x1W1, x2W2, ...]
        returns an array of firing rates
        '''


class MD(Model):
    name = 'MD'

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.neurons = [1]  # np.zeros(shape=(config.Nmd))

    def step(self, xWs):
        self.neurons = xWs
        print('MD', self.neurons)
        return self.neurons

    def trial_end(self, xWs):
        print('MD trial end called')
        return self.neurons


class PFC(Model):
    name = 'PFC'

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.neurons = [1]  # np.zeros(shape=(config.Npfc))

    def step(self, xWs):
        self.neurons = xWs
        print('PFC firing', self.neurons)
        return self.neurons

    def trial_end(self, xWs):
        print('PFC trial end called')
        return self.neurons

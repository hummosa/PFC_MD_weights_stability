import uuid
import numpy as np


class Model:
    name = ''

    def __init__(self, config):
        self.uid = uuid.uuid1()
        self.neurons = []
        self.weights = []
        self.config = None

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
        self.neurons = np.zeros(shape=(2, 0))

    def step(self, xWs):
        if xWs[0] > xWs[1]:
            self.neurons = np.array([1, 0])
        else:
            self.neurons = np.array([0, 1])
        return self.neurons

    def trial_end(self, xWs):
        return self.neurons


class Output(Model):
    name = 'Output'

    def __init__(self, config):
        super().__init__(config)
        self.neurons = np.zeros([0, 0])

    def step(self, xWs):
        self.neurons = xWs
        return self.neurons

    def trial_end(self, xWs):
        return self.neurons


class OFC(Model):
    name = 'OFC'

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # TODO set this to start at 0,0 -- once global signals are working
        self.neurons = np.array([0.7, 0.3])

    def step(self, xWs):
        return self.neurons

    def trial_end(self, xWs):
        # TODO compute average trace
        return self.neurons

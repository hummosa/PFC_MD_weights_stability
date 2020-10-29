class Model:
    name = ''

    def __init__(self, config):
        return None

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
        self.config = config
        self.neurons = [0]

    def step(self, xWs):
        self.neurons = xWs * 2
        print('MD', self.neurons)
        return self.neurons

    def trial_end(self, xWs):
        print('MD trial end called')
        return self.neurons


class PFC(Model):
    name = 'PFC'

    def __init__(self, config):
        self.config = config
        self.neurons = [0]

    def step(self, xWs):
        self.neurons = xWs * 2
        print('PFC', self.neurons)
        return self.neurons

    def trial_end(self, xWs):
        print('PFC trial end called')
        return self.neurons

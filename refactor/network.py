class Network:
    def __init__(self):
        self.conn_external = {'n': 0, 'W': [], 'model': None}
        self.conn_internal = {}  # {m1_name: [(m1, W12, m2, fW12), ...]}

    def define_inputs(self, n, W, model):
        '''
        Define external inputs.
        n: # of external inputs
        model: model that receives the inputs
        W: weights from inputs to model
        '''
        self.conn_external = {n, W, model}

    def connect(self, model1, model2, W, update_W):
        '''
        Connect models to create a network.
        model1: from model
        model2: to model
        W: weights that connect model1 to model2
        update_W: a lambda function specifying weights update rule
        '''
        new = (model1, W, model2, update_W)
        arr = self.conn_internal[model1.name]
        if len(arr) == 0:
            self.conn_internal[model1.name] = [new]
        else:
            self.conn_internal[model1.name].append(new)

    def step(self, external_inputs, plasticity=True):
        xW_ext = external_inputs * self.conn_external['W']
        m = self.conn_external['model']
        self.conn_internal[m.name]
        # TODO finish step implementation

        '''
        -> do xW on external inputs to model m_in
        -> update m_in, update W_in
        -> for all m_k with inputs from m_in
            -> update m_k, update W_in_k ... repete
        '''

    def trial_end(self):
        return None


class Simulation:
    def __init__(self, network):
        self.network = network

    def train(self, cb, params):
        # for t in range(...)
        #   network.step(external_inputs)
        #   cb(network)
        # network.trial_end()
        return None

    def test(self, cb, params):
        return None

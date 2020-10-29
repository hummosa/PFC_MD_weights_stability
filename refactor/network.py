import numpy as np


class Network:
    def __init__(self):
        self.conn_external = {'n': 0, 'W': [], 'model': None}
        self.children_conn = {}  # {parent_uid: [(child, Wpc, fW), ...]}
        self.parents_conn = {}  # {child_uid: [(parent, Wpc, fW), ...]}

    def define_inputs(self, n, W, model):
        '''
        Define external inputs.
        n: # of external inputs
        model: model that receives the inputs
        W: weights from inputs to model
        '''

        self.conn_external = {'n': n, 'W': W, 'model': model}

    def connect(self, parent, child, W, update_W):
        '''
        Connect models to create a network.
        model1: from model
        model2: to model
        W: weights that connect model1 to model2
        update_W: a lambda function specifying weights update rule
        '''

        new_child = (child, W, update_W)
        new_parent = (parent, W, update_W)

        if parent.uid in self.children_conn:
            self.children_conn[parent.uid].append(new_child)
        else:
            self.children_conn[parent.uid] = [new_child]

        if child.uid in self.parents_conn:
            self.parents_conn[child.uid].append(new_parent)
        else:
            self.parents_conn[child.uid] = [new_parent]

    def step(self, external_inputs, plasticity=True):
        # TODO add plasticity updates
        print('input firing', external_inputs)

        tracker = set()

        m = self.conn_external['model']
        W = np.zeros(len(m.neurons))
        W += external_inputs * self.conn_external['W']

        parents = self.parents_conn[m.uid]
        for parent in parents:
            parent_model = parent[0]
            W_pc = parent[1]
            W += parent_model.neurons * W_pc
        m.step(W)
        tracker.add(m.uid)

        if m.uid not in self.children_conn:
            return

        children = self.children_conn[m.uid]
        for child in children:
            self._step(child[0], tracker)

    def _step(self, model, tracker):
        if model.uid in tracker:
            return

        W = np.zeros(len(model.neurons))
        parents = self.parents_conn[model.uid]
        for parent in parents:
            parent_model = parent[0]
            W_pc = parent[1]
            W += parent_model.neurons * W_pc
        model.step(W)
        tracker.add(model.uid)

        if model.uid not in self.children_conn:
            return

        children = self.children_conn[model.uid]
        for child in children:
            self._step(child[0], tracker)

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

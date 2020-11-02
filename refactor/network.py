import numpy as np


class Network:
    def __init__(self):
        self.models = []
        self.conn_external = {'n': 0, 'W': [], 'model': None}
        self.children_conn = {}  # {parent_uid: [(child, Wpc, update_W), ...]}
        self.parents_conn = {}  # {child_uid: [(parent, Wpc, update_W), ...]}

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
        update_W: (parent, child, W, ('STEP'|'TRIAL_END', time)) -> W'
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

        if parent not in self.models:
            self.models.append(parent)
        if child not in self.models:
            self.models.append(child)

    def step(self, external_inputs, tstep, plasticity=True):
        print('timestep =', tstep)
        print('input firing', external_inputs)

        tracker = set()

        m = self.conn_external['model']
        xW = self.compute_incoming_activity(m)
        xW += np.dot(self.conn_external['W'], external_inputs)
        m.step(xW)

        if plasticity:
            self.update_W(m, ('STEP', tstep))

        tracker.add(m.uid)

        if m.uid not in self.children_conn:
            return

        children = self.children_conn[m.uid]
        for child in children:
            self._step(child[0], tracker, tstep, plasticity)

    def _step(self, model, tracker, tstep, plasticity):
        if model.uid in tracker:
            return

        xW = self.compute_incoming_activity(model)
        model.step(xW)

        if plasticity:
            self.update_W(model, ('STEP', tstep))

        tracker.add(model.uid)

        if model.uid not in self.children_conn:
            return

        children = self.children_conn[model.uid]
        for child in children:
            self._step(child[0], tracker, tstep, plasticity)

    def trial_end(self, plasticity=True):
        for model in self.models:
            xW = self.compute_incoming_activity(model)
            model.trial_end(xW)

            if plasticity:
                self.update_W(model, ('TRIAL_END', None))

    def compute_incoming_activity(self, model):
        xW = np.zeros(len(model.neurons))
        parents = self.parents_conn[model.uid]
        for parent in parents:
            parent_model, W_pc, _ = parent
            xW += np.dot(W_pc, parent_model.neurons)
        return xW

    def update_W(self, model, t):
        parents = self.parents_conn[model.uid]
        for i, (parent, W_pc, update_W) in enumerate(parents):
            W_new = update_W(parent, model, W_pc, t)
            self.parents_conn[model.uid][i] = (parent, W_new, update_W)
            parent_children = self.children_conn[parent.uid]
            for j, parent_child in enumerate(parent_children):
                if parent_child[0].uid != model.uid:
                    continue
                self.children_conn[parent.uid][j] = (model, W_new, update_W)


class Simulation:
    def __init__(self, network):
        self.network = network

    def run_trials(self, trial_setup, get_input, n_trials, cb):
        '''
        trial: [("STEP_NAME", n_steps, plasticity), ...]
        get_input: ("STEP_NAME", timestep) -> input vector
        n: number of trials to run
        cb: (trial_num, "STEP_NAME", timestep, network) -> None
        '''
        for trial_num in range(1, n_trials+1):
            timestep = 0
            for (step_name, n_steps, is_plastic) in trial_setup:
                for sub_step in range(n_steps):
                    timestep += 1
                    inp = get_input(trial_num, step_name, timestep)
                    self.network.step(inp, timestep, is_plastic)
                    cb(trial_num, step_name, timestep, self.network)
            self.network.trial_end()
            cb(trial_num, "TRIAL_END", timestep, self.network)

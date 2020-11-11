import numpy as np


class Network:
    def __init__(self, fn_global_compute):
        self.models = {}  # { name: model }
        self.conn_external = {'W': [], 'model': None}
        # {parent_name: [(child, Wpc, update_W, state), ...]}
        self.children_conn = {}
        # {child_name: [(parent, Wpc, update_W, state), ...]}
        self.parents_conn = {}
        # Function to compute global signals
        self.compute_global_signals = fn_global_compute

    def define_inputs(self, W, model):
        '''
        Define external inputs.
        n: # of external inputs
        model: model that receives the inputs
        W: weights from inputs to model
        '''
        self.conn_external = {'W': W, 'model': model}
        self.parents_conn[model.name] = []

    def connect(self, parent, child, W, update_W, state):
        '''
        Connect models to create a network.
        model1: from model
        model2: to model
        W: weights that connect model1 to model2
        update_W: {f: (parent, child, W, ('STEP'|'TRIAL_END', time)) -> W', dat: {}}
        '''

        new_child = (child, W, update_W, state)
        new_parent = (parent, W, update_W, state)

        if parent.name in self.children_conn:
            self.children_conn[parent.name].append(new_child)
        else:
            self.children_conn[parent.name] = [new_child]

        if child.name in self.parents_conn:
            self.parents_conn[child.name].append(new_parent)
        else:
            self.parents_conn[child.name] = [new_parent]

        if parent.name not in self.models:
            self.models[parent.name] = parent
        if child.name not in self.models:
            self.models[child.name] = child

    def step(self, external_inputs, tstep, plasticity=True):
        tracker = set()

        model = self.conn_external['model']
        xW = self.compute_incoming_activity(model)
        xW += np.dot(self.conn_external['W'], external_inputs)
        model.step(xW, plasticity)

        if plasticity:
            self.update_W(model, ('STEP', tstep))

        tracker.add(model.name)

        if model.name not in self.children_conn:
            return

        children = self.children_conn[model.name]
        for child in children:
            self._step(child[0], tracker, tstep, plasticity)

    def _step(self, model, tracker, tstep, plasticity):
        if model.name in tracker:
            return

        xW = self.compute_incoming_activity(model)
        model.step(xW, plasticity)

        if plasticity:
            self.update_W(model, ('STEP', tstep))

        tracker.add(model.name)

        if model.name not in self.children_conn:
            return

        children = self.children_conn[model.name]
        for child in children:
            self._step(child[0], tracker, tstep, plasticity)

    def trial_end(self, expected_output, plasticity=True):
        global_signals = self.compute_global_signals(
            self, expected_output, plasticity)
        for model in self.models.values():
            model.trial_end(global_signals, plasticity)
            if plasticity and model.name in self.parents_conn:
                self.update_W(model, ('TRIAL_END', global_signals))

    def compute_incoming_activity(self, model):
        xW = np.zeros(len(model.neurons))
        parents = self.parents_conn[model.name]
        for parent in parents:
            parent_model, W_pc, _, _ = parent
            xW += np.dot(W_pc, parent_model.neurons)
        return xW

    def update_W(self, model, step):
        parents = self.parents_conn[model.name]
        for i, (parent, W_pc, update_W, state) in enumerate(parents):
            (state_new, W_new) = update_W(step, state, W_pc, parent, model)
            self.parents_conn[model.name][i] = (
                parent, W_new, update_W, state_new)
            parent_children = self.children_conn[parent.name]
            for j, parent_child in enumerate(parent_children):
                if parent_child[0].name != model.name:
                    continue
                self.children_conn[parent.name][j] = (
                    model, W_new, update_W, state_new)


class Simulation:
    def __init__(self, network):
        self.network = network

    def run_trials(self, trial_setup, get_input, get_output, n_trials, cb):
        '''
        trial: [("STEP_NAME", n_steps, plasticity), ...]
        get_input: (trial_num, "STEP_NAME", timestep) -> input vector
        n: number of trials to run
        cb: (trial_num, "STEP_NAME", timestep, network) -> None
        '''
        for trial_num in range(1, n_trials+1):
            timestep = 0
            for (step_name, n_steps, is_plastic) in trial_setup:
                for _ in range(n_steps):
                    timestep += 1
                    inp = get_input(trial_num, step_name, timestep)
                    self.network.step(inp, timestep, is_plastic)
                    expected_output = get_output(trial_num, inp)
                    cb(trial_num, step_name, timestep,
                       inp, expected_output, self.network)
            expected_output = get_output(trial_num, inp)
            self.network.trial_end(expected_output)
            cb(trial_num, "TRIAL_END", timestep,
               inp, expected_output, self.network)

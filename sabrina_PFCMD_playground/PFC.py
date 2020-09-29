import numpy as np

G_BASE = 0.75
DT_DIV_TAU = 1. / 30  # dt = 1, tau = 30 miliseconds
ALPHA_EXCT = 0.05  # excitation running average factor
ALPHA_TRACE = 0.33
ETA = 0.5

N_NEURONS = 200
N_INPUT_NEURONS = 2
G = 1.5


class PFC():
    def __init__(self):
        self.n_neurons = N_NEURONS
        size_PFC = (self.n_neurons, self.n_neurons)
        self.firing_rates = np.zeros(size_PFC)

        # initialize PFC internal connections
        weights_factor = G**2 / N_NEURONS
        self.weights_PFC = np.random.normal(size=size_PFC) * weights_factor

        # initialize input weights
        self.input_weights = np.random.uniform(-1, 1, (2, self.n_neurons))

        # assign a random output neuron
        self.output_neuron = np.random.randint(N_NEURONS, size=2)

        # variables tracked over trials & re-inialized each trial
        self.excitation = np.random.uniform(-0.1, 0.1, size_PFC)
        self.excitation_avg = np.zeros(size_PFC)
        self.eligibility_trace = np.zeros(size_PFC)
        self.output_activity = []

    def initialize_trial(self):
        size_PFC = (self.n_neurons, self.n_neurons)
        self.excitation = np.random.uniform(-0.1, 0.1, size_PFC)
        self.firing_rates = self.activate(self.excitation)
        self.excitation_avg = np.zeros(size_PFC)
        self.eligibility_trace = np.zeros(size_PFC)
        self.output_activity = []

    def activate(self, x):
        return np.tanh(x)

    def step_ms(self, input_weights=None, input_activity=None, reward_on=False):
        # compute PFC activity recurrence to PFC
        excitation_add = np.dot(self.weights_PFC, self.firing_rates)

        if input_activity is not None:
            excitation_add += np.dot(input_activity, input_weights)
        self.excitation += DT_DIV_TAU * (-self.excitation + excitation_add)

        prev_firing_rates = self.firing_rates
        self.firing_rates = self.activate(self.excitation)

        excitation_delta = self.excitation - self.excitation_avg
        self.excitation_avg = ALPHA_EXCT * self.excitation_avg + \
            (1.0 - ALPHA_EXCT) * self.excitation

        # NOTE: As of Python 3.5+, @ does matrix multiplication
        potential = (prev_firing_rates @ excitation_delta) ** 3
        self.eligibility_trace = self.eligibility_trace + potential

        if reward_on:
            (i, j) = self.output_neuron
            output_activity = self.firing_rates[i, j]
            self.output_activity.append(output_activity)

    def update(self, expected_output):
        output_avg = np.mean(self.output_activity)
        print(f'output avg: {output_avg}, expected: {expected_output}')
        error = expected_output - output_avg
        self.weights_PFC += ETA * self.eligibility_trace * error
        return error

    def run_trial(self, stimulus1, stimulus2, expected_output):
        self.initialize_trial()

        for _ in range(200):
            self.step_ms(self.input_weights, stimulus1)

        for _ in range(200):
            self.step_ms()

        for _ in range(200):
            self.step_ms(self.input_weights, stimulus2)

        for _ in range(200):
            self.step_ms()

        for _ in range(200):
            self.step_ms(reward_on=True)

        return self.update(expected_output)

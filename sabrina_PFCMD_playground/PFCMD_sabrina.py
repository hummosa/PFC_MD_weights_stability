# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Model of PFC-MD with learning on output weights as in Rikhye, Gilra and Halassa, Nature Neuroscience 2018."""

# sabrina

import numpy as np

G_BASE = 0.75
G_PFC = 6.


class PFCMD():
    def __init__(self, do_save_data):
        np.random.seed([0])

        self.n_cues = 2
        self.match_contexts = [1]
        self.n_contexts = len(self.match_contexts)
        self.n_neurons_per_cue = 200
        # TODO include n_contexts into n_neurons
        self.n_neurons = self.n_neurons_per_cue * (self.n_cues + 1)
        self.n_output_neurons = 2

        self.tau = 0.02
        self.dt = 0.001
        self.trial_timesteps = 200
        self.cue_timesteps = 100
        self.nose_std = 1e-3
        self.learning_rate = 5e-6
        self.tau_error = 0.001

        self.do_save_data = do_save_data

        # construct the network
        self.init_PFC_MD()
        self.init_MD_PFC()
        self.init_PFC()
        self.init_inputs()

    def init_PFC_MD(self):
        # initialize PFC -> MD connections and weights
        self.weights_PFC_MD = np.zeros((self.n_cues, self.n_neurons))
        for i_task in np.arange(self.n_cues):
            s = i_task * self.n_neurons_per_cue * 2
            t = s + self.n_neurons_per_cue * 2
            self.weights_PFC_MD[i_task, s:t] = 1. / self.n_neurons_per_cue

    def init_MD_PFC(self):
        # initialize MD -> PFC connections and weights
        self.weights_MD_PFC = np.ones((self.n_neurons, self.n_cues)) * -10.
        for i_task in np.arange(self.n_cues):
            s = i_task * self.n_neurons_per_cue * 2
            t = s + self.n_neurons_per_cue * 2
            self.weights_MD_PFC[s:t, i_task] = G_PFC / G_BASE
        self.tau_MD = self.tau

    def init_PFC(self):
        # initialize PFC internal connections
        size = (self.n_neurons, self.n_neurons)
        self.weights_PFC = np.random.normal(size=size)
        self.weights_PFC *= G_BASE / np.sqrt(self.n_neurons_per_cue * 2)
        # make mean input to each row zero (avoids saturation for positive-only rates)
        #   see Nicola & Clopath 2016
        self.weights_PFC -= np.mean(self.weights_PFC, axis=1)[:, np.newaxis]

    def init_inputs(self):
        # initialize inputs weights
        self.weights_in = np.zeros((self.n_neurons, self.n_cues))
        self.cues = [0.5, 1.]
        cue1, cue2 = self.cues
        cue_factor = 1.5
        for i_cue in np.arange(self.n_cues):
            s = i_cue * self.n_neurons_per_cue
            t = s + self.n_neurons_per_cue
            size = self.n_neurons_per_cue
            weights = np.random.uniform(cue1, cue2, size=size) * cue_factor
            self.weights_in[s:t, i_cue] = weights

    def activation(self, x):
        ''' Neuron activation function - only allow positive rates'''
        return np.clip(np.tanh(x), 0, None)

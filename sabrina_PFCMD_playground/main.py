import numpy as np
import random
import matplotlib.pyplot as plt

from PFCMD_sabrina import PFCMD


def train(model, n_ittr_per_task):
    n_ittr = n_ittr_per_task * model.n_cues

    # initialize the output weights
    size = (model.n_output_neurons, model.n_neurons)
    output_weights = np.random.uniform(-1, 1, size=size) / model.n_neurons

    MSEs = np.zeros(n_ittr)
    for i_ittr in range(n_ittr):
        print(f'Running training itteration {i_ittr} / {n_ittr}')

        # NOTE: For now I am randomly selecting a cue for training
        cue_idx = np.random.randint(0, len(model.cues) - 1)
        cue = model.cues[cue_idx]
        target = np.equal(model.cues, cue).astype(float)

        output_weights, errors = simulate_cue(
            model, output_weights, cue, cue_idx, target)
        MSEs[i_ittr] = np.mean(errors * errors)
        # print(errors[1], (errors * errors)[1])
    plt.plot(MSEs)
    plt.show()


def simulate_cue(model, output_weights, cue, cue_idx, target):
    errors = np.empty((model.trial_timesteps, model.n_output_neurons))
    error_smoothed = np.empty(model.n_output_neurons)

    cue_activity = cue * target
    x_inp = np.random.uniform(0, 0.1, size=(model.n_neurons))
    out_inp = np.zeros(shape=model.n_output_neurons)
    for timestep in range(model.trial_timesteps):
        # compute output layer input from random PFC activity
        PFC_rand_out = model.activation(x_inp)

        out_add = np.dot(output_weights, PFC_rand_out)

        # compute activity into PFC neurons from MD
        out_MD = np.zeros(model.n_cues)
        out_MD[cue_idx] = 1.
        MD_PFC_dot = np.dot(model.weights_MD_PFC, out_MD)
        # compute activity into PFC from PFC (random activity)
        PFC_PFC_dot = np.dot(model.weights_PFC, PFC_rand_out)
        # compute total input activity to PFC
        in_PFC = MD_PFC_dot + PFC_PFC_dot

        if timestep < model.cue_timesteps:
            in_PFC += np.dot(model.weights_in, cue_activity)

        x_inp += model.dt/model.tau * (-x_inp + in_PFC)

        out_inp += model.dt/model.tau * (-out_inp + out_add)
        output = model.activation(out_inp)
        error = target - output
        errors[timestep, :] = error

        # apply training
        error_smoothed += model.dt / \
            model.tau_error * (-error_smoothed + error)
        output_weights += model.learning_rate * \
            np.outer(error_smoothed, PFC_rand_out)

    return output_weights, errors


model = PFCMD(False)
train(model, 100)

import numpy as np
# from psychopy.visual.grating import GratingStim
from scipy.stats import truncnorm


def create_block_stimuli(num_trials,
                         block_side_bias_probabilities,
# in tactile task, we fix the trial strength, no need to have probability any more.
                    #     possible_block_mu,
                    #     possible_block_mu_probs,
                         max_rnn_steps_per_trial):

    # sample standard normal noise for both left and right stimuli
    # in tactile task, there is no noise
    sampled_stimuli=np.zeros((num_trials,max_rnn_steps_per_trial,2))
    # sampled_stimuli = np.random.normal(
    #    loc=0.,
    #    scale=1.,
    #    size=(num_trials, max_rnn_steps_per_trial, 2))

    # now, determine which sides will have signal
    # -1 is down, +1 is up
    # these values also control the means of the distributions, not the case in tactile task
    temp_number=np.random.choice([0.5,0.7,0.9])#association level can be 0.5, 0.7, or 0.9
    block_side_bias_probabilities=(temp_number, 1-temp_number)
    print(block_side_bias_probabilities)
    print(num_trials)
    signal_sides_indices = np.random.choice(
        [0, 1],
        p=block_side_bias_probabilities,
        size=(num_trials, 1))
    signal_sides_indices = np.repeat(
        signal_sides_indices,
        axis=-1,
        repeats=max_rnn_steps_per_trial)

    trial_sides = 2*signal_sides_indices - 1

#     block_mu = np.random.choice(
#        possible_block_mu,
#        p=possible_block_mu_probs,
#        size=(num_trials, 1))
    # in tactile task, set the trial strength to be constant.
    trial_strengths = 0.5*np.ones((num_trials,1));
    # hold trial strength constant for duration of trial
    trial_strengths = np.repeat(
        a=trial_strengths,
        repeats=max_rnn_steps_per_trial,
        axis=1)

 #   signal = np.random.normal(
 #       loc=trial_strengths,
 #       scale=np.ones_like(trial_strengths))

    # add signal to noise
    # rely on nice identity matrix trick for converting boolean signal_side_indices
    # to one-hot encoded for indexing
    #sampled_stimuli[np.eye(2)[signal_sides_indices].astype(bool)] = signal.flatten()
    sampled_stimuli[np.eye(2)[signal_sides_indices].astype(bool)] = trial_strengths.flatten()

    output = dict(
        stimuli=sampled_stimuli,
        stimuli_strengths=trial_strengths,
        trial_sides=trial_sides)

    return output


# class GratingCreator(StimulusCreator):
#
# TODO: fix this to create Gabor patches as specified
#
#     def __init__(self,
#                  trial_strengths=None,
#                  trial_strength_probs=None):
#
#         # defaults
#         if trial_strengths is None:
#             trial_strengths = [1, 0.5, 0.25, 0.125, 0.06, 0]
#         if trial_strength_probs is None:
#             trial_strength_probs = [2 / 11, 2 / 11, 2 / 11, 2 / 11, 2 / 11, 1 / 11]
#
#         super(GratingCreator, self).__init__(
#             trial_strengths=trial_strengths,
#             trial_strength_probs=trial_strength_probs)
#
#     def create_block_stimuli(self,
#                              block_num_trials,
#                              block_side_bias_probabilities):
#
#         sampled_strength = np.random.choice(
#             self.trial_strengths,
#             p=self.trial_strength_probs)
#
#
#         if side == 'left':
#             return GratingStim(tex='sin', mask='gauss')
#         elif side == 'right':
#             return GratingStim(tex='sin', mask='gauss')
#         else:
#             raise ValueError('Impermissible side: ', side)

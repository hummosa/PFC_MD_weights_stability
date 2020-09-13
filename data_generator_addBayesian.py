import numpy as np

class data_generator():
    def __init__(self, local_Ntrain):
        self.matches = { #cause a non-match (1.) every so many matches.
        '90': np.array([0. if (i+1)%10!=0 else 1. for i in range(local_Ntrain) ]),
        '75': np.array([0. if (i+1)%4!=0  else 1. for i in range(local_Ntrain)  ]),
        '50': np.array([0. if (i+1)%2!=0  else 1. for i in range(local_Ntrain)  ]),
        '25': np.array([1. if (i+1)%4!=0  else 0. for i in range(local_Ntrain)  ]),
        '10': np.array([1. if (i+1)%10!=0 else 0. for i in range(local_Ntrain) ]),
         }

        self.task_data_gen = {
        0: self.trial_generator(self.matches['90']),
        1: self.trial_generator(self.matches['10']),
        2: self.trial_generator(self.matches['50']),
        3: self.trial_generator(self.matches['25']),
        4: self.trial_generator(self.matches['75']),
        }

    def trial_generator(self, non_matches):
        for non_match in non_matches:
            yield (non_match)



def compute_optimal_bayesian_coupled_observer(session_data):
    # see https://github.com/bayespy/bayespy/issues/28
    non_blank_data = session_data[(session_data.up_stimulus != 0) &
                                  (session_data.down_stimulus != 0)]

    from bayespy.nodes import Categorical, CategoricalMarkovChain, Dirichlet, \
        Gaussian, Mixture, Wishart

    num_latent_variables = 4
    initial_state_probs = Dirichlet(np.array([.4, .1, .1, .4]))
    mu=np.random.choice([0.5, 0.7, 0.9])
    transition_probs = Dirichlet(10 * np.array([
        [0.98 * mu, 0.02 * mu, 0.98 * mu, 0.02 * (1-mu)],  # b_n = L, s_n = L
        [0.02 * (1-mu), 0.98 * (1-mu), 0.02 * (1-mu), 0.98 * (1-mu)],  # b_n = R, s_n = L
        [0.98 * (1-mu), 0.02 * (1-mu), 0.98 * (1-mu), 0.02 * (1-mu)],  # b_n = L, s_n = R
        [0.02 * mu, 0.98 * mu, 0.02 * mu, 0.98 * mu],  # b_n = R, s_n = R
    ]))

    latents = CategoricalMarkovChain(
        pi=initial_state_probs,
        A=transition_probs,
        states=len(non_blank_data))

    # approximate observation as mixture 
    #mu = Gaussian(
    #    np.zeros(1),
    #    np.identity(1),
    #    plates=(num_latent_variables,))
    #Lambda = Wishart(
    #    1,
    #    1e-6 * np.identity(1))

    observations = Mixture(latents, mu)

    diff_obs = non_blank_data['up_stimulus'] - non_blank_data['down_stimulus']
    # reshape to (number of non-blank dts, 1)
    diff_obs = np.expand_dims(diff_obs.values, axis=1)
    observations.observe(diff_obs)

    # I want to specify the means and variance of mu, but I can't figure out how
    avg_diff = np.mean(diff_obs[non_blank_data.trial_side == 1.])
    #mu.u[0] = avg_diff * np.array([-1., -1., 1., 1.])[:, np.newaxis]  # shape (4, 1)
    #mu.u[1] = np.ones(shape=(num_latent_variables, 1, 1))  # shape (4, 1, 1)

    # Reasonable initialization for Lambda
    #Lambda.initialize_from_value(np.identity(1))

    from bayespy.inference import VB
    #Q = VB(observations, latents, transition_probs, initial_state_probs, Lambda)
    Q = VB(observations, latents, transition_probs, initial_state_probs)

    # use deterministic annealing to reduce sensitivity to initial conditions
    # https://www.bayespy.org/user_guide/advanced.html#deterministic-annealing
    beta = 0.1
    while beta < 1.0:
        beta = min(beta * 1.5, 1.0)
        Q.set_annealing(beta)
        Q.update(repeat=250, tol=1e-4)

    # recover transition posteriors
    logging.info('Coupled Bayesian Observer State Space:\n'
                 'b_n=L & s_n=L\n'
                 'b_n=R & s_n=L\n'
                 'b_n=L & s_n=R\n'
                 'b_n=R & s_n=R')
    logging.info('True Initial block side: {}\tTrue Initial trial side: {}'.format(
        session_data.loc[0, 'block_side'],
        session_data.loc[0, 'trial_side']))
    initial_state_probs_posterior = Categorical(initial_state_probs).get_moments()[0]
    logging.info(f'Coupled Bayesian Observer Initial State Posterior: \n{str(initial_state_probs_posterior)}')
    transition_probs_posterior = Categorical(transition_probs).get_moments()[0]
    logging.info(f'Coupled Bayesian Observer Transition Parameters Posterior: \n{str(transition_probs_posterior)}')

    from bayespy.inference.vmp.nodes.categorical_markov_chain import CategoricalMarkovChainToCategorical
    latents_posterior = CategoricalMarkovChainToCategorical(latents).get_moments()[0]

    optimal_coupled_observer_results = dict(
        coupled_observer_initial_state_posterior=initial_state_probs_posterior,
        coupled_observer_transition_posterior=transition_probs_posterior,
        coupled_observer_latents_posterior=latents_posterior
    )

    return optimal_coupled_observer_results


def compute_optimal_bayesian_observer_block_side(session_data,
                                                 env):
    initial_state_probs = np.array([
        0.5, 0.5])

    transition_probs = np.array([
        [0.98, 0.02],
        [0.02, 0.98]])

    emission_probs = np.array([
        [mu, 1-mu],
        [1-mu, mu]])

    trial_end_data = session_data[session_data.trial_end == 1.]
    latent_conditional_probs = np.zeros(shape=(len(trial_end_data), 2))
    trial_sides = ((1 + trial_end_data.trial_side.values) / 2).astype(np.int)

    # joint probability p(x_1, y_1)
    curr_joint_prob = np.multiply(
        emission_probs[trial_sides[0], :],
        initial_state_probs)

    for i, trial_side in enumerate(trial_sides[:-1]):
        # normalize to get P(b_n | s_{<=n})
        # np.sum(curr_joint_prob) is marginalizing over b_{n} i.e. \sum_{b_n} P(b_n, s_n |x_{<=n-1})
        curr_latent_conditional_prob = curr_joint_prob / np.sum(curr_joint_prob)
        latent_conditional_probs[i] = curr_latent_conditional_prob

        # P(y_{t+1}, x_{t+1} | x_{<=t})
        curr_joint_prob = np.multiply(
            emission_probs[trial_sides[i + 1], :],
            np.matmul(transition_probs, curr_latent_conditional_prob))

    # right block posterior, right block prior
    session_data['bayesian_observer_block_posterior_right'] = np.nan
    session_data.loc[trial_end_data.index, 'bayesian_observer_block_posterior_right'] = \
        latent_conditional_probs[:, 1]
    session_data['bayesian_observer_block_prior_right'] = \
        session_data['bayesian_observer_block_posterior_right'].shift(1)

    # right stimulus prior
    session_data['bayesian_observer_stimulus_prior_right'] = np.nan
    block_prior_indices = ~pd.isna(session_data['bayesian_observer_block_prior_right'])
    bayesian_observer_stimulus_prior_right = np.matmul(latent_conditional_probs[:-1, :], emission_probs.T)
    session_data.loc[
        block_prior_indices, 'bayesian_observer_stimulus_prior_right'] = bayesian_observer_stimulus_prior_right[:, 1]

    # manually specify that first block prior and first stimulus prior should be 0.5
    # before evidence, this is the correct prior
    session_data.loc[0, 'bayesian_observer_block_prior_right'] = 0.5
    session_data.loc[0, 'bayesian_observer_stimulus_prior_right'] = 0.5


def compute_optimal_bayesian_observer_trial_side(session_data,
                                                 env):
    #strength_means = np.sort(session_data.signed_trial_strength.unique())
    mu_means = np.sort([0.5,0.7,0.9])
    prob_mu = env.possible_block_mu_probs

    # P(mu_n | s_n) as a matrix with shape (2 * number of stimulus strengths - 1, 2)
    # - 1 is for stimulus strength 0, which both stimulus sides can generate
    #prob_mu_given_stim_side = np.zeros(shape=(len(strength_means), 2))
    prob_mu_given_stim_side = np.zeros(shape=(3, 2))
    prob_mu_given_stim_side[:len(prob_mu), 0] = prob_mu[::-1]
    #prob_mu_given_stim_side[len(prob_mu) - 1:, 1] = 1-prob_mu[1]
    prob_mu_given_stim_side[:len(prob_mu) , 1] = [1-prob_mu[0],1-prob_mu[1],1-prob_mu[2]]

    diff_obs = session_data['down_stimulus'] - session_data['up_stimulus']

    session_data['bayesian_observer_stimulus_posterior_right'] = np.nan
    for (session_idx, block_idx, trial_idx), trial_data in session_data.groupby([
        'session_index', 'block_index', 'trial_index']):
        bayesian_observer_stimulus_prior_right = trial_data[
            'bayesian_observer_stimulus_prior_right'].iloc[0]
        optimal_stim_prior = np.array([
            1 - bayesian_observer_stimulus_prior_right,
            bayesian_observer_stimulus_prior_right])

        # P(\mu_n, s_n | history) = P(\mu_n | s_n) P(s_n | history)
        # shape = (# of possible signed stimuli strengths, num trial sides)
        stim_side_strength_joint_prob = np.einsum(
            'ij,j->ij',
            prob_mu_given_stim_side,
            optimal_stim_prior)

        # exclude blank dts
        dt_indices = trial_data.iloc[env.rnn_steps_before_obs:].index
        trial_diff_obs = diff_obs[trial_data.index].values[
                         env.rnn_steps_before_obs:]

        # P(o_t | \mu_n, s_n) , also = P(o_t | \mu_n)
        # shape = (num of observations, # of possible signed stimuli strengths)
        individual_diff_obs_likelihood = scipy.stats.norm.pdf(
            np.expand_dims(trial_diff_obs, axis=1),
            loc=mu_means,
            scale=np.sqrt(2) * np.ones_like(mu_means))  # scale is std dev

        # P(o_{<=t} | \mu_n, s_n) = P(o_{<=t} | \mu_n)
        # shape = (num of observations, # of possible signed stimuli strengths)
        running_diff_obs_likelihood = np.cumprod(
            individual_diff_obs_likelihood,
            axis=0)

        # P(o_{<=t}, \mu_n, s_n | history) = P(o_{<=t} | \mu_n, s_n) P(\mu_n, s_n | history)
        # shape = (num of observations, # of possible signed stimuli strengths, # of trial sides i.e. 2)
        running_diff_obs_stim_side_strength_joint_prob = np.einsum(
            'ij,jk->ijk',
            running_diff_obs_likelihood,
            stim_side_strength_joint_prob)
        assert len(running_diff_obs_stim_side_strength_joint_prob.shape) == 3

        # marginalize out mu_n
        # shape = (num of observations, # of trial sides i.e. 2)
        running_diff_obs_stim_side_joint_prob = np.sum(
            running_diff_obs_stim_side_strength_joint_prob,
            axis=1)
        assert len(running_diff_obs_stim_side_joint_prob.shape) == 2

        # normalize by p(o_{<=t})
        # shape = (num of observations, # of trial sides i.e. 2)
        running_diff_obs_marginal_prob = np.sum(
            running_diff_obs_stim_side_joint_prob,
            axis=1)
        assert len(running_diff_obs_marginal_prob.shape) == 1

        # shape = (num of observations, # of trial sides i.e. 2)
        optimal_stim_posterior = np.divide(
            running_diff_obs_stim_side_joint_prob,
            np.expand_dims(running_diff_obs_marginal_prob, axis=1)  # expand to broadcast
        )
        assert np.allclose(
            np.sum(optimal_stim_posterior, axis=1),
            np.ones(len(optimal_stim_posterior)))

        session_data.loc[dt_indices, 'bayesian_observer_stimulus_posterior_right'] = \
            optimal_stim_posterior[:, 1]

    # determine whether action was taken
    session_data['bayesian_observer_action_taken'] = \
        ((session_data['bayesian_observer_stimulus_posterior_right'] > 0.9) \
         | (session_data['bayesian_observer_stimulus_posterior_right'] < 0.1)
         ).astype(np.float)

    # next, determine which action was taken (if any)
    session_data['bayesian_observer_action_side'] = \
        2 * session_data['bayesian_observer_stimulus_posterior_right'].round() - 1
    # keep only trials in which action would have actually been taken
    # session_data.loc[session_data['bayesian_observer_action_taken'] == 0.,
    #                  'bayesian_observer_action_side'] = np.nan

    # next, determine whether action was correct
    session_data['bayesian_observer_correct_action_taken'] = \
        session_data['bayesian_observer_action_side'] == session_data['trial_side']
    session_data['bayesian_observer_reward'] = \
        2. * session_data['bayesian_observer_correct_action_taken'] - 1.
    # if action was not taken, set correct to 0
    # session_data.loc[session_data['bayesian_observer_action_taken'] == 0.,
    #                  'bayesian_observer_correct_action_taken'] = 0.

    # logging fraction of correct actions
    logging.info('Bayesian Observer')
    bayesian_observer_correct_action_taken_by_total_trials = session_data[
        session_data.trial_end == 1.]['bayesian_observer_correct_action_taken'].mean()
    logging.info(f'# Correct Trials / # Total Trials: '
                 f'{bayesian_observer_correct_action_taken_by_total_trials}')
    # bayesian_observer_action_taken_by_total_trials = session_data[
    #     session_data.trial_end == 1.]['bayesian_observer_action_taken'].mean()
    # logging.info(f'# Correct Trials / # Total Trials: '
    #              f'{bayesian_observer_action_taken_by_total_trials}')


            
import matplotlib.pyplot as plt
# plt.get_backend()
# 'Qt5Agg'
# fig, ax = plt.subplots()
# mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
# mngr.window.setGeometry(50,100,640, 545)
    
# # note that instead of mngr = get_current_fig_manager(), we can also use fig.canvas.manager 

# geom = mngr.window.geometry()
# x,y,dx,dy = geom.getRect()

def move_figure(figh, col=1, position="top"):
    '''
    Move and resize a window to a set of standard positions on the screen.
    Possible positions are:
    top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
    '''

    mgr = figh.canvas.manager
    
    fig_h = mgr.canvas.height()
    fig_w = mgr.canvas.width()
    mgr.full_screen_toggle()  # primitive but works to get screen size
    py = mgr.canvas.height()
    px = mgr.canvas.width()

    d = 10  # width of the window border in pixels
    num_of_cols            = 6
    w_col = (px//num_of_cols) + d*2

    top = (d*4)
    bottom = py-fig_h-(d*4) 
    vertical_pos = top if position is 'top' else bottom
    mgr.window.setGeometry(d+(w_col*col), vertical_pos , fig_w, fig_h)

    # if position == "col1":
    #     # x-top-left-corner, y-top-left-corner, x-width, y-width (in pixels)
    #     mgr.window.setGeometry(d, 4*d, px - 2*d, py/2 - 4*d)
    # elif position == "bottom":
    #     mgr.window.setGeometry(d, py/2 + 5*d, px - 2*d, py/2 - 4*d)
    # elif position == "left":
    #     mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py - 4*d)
    # elif position == "right":
    #     mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py - 4*d)
    # elif position == "top-left":
    #     mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "top-right":
    #     mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "bottom-left":
    #     mgr.window.setGeometry(d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "bottom-right":
    #     mgr.window.setGeometry(px/2 + d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)

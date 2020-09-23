from data_generator import data_generator
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats, optimize, interpolate
from scipy.ndimage.interpolation import shift
def __init__(self):
    self.RNGSEED = 1
    np.random.seed([self.RNGSEED])
    self.data_generator=data_generator

def get_next_target(self, taski):
    return next(self.data_generator.task_data_gen[taski])


def compute_optimal_bayesian_coupled_observer(cue, target):
    # see https://github.com/bayespy/bayespy/issues/28
 #   non_blank_data = session_data[(session_data.up_stimulus != 0) &
 #                                 (session_data.down_stimulus != 0)]

    from bayespy.nodes import Categorical, CategoricalMarkovChain, Dirichlet, \
        Gaussian, Mixture, Wishart
    #latent variables dimension is 4 because b_n can be match or nonmatch, while s_n also can be match or nonmatch
    num_latent_variables = 4
    initial_state_probs = Dirichlet(np.array([.4, .1, .1, .4]))
    #mu is the association level
    muini=np.random.choice([0.5, 0.7, 0.9])
    transition_probs = Dirichlet(10 * np.array([
        [0.98 * muini, 0.02 * muini, 0.98 * muini, 0.02 * (1-muini)],  # b_n = L, s_n = L
        [0.02 * (1-muini), 0.98 * (1-muini), 0.02 * (1-muini), 0.98 * (1-muini)],  # b_n = R, s_n = L
        [0.98 * (1-muini), 0.02 * (1-muini), 0.98 * (1-muini), 0.02 * (1-muini)],  # b_n = L, s_n = R
        [0.02 * muini, 0.98 * muini, 0.02 * muini, 0.98 * muini],  # b_n = R, s_n = R
    ]))

    latents = CategoricalMarkovChain(
        pi=initial_state_probs,
        A=transition_probs,
        states=len(cue))

    # approximate observation as mixture 
    mu = Gaussian(
        np.zeros(1),
        np.identity(1),
        plates=(num_latent_variables,))
    Lambda = Wishart(
        1,
        1e-6 * np.identity(1))

    observations = Mixture(latents, Gaussian, mu, Lambda)
#    observations = Mixture(latents, mu)
    targetdata=np.zeros(shape=(500,1))
    diff_obs=np.zeros(shape=(500, 1))
    for i in range(500):
#       if target[i]==0:
#        targetdata[i]=cue[i]
#       if target[i]==1:
#        targetdata[i]=1-cue[i]
       if cue[i]>=target[i]:
        diff_obs[i] = cue[i] - target[i]
       if cue[i]<target[i]:
        diff_obs[i] = target[i] - cue[i]
    # reshape to (number of non-blank dts, 1)
    # diff_obs = np.expand_dims(diff_obs.values, axis=1)
    observations.observe(diff_obs)

    # I want to specify the means and variance of mu, but I can't figure out how
    #avg_diff = np.mean(diff_obs[non_blank_data.trial_side == 1.])
    #mu.u[0] = avg_diff * np.array([-1., -1., 1., 1.])[:, np.newaxis]  # shape (4, 1)
    #mu.u[1] = np.ones(shape=(num_latent_variables, 1, 1))  # shape (4, 1, 1)

    # Reasonable initialization for Lambda
    Lambda.initialize_from_value(np.identity(1))

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
#    logging.info('Coupled Bayesian Observer State Space:\n'
#                 'b_n=L & s_n=L\n'
#                 'b_n=R & s_n=L\n'
#                 'b_n=L & s_n=R\n'
#                 'b_n=R & s_n=R')
#    logging.info('True Initial block side: {}\tTrue Initial trial side: {}'.format(
#        session_data.loc[0, 'block_side'],
#        session_data.loc[0, 'trial_side']))
    initial_state_probs_posterior = Categorical(initial_state_probs).get_moments()[0]
#    logging.info(f'Coupled Bayesian Observer Initial State Posterior: \n{str(initial_state_probs_posterior)}')
    transition_probs_posterior = Categorical(transition_probs).get_moments()[0]
#    logging.info(f'Coupled Bayesian Observer Transition Parameters Posterior: \n{str(transition_probs_posterior)}')

    from bayespy.inference.vmp.nodes.categorical_markov_chain import CategoricalMarkovChainToCategorical
    latents_posterior = CategoricalMarkovChainToCategorical(latents).get_moments()[0]

    optimal_coupled_observer_results = dict(
        coupled_observer_initial_state_posterior=initial_state_probs_posterior,
        coupled_observer_transition_posterior=transition_probs_posterior,
        coupled_observer_latents_posterior=latents_posterior
    )

    return optimal_coupled_observer_results

def compute_optimal_bayesian_observer_block_side(cue, target):
    initial_state_probs = np.array([
        0.5, 0.5])

    transition_probs = np.array([
        [0.98, 0.02],
        [0.02, 0.98]])
    
    muini=np.random.choice([0.5, 0.7, 0.9])

    emission_probs = np.array([
        [muini, 1-muini],
        [1-muini, muini]])

    #trial_end_data = session_data[session_data.trial_end == 1.]
    #latent_conditional_probs = np.zeros(shape=(len(trial_end_data), 2))
    latent_conditional_probs = np.zeros(shape=(len(cue), 2))
    diff_obs=np.zeros(shape=(500, 2))
    #for i in range(500):
    #   print(cue[i],target[i])
    for i in range(500):
       if target[i]==1:
          diff_obs[i,0] = 1 
          diff_obs[i,1] = 0 
       if target[i]!=1:
          diff_obs[i,0] = 0 
          diff_obs[i,1] = 1 
    #trial_sides = ((1 + trial_end_data.trial_side.values) / 2).astype(np.int)
    trial_sides = diff_obs.astype(np.int)
    print('result is', trial_sides[0])
    # joint probability p(x_1, y_1)
    curr_joint_prob = np.multiply(
        emission_probs[trial_sides[0], :],
        initial_state_probs)
    print(curr_joint_prob)

    for i, trial_side in enumerate(trial_sides[:-1]):
        # normalize to get P(b_n | s_{<=n})
        # np.sum(curr_joint_prob) is marginalizing over b_{n} i.e. \sum_{b_n} P(b_n, s_n |x_{<=n-1})
        curr_latent_conditional_prob = curr_joint_prob[0] / np.sum(curr_joint_prob)
        latent_conditional_probs[i] = curr_latent_conditional_prob
    #    print(len(curr_latent_conditional_prob))
        # P(y_{t+1}, x_{t+1} | x_{<=t})
        curr_joint_prob = np.multiply(
            emission_probs[trial_sides[i + 1], :],
            np.matmul(transition_probs, curr_latent_conditional_prob))

    # right block posterior, right block prior
    bayesian_observer_block_posterior_nonmatch = latent_conditional_probs[:, 1]
    bayesian_observer_block_prior_nonmatch = shift(bayesian_observer_block_posterior_nonmatch, 1)

    # right stimulus prior
    bayesian_observer_stimulus_prior_nonmatch = np.matmul(latent_conditional_probs[:-1, :], emission_probs.T)

    # manually specify that first block prior and first stimulus prior should be 0.5
    # before evidence, this is the correct prior
    bayesian_observer_block_prior_nonmatch[0] = 0.5
    #bayesian_observer_stimulus_prior_nonmatch[0] = 0.5

    optimal_coupled_observer_block_side_result = dict(
    a= bayesian_observer_block_posterior_nonmatch,
    b= bayesian_observer_block_prior_nonmatch ,
    c= bayesian_observer_stimulus_prior_nonmatch 
    )

 
    return optimal_coupled_observer_block_side_result

def compute_optimal_bayesian_observer_trial_side(cue,target, block_side_result):
    mu_means = np.sort([0.5,0.7,0.9])
    prob_mu = [0.5,0.7,0.9]

    # P(mu_n | s_n) as a matrix with shape (2 * number of stimulus strengths - 1, 2)
    # - 1 is for stimulus strength 0, which both stimulus sides can generate
    #prob_mu_given_stim_side = np.zeros(shape=(len(strength_means), 2))
    prob_mu_given_stim_side = np.zeros(shape=(3, 2))
    prob_mu_given_stim_side[:len(prob_mu), 0] = prob_mu[::-1]
    #prob_mu_given_stim_side[len(prob_mu) - 1:, 1] = 1-prob_mu[1]
    prob_mu_given_stim_side[:len(prob_mu) , 1] = [1-prob_mu[0],1-prob_mu[1],1-prob_mu[2]]

    targetdata=np.zeros(shape=(500,1))
    diff_obs=np.zeros(shape=(500, 1))
    for i in range(500):
       if cue[i]>=target[i]:
        diff_obs[i] = cue[i] - target[i]
       if cue[i]<target[i]:
        diff_obs[i] = target[i] - cue[i]
    bayesian_observer_stimulus_posterior_nonmatch = np.zeros(shape=(500,1))

    for i in range(500):
        bayesian_observer_stimulus_prior_nonmatch = block_side_result['b'][i]
        optimal_stim_prior = np.array([
            1 - bayesian_observer_stimulus_prior_nonmatch,
            bayesian_observer_stimulus_prior_nonmatch])

        # P(\mu_n, s_n | history) = P(\mu_n | s_n) P(s_n | history)
        # shape = (# of possible signed stimuli strengths, num trial sides)
    #    print(len(prob_mu_given_stim_side))
    #    print(len(optimal_stim_prior))
        stim_side_strength_joint_prob = np.einsum(
            'ij,j->ij',
            prob_mu_given_stim_side,
            optimal_stim_prior)


        # P(o_t | \mu_n, s_n) , also = P(o_t | \mu_n)
        # shape = (num of observations, # of possible signed stimuli strengths)
        individual_diff_obs_likelihood = scipy.stats.norm.pdf(
            np.expand_dims(diff_obs[i], axis=1),
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

        bayesian_observer_stimulus_posterior_nonmatch[i] = optimal_stim_posterior[:, 1]

    # determine whether action was taken
    bayesian_observer_action_taken = np.zeros(shape=(500,1))
    for i in range(500):
       bayesian_observer_action_taken[i] = (bayesian_observer_stimulus_posterior_nonmatch[i] > 0.9 or bayesian_observer_stimulus_posterior_nonmatch[i] < 0.1).astype(np.float)

    # next, determine which action was taken (if any)
    bayesian_observer_action_side = 2 * bayesian_observer_stimulus_posterior_nonmatch.round() - 1

    # next, determine whether action was correct
    bayesian_observer_correct_action_taken=np.zeros(shape=(500,1))
    for i in range(500):
     bayesian_observer_correct_action_taken[i] = (bayesian_observer_action_side[i]==target[i])
    bayesian_observer_reward = 2. * bayesian_observer_correct_action_taken - 1.
    
    optimal_coupled_observer_stimulus_side_result = dict(
    a= bayesian_observer_action_taken,
    b= bayesian_observer_action_side ,
    c= bayesian_observer_correct_action_taken
    )

    return optimal_coupled_observer_stimulus_side_result

def main():
     Ncues=500
     main_data_generator = data_generator(local_Ntrain = 10000)
     cue = np.zeros(Ncues) #reset cue
     target = np.zeros(Ncues) #reset cue
     task = np.zeros(Ncues) #reset cue
    # cuei = np.random.randint(0,2) #up or down

     for i in range(Ncues):
       cuei = np.random.randint(0,2) #up or down
       if i<=Ncues/3:
         taski=2
       if i<=Ncues/3*2 and i>Ncues/3:
         taski=1
       if i>Ncues/3*2:
         taski=3
       non_match = main_data_generator.task_data_gen[taski] #get a match or a non-match response from the data_generator class
       if non_match: #flip
          targeti = 0 if cuei ==1 else 1
       else:
          targeti = cuei
       cue[i]=cuei
       target[i]=targeti
       task[i]=taski
       print(i, cuei, targeti, taski, sep=", ") 
     optimal_coupled_observer_result=compute_optimal_bayesian_coupled_observer(cue, target)
     optimal_coupled_observer_block_side_result=compute_optimal_bayesian_observer_block_side(cue, target)
     optimal_coupled_observer_stimulus_side_result=compute_optimal_bayesian_observer_trial_side(cue, target, optimal_coupled_observer_block_side_result)
     print(optimal_coupled_observer_result['coupled_observer_latents_posterior'])
     print(optimal_coupled_observer_block_side_result)
#     plt.plot(optimal_coupled_observer_block_side_result['a'])
#     plt.show()
#     print(cue, target, task) 


if __name__ == "__main__":
    main()


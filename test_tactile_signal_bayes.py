from data_generator import data_generator
import numpy as np
import matplotlib.pyplot as plt

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
    diff_obs=np.zeros(shape=(500, 1)) 
    for i in range(500):
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
     print(optimal_coupled_observer_result)
#     print(cue, target, task) 


if __name__ == "__main__":
    main()


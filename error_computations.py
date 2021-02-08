import numpy as np
import matplotlib.pyplot as plt


class OFC:
    ASSOCIATION_RANGE = np.linspace(0, 1, 21)

    def __init__(self):
        self.contexts = {}
        self.ctx = None
        self.prior = []

    def get_v(self):
        if (len(self.prior) == 0):
            return np.array([0.5, 0.5])
        else:
            idx_MAPs = np.where(self.prior == max(self.prior))[0]
            idx_MAP = np.random.choice(idx_MAPs)
            v1 = self.ASSOCIATION_RANGE[idx_MAP]
            v2 = 1 - v1
            return np.array([v1, v2])

    def set_context(self, ctx):
        self.contexts[self.ctx] = self.prior

        if (ctx in self.contexts):
            self.prior = self.contexts[ctx]
        else:
            n = len(self.ASSOCIATION_RANGE)
            self.prior = np.ones(n) / n  # Assume a uniform prior
        self.ctx = ctx

    def update_v(self, stimulus, choice, target):
        trial_type = "MATCH" if (stimulus == target).all() else "NON-MATCH"

        likelihood = list(map(lambda x:
                              x if trial_type == "MATCH" else (1-x), self.ASSOCIATION_RANGE))
        posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        self.prior = posterior

class Error_computations:
    ASSOCIATION_RANGE = np.linspace(0, 1, 2)

    def __init__(self, config):
        self.config = config
        self.contexts = {}
        self.ctx = None
        self.prior = np.array([0.5, 0.5])
        self.horizon = config.horizon
        self.trial_history = ["MATCH"] *2 

        self.follow = 'behavioral_context' # 'association_levels'
        if self.follow == 'association_levels':
            contexts = 5
            self.association_levels_ids = {'90':0, '70':1, '50':2, '30':3, '10':4}
        elif self.follow == 'behavioral_context':
            contexts = 2
            self.match_association_levels = {'90', '70', '50'}
                
        self.baseline_err = np.zeros(shape=contexts)
        self.Q_values = [0., 0.]
        self.Sabrina_Q_values = [0.5, 0.5]
        self.p_sm_snm_ns = np.array ([1/3, 1/3, 1/3])

    def get_v(self):
        return (np.array(self.prior))

    def set_context(self, ctx):
        self.prior = np.array([0.5, 0.5])

    def update_v(self, stimulus, choice, target):
        trial_type = "MATCH" if (stimulus == target).all() else "NON-MATCH"
        self.trial_history.append(trial_type)
        if len(self.trial_history) > self.horizon: self.trial_history = self.trial_history[-self.horizon:]

        likelihood = list(map(lambda trial_type:
                              np.array([0.45, 0.55]) if trial_type is "MATCH" else np.array([0.55, 0.45]), self.trial_history))
                            #   np.array([0.55, 0.45]) if trial_type == "MATCH" else np.array([0.45, 0.55]), self.trial_history))
        likelihood = np.prod(np.array(likelihood), axis=0)
        posterior = (likelihood * np.array([0.5, 0.5])) / np.sum(likelihood * np.array([0.5, 0.5]))
        # posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        # print(self.trial_history, posterior)
        T = len(self.trial_history)
        self.prior = posterior
        
        # TODO consider which model is used to estimte p(r|action), currently just using our MLE based estimator above.
        # no but the above is the P(match_context)!! Not v1.... For that, I should use Sabrina's.
        p_match, p_non_match = self.Sabrina_Q_values


        p_sm, p_snm, p_ns = [], [], []
        for t in range(T):
            l_sm = np.math.pow(p_non_match, t) * np.math.pow(p_match, T-t)  #likelihood(switch_to_Match| horizon trials)     
            l_snm= np.math.pow(p_match, t)     * np.math.pow(p_non_match, T-t) # likelihood(switch_to_Non-Match| trials) 
            l_ns  = np.math.pow(p_non_match, T)                                 # likelihood(no_switch|trials)
            # what about priors? p(switch_to_match)? that is affected by the last block change, and belief about current context.
            # Or use priors as probabilities from previous trials. No_switch will be the biggest, but should keep it evolving over a short horizon. 
            z = (l_sm + l_snm + l_ns)
            p_sm.append(l_sm / z)
            p_snm.append(l_snm / z)
            p_ns.append(l_ns / z)
            #p(r(0,t)| switch_at_t) p(switch_at_t)  / p(r(0,t))
        
        #Integrating from all horizon:
        p_sm_T = np.sum(p_sm)
        p_snm_T = np.sum(p_snm)
        p_ns_T = np.sum(p_ns)

        self.p_sm_snm_ns = np.array ([p_sm_T, p_snm_T, p_ns_T])

        # ALTERNATIVELY:
        # v1, v2 = self.Sabrina_Q_values
        current_context = "MATCH"
        horizon = [t == "MATCH" for t in self.trial_history]
        choices = self.Sabrina_Q_values if current_context is "MATCH" else np.flip(self.Sabrina_Q_values)
        choices_other = np.flip(choices)

        stay_votes = np.choose(horizon, choices)
        leave_votes = 1- stay_votes
        ratio_switch_t = []
        for t in range(T):
            like_switch = np.prod(stay_votes[:t]) * np.prod(leave_votes[t:])
            like_stay   = np.prod(stay_votes)
            ratio_switch_t.append( like_switch/ (like_switch + like_stay) )

        #Integrating from all horizon:
        ratio_switch = np.array(ratio_switch_t).mean()
        p_sm_T = ratio_switch 
        p_snm_T = ratio_switch 
        p_ns_T = 1-ratio_switch 
        self.p_sm_snm_ns = np.array ([p_sm_T, p_snm_T, p_ns_T])

        if ratio_switch > 1.2: #flip context
            if current_context is "MATCH": current_context = "NON-MATCH"
            else: current_context = "MATCH"


        #OTHER ATTEMPT:
        # p_r = # avg last ten rewards #probability of reward in current context using current action. simple average of recent rewards, but it might become unstable around change points. 

        # p_switch, p_stay = [], []
        # for t in range(T):
        #     p_stay = np.math.pow(p_r, T)
        #     p_switch = np.math.pow(p_r, t) * np.math.pow(p_r, T-t)


    def get_cid(self, association_level):
        if self.config.follow == 'association_levels':
            cid = self.association_levels_ids[association_level]
        elif self.follow == 'behavioral_context':
            if association_level in self.match_association_levels:
                cid = 0 # Match context
            else: cid = 1 # Non-Match context
        return (cid)

    def get_trial_err(self, errors, association_level):
        # error calc
        cid = self.get_cid(association_level)
        if self.config.response_delay:
            response_start = self.config.cuesteps + self.config.response_delay
            errorEnd = np.mean(errors[response_start:]*errors[response_start:]) 
        else:
            errorEnd = np.mean(errors*errors) # errors is [tsteps x Nout]

        all_contexts_err = np.array([errorEnd, 1-errorEnd]) if cid==0. else np.array([errorEnd-1, errorEnd ])
        return (errorEnd, all_contexts_err)
        
    def update_baseline_err(self, trial_err):

        self.baseline_err =  (1.0 - self.config.decayErrorPerTrial) * self.baseline_err + \
                 self.config.decayErrorPerTrial * trial_err

        return self.baseline_err

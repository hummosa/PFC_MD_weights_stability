import numpy as np

class data_generator():
    def __init__(self, training_schedule=None):
        self.training_schedule = training_schedule

    def trial_generator(self, association_level):
        '''
        generates trials given association_level in current block
        input:
        association level as one of ['30', '70', '10', '50', '90']
        returns: a tuple (input, output) e.g. in: up, out: up  ([1,0], [1,0])
        '''
        self.match_trial_probability={'90':0.9, '70':.7, '50':.5, '30':.3, '10':.1}
        prob = self.match_trial_probability[association_level]
        #get Input: random up or down cue
        inp = np.random.choice(np.array([1., 0.])) 
        # get Output: match or non-match
        out = inp if np.random.uniform() < prob else (1 - inp)
        return (np.array([inp, 1-inp]), np.array([out, 1-out]))

    def block_generator(self,blocki):
        self.association_levels = np.array(['90', '70', '50', '30', '10'])
        if self.training_schedule is None:
            self.block_schedule = ['90', '10','90', '10','90', '30', '70', '10', '50', '10']
            self.ofc_control_schedule= ['off'] *5 + ['match', 'non-match'] *3
            
        else:
            self.block_schedule=self.training_schedule
        self.strategy_schedule = ['match' if bs in ['90', '70', '50'] else 'non-match' for bs in self.block_schedule]
        if blocki < len(self.block_schedule):
            yield (self.block_schedule[blocki], self.ofc_control_schedule[blocki])
        else:
            yield (np.random.choice(self.association_levels), 'off')
            
import numpy as np

class Config():
    def __init__(self):
        #enviroment parameters:
        self.debug = False
        self.saveData = True
        self.RNGSEED = 1                    # Seed for the np random number generator
        np.random.seed([self.RNGSEED])
        self.cuda = False
        # self.args = args_dict               # dict of args label:value

        #Experiment parameters:
        self.Ntasks = 2                     # Ambiguous variable name, replacing with appropriate ones below:  # number of contexts 
        self.Ncontexts = 2                  # number of contexts (match block or non-match block)
        self.Nblocks = 4                    # number of blocks for the simulation
        self.tau = 0.02
        self.dt = 0.001
        self.tsteps = 300                   # number of timesteps in a trial
        self.cuesteps = 100                 # number of time steps for which cue is on
        self.noiseSD = 1e-3
        self.learning_rate = 5e-6  # too high a learning rate makes the output weights change too much within a trial / training cycle,
                  
        #Network architecture
        self.Ninputs = 4                      # total number of inputs
        self.Ncues = 2                     # How many of the inputs are task cues (UP, DOWN)
        self.Nmd    = 2                       # number of MD cells.
        self.Npfc = 1000                      # number of pfc neurons
        self.Nofc = 1000                      # number of ofc neurons
        self.Nsub = 200                     # number of neurons per cue
        self.Nout = 2                       # number of outputs
        self.G = 1.5                        # Controls level of excitation in the net

                          #  then the output interference depends on the order of cues within a cycle typical values is 1e-5, can vary from 1e-4 to 1e-6
        self.training_schedule = lambda x: x%self.Ncontexts 
                                            # Creates a training_schedule. Specifies task context for each block 
                                            # Currently just loops through available contexts
        self.tauError = 0.001            # smooth the error a bit, so that weights don't fluctuate
        self.modular  = True                # Assumes PFC modules and pass input to only one module per tempral context.
        self.MDeffect = True                # whether to have MD present or not
        self.MDamplification = 3.           # Factor by which MD amplifies PFC recurrent connections multiplicatively
        self.MDlearningrate = 1e-7

        self.delayed_response = 0 #50       # in ms, Reward model based on last 50ms of trial, if 0 take mean error of entire trial. Impose a delay between cue and stimulus.

        # OFC
        self.OFC_reward_hx = True           # model ofc as keeping track of current strategy and recent reward hx for each startegy.
        self.use_context_belief_to_route_input =False  # input routing per current context or per context belief
        self.use_context_belief_to_switch_MD = True  # input routing per current context or per context belief


        self.noisePresent = True           # add noise to all reservoir units
        self.positiveRates = True           # whether to clip rates to be only positive, G must also change

        self.reinforce = True              # use reinforcement learning (node perturbation) a la Miconi 2017
        self.MDreinforce = True            #  instead of error-driven learning
                                            
        if self.reinforce:
            self.perturbProb = 50./self.tsteps
                                            # probability of perturbation of each output neuron per time step
            self.perturbAmpl = 10.          # how much to perturb the output by
            self.meanErrors = np.zeros(self.Ncontexts)#*self.inpsPerContext) #Ali made errors per context rather than per context*cue
                                            # vector holding running mean error for each cue
            self.decayErrorPerTrial = 0.1   # how to decay the mean errorEnd by, per trial
            self.learning_rate *= 10        # increase learning rate for reinforce
            self.reinforceReservoir = False # learning on reservoir weights also?
            if self.reinforceReservoir:
                self.perturbProb /= 10

        # self.monitor = monitor(['context_belief', 'error_cxt1', 'error_cxt2', 'error_dif']) #monior class to track vars of interest

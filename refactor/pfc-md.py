import numpy as np

from network import Network
import models
import torch

import sys,shelve, tqdm, time
# from data_generator import data_generator
from plot_figures import monitor


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

        self.monitor = monitor(['context_belief', 'error_cxt1', 'error_cxt2', 'error_dif']) #monior class to track vars of interest

config = Config()

md = models.MD({})
pfc = models.PFC({})
ofc = models.PFC({})

ofc.current_context_belief = 0 # Which context is the network assuming currently
ofc.pcontext = np.ones(config.Ncontexts)/ config.Ncontexts  # prob of being in each context.
ofc.recent_error = np.zeros(config.Ncontexts)           # Recent reward accumulator for each context
ofc.recent_error_history = []  # List to keep track of entire error history


# Initialize weights
wPFC2MD = np.random.normal(size=(config.Nmd, config.Npfc))\
           *config.G/np.sqrt(config.Nsub*2)
wPFC2MD -= np.mean(wPFC2MD,axis=1)[:,np.newaxis] # same as res rec, substract mean from each row.
wMD2PFC = np.random.normal(size=(config.Npfc, config.Nmd))\
           *config.G/np.sqrt(config.Nsub*2)
wMD2PFC -= np.mean(wMD2PFC,axis=1)[:,np.newaxis] # same as res rec, substract mean from each row.
wMD2PFCMult = wMD2PFC # Get the exact copy to init mult weights
initial_norm_wPFC2MD = np.linalg.norm(wPFC2MD)
initial_norm_wMD2PFC = np.linalg.norm(wMD2PFC) # Keep the initial norm for synaptic scaling later.
MDpreTrace = np.zeros(shape=(config.Npfc)) # Used to hold pre activity for Hebbian learning at MD-PFC synapses

Jrec = np.random.normal(size=(config.Npfc, config.Npfc))\
            *config.G/np.sqrt(config.Nsub*2)
if config.cuda:
    Jrec = torch.Tensor(Jrec).cuda()

# W_in: brreak down into cues, which get projected to 200 neurons, and then to v1 v2 inputs, which we'll spread using a gaussian. I see all being a gaussian eventually but for now let's keep some structure
Cues_wIn = np.zeros((config.Npfc,config.Ncues))
config.cueFactor = 0.75#1.5 Ali halved it when I added cues going to both PFC regions, i.e two copies of input. But now working ok even with only one copy of input.
if config.positiveRates: lowcue,highcue = 0.5,1.   #NOT really sure this applies here. Input weights can be negative!
else: lowcue,highcue = -1.,1

for cuei in np.arange(config.Ncues):
    Cues_wIn[config.Nsub*cuei:config.Nsub*(cuei+1),cuei] = \
            np.random.uniform(lowcue,highcue,size=config.Nsub) \
                    *config.cueFactor

Other_wIn = np.random.normal( size=(config.Npfc,config.Ninputs-config.Ncues) ) * config.cueFactor

inputs = np.array(np.ones(shape=(config.Ninputs,1)))
W_in = np.hstack((Cues_wIn, Other_wIn))
W_pfc_md = wPFC2MD
W_md_pfc = wMD2PFC


def update_W_md_pfc(md, pfc, W, tstep):
    k, t = tstep
    if k == 'STEP':
        W_new = W * 2
        print('STEP: MD->PFC W', W, W_new)
        return W_new
    elif k == 'TRIAL_END':
        print('TRIAL_END: MD->PFC W', W)
        return W


def update_W_pfc_md(pfc, md, W, tstep):
    k, t = tstep
    if k == 'STEP':
        W_new = W * 2
        print('STEP: PFC->MD W', W, W_new)
        return W_new
    elif k == 'TRIAL_END':
        print('TRIAL_END: PFC->MD W', W)
        return W


network = Network()
network.define_inputs(len(inputs), W_in, pfc)
network.connect(pfc, md, W_pfc_md, update_W_pfc_md)
network.connect(md, pfc, W_md_pfc, update_W_md_pfc)

for t in range(0, 3):
    inputs = [t]
    network.step(inputs, t)
network.trial_end()

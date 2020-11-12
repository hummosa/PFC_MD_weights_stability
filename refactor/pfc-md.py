import numpy as np

from network import Network
import models
import torch

import sys,shelve, tqdm, time
# from data_generator import data_generator
from plot.plot_figures import monitor
from config import Config

config = Config()

md = models.MD(config, 'MD')
out = models.MD(config, 'OUT')
pfc = models.PFC(config, 'PFC')
ofc = models.PFC(config, 'OFC')

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
W_pfc_out  = np.random.normal( size=(config.Npfc,config.Nout) )

weight_state = {'initial_norm_wMD2PFC': initial_norm_wMD2PFC, 
                'initial_norm_wPFC2MD': initial_norm_wPFC2MD}
ofc_state = {'match_expected_value'    : 0.,
             'non_match_expected_value': 0.,
             'current_reward'          : 0.,
             'current_strategy_belief' : 0.,
             }

def update_W_Hebbian(tstep, state, W, pre, post):
    k, t = tstep
    W_new = W
    pre_trace = state['pre_trace']
    if k == 'STEP':
        hebbian_learning_bias = 0.13
        Allowable_Weight_range = 0.1#0.06

        pre_trace += 1./200/10. * (-pre_trace + pre ) # as computed by Gilra. 200 is tsteps in trial
        pre_activity = pre.neurons - hebbian_learning_bias
        post_activity = post.neurons
        W_delta = config.MDlearningrate* np.outer(post_activity, pre_activity)
        W_new = np.clip(W +W_delta,  -Allowable_Weight_range ,Allowable_Weight_range ) # Clip to specified range
            # else:
        # print('STEP: MD->PFC W', W, W_new)
    elif k == 'TRIAL_END':
        # Synaptic scaling
        initial_norm_w = state['initial_norm']
        W /= np.linalg.norm(W)/ initial_norm_w

        print('TRIAL_END: MD->PFC W', W)

    state['pre_trace'] = pre_trace
    return (state, W_new)


def update_node_perturbation(tstep, state, W, pre, post):
    k, global_signals = tstep
    W_new = W
    pre_trace = state['pre_trace']

    if k == 'STEP':
        # add perturbation to post neurons  Exploratory perturbations a la Miconi 2017
        # Perturb each output neuron independently  with probability perturbProb
        perturbationOff = np.where(
                np.random.uniform(size=len(post.neurons))>=config.perturbProb )
        perturbation = np.random.uniform(-1,1,size=len(post.neurons))
        perturbation[perturbationOff] = 0.
        perturbation *= config.perturbAmpl
        post.neurons += perturbation
        # accummulate eligibility trace (pre*perturbation) pending reward feedback
        # TODO where does HebbTrace live (persist)?
        
        pre_trace += np.outer(pre.neurons , perturbation)

        print('STEP: PFC->OUT W', W, W_new)
        return W_new
    elif k == 'TRIAL_END':
        # use reward info to update weights based on pre-activity * perturbation eligibility trace
        # with learning using REINFORCE / node perturbation (Miconi 2017),
        #  the weights are only changed once, at the end of the trial
        # apart from eta * (err-baseline_err) * hebbianTrace,
        #  the extra factor baseline_err helps to stabilize learning  as per Miconi 2017's code,
        #  but I found that it destabilized learning, so not using it. 
        # TODO (ALI: There was a bug in calculating baseline_err, worth trying to add it back in)
        
        # TODO pass error information to this code. Through class? as input to the update fxn?
        current_RPE = global_signals 

        W_new += config.learning_rate * current_RPE * pre_trace                        
        state['pre_trace'] = pre_trace
        print('TRIAL_END: PFC->OUT W', W)
    return (state, W_new)

def compute_global_signals(network, expected_output, plasticity):
    output = network.models['Output'].neurons
    error = output - expected_output
    return {"error": error}


network = Network(compute_global_signals)
network.define_inputs(W_in, pfc)
network.connect(pfc, md, W_pfc_md, update_W_Hebbian, config)
network.connect(md, pfc, W_md_pfc, update_W_Hebbian, config)
network.connect(pfc, out, W_pfc_out, update_node_perturbation, config)



# Simulation


dat_inp = []
dat_output = []
dat_output_expected = []
dat_OFC = []
dat_PFC = []


def cb(trial_num, step_name, timestep, inp, expected_output, network):
    if step_name == 'CUE' and timestep == 1:
        dat_inp.append(inp)

    if step_name == 'TRIAL_END':
        dat_output_expected.append(expected_output)
        for model in network.models.values():
            if model.name == 'OFC':
                dat_OFC.append(model.neurons)
            elif model.name == 'PFC':
                dat_PFC.append(model.neurons)
            elif model.name == 'Output':
                dat_output.append(model.neurons)


def get_input(trial_num, step_name, timestep, prev_inp):
    # [UP, DOWN]
    if step_name == 'CUE' and timestep == 1:
        return np.array([1., 0.]) if random.random() < 0.5 else np.array([0., 1.])
    else:
        return prev_inp


def get_output(trial_num, inps_arr):
    inp = inps_arr[0]
    if trial_num <= 25:  # Match context
        return inp if random.random() < 0.7 else (1 - inp)
    elif trial_num <= 50:
        return 1 - inp if random.random() < 0.7 else (1 - inp)
    else:
        return inp if random.random() < 0.7 else (1 - inp)


N = 75
simulation = Simulation(network)
trial_setup = [("CUE", 5, False)]
simulation.run_trials(trial_setup, get_input, get_output, N, cb)

# Plot results
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
# Plot input
dat_inp = np.array(dat_inp)
inp_y = dat_inp[:, 1] * 4
ax1.plot(np.arange(len(inp_y)), inp_y, 'ko', label='Input')
dat_output_expected = np.array(dat_output_expected)
dat_output = np.array(dat_output)
for i in range(0, len(dat_output)):
    icon = 'go'
    if not np.array_equal(dat_output[i], dat_output_expected[i]):
        icon = 'gx'
    ax1.plot(i, .3 + 4 * dat_output[i, 1], icon)
ax1.legend()
ax1.set_title('Behavior')
# Plot OFC
dat_OFC = np.array(dat_OFC)
ax2.plot(dat_OFC[:, 0], label='Favor match')
ax2.plot(dat_OFC[:, 1], label='Favor nonmatch')
ax2.legend()
ax2.set_title('OFC')
# Plot PFC
dat_PFC = np.array(dat_PFC)
ax3.plot(dat_PFC[:, 0], label='Chose up')
ax3.plot(dat_PFC[:, 1], label='Chose down')
ax3.legend()
ax3.set_title('PFC')
plt.savefig('out_Fig.jpeg')

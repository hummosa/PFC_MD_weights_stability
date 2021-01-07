# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import os
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
# from IPython import embed; embed()
# import pdb; pdb.set_trace()
from scipy.io import savemat
import sys,shelve, tqdm, time
import plot_utils as pltu
from data_generator import data_generator
from plot_figures import *
import argparse
cuda = False
if cuda: import torch

from refactor.ofc_mle import OFC, OFC_dumb
ofc = OFC_dumb(horizon=40)
ofc.set_context("0.7")

from config import Config

class PFCMD():
    def __init__(self,config,args_dict={}):
        
        self.monitor = monitor(['context_belief', 'error_cxt1', 'error_cxt2', 'error_dif']) #monior class to track vars of interest


        #* Adjust network excitation levels based on MD effect, Positive Rates, and activation fxn
        if not config.MDeffect:
            config.G *= config.MDremovalCompensationFactor

        # I don't want to have an if inside activation  as it is called at each time step of the simulation
        # But just defining within __init__ doesn't make it a member method of the class,
        #  hence the special config.__class__. assignment
        if config.positiveRates:
            # only +ve rates
            def activation(self,inp):
                return np.clip(np.tanh(inp),0,None)
        else:
            # both +ve/-ve rates as in Miconi
            def activation(self,inp):
                return np.tanh(inp)
        self.__class__.activation = activation
        # Choose G based on the type of activation function
        # unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if config.positiveRates:
            config.tauMD = config.tau
        else:
            config.G /= 2.
            config.MDthreshold = 0.4
            config.tauMD = config.tau*10

                
        if config.saveData:
            self.fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD'+\
                                    +str(self.RNGSEED)+\
                                    ('')+'.shelve')

            
        #*# init weights:
        #### MD PFC weights 
        self.wPFC2MD = np.zeros(shape=(config.Nmd,config.Npfc))
        
        self.wPFC2MD = np.random.normal(size=(config.Nmd, config.Npfc))\
                        *config.MDrange #*config.G/np.sqrt(config.Nsub*2)
        self.wPFC2MD -= np.mean(self.wPFC2MD,axis=1)[:,np.newaxis] # same as res rec, substract mean from each row.
        self.wMD2PFC = np.random.normal(size=(config.Npfc, config.Nmd))\
                        *config.MDrange #*config.G/np.sqrt(config.Nsub*2)
        self.wMD2PFC -= np.mean(self.wMD2PFC,axis=1)[:,np.newaxis] # same as res rec, substract mean from each row.
        self.wMD2PFCMult = self.wMD2PFC # Get the exact copy to init mult weights
        self.initial_norm_wPFC2MD = np.linalg.norm(self.wPFC2MD) * .6
        self.initial_norm_wMD2PFC = np.linalg.norm(self.wMD2PFC) * .6
        #### Recurrent weights
        self.Jrec = np.random.normal(size=(config.Npfc, config.Npfc))\
                        *config.G/np.sqrt(config.Nsub*2)
        if cuda:
            self.Jrec = torch.Tensor(self.Jrec).cuda()
        #### Output weights
        self.wOut = np.random.uniform(-1,1,
                        size=(config.Nout,config.Npfc))/config.Npfc
        #### Input weights
        self.wV = np.zeros((config.Npfc,2))
        self.wIn = np.zeros((config.Npfc,config.Ncues))
        
        if config.positiveRates: lowcue,highcue = 0.5,1.
        else: lowcue,highcue = -1.,1
        for cuei in np.arange(config.Ncues):
            self.wIn[config.Nsub*cuei:config.Nsub*(cuei+1),cuei] = \
                    np.random.uniform(lowcue,highcue,size=config.Nsub) \
                            *config.cueFactor * 0.8 # to match that the max diff between v1 v2 is 0.8
            if config.wV_structured:
                self.wV[config.Nsub*cuei:config.Nsub*(cuei)+config.Nsub//2,0] = \
                        np.random.uniform(lowcue,highcue,size=config.Nsub//2) \
                                * config.cueFactor
                self.wV[config.Nsub*(cuei)+config.Nsub//2:config.Nsub*(cuei+1) ,1] = \
                        np.random.uniform(lowcue,highcue,size=config.Nsub//2) \
                                * config.cueFactor

            else:
                input_variance = 1.5
                self.wV = np.random.normal(size=(config.Npfc, 2 ), loc=(lowcue+highcue)/2, scale=input_variance) *config.cueFactor # weights of value input to pfc
                self.wV = np.clip(self.wV, 0, 1)
                self.wIn = np.random.normal(size=(config.Npfc, config.Ncues), loc=(lowcue+highcue)/2, scale=input_variance) *config.cueFactor 
                self.wIn = np.clip(self.wIn, 0, 1)

        self.MDpreTrace = np.zeros(shape=(config.Npfc))

        # make mean input to each row zero, helps to avoid saturation (both sides) for positive-only rates.
        #  see Nicola & Clopath 2016 mean of rows i.e. across columns (axis 1), then expand with np.newaxis
        #   so that numpy's broadcast works on rows not columns
        if cuda:
            with torch.no_grad():
                self.Jrec -= torch.mean(self.Jrec, dim=1, keepdim=True)
        else:
            self.Jrec -= np.mean(self.Jrec,axis=1)[:,np.newaxis]
        
        

    def sim_cue(self,contexti,cuei,cue,target,MDeffect=True,
                    MDCueOff=False,MDDelayOff=False,
                    train=True,routsTarget=None):
        '''
        config.reinforce trains output weights
         using REINFORCE / node perturbation a la Miconi 2017.'''
        cues = np.zeros(shape=(config.tsteps,config.Ncues))

        xinp = np.random.uniform(0,0.1,size=(config.Npfc))
        xadd = np.zeros(shape=(config.Npfc))
        MDinp = np.random.uniform(0,0.1,size=config.Nmd)
        MDinps = np.zeros(shape=(config.tsteps, config.Nmd))
        routs = np.zeros(shape=(config.tsteps,config.Npfc))
        MDouts = np.zeros(shape=(config.tsteps,config.Nmd))
        outInp = np.zeros(shape=config.Nout)
        outs = np.zeros(shape=(config.tsteps,config.Nout))
        out = np.zeros(config.Nout)
        errors = np.zeros(shape=(config.tsteps,config.Nout))
        errors_other = np.zeros(shape=(config.tsteps,config.Nout))
        error_smooth = np.zeros(shape=config.Nout)

        if config.reinforce:
            HebbTrace = np.zeros(shape=(config.Nout,config.Npfc))
            if config.reinforceReservoir:
                if cuda:
                    HebbTraceRec = torch.Tensor(np.zeros(shape=(config.Npfc,config.Npfc))).cuda()
                else:

                    HebbTraceRec = np.zeros(shape=(config.Npfc,config.Npfc))
            if config.MDreinforce:
                HebbTraceMD = np.zeros(shape=(config.Nmd,config.Npfc))

        for i in range(config.tsteps):
            rout = self.activation(xinp)
            routs[i,:] = rout
            outAdd = np.dot(self.wOut,rout)

            if MDeffect:
                # MD decays 10x slower than PFC neurons, so as to somewhat integrate PFC input
                if self.ofc_to_md_active: 
                    #MDout = np.array([0,1]) if self.current_context_belief==0 else np.array([1,0]) #MD 0 for cxt belief 1
                    # MDout = np.array([0,1]) if contexti==0 else np.array([1,0]) #MD 0 for cxt belief 1
                    # MDout = np.array([1,0]) if contexti==0 else np.array([0,1]) #MD 1 for cxt belief 1
                    MDinp += np.array([.6,-.6]) if contexti==0 else np.array([-.6,.6]) 

                if self.positiveRates:
                    MDinp +=  self.dt/self.tauMD * \
                            ( -MDinp + np.dot(self.wPFC2MD,rout) )
                else: # shift PFC rates, so that mean is non-zero to turn MD on
                    MDinp +=  self.dt/self.tauMD * \
                            ( -MDinp + np.dot(self.wPFC2MD,(rout+1./2)) )

                # MD off during cue or delay periods:
                if MDCueOff and i<self.cuesteps:
                    MDinp = np.zeros(self.Nmd)
                    #MDout /= 2.
                if MDDelayOff and i>self.cuesteps and i<config.tsteps:
                    MDinp = np.zeros(self.Nmd)

                # MD out either from MDinp or forced
                if self.MDstrength is not None:
                    MDout = np.zeros(self.Nmd)
                    # MDout[contexti] = 1. # No longer feeding context information directly to MD
                    MDout = (np.tanh( (MDinp-self.MDthreshold)/0.1 ) + 1) / 2.
                # if MDlearn then force "winner take all" on MD output
                if train and self.MDlearn:
                    #MDout = (np.tanh(MDinp-self.MDthreshold) + 1) / 2.
                    # winner take all on the MD
                    #  hardcoded for self.Nmd = 2
                    if MDinp[0] > MDinp[1]: MDout = np.array([1,0])
                    else: MDout = np.array([0,1])

                    ########################################################
                    ########################################################
                    #######################################################
                    # MDout = np.array([1,0]) #! NOTE clammped MD!!!!!!!!!!!!!!!!!!!!!!!!!!!!##
                    ########################################################
                    ########################################################
                    ########################################################

                MDouts[i,:] = MDout
                MDinps[i, :]= MDinp

                if self.useMult:
                    self.MD2PFCMult = np.dot(self.wMD2PFCMult * self.MDamplification,MDout)
                    if cuda:
                        with torch.no_grad():
                            xadd = (1.+self.MD2PFCMult) * torch.matmul(self.Jrec, torch.Tensor(rout).cuda()).detach().cpu().numpy()
                    else:
                        xadd = (1.+self.MD2PFCMult) * np.dot(self.Jrec,rout)
                else:
                    #startt = time.time()
                    if cuda:
                        with torch.no_grad():
                            xadd = torch.matmul(self.Jrec, torch.Tensor(rout).cuda()).detach().cpu().numpy()
                    else:
                        xadd = np.dot(self.Jrec, rout)
                    #print(time.time() * 10000- startt * 10000)
                xadd += np.dot(self.wMD2PFC,MDout)

                if train and self.MDlearn:# and not self.MDreinforce:
                    # MD presynaptic traces filtered over 10 trials
                    # Ideally one should weight them with MD syn weights,
                    #  but syn plasticity just uses pre*post, but not actualy synaptic flow.
                    self.MDpreTrace += 1./config.tsteps/10. * \
                                        ( -self.MDpreTrace + rout )
                    self.MDlearningBias = self.MDlearningBiasFactor * np.mean(self.MDpreTrace)
                    
                    # wPFC2MDdelta = 1e-4*np.outer(MDout-0.5,self.MDpreTrace-0.11) # Ali changed from 1e-4 and thresh from 0.13
                    wPFC2MDdelta = np.outer(MDout-0.5,self.MDpreTrace-self.MDlearningBias) # Ali changed from 1e-4 and thresh from 0.13
                    fast_delta = self.MDlearningrate*wPFC2MDdelta
                    slow_delta = 1e-1*self.MDlearningrate*wPFC2MDdelta
                    # wPFC2MDdelta *= self.wPFC2MD # modulate it by the weights to get supralinear effects. But it'll actually be sublinear because all values below 1
                    MDrange = self.MDrange #0.05#0.1#0.06
                    MDweightdecay = 1.#0.996
                    self.wPFC2MD = np.clip(self.wPFC2MD +fast_delta,  -MDrange , MDrange ) # Ali lowered to 0.01 from 1. 
                    # self.wMD2PFC = np.clip(self.wMD2PFC +fast_delta.T,-MDrange , MDrange ) # lowered from 10.
                    # self.wMD2PFCMult = np.clip(self.wMD2PFCMult+ slow_delta.T,-2*MDrange /self.G, 2*MDrange /self.G) 
                    # self.wMD2PFCMult = np.clip(self.wMD2PFC,-2*MDrange /self.G, 2*MDrange /self.G) * self.MDamplification
            else:
                if cuda:
                    with torch.no_grad():  
                        xadd = torch.matmul(self.Jrec, torch.Tensor(rout).cuda()).detach().cpu().numpy()
                else:
                        xadd = np.dot(self.Jrec,rout)

            if i < self.cuesteps:
                ## add an MDeffect on the cue
                #if MDeffect and useMult:
                #    xadd += self.MD2PFCMult * np.dot(self.wIn,cue)
                # baseline cue is always added
                xadd += np.dot(self.wIn,cue)
                if self.use_recent_reward_to_pfc_inputs:
                    xadd += np.dot(self.wV,self.recent_error)
                cues[i,:] = cue
                if self.dirConn:
                    if self.outExternal:
                        outAdd += np.dot(self.wDir,cue)
                    else:
                        xadd[-config.Nout:] += np.dot(self.wDir,cue)

            if self.reinforce:
                # Exploratory perturbations a la Miconi 2017
                # Perturb each output neuron independently
                #  with probability perturbProb
                perturbationOff = np.where(
                        np.random.uniform(size=config.Nout)>=self.perturbProb )
                perturbation = np.random.uniform(-1,1,size=config.Nout)
                perturbation[perturbationOff] = 0.
                perturbation *= self.perturbAmpl
                outAdd += perturbation
            
                if self.reinforceReservoir:
                    perturbationOff = np.where(
                            np.random.uniform(size=config.Npfc)>=self.perturbProb )
                    perturbationRec = np.random.uniform(-1,1,size=config.Npfc)
                    perturbationRec[perturbationOff] = 0.
                    # shouldn't have MD mask on perturbations,
                    #  else when MD is off, perturbations stop!
                    #  use strong subtractive inhibition to kill perturbation
                    #   on task irrelevant neurons when MD is on.
                    #perturbationRec *= self.MD2PFCMult  # perturb gated by MD
                    perturbationRec *= self.perturbAmpl
                    xadd += perturbationRec
                
                if self.MDreinforce:
                    perturbationOff = np.where(
                            np.random.uniform(size=self.Nmd)>=self.perturbProb )
                    perturbationMD = np.random.uniform(-1,1,size=self.Nmd)
                    perturbationMD[perturbationOff] = 0.
                    perturbationMD *= self.perturbAmpl
                    MDinp += perturbationMD

            xinp += self.dt/self.tau * (-xinp + xadd)
            
            if self.noisePresent:
                xinp += np.random.normal(size=(config.Npfc))*self.noiseSD \
                            * np.sqrt(self.dt)/self.tau
            
            outInp += self.dt/self.tau * (-outInp + outAdd)
            out = self.activation(outInp)                

            error = out - target
            errors[i,:] = error
            outs[i,:] = out
            error_smooth += self.dt/self.tauError * (-error_smooth + error)
            
            if train:
                if self.reinforce:
                    # note: rout is the activity vector for previous time step
                    HebbTrace += np.outer(perturbation,rout)
                    if self.reinforceReservoir:
                        if cuda:
                            with torch.no_grad():
                                HebbTraceRec += torch.ger(torch.Tensor(perturbationRec).cuda(),torch.Tensor(rout).cuda())
                        else:
                            HebbTraceRec += np.outer(perturbationRec,rout)
                    if self.MDreinforce:
                        HebbTraceMD += np.outer(perturbationMD,rout)
                else:
                    # error-driven i.e. error*pre (perceptron like) learning
                    self.wOut += -self.learning_rate \
                                    * np.outer(error_smooth,rout)
   
        # inpi = contexti*self.inpsPerContext + cuei
        inpi = contexti
        if train and self.reinforce:
            # with learning using REINFORCE / node perturbation (Miconi 2017),
            #  the weights are only changed once, at the end of the trial
            # apart from eta * (err-baseline_err) * hebbianTrace,
            #  the extra factor baseline_err helps to stabilize learning
            #   as per Miconi 2017's code,
            #  but I found that it destabilized learning, so not using it.
            if self.delayed_response:
                errorEnd = np.mean(errors[-50:]*errors[-50:]) 
            else:
                errorEnd = np.mean(errors*errors) # errors is [tsteps x Nout]

            self.wOut -= self.learning_rate * \
                    (errorEnd-self.meanErrors[inpi]) * \
                        HebbTrace #* self.meanErrors[inpi]

            if self.reinforceReservoir:
                if cuda:
                    with torch.no_grad():
                        self.Jrec -= self.learning_rate * \
                                (errorEnd-self.meanErrors[inpi]) * \
                                    HebbTraceRec #* self.meanErrors[inpi]  
                else:
                    self.Jrec -= self.learning_rate * \
                            (errorEnd-self.meanErrors[inpi]) * \
                                HebbTraceRec #* self.meanErrors[inpi]                
            if self.MDreinforce:
                self.wPFC2MD -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceMD * 10. # changes too small Ali amplified #* self.meanErrors[inpi]                
                self.wMD2PFC -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceMD.T * 10. #* self.meanErrors[inpi]  
                                          
            if self.MDlearn: # after all Hebbian learning within trial and reinforce after trial, re-center MD2PFC and PFC2MD weights This will introduce 
                #synaptic competition both ways.
                self.wMD2PFC = MDweightdecay* (self.wMD2PFC)
                self.wPFC2MD = MDweightdecay* (self.wPFC2MD)
                self.wPFC2MD /= np.linalg.norm(self.wPFC2MD)/ self.initial_norm_wPFC2MD
                self.wMD2PFC /= np.linalg.norm(self.wMD2PFC)/ self.initial_norm_wMD2PFC

                # self.wMD2PFC -= np.mean(self.wMD2PFC)
                # self.wMD2PFC *= self.G/np.sqrt(self.Nsub*2)/np.std(self.wMD2PFC) # div weights by their std to get normalized dist, then mul it by desired std
                # self.wPFC2MD -= np.mean(self.wPFC2MD)
                # self.wPFC2MD *= self.G/np.sqrt(self.Nsub*2)/np.std(self.wPFC2MD) # div weights by their std to get normalized dist, then mul it by desired std

            # cue-specific mean error (low-pass over many trials)
            # self.meanErrors[inpi] = \
            #     (1.0 - self.decayErrorPerTrial) * self.meanErrors[inpi] + \
            #      self.decayErrorPerTrial * errorEnd
            
            # hack and slash calculate error had the model taken the target been the oppisite direction (or the model had taken the other startegy)
            errors_other = errors + target - np.abs(target -1.)
            if self.delayed_response:
                errorEnd_other = np.mean(errors_other[-50:]*errors_other[-50:]) 
            else:
                errorEnd_other = np.mean(errors_other*errors_other) # errors is [tsteps x Nout]
            # arrange errors into a matrix depending on which error is match and non-match
            errorEnd_m = np.array([errorEnd, errorEnd_other]) if inpi==0. else np.array([errorEnd_other, errorEnd ])

            self.meanErrors = \
                (1.0 - self.decayErrorPerTrial) * self.meanErrors + \
                 self.decayErrorPerTrial * errorEnd_m

            # self.recent_error[self.current_context_belief] = self.meanErrors
            if self.use_context_belief:
            #     self.recent_error[self.current_context_belief] =(1.0 - self.decayErrorPerTrial) * self.recent_error[self.current_context_belief] + \
            #      self.decayErrorPerTrial * self.meanErrors
            # else:
                self.recent_error = self.meanErrors # TODO temporarily not using context belief
                # I start with match context 0.9. So match startegy would have 0.1 error
                self.recent_error = np.array([0.1, 0.9]) if inpi==0 else np.array([0.9, 0.1])
                if self.get_v1_v2_from_ofc: 
                    self.recent_error = np.array(ofc.get_v() ) 

            ofc.update_v(cue[:2], out, target)
        
        self.meanAct[inpi,:,:] += routs

        # TODO Flip belief about context if recent errors differ more than a threshold:
        dif_err = np.abs(self.recent_error[0]-self.recent_error[1])
        if dif_err > 0.2:
            self.current_context_belief = np.argmin(self.recent_error)

        self.monitor.log([self.current_context_belief, self.recent_error[0], self.recent_error[1], dif_err])    

        return cues, routs, outs, MDouts, MDinps, errors

    def get_cues_order(self,cues):
        cues_order = np.random.permutation(cues)
        return cues_order

    def get_cue_target(self,contexti,cuei):
        cue = np.zeros(self.Ncues)
        
        if self.tactileTask:
            cue = np.zeros(self.Ncues) #reset cue 
            cuei = np.random.randint(0,2) #up or down
            non_match = self.get_next_target(contexti) #get a match or a non-match response from the data_generator class
            if non_match: #flip
                targeti = 0 if cuei ==1 else 1
            else:
                targeti = cuei 
            
            if self.modular:
                if self.use_context_belief_to_route_input:
                    cue[self.current_context_belief*2+cuei] = 1. # Pass cue according to context belief 
                else:
                    cue[0+cuei] = 1. # Pass cue to the first PFC region 
            else:
                cue[0+cuei] = 1. # Pass cue to the first PFC region 
                cue[2+cuei] = 1. # Pass cue to the second PFC region
            
            target = np.array((1.,0.)) if targeti==0  else np.array((0.,1.))

        return cue, target

    def get_cue_list(self,contexti=None):
        if contexti is not None:
            # (contexti,cuei) combinations for one given contexti
            cueList = np.dstack(( np.repeat(contexti,self.inpsPerContext),
                                    np.arange(self.inpsPerContext) ))
        else:
            # every possible (contexti,cuei) combination
            cueList = np.dstack(( np.repeat(np.arange(self.Ncontexts),self.inpsPerContext),
                                    np.tile(np.arange(self.inpsPerContext),self.Ncontexts) ))
        return cueList[0]
    
    def get_next_target(self, contexti):
        
        return next(self.data_generator.task_data_gen[contexti])

    def train(self):
        MDeffect = self.MDeffect
        Ntrain = self.trials_per_block * self.Nblocks

        # Containers to save simulation variables
        wOuts = np.zeros(shape=(Ntrain,config.Nout,config.Npfc))
        wPFC2MDs = np.zeros(shape=(Ntrain,2,config.Npfc))
        wMD2PFCs = np.zeros(shape=(Ntrain,config.Npfc,2))
        wMD2PFCMults = np.zeros(shape=(Ntrain,config.Npfc,2))
        MDpreTraces = np.zeros(shape=(Ntrain,config.Npfc))
        wJrecs   = np.zeros(shape=(Ntrain, 40, 40))
        PFCrates = np.zeros( (Ntrain, config.tsteps, config.Npfc ) )
        MDinputs = np.zeros( (Ntrain, config.tsteps, self.Nmd) )
        MDrates  = np.zeros( (Ntrain, config.tsteps, self.Nmd) )
        Outrates = np.zeros( (Ntrain, config.tsteps, config.Nout  ) )
        Inputs   = np.zeros( (Ntrain, self.inpsPerContext))
        Targets  =  np.zeros( (Ntrain, config.Nout))
        MSEs = np.zeros(Ntrain)
        
        for traini in tqdm.tqdm(range(Ntrain)):

            blocki = traini // self.trials_per_block
                # USE context beliefe
            contexti = self.training_schedule(blocki)# Get the context index for this current block
            # if traini % self.trials_per_block == 0:
                # contexti = next(self.data_generator.training_schedule_gen)# Get the context index for this current block
                # print('context i: ', contexti)
                cueList = self.get_cue_list(contexti=contexti) # Get all the possible cue combinations for the current context
            cues_order = self.get_cues_order(cueList) # randomly permute them. 
            
            contexti,cuei = cues_order[0] # No need for this loop, just pick the first cue, this list is ordered randomly
            cue, target = \
                self.get_cue_target(contexti,cuei)
            if self.debug:
                print('cue:', cue)
                print('target:', target)

            #testing on the last 5 trials
            self.get_v1_v2_from_ofc = True
            self.no_of_trials_with_ofc_signal = int(args_dict['switches']) #lengths_of_directed_trials[blocki - self.Nblocks +6] #200-(40*(blocki-self.Nblocks + 6)) #decreasing no of instructed trials
            if (blocki > self.Nblocks - 8) and ((traini%self.trials_per_block) < self.no_of_trials_with_ofc_signal):
                self.ofc_to_md_active = True#bool(args_dict['switches'])
                print(f'traini: {traini}')
                # import pdb; pdb.set_trace()    
            else:
                self.ofc_to_md_active = False
            if (traini%self.trials_per_block ==0) and self.use_context_belief_to_switch_MD:
                self.hx_of_ofc_signal_lengths.append((blocki, self.no_of_trials_with_ofc_signal))
            # elif (traini%self.trials_per_block ==0):
                # self.use_context_belief_to_switch_MD = False

            cues, routs, outs, MDouts, MDinps, errors = \
                self.sim_cue(contexti,cuei,cue,target,MDeffect=MDeffect,
                train=True)
                
            PFCrates[traini, :, :] = routs
            MDinputs[traini, :, :] = MDinps
            MDrates [traini, :, :] = MDouts
            Outrates[traini, :, :] = outs
            Inputs  [traini, :]    = np.clip((cue[:2] + cue[2:]), 0., 1.) # go get the input going to either PFC regions. (but clip in case both regions receiving same input)
            Targets [traini, :]    = target

            MSEs[traini] += np.mean(errors*errors)

            wOuts[traini,:,:] = self.wOut
            if self.plotFigs and self.outExternal:
                if self.MDlearn:
                    wPFC2MDs[traini,:,:] = self.wPFC2MD
                    wMD2PFCs[traini,:,:] = self.wMD2PFC
                    wMD2PFCMults[traini,:,:] = self.wMD2PFCMult
                    MDpreTraces[traini,:] = self.MDpreTrace
                if self.reinforceReservoir:
                        wJrecs[traini,:,:] = self.Jrec[:40, 0:25:1000].detach().cpu().numpy() # saving the whole rec is too large, 1000*1000*2200
        self.meanAct /= Ntrain

        if self.saveData:
            self.fileDict['MSEs'] = MSEs
            self.fileDict['wOuts'] = wOuts


        if self.plotFigs:

            # plot output weights evolution
            
            weights= [wOuts, wPFC2MDs, wMD2PFCs,wMD2PFCMults,  wJrecs, MDpreTraces]
            rates =  [PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs]
            plot_weights(self, weights)
            plot_rates(self, rates)
            plot_what_i_want(self, weights, rates)
            #from IPython import embed; embed()
            dirname="results/"+self.args['exp_name']+"/"
            parm_summary= str(list(self.args.values())[0])+"_"+str(list(self.args.values())[1])+"_"+str(list(self.args.values())[2])
            if not os.path.exists(dirname):
                    os.makedirs(dirname)
            filename1=os.path.join(dirname, 'fig_weights_{}_{}.'+self.figure_format)
            filename2=os.path.join(dirname, 'fig_behavior_{}_{}.'+self.figure_format)
            filename3=os.path.join(dirname, 'fig_rates_{}_{}.'+self.figure_format)
            filename4=os.path.join(dirname, 'fig_monitored_{}_{}.'+self.figure_format)
            filename5=os.path.join(dirname, 'fig_trials_{}_{}.'+self.figure_format)
            filename6=os.path.join(dirname, 'fig_custom_{}_{}.'+self.figure_format)
            self.fig3.savefig     (filename1.format(parm_summary, time.strftime("%Y%m%d-%H%M%S")),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=self.figure_format)
            self.figOuts.savefig  (filename2.format(parm_summary, time.strftime("%Y%m%d-%H%M%S")),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=self.figure_format)
            self.figRates.savefig (filename3.format(parm_summary, time.strftime("%Y%m%d-%H%M%S")),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=self.figure_format)
            self.fig_monitor = plt.figure()
            self.monitor.plot(self.fig_monitor, self)
            if self.debug:
                self.figTrials.savefig(filename5.format(parm_summary, time.strftime("%Y%m%d-%H%M%S")),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=self.figure_format)
                self.figCustom.savefig(filename6.format(parm_summary, time.strftime("%Y%m%d-%H%M%S")),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=self.figure_format)
                self.fig_monitor.savefig(filename4.format(parm_summary, time.strftime("%Y%m%d-%H%M%S")),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=self.figure_format)

            # output some variables of interest:
            # md ampflication and % correct responses from model.
            filename7=os.path.join(dirname, 'values_of_interest.txt')
            filename7exits = os.path.exists(filename7)
            with open(filename7, 'a') as f:
                if not filename7exits:
                    [f.write(head+'\t') for head in ['switches', 'LR', 'HebbT', '1st', '2nd', '3rd', '4th', 'avg1-3', 'mean']]
                [f.write('{}\t '.format(val)) for val in  [*self.args.values()][:3]]
                # {:.2e} \t {:.2f} \t'.format(self.args['switches'], self.args['MDlr'],self.args['MDactive'] ))
                for score in self.score:
                    f.write('{:.2f}\t'.format(score)) 
                f.write('\n')
            

            filename8=os.path.join(dirname, 'Corrects{}_{}')
            # np.save(filename8.format(parm_summary, time.strftime("%Y%m%d-%H%M%S")), self.corrects)
            if 1==2: # output massive weight and rate files
                import pickle
                filehandler = open(os.path.join(dirname, 'Rates{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), 'wb')
                pickle.dump(rates, filehandler)
                filehandler.close()
                filehandler = open(os.path.join(dirname, 'Weights{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), 'wb')
                pickle.dump(weights, filehandler)
                filehandler.close()

            # np.save(os.path.join(dirname, 'Rates{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), rates)
            # np.save(os.path.join(dirname, 'Weights{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), weights)

    def load(self,filename):
        d = shelve.open(filename) # open
        if self.outExternal:
            self.wOut = d['wOut']
        else:
            self.Jrec[-config.Nout:,:] = d['JrecOut']
        if self.dirConn:
            self.wDir = d['wDir']

        if self.MDlearn:
            self.wMD2PFC     = d['MD2PFC']
            self.wMD2PFCMult = d['MD2PFCMult'] 
            self.wPFC2MD     = d['PFC2MD'] 
                         
        d.close()
        return None

    def save(self):
        if self.outExternal:
            self.fileDict['wOut'] = self.wOut
        else:
            self.fileDict['JrecOut'] = self.Jrec[-config.Nout:,:]
        if self.dirConn:
            self.fileDict['wDir'] = self.wDir
        if self.MDlearn:
            self.fileDict['MD2PFC'] = self.wMD2PFC
            self.fileDict['MD2PFCMult'] = self.wMD2PFCMult
            self.fileDict['PFC2MD'] = self.wPFC2MD
            
            

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_argument("exp_name", default= "finals_switch_and_no_switch", nargs='?',  type=str, help="pass a str for experiment name")
    group=parser.add_argument("x", default= 30., nargs='?',  type=float, help="arg_1")
    group=parser.add_argument("y", default= 1, nargs='?', type=float, help="arg_2")
    group=parser.add_argument("z", default= 1.0, nargs='?', type=float, help="arg_2")
    args=parser.parse_args()
    # can  assign args.x and args.y to vars
    args_dict = {'switches': args.x, 'MDlr': args.y, 'MDactive': args.z, 'exp_name': args.exp_name, 'seed': int(args.y)}
   
    config = Config(args_dict)

    # redefine some parameters for quick experimentation here.
    config.MDamplification = 30. #args_dict['switches']
    config.MDlearningrate = 5e-5
    config.MDlearningBiasFactor = args_dict['MDactive']

    pfcmd = PFCMD(config, args_dict=args_dict)

    
    if not config.reLoadWeights:
        t = time.perf_counter()
        pfcmd.train()
        print('training_time', (time.perf_counter() - t)/60, ' minutes')

        if config.saveData:
            pfcmd.save()
    else:
        filename = 'dataPFCMD/data_reservoir_PFC_MD'+str(pfcmd.MDstrength)+'_R'+str(pfcmd.RNGSEED)+ '.shelve'
        pfcmd.load(filename)
        # all 4cues in a block
        pfcmd.train()
     
    if config.saveData:
        pfcmd.fileDict.close()
    
    plt.show()
    

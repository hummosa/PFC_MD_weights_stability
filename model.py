# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Extends code by Aditya Gilra. Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import os
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
# from IPython import embed; embed()
# import pdb; pdb.set_trace()
from scipy.io import savemat
import sys,shelve, tqdm, time
import plot_utils as pltu
import argparse
cuda = False
if cuda: import torch
import torch

from refactor.data_generator import data_generator
from refactor.plot_figures import *
from refactor.ofc_mle import OFC, OFC_dumb
from refactor.config import Config

data_generator = data_generator()

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
                                    str(config.RNGSEED)+\
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
                        *config.G/np.sqrt(config.Nsub)
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
        
        

    def run_trial(self,association_level,ofc_signal,cue,target,MDeffect=True,
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
        error_smooth = np.zeros(shape=config.Nout)

        #init a Hebbian Trace for node perturbation to keep track of eligibilty trace.
        if config.reinforce:
            HebbTrace = np.zeros(shape=(config.Nout,config.Npfc))
            if config.reinforceReservoir:
                if cuda:
                    HebbTraceRec = torch.Tensor(np.zeros(shape=(config.Npfc,config.Npfc))).cuda()
                else:

                    HebbTraceRec = np.zeros(shape=(config.Npfc,config.Npfc))
            if config.MDreinforce:
                HebbTraceMD = np.zeros(shape=(config.Nmd,config.Npfc))


        ofc.Q_values = np.array(ofc.get_v() ) 

        for i in range(config.tsteps):
            rout = self.activation(xinp)
            routs[i,:] = rout
            outAdd = np.dot(self.wOut,rout)

            # Gather MD inputs
            if config.ofc_to_md_active: # TODO make an ofc that keeps track of MD neurons specialization
                # MDinp += np.array([.6,-.6]) if association_level in ofc.match_association_levels else np.array([-.6,.6]) 
                MDinp += np.array([-.6,.6]) if association_level in ofc.match_association_levels else np.array([.6,-.6]) 

            if config.positiveRates:
                MDinp +=  config.dt/config.tau * \
                        ( -MDinp + np.dot(self.wPFC2MD,rout) )
            else: # shift PFC rates, so that mean is non-zero to turn MD on
                MDinp +=  config.dt/config.tau * 10. * \
                        ( -MDinp + np.dot(self.wPFC2MD,(rout+1./2)) )

            # winner take all on the MD hardcoded for config.Nmd = 2
            if MDinp[0] > MDinp[1]: MDout = np.array([1,0])
            else: MDout = np.array([0,1])

            MDouts[i,:] = MDout
            MDinps[i, :]= MDinp

            # Gather PFC inputs

            if MDeffect:
                 # Add multplicative amplification of recurrent inputs.
                self.MD2PFCMult = np.dot(self.wMD2PFCMult * config.MDamplification,MDout)
                if cuda:
                    with torch.no_grad():
                        xadd = (1.+self.MD2PFCMult) * torch.matmul(self.Jrec, torch.Tensor(rout).cuda()).detach().cpu().numpy()
                else:
                    xadd = (1.+self.MD2PFCMult) * np.dot(self.Jrec,rout)
                # Additive MD input to PFC
                xadd += np.dot(self.wMD2PFC,MDout) 
                
            if i < config.cuesteps:
            #if MDeffect and useMult:
            #    xadd += self.MD2PFCMult * np.dot(self.wIn,cue)
                xadd += np.dot(self.wIn,cue)
                xadd += np.dot(self.wV,ofc.Q_values)

            # MD Hebbian learning
            if train and not config.MDreinforce:
                # MD presynaptic traces evolve dyanamically during trial and across trials
                # to decrease fluctuations.
                self.MDpreTrace += 1./config.tsteps/10. * \
                                    ( -self.MDpreTrace + rout )
                MDlearningBias = config.MDlearningBiasFactor * np.mean(self.MDpreTrace)
                MDrange = config.MDrange #0.05#0.1#0.06
                wPFC2MDdelta = np.outer(MDout-0.5,self.MDpreTrace-MDlearningBias) # Ali changed from 1e-4 and thresh from 0.13
                self.wPFC2MD = np.clip(self.wPFC2MD +config.MDlearningrate*wPFC2MDdelta,  -MDrange , MDrange ) # Ali lowered to 0.01 from 1. 
                # self.wMD2PFC = np.clip(self.wMD2PFC +fast_delta.T,-MDrange , MDrange ) # lowered from 10.
                # self.wMD2PFCMult = np.clip(self.wMD2PFCMult+ slow_delta.T,-2*MDrange /self.G, 2*MDrange /self.G) 
                # self.wMD2PFCMult = np.clip(self.wMD2PFC,-2*MDrange /self.G, 2*MDrange /self.G) * self.MDamplification
        
            # Add random perturbations to neurons 
            if config.reinforce:
                # Exploratory perturbations a la Miconi 2017 Perturb each output neuron independently
                #  with probability perturbProb
                perturbationOff = np.where(
                        np.random.uniform(size=config.Nout)>=config.perturbProb )
                perturbation = np.random.uniform(-1,1,size=config.Nout)
                perturbation[perturbationOff] = 0.
                perturbation *= config.perturbAmpl
                outAdd += perturbation
            
                if config.reinforceReservoir:
                    perturbationOff = np.where(
                            np.random.uniform(size=config.Npfc)>=config.perturbProb )
                    perturbationRec = np.random.uniform(-1,1,size=config.Npfc)
                    perturbationRec[perturbationOff] = 0.
                    perturbationRec *= config.perturbAmpl
                    xadd += perturbationRec
                
                if config.MDreinforce:
                    perturbationOff = np.where(
                            np.random.uniform(size=config.Nmd)>=config.perturbProb )
                    perturbationMD = np.random.uniform(-1,1,size=config.Nmd)
                    perturbationMD[perturbationOff] = 0.
                    perturbationMD *= config.perturbAmpl
                    MDinp += perturbationMD

            # Evolve inputs dynamically to cells before applying activations
            xinp += config.dt/config.tau * (-xinp + xadd)
            # Add noise
            xinp += np.random.normal(size=(config.Npfc))*config.noiseSD \
                        * np.sqrt(config.dt)/config.tau
            # Activation of PFC cells happens at the begnining of next timestep
            outInp += config.dt/config.tau * (-outInp + outAdd)
            out = self.activation(outInp)                

            error = out - target
            errors[i,:] = error
            outs[i,:] = out
            error_smooth += config.dt/config.tauError * (-error_smooth + error)
            
            if train: # Get the pre*post activity for the preturbation trace
                if config.reinforce:
                    # note: rout is the activity vector for previous time step
                    HebbTrace += np.outer(perturbation,rout)
                    if config.reinforceReservoir:
                        if cuda:
                            with torch.no_grad():
                                HebbTraceRec += torch.ger(torch.Tensor(perturbationRec).cuda(),torch.Tensor(rout).cuda())
                        else:
                            HebbTraceRec += np.outer(perturbationRec,rout)
                    if config.MDreinforce:
                        HebbTraceMD += np.outer(perturbationMD,rout)
                else:
                    # error-driven i.e. error*pre (perceptron like) learning
                    self.wOut += -config.learning_rate \
                                    * np.outer(error_smooth,rout)
        #* At trial end:
        #################
        cid = ofc.get_cid(association_level) # get inferred context id from ofc
        trial_err, all_contexts_err = ofc.get_trial_err(errors, association_level)
        baseline_err = ofc.baseline_err
            
        if train: #and config.reinforce:
            # with learning using REINFORCE / node perturbation (Miconi 2017),
            #  the weights are only changed once, at the end of the trial
            # apart from eta * (err-baseline_err) * hebbianTrace,
            #  the extra factor baseline_err helps to stabilize learning
            #   as per Miconi 2017's code,
            #  but I found that it destabilized learning, so not using it.
            self.wOut -= config.learning_rate * \
                    (trial_err-baseline_err[cid]) * \
                        HebbTrace #* baseline_err[cid]

            if config.reinforceReservoir:
                if cuda:
                    with torch.no_grad():
                        self.Jrec -= config.learning_rate * \
                                (trial_err-baseline_err[cid]) * \
                                    HebbTraceRec #* baseline_err[cid]  
                else:
                    self.Jrec -= config.learning_rate * \
                            (trial_err-baseline_err[cid]) * \
                                HebbTraceRec #* baseline_err[cid]                
            if config.MDreinforce:
                self.wPFC2MD -= config.learning_rate * \
                        (trial_err-baseline_err[cid]) * \
                            HebbTraceMD * 10. # changes too small Ali amplified #* baseline_err[cid]                
                self.wMD2PFC -= config.learning_rate * \
                        (trial_err-baseline_err[cid]) * \
                            HebbTraceMD.T * 10. #* baseline_err[cid]  
                                          
            baseline_err = ofc.update_baseline_err( all_contexts_err)

            #synaptic scaling and competition both ways at MD-PFC synapses.
            self.wPFC2MD /= np.linalg.norm(self.wPFC2MD)/ self.initial_norm_wPFC2MD
            self.wMD2PFC /= np.linalg.norm(self.wMD2PFC)/ self.initial_norm_wMD2PFC


        ofc.update_v(cue, out, target)
        
        # self.monitor.log({'qvalue0':ofc.Q_values[0], 'qvalue1':ofc.Q_values[1]})

        return cues, routs, outs, MDouts, MDinps, errors

    def train(self, data_gen):
        Ntrain = config.trials_per_block * config.Nblocks

        # Containers to save simulation variables
        wOuts = np.zeros(shape=(Ntrain,config.Nout,config.Npfc))
        wPFC2MDs = np.zeros(shape=(Ntrain,2,config.Npfc))
        wMD2PFCs = np.zeros(shape=(Ntrain,config.Npfc,2))
        wMD2PFCMults = np.zeros(shape=(Ntrain,config.Npfc,2))
        MDpreTraces = np.zeros(shape=(Ntrain,config.Npfc))
        wJrecs   = np.zeros(shape=(Ntrain, 40, 40))
        PFCrates = np.zeros( (Ntrain, config.tsteps, config.Npfc ) )
        MDinputs = np.zeros( (Ntrain, config.tsteps, config.Nmd) )
        MDrates  = np.zeros( (Ntrain, config.tsteps, config.Nmd) )
        Outrates = np.zeros( (Ntrain, config.tsteps, config.Nout  ) )
        Inputs   = np.zeros( (Ntrain, config.Ninputs))
        Targets  =  np.zeros( (Ntrain, config.Nout))
        self.hx_of_ofc_signal_lengths = []
        MSEs = np.zeros(Ntrain)

        for traini in tqdm.tqdm(range(Ntrain)):
            if traini % config.trials_per_block == 0:
                blocki = traini // config.trials_per_block  
                association_level, ofc_signal = next(data_gen.block_generator(blocki)) # Get the context index for this current block  
            if config.debug:
                print('context i: ', association_level)
            
            cue, target = data_gen.trial_generator(association_level)

            # trigger OFC switch signal
            config.no_of_trials_with_ofc_signal = 200 #int(args_dict['switches']) #lengths_of_directed_trials[blocki - config.Nblocks +6] #200-(40*(blocki-config.Nblocks + 6)) #decreasing no of instructed trials
            
            if ofc_signal is not 'off' and ((traini%config.trials_per_block) < config.no_of_trials_with_ofc_signal):
                config.ofc_to_md_active = True 
                if traini % config.trials_per_block == 0:
                    self.hx_of_ofc_signal_lengths.append((blocki, config.no_of_trials_with_ofc_signal))
            else:
                config.ofc_to_md_active = False

            _, routs, outs, MDouts, MDinps, errors = \
                self.run_trial(association_level,ofc_signal,cue,target,MDeffect=config.MDeffect,
                train=True)

            #Collect variables for analysis, plotting, and saving to disk    
            PFCrates[traini, :, :] = routs
            MDinputs[traini, :, :] = MDinps
            MDrates [traini, :, :] = MDouts
            Outrates[traini, :, :] = outs
            Inputs  [traini, :]    = np.concatenate((cue,ofc.Q_values) )
            Targets [traini, :]    = target
            wOuts   [traini,:,:] = self.wOut
            wPFC2MDs[traini,:,:] = self.wPFC2MD
            wMD2PFCs[traini,:,:] = self.wMD2PFC
            wMD2PFCMults[traini,:,:] = self.wMD2PFCMult
            MDpreTraces[traini,:] = self.MDpreTrace
            MSEs[traini] += np.mean(errors*errors)
            if config.reinforceReservoir:
                wJrecs[traini,:,:] = self.Jrec[:40, 0:25:1000].detach().cpu().numpy() # saving the whole rec is too large, 1000*1000*2200

        if config.plotFigs: #Plotting and writing results. All needs cleaned up.
            weights= [wOuts, wPFC2MDs, wMD2PFCs,wMD2PFCMults,  wJrecs, MDpreTraces]
            rates =  [PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs]
            plot_weights(self, weights, config)
            plot_rates(self, rates, config)
            plot_what_i_want(self, weights, rates, config)
            #from IPython import embed; embed()
            dirname="results/"+config.args_dict['exp_name']+"/"
            parm_summary= str(list(config.args_dict.values())[0])+"_"+str(list(config.args_dict.values())[1])+"_"+str(list(config.args_dict.values())[2])
            if not os.path.exists(dirname):
                    os.makedirs(dirname)
            fn = lambda fn_str:os.path.join(dirname, 'fig_{}_{}_{}.{}'.format(fn_str,parm_summary, time.strftime("%Y%m%d-%H%M%S"), config.figure_format) )
            self.figWeights.savefig     (fn('weights'), dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)
            self.figOuts.savefig  (fn('behavior'),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)
            self.figRates.savefig (fn('rates'),   dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)
            self.figTrials.savefig(fn('trials'),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)
            if config.debug:
                self.fig_monitor = plt.figure()
                self.monitor.plot(self.fig_monitor, self)
                self.figCustom.savefig(fn('custom'),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)
                self.fig_monitor.savefig(fn('monitor'),dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)

            # output some variables of interest:
            # md ampflication and % correct responses from model.
            filename7=os.path.join(dirname, 'values_of_interest.txt')
            filename7exits = os.path.exists(filename7)
            with open(filename7, 'a') as f:
                if not filename7exits:
                    [f.write(head+'\t') for head in ['switches', 'LR', 'HebbT', '1st', '2nd', '3rd', '4th', 'avg1-3', 'mean']]
                [f.write('{}\t '.format(val)) for val in [*config.args_dict.values()][:3]]
                # {:.2e} \t {:.2f} \t'.format(config.args_dict['switches'], config.args_dict['MDlr'],config.args_dict['MDactive'] ))
                for score in self.score:
                    f.write('{:.2f}\t'.format(score)) 
                f.write('\n')
            
            if config.saveData: # output massive weight and rate files
                np.save(fn('saved_Corrects')[:-4]+'.npy', self.corrects)
                import pickle
                filehandler = open(fn('saved_rates')[:-4]+'.pickle', 'wb')
                pickle.dump(rates, filehandler)
                filehandler.close()
                filehandler = open(fn('saved_weights')[:-4]+'.pickle', 'wb')
                pickle.dump(weights, filehandler)
                filehandler.close()

                # np.save(os.path.join(dirname, 'Rates{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), rates)
                # np.save(os.path.join(dirname, 'Weights{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), weights)

    def load(self,filename):
        d = shelve.open(filename) # open
        self.wOut = d['wOut']
        self.wMD2PFC     = d['MD2PFC']
        self.wMD2PFCMult = d['MD2PFCMult'] 
        self.wPFC2MD     = d['PFC2MD'] 
                         
        d.close()
        return None

    def save(self):
        self.fileDict['wOut'] = self.wOut
        self.fileDict['MD2PFC'] = self.wMD2PFC
        self.fileDict['MD2PFCMult'] = self.wMD2PFCMult
        self.fileDict['PFC2MD'] = self.wPFC2MD
            

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_argument("exp_name", default= "new_code", nargs='?',  type=str, help="pass a str for experiment name")
    group=parser.add_argument("x", default= 30., nargs='?',  type=float, help="arg_1")
    group=parser.add_argument("y", default= 7, nargs='?', type=float, help="arg_2")
    group=parser.add_argument("z", default= 1.0, nargs='?', type=float, help="arg_2")
    args=parser.parse_args()
    # can  assign args.x and args.y to vars
    args_dict = {'switches': args.x, 'MDlr': args.y, 'MDactive': args.z, 'exp_name': args.exp_name, 'seed': int(args.y)}
   
    config = Config(args_dict)
    ofc = OFC_dumb(config)
    ofc.set_context("0.7")

    # redefine some parameters for quick experimentation here.
    config.MDamplification = 30. #args_dict['switches']
    config.MDlearningrate = 5e-5
    config.MDlearningBiasFactor = args_dict['MDactive']

    pfcmd = PFCMD(config)
    
    if not config.reLoadWeights:
        t = time.perf_counter()
        pfcmd.train(data_generator)
        print('training_time', (time.perf_counter() - t)/60, ' minutes')

        # if config.saveData:
        #     pfcmd.save()
        #     pfcmd.fileDict.close()
    else:
        filename = 'dataPFCMD/data_reservoir_PFC_MD'+'_R'+str(pfcmd.RNGSEED)+ '.shelve'
        pfcmd.load(filename)
        pfcmd.train(data_generator)
       
    plt.show()
    

# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Extends code by Aditya Gilra. Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import torch
import json
from config import Config
from error_computations import Error_computations
# from refactor.ofc_trailtype import OFC as OFC_Trial
from vmPFC_k_means import OFC
from plot_figures import *
from data_generator import data_generator
import os
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
# from IPython import embed; embed()
# import pdb; pdb.set_trace()
from scipy.io import savemat
import tqdm
import time
import plot_utils as pltu
import argparse
from model import PFCMD



def train(areas, data_gen, config):
    pfcmd, vmPFC = areas
    Ntrain = config.trials_per_block * config.Nblocks

    # Containers to save simulation variables
    wOuts = np.zeros(shape=(Ntrain, config.Nout, config.Npfc))
    wPFC2MDs = np.zeros(shape=(Ntrain, 2, config.Npfc))
    wMD2PFCs = np.zeros(shape=(Ntrain, config.Npfc, 2))
    wMD2PFCMults = np.zeros(shape=(Ntrain, config.Npfc, 2))
    MDpreTraces = np.zeros(shape=(Ntrain, config.Npfc))
    wJrecs = np.zeros(shape=(Ntrain, 40, 40))
    PFCrates = np.zeros((Ntrain, config.tsteps, config.Npfc))
    MDinputs = np.zeros((Ntrain, config.tsteps, config.Nmd))
    if config.neuralvmPFC:
        vm_MDinputs = np.zeros((Ntrain, config.tsteps, config.Nmd))
        vm_Outrates = np.zeros((Ntrain, vm_config.tsteps, vm_config.Nout))
    MDrates = np.zeros((Ntrain, config.tsteps, config.Nmd))
    Outrates = np.zeros((Ntrain, config.tsteps, config.Nout))
    Inputs = np.zeros((Ntrain, config.Ninputs+3)) # Adding OFC latents temp #TODO remove this.
    Targets = np.zeros((Ntrain, config.Nout))
    pfcmd.hx_of_ofc_signal_lengths = []
    MSEs = np.zeros(Ntrain)

    for traini in tqdm.tqdm(range(Ntrain)):
        if traini % config.trials_per_block == 0:
            blocki = traini // config.trials_per_block
            association_level, ofc_signal = next(data_gen.block_generator(
                blocki))  # Get the context index for this current block
        if config.debug:
            print('context i: ', association_level)

        cue, target = data_gen.trial_generator(association_level)

        # trigger OFC switch signal for a number of trials in the block
        
        ofc_signal_delay = 100
        bi = traini % config.trials_per_block
        if ofc_signal is not 'off' and ((bi > ofc_signal_delay) and (bi < config.no_of_trials_with_ofc_signal+ofc_signal_delay)):
            config.ofc_to_md_active = True
            if traini % config.trials_per_block == 0:
                pfcmd.hx_of_ofc_signal_lengths.append(
                    (blocki+.25, config.no_of_trials_with_ofc_signal))
        else:
            config.ofc_to_md_active = False
        vmPFC.hx_of_ofc_signal_lengths = []
        
        q_values_before = ofc.get_v()

        _, routs, outs, MDouts, MDinps, errors = \
            pfcmd.run_trial(association_level, ofc, error_computations, cue, target, config, MDeffect=config.MDeffect,
                            train=config.train)
        q_values_after = ofc.get_v()

        if config.neuralvmPFC:
            vmPFC_input = np.array([errors.mean(), q_values_before[0]])
            # _, _, vm_outs, _, vm_MDinps,_ =\
        _, routs, vm_outs, MDouts, MDinps, _ =\
            vmPFC.run_trial(association_level, ofc_vmPFC, error_computations_vmPFC,
                                vmPFC_input, q_values_after, vm_config, MDeffect=config.MDeffect,
                                train=config.train)

        # Collect variables for analysis, plotting, and saving to disk
        area_to_plot = vmPFC
        PFCrates[traini, :, :] = routs
        MDinputs[traini, :, :] = MDinps
        if config.neuralvmPFC:
            vm_MDinputs[traini, :, :] = vm_MDinps if not area_to_plot is vmPFC else MDinps
            vm_Outrates[traini, :, :] = vm_outs 
        MDrates[traini, :, :] = MDouts
        Outrates[traini, :, :] = outs
        Inputs[traini, :] = np.concatenate([cue, ofc.Q_values, error_computations.p_sm_snm_ns])
        Targets[traini, :] = target
        wOuts[traini, :, :] = area_to_plot.wOut
        wPFC2MDs[traini, :, :] = area_to_plot.wPFC2MD
        wMD2PFCs[traini, :, :] = area_to_plot.wMD2PFC
        wMD2PFCMults[traini, :, :] = area_to_plot.wMD2PFCMult
        MDpreTraces[traini, :] = area_to_plot.MDpreTrace
        MSEs[traini] += np.mean(errors*errors)
        if config.reinforceReservoir:
            # saving the whole rec is too large, 1000*1000*2200
            wJrecs[traini, :, :] = area_to_plot.Jrec[:40,
                                                0:25:1000].detach().cpu().numpy()

        # Saves a data file per each trial
        # TODO possible variables to add for Mante & Sussillo condition analysis:
        #   - association level, OFC values
        if config.args_dict["save_data_by_trial"]:
            trial_weights = {
                "w_outputs": wOuts[traini].tolist(),
                "w_PFC2MD": wPFC2MDs[traini].tolist(),
                "w_MD2PFCs": wMD2PFCs[traini].tolist(),
                "w_MD2PFC_mults": wMD2PFCMults[traini].tolist(),
                "w_MD_pretraces": MDpreTraces[traini].tolist()
            }
            trial_rates = {
                "r_PFC": PFCrates[traini].tolist(),
                "MD_input": MDinputs[traini].tolist(),
                "r_MD": MDrates[traini].tolist(),
                "r_output": Outrates[traini].tolist(),
            }
            trial_data = {
                "input": Inputs[traini].tolist(),
                "target": Targets[traini].tolist(),
                "mse": MSEs[traini]
            }

            d = f"{config.args_dict['outdir']}/{config.args_dict['exp_name']}/by_trial"
            if not os.path.exists(d):
                os.makedirs(d)
            with open(f"{d}/{traini}.json", 'w') as outfile:
                json.dump({"trial_data": trial_data,
                            "network_weights": trial_weights,
                            "network_rates": trial_rates}, outfile)

    # collect input from OFC and add it to Inputs for outputting.
    # if ofc is off, the trial gets 0, if it is stimulating the 'match' side, it gets 1
    # and 'non-match' gets -1. Although currently match and non-match have no meaning,
    # as MD can be responding to either match or non-match. The disambiguation happens in post analysis
    ofc_inputs = np.zeros((Ntrain,1))
    tpb = config.trials_per_block
    if len(pfcmd.hx_of_ofc_signal_lengths) > 0:
        for bi in range(config.Nblocks):
            ofc_hx = np.array(pfcmd.hx_of_ofc_signal_lengths)
            if bi in ofc_hx[:,0]:
                if data_generator.ofc_control_schedule[bi] is 'match':
                    ofc_inputs[bi*tpb:bi*tpb+config.no_of_trials_with_ofc_signal] = np.ones((config.no_of_trials_with_ofc_signal, 1))
                else:
                    ofc_inputs[bi*tpb:bi*tpb+config.no_of_trials_with_ofc_signal] = -np.ones((config.no_of_trials_with_ofc_signal, 1))
    Inputs = np.concatenate((Inputs, ofc_inputs), axis=-1)

    if config.plotFigs:  # Plotting and writing results. Needs cleaned up.
        weights = [wOuts, wPFC2MDs, wMD2PFCs,
                    wMD2PFCMults,  wJrecs, MDpreTraces]
        rates = [PFCrates, MDinputs, MDrates,
                    Outrates, Inputs, Targets, MSEs]
        plot_q_values([vm_Outrates, vm_MDinputs])
        plot_weights(area_to_plot, weights, config)
        plot_rates(area_to_plot, rates, config)
        plot_what_i_want(area_to_plot, weights, rates, config)
        #from IPython import embed; embed()
        dirname = config.args_dict['outdir'] + \
            "/"+config.args_dict['exp_name']+"/"
        parm_summary = str(list(config.args_dict.values())[0])+"_"+str(
            list(config.args_dict.values())[1])+"_"+str(list(config.args_dict.values())[2])
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        def fn(fn_str): return os.path.join(dirname, 'fig_{}_{}_{}.{}'.format(
            fn_str, parm_summary, time.strftime("%Y%m%d-%H%M%S"), config.figure_format))
        area_to_plot.figWeights.savefig(fn('weights'), dpi=pltu.fig_dpi,
                                facecolor='w', edgecolor='w', format=config.figure_format)
        area_to_plot.figOuts.savefig(fn('behavior'), dpi=pltu.fig_dpi,
                                facecolor='w', edgecolor='w', format=config.figure_format)
        area_to_plot.figRates.savefig(fn('rates'),   dpi=pltu.fig_dpi,
                                facecolor='w', edgecolor='w', format=config.figure_format)
        area_to_plot.figTrials.savefig(fn('trials'), dpi=pltu.fig_dpi,
                                facecolor='w', edgecolor='w', format=config.figure_format)
        if config.debug:
            area_to_plot.fig_monitor = plt.figure()
            area_to_plot.monitor.plot(area_to_plot.fig_monitor, area_to_plot)
            area_to_plot.figCustom.savefig(
                fn('custom'), dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)
            area_to_plot.fig_monitor.savefig(
                fn('monitor'), dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)

        # output some variables of interest:
        # md ampflication and % correct responses from model.
        filename7 = os.path.join(dirname, 'values_of_interest.txt')
        filename7exits = os.path.exists(filename7)
        with open(filename7, 'a') as f:
            if not filename7exits:
                [f.write(head+'\t') for head in ['switches', 'LR',
                                                    'HebbT', '1st', '2nd', '3rd', '4th', 'avg1-3', 'mean']]
            [f.write('{}\t '.format(val))
                for val in [*config.args_dict.values()][:3]]
            # {:.2e} \t {:.2f} \t'.format(config.args_dict['switches'], config.args_dict['MDlr'],config.args_dict['MDactive'] ))
            for score in area_to_plot.score:
                f.write('{:.2f}\t'.format(score))
            f.write('\n')

        if config.saveData:  # output massive weight and rate files
            np.save(fn('saved_Corrects')[:-4]+'.npy', area_to_plot.corrects)
            import pickle
            filehandler = open(fn('saved_rates')[:-4]+'.pickle', 'wb')
            pickle.dump(rates, filehandler)
            filehandler.close()
            filehandler = open(fn('saved_weights')[:-4]+'.pickle', 'wb')
            pickle.dump(weights, filehandler)
            filehandler.close()

            # np.save(os.path.join(dirname, 'Rates{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), rates)
            # np.save(os.path.join(dirname, 'Weights{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), weights)



###################################################################################
###################################################################################
###################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument("exp_name", default="new_code",
                                nargs='?',  type=str, help="pass a str for experiment name")
    group = parser.add_argument(
        "x", default=30., nargs='?',  type=float, help="arg_1")
    group = parser.add_argument(
        "y", default=8, nargs='?', type=float, help="arg_2")
    group = parser.add_argument(
        "z", default=1.0, nargs='?', type=float, help="arg_2")
    group = parser.add_argument("--outdir", default="./results",
                                nargs='?',  type=str, help="pass a str for data directory")
    args = parser.parse_args()
    # can  assign args.x and args.y to vars
    # OpenMind shared directory: "/om2/group/halassa/PFCMD-ali-sabrina"
    args_dict = {'switches': args.x, 'MDlr': args.y, 'MDactive': args.z,
                 'outdir':  args.outdir, 'exp_name': args.exp_name, 'seed': int(args.y),
                 "save_data_by_trial": False}

    config = Config(args_dict)
    vm_config = Config(args_dict)
    vm_config.Ninputs = 6
    data_generator = data_generator(config)


    ofc = OFC()
    ofc_vmPFC = OFC()
    error_computations = Error_computations(config)
    error_computations_vmPFC = Error_computations(vm_config)

    # redefine some parameters for quick experimentation here.
    config.no_of_trials_with_ofc_signal = int(args_dict['switches'])
    config.MDamplification = 30.  # args_dict['switches']
    config.MDlearningrate = 5e-5
    config.MDlearningBiasFactor = args_dict['MDactive']

    pfcmd = PFCMD(config)
    if config.neuralvmPFC:
        vmPFC = PFCMD(vm_config)
    else:
        vmPFC = []

    if not config.reLoadWeights:
        t = time.perf_counter()
        # pfcmd.train(data_generator)
        train((pfcmd, vmPFC), data_generator, config)
        print('training_time', (time.perf_counter() - t)/60, ' minutes')

        # if config.saveData:
        #     pfcmd.save()
        #     pfcmd.fileDict.close()
    else:
        filename = 'dataPFCMD/data_reservoir_PFC_MD' + \
            '_R'+str(pfcmd.RNGSEED) + '.shelve'
        pfcmd.load(filename)
        # pfcmd.train(data_generator)
        train(pfcmd, data_generator, config)


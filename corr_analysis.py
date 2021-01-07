##############################################################
############### Correlation 2 WITH SWITCH  ################################
##############################################################
# multiple seedds

##############################################################
############### Correlation 2 WITH SWITCH  ################################
##############################################################
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

file_format= 'EPS'

exp_name = 'switch_presentation'

files = os.listdir('./results/' + exp_name+'/')
rates_files = [file for file in files if 'Rates' in file]
weights_files = [file for file in files if 'Weights' in file]

# strings to look for in filtering file for each experiemnt to compare 
comp_one = '180.0_' # with with OFC switches 180
comp_two = 's0.0_' # with MD clampped to one neuron

comp_one_files = [os.path.join('./results/'+exp_name,file) for file in rates_files if comp_one in file]
comp_two_files = [os.path.join('./results/'+exp_name,file) for file in rates_files if (comp_two in file)]



for fi, file_name in tqdm(enumerate(comp_one_files)):

    filehandler = open(file_name, 'rb')
    rate
    
    
    
    
    
    s_one = pickle.load(filehandler)
    filehandler.close()
    PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs= rates_one

    prates= np.mean(PFCrates, axis=1)

    # Corre btn MD 0 and PFC V1 vs whole of PFC
    Trials, Tsteps, Neurons = PFCrates.shape

    corrs_switch = np.zeros(shape=(len(comp_one_files), Trials, Neurons))
    for t in (range(Trials)):
        for n in range(Neurons):
            corrs_switch[fi,t,n] = np.corrcoef(MDinputs[t,:,0], PFCrates[t,:,n])[0,1]
    del rates_one
    del PFCrates
    del prates
        
ax = plt.gca()
ax.plot(np.convolve( np.mean(corrs_switch[:,:,150:200], axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ))
ax.plot(np.convolve( np.mean(corrs_switch[:,:,50:100], axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ))
# ax.set_xlim([1900,2200])
# plt.savefig(f'corr_input_populations_switch2.{file_format}', format=file_format)
try:
    np.save('corrs_switch_180', corrs_switch, )
except:
    pass

fig = plt.figure()
fig.set_size_inches([12,3])
ax = plt.gca()

# ax.plot(np.convolve( np.mean(corrs_switch[opposites,:, 0:50 ], axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ), label='PFC Down-V1')
# ax.plot(np.convolve( np.mean(corrs_switch[opposites,:,50:100], axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ), label='PFC Up-V2')

# All
# ax.plot(np.convolve( np.mean(np.concatenate((corrs_switch[opposites,:,:250],corrs_switch[opposites,:,250:]), axis=2), axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ), label='PFC', linewidth=1, alpha=0.8)

# Populations v1 v2
ax.plot(np.convolve( np.mean(np.concatenate((corrs_switch[:,:,0:50],corrs_switch[:,:,100:150]), axis=2), axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ), label='PFC V1', linewidth=1, alpha=0.8)
ax.plot(np.convolve( np.mean(np.concatenate((corrs_switch[:,:,50:100],corrs_switch[:,:,150:200]), axis=2), axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ), label='PFC V2', linewidth=1, alpha=0.8)

# select only trials where ofc went against md
# opposites = [1,4,4,4,2,3]#[1,3,4,6,8,9] # NOT SURE THIS IS WORKING trials where ofc switched pfc against current strategy
# ax.plot(np.convolve( np.mean(np.concatenate((corrs_switch[opposites,:,0:50],corrs_switch[opposites,:,100:150]), axis=2), axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ), label='PFC V1', linewidth=1, alpha=0.8)
# ax.plot(np.convolve( np.mean(np.concatenate((corrs_switch[opposites,:,50:100],corrs_switch[opposites,:,150:200]), axis=2), axis=2).mean(axis=0), np.ones((5,))/5, mode='valid' ), label='PFC V2', linewidth=1, alpha=0.8)


Nblocks = 12
tpb = 400
for ib in range(1, Nblocks,2):
            ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')

# ax.set_xlim([1900,3600])
# ax.set_ylim([-0.02, 0.02])
ax.set_xlabel('trials')
ax.set_ylabel('Corr')
ax.set_title('Correlation to MD 0')
ax.legend()
# file_format='EPS'
# plt.savefig(f'corr_final_v1_v2.{file_format}', format=file_format)



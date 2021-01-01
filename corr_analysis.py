##############################################################
############### Correlation 2 WITH SWITCH  ################################
##############################################################
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

file_format= 'PNG'

exp_name = 'ofc_v1v2_switch'

files = os.listdir('./results/' + exp_name+'/')
rates_files = [file for file in files if 'Rates' in file]
weights_files = [file for file in files if 'Weights' in file]

# strings to look for in filtering file for each experiemnt to compare 
comp_one = '0_1.0' # with MD
# comp_two = '0_0.3' # with MD clampped to one neuron

comp_one_files = [os.path.join('./results/'+exp_name,file) for file in rates_files if comp_one in file]
# comp_two_files = [os.path.join('./results/'+exp_name,file) for file in rates_files if (comp_two in file)]


filehandler = open(comp_one_files[0], 'rb')
rates_one = pickle.load(filehandler)
filehandler.close()
PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs= rates_one

PFCrates_first_block = np.mean(PFCrates[:500, :, :], axis=1)
pr1 = np.mean(PFCrates_first_block, axis=0)
high1 = pr1.argsort()[-10:]

PFCrates_second_block = np.mean(PFCrates[500:1000, :, :], axis=1)
pr2 = np.mean(PFCrates_second_block, axis=0)
high2 = pr2.argsort()[-10:]

prates= np.mean(PFCrates, axis=1)


# Corre btn MD 0 and PFC V1 vs whole of PFC
# Trial data
Trials, Tsteps, Neurons = PFCrates.shape

corrs_switch = np.zeros(shape=(Trials, Neurons))
for t in tqdm(range(Trials)):
    for n in range(Neurons):
        corrs_switch[t,n] = np.corrcoef(MDinputs[t,:,0], PFCrates[t,:,n])[0,1]
        
ax = plt.gca()
ax.plot(np.convolve( np.mean(corrs_switch[:,150:200], axis=1), np.ones((5,))/5, mode='valid' )) #this is the down v2 population
ax.plot(np.convolve( np.mean(corrs_switch[:,50:100], axis=1), np.ones((5,))/5, mode='valid' ))  # this the up v2 population
# ax.set_xlim([1900,2200])
plt.savefig(f'corr_input_populations_switch2.{file_format}', format=file_format)
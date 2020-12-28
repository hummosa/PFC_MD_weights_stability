#%%Analysis.py

import matplotlib.pyplot as plt
import numpy as np
import os
file_format= 'PNG'


#%%
exp_name = 'switch_prob_runs'
exp_name = 'seeds_seeds_seeds'

#get files
files = os.listdir('./results/' + exp_name+'/')
corrects_files = [file for file in files if 'Corrects' in file]

# comp_one = '0200' # MD clamped to one neuron active
comp_one = '.2'
comp_two = '1.0_20'

comp_one_files = [os.path.join('./results/'+exp_name,file) for file in corrects_files if comp_one in file]
comp_two_files = [os.path.join('./results/'+exp_name,file) for file in corrects_files if (comp_two in file) and ('0200' not in file)]

corr_one = [np.convolve(np.load(c1f), np.ones((40,))/40, mode='valid') for c1f in comp_one_files]
corr_two = [np.convolve(np.load(c2f), np.ones((40,))/40, mode='valid') for c2f in comp_two_files]

corrects_one = np.stack(corr_one)
corrects_two = np.stack(corr_two) # shape 2, 2500 [no of files, no of trials]

mean_one = np.mean(corrects_one, axis= 0)
std_one  = np.std (corrects_one, axis= 0)

mean_two = np.mean(corrects_two, axis= 0)
std_two  = np.std (corrects_two, axis= 0)
# %%

cog_flex_avg_one = [np.mean(corrects_one[:,s:s+100]) for s in range(0, 3000, 500)]
cog_flex_avg_two = [np.mean(corrects_two[:,s:s+100]) for s in range(0, 3000, 500)]
cog_flex_avg_one[-1]=(np.mean(corrects_one))
cog_flex_avg_two[-1]=(np.mean(corrects_two))

cog_flex_std_one = np.array([np.std(corrects_one[:,s:s+100]) for s in range(0, 3000, 500)])
cog_flex_std_two = np.array([np.std(corrects_two[:,s:s+100]) for s in range(0, 3000, 500)])
cog_flex_std_one[-1]=np.std(corrects_one)
cog_flex_std_two[-1]=np.std(corrects_two)

plt.bar(range(1,19,3),cog_flex_avg_one)
plt.errorbar(range(1,19,3),cog_flex_avg_one, yerr=cog_flex_std_one*1.65 / np.sqrt(10), fmt='o', color='black')
# *1.65 ttimes std div by sqrt(n) to get CI 
plt.bar(range(2,19,3),cog_flex_avg_two, color='tab:orange')
plt.errorbar(range(2,19,3),cog_flex_avg_two, yerr=cog_flex_std_two*1.65 / np.sqrt(10), fmt='o', color='black')
plt.savefig(f'error_bars.{file_format}', format=file_format)

plt.figure()
plt.plot(range(len(mean_one)), mean_one)
plt.fill_between(range(len(mean_one)), mean_one-std_one, mean_one+std_one, alpha=.4)

plt.plot(range(len(mean_two)), mean_two)
plt.fill_between(range(len(mean_two)), mean_two-std_two, mean_two+std_two, alpha=.4)

plt.savefig(f'average_correct_w_wout_MD.{file_format}', format=file_format)

#%% rates
##############################################################
################################################ PFC rates
##############################################################
exp_name = 'switch_prob_runs'

files = os.listdir('./results/' + exp_name+'/')
rates_files = [file for file in files if 'Rates' in file]
weights_files = [file for file in files if 'Weights' in file]

# strings to look for in filtering file for each experiemnt to compare 

comp_one = '1_1.0'
comp_two = '1_0.3'

comp_one_files = [os.path.join('./results/'+exp_name,file) for file in rates_files if comp_one in file]
comp_two_files = [os.path.join('./results/'+exp_name,file) for file in rates_files if (comp_two in file)]

comp_one_weights_files = [os.path.join('./results/'+exp_name,file) for file in weights_files if comp_one in file]
comp_two_weights_files = [os.path.join('./results/'+exp_name,file) for file in weights_files if (comp_two in file)]

import pickle
filehandler = open(comp_one_files[0], 'rb')
rates_one = pickle.load(filehandler)
filehandler.close()

filehandler = open(comp_one_weights_files[0], 'rb')
weights_one = pickle.load(filehandler)
filehandler.close()

PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs= rates_one
wOuts, wPFC2MDs, wMD2PFCs, wMD2PFCMults, wJrecs, MDpreTraces = weights_one


PFCrates_first_block = np.mean(PFCrates[:500, :, :], axis=1)
pr1 = np.mean(PFCrates_first_block, axis=0)
high1 = pr1.argsort()[-10:]

PFCrates_second_block = np.mean(PFCrates[500:1000, :, :], axis=1)
pr2 = np.mean(PFCrates_second_block, axis=0)
high2 = pr2.argsort()[-10:]

prates= np.mean(PFCrates, axis=1)

fig, axes,  = plt.subplots(2,1, sharey=True)
ax1, ax2= axes
ax1.plot(range(2500),np.array([prates[:,high1].mean(axis=1), prates[:,high2].mean(axis=1)]).T, alpha =0.7)
ax2.plot(np.mean(MDrates, 1), alpha=0.6)

fig.savefig(f'rates.{file_format}', format=file_format)

#%% Individual neuron rates:
ninp = np.array(Inputs)
fig, axes,  = plt.subplots(4,4, sharey=True)
fig.set_size_inches([9,7])
faxes = axes.flatten()
for xi, ai in enumerate(range(16)):
    ax = faxes[xi]
    # ax.plot(range(1000,2500), prates[1000:,ai*2+2], '.',markersize=0.5)
    # ax.plot(range(1000,2500), prates[1000:,ai*2+2], '.',markersize=0.5)
    
    ax.plot(np.arange(1000,2500)[ninp[1000:2500,0]==1.],prates[1000:,ai*2+1][ninp[1000:2500,0]==1.], '.', markersize =0.5, color='tab:blue', label='Up')
    ax.plot(np.arange(1000,2500)[ninp[1000:2500,0]==0.],prates[1000:,ai*2+1][ninp[1000:2500,0]==0.], '.', markersize =0.5, color='tab:red',  label='Down')
    # ax.legend()
    
    ax.set_ylim([0,1])
fig.savefig(f'individual_neurons2.{file_format}', format=file_format)    
#%% Weights
# plot high1 to out
# plt.plot(wOuts[:,0,high1], color='tab:blue')
plt.plot(wOuts[:,0,high2], color='tab:red')
down = Inputs[:,1]
up = Inputs[:,0]
# plt.plot(np.mean(wOuts[:,1,high1], axis=-1), color='tab:blue')
# plt.plot(np.mean(wOuts[:,1,high2], axis=-1), color='tab:red')

down = down==1.0
down_rates = prates[down,:]
down_means = np.mean(down_rates, 0)

up = up==1.0
up_rates = prates[up, :]
up_means = np.mean(up_rates, axis =0)

diff = up_means - down_means
sort_dif = np.argsort(diff, axis=0)

#%% Correlation analysis

# Corre btn MD 0 and PFC V1 vs whole of PFC
# Trial data
Trials, Tsteps, Neurons = PFCrates.shape

corrs = np.zeros(shape=(Trials, Neurons))
for t in range(Trials):
    for n in range(Neurons):
        corrs[t,n] = np.corrcoef(MDinputs[t,:,0], PFCrates[t,:,n])[0,1]
#%%
plt.ion()
plt.plot(np.convolve( np.mean(corrs[:,150:200], axis=1), np.ones((5,))/5, mode='valid' ))
plt.plot(np.convolve( np.mean(corrs[:,50:100], axis=1), np.ones((5,))/5, mode='valid' ))

# Averages data

# %%

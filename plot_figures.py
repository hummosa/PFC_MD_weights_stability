import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pltu

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def stats(var, var_name=None):
    if type(var) == type([]): # if a list
        var = np.array(var)
    elif type(var) == type(np.array([])):
        pass #if already a numpy array, just keep going.
    else: #assume torch tensor
        pass
        # var = var.detach().cpu().numpy()
    if var_name:
        print(var_name, ':')   
    out = ('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}, norm {}'.format(var.mean(), var.var(), var.min(), var.max(),np.linalg.norm(var) ))
    print(out)
    return (out)



def plot_rates(pfcmd, rates, labels = ['wAto0(r) wAto1(b)', 'wBto0(r) wBto1(b)', 'wCto0(r) wCto1(b)']):
    PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs= rates
    # these tensors are  training_i x tsteps x no_neuron 
    p = pfcmd.Nsub
    Ntrain = PFCrates[:,:, :5].shape[0]
    yticks = (0, 0.5,1)
    xticks = [0, 1000, 2000]
    pfcmd.figRates, axes = plt.subplots(4,3)#, sharex=True)# , sharey=True)
    pfcmd.figRates.set_size_inches([9,7])

    ax = axes[0,0]
    ax.plot(range(Ntrain),np.mean( PFCrates[:,:,:5], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','Mean FR')
    ax.set_title('PFC Cue 1')
    
    ax = axes[0,1]
    ax.plot(range(Ntrain),np.mean( PFCrates[:, :,p:p+5], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_title('PFC Cue 2')

    ax = axes[0,2]
    ax.plot(range(Ntrain),np.mean( PFCrates[:, :,p*2:p*2+5], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False)
    pltu.axes_labels(ax,'','')
    ax.set_title('PFC Cue 3')

    ax = axes[1,0]
    ax.plot(range(Ntrain),np.mean( MDrates[:,:,0], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','mean FR')
    ax.set_title('MD 0')
    
    ax = axes[1,1]
    ax.plot(range(Ntrain),np.mean( MDrates[:,:,1], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_title('MD 1')
    
    ax = axes[1,2]
    ax.plot(range(Ntrain),np.mean( MDinputs[:, :,:], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_title('MD avg inputs')
    
    ax = axes[2,0]
    ax.plot(range(Ntrain),np.mean( Outrates[:,:,0], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','mean FR')
    ax.set_title('Out 0')
    
    ax = axes[2,1]
    ax.plot(range(Ntrain),np.mean( Outrates[:,:,1], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_title('Out 1')
    
    ax = axes[2,2]
    ax.plot(range(Ntrain),np.mean( Outrates[:, :,:], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_title('Out 0 and 1')

    ax = axes[3,0]
    # Plot MSE
    ax.plot(MSEs)
    ax.plot(smooth(MSEs, 8), 'tab:orange', linewidth= pltu.linewidth)
    pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
    pltu.axes_labels(ax,'Trials','MSE')
    ax.set_title('MSE')

    # ax.plot(range(Ntrain),Inputs[:, 0] + np.random.uniform(-0.1, 0.1, size=(Ntrain,1)) , 'o', markersize =0.5)
    # ax.plot(range(Ntrain),Targets[:, 0] + np.random.uniform(-0.1, 0.1, size=(Ntrain,1)) , 'o', markersize =0.5)
    # pltu.beautify_plot(ax,x0min=False,y0min=False)
    # pltu.axes_labels(ax,'','Inputs')
    

    # pfcmd.figOuts = plt.figure()
    # ax = pfcmd.figOuts.add_subplot(311)
    # ax.plot(range(Ntrain), 1.*(Targets[:,0] == out_higher_mean) -0.3+ np.random.uniform(-0.01, 0.01, size=(Ntrain,) ) , '.', markersize =0.5)
    # ax.set_title('Percent correct answers smoothened over 20 trials')
    # ax = pfcmd.figOuts.add_subplot(312)
    # ax.plot(smooth((Targets[:,0] == out_higher_mean)*1., 20), linewidth=pltu.linewidth)
    # pltu.axes_labels(ax, 'Trials', '% Correct')
    out_higher_mean = 1.*( np.mean( Outrates[:, :,0], axis=1) > np.mean( Outrates[:, :,1], axis=1) )
    out_higher_endFR =1.*( Outrates[:, -1 ,0] >  Outrates[:, -1 ,1]                                )

    Matches =  1. * (Targets[:,0] == Inputs[:,0])                   #+ np.random.uniform(-noise, noise, size=(Ntrain,) )
    Responses= 1.* (out_higher_mean == Inputs[:,0]) * 0.8 + 0.1     #+ np.random.uniform(-noise, noise, size=(Ntrain,) )

    noise = 0.1
    ax = axes[3,1]
    ax.plot(Matches  + np.random.uniform(-noise, noise, size=(Ntrain,) ),  'o', markersize = 2)
    ax.plot(Responses+ np.random.uniform(-noise, noise, size=(Ntrain,) ),  'x', markersize = 2)
    pltu.axes_labels(ax, 'Trials', 'non-match    Match')
    # ax.set_title('Blue o: Correct    Orange x: response')
    ax.set_ylim([-0.3, 1.3])
    ax.set_xlim([0, 2200])
    
    ax = axes[3,2]

    ax.plot(Matches  + np.random.uniform(-noise, noise, size=(Ntrain,) ),  'o', markersize = 2)
    ax.plot(Responses+ np.random.uniform(-noise, noise, size=(Ntrain,) ),  'x', markersize = 2)
    pltu.axes_labels(ax, 'Trials', 'non-match    Match')
    # ax.set_title('Blue o: Correct    Orange x: response')
    ax.set_ylim([-0.3, 1.3])
    ax.set_xlim([970, 1100])


    pfcmd.figOuts = plt.figure()

    noise = 0.05
    ax = pfcmd.figOuts.add_subplot(311)
    ax.plot(Matches + np.random.uniform(-noise, noise, size=(Ntrain,)  ),    'o', markersize = 1)
    ax.plot(Responses+ np.random.uniform(-noise, noise, size=(Ntrain,) ),  'x', markersize = 1)
    pltu.axes_labels(ax, 'Trials', 'non-match    Match')
    ax.set_title('Blue o: Correct    Orange x: response')
    ax.set_ylim([-0.3, 1.3])
    ax.set_xlim([0, 2200])

    ax = pfcmd.figOuts.add_subplot(312)
    ax.plot(Matches,    'o', markersize = 2)
    ax.plot(Responses,  'x', markersize = 2)
    pltu.axes_labels(ax, 'Trials', 'non-match    Match')
    ax.set_ylim([-0.3, 1.3])
    ax.set_xlim([960, 1230])

    ax = pfcmd.figOuts.add_subplot(313)
    ax.plot(Matches,    'o', markersize = 3)
    ax.plot(Responses,  'x', markersize = 3)
    pltu.axes_labels(ax, 'Trials', 'non-match    Match')
    ax.set_ylim([-0.3, 1.3])
    ax.set_xlim([1970, 2050])

    pfcmd.figRates.tight_layout()
    

def plot_weights(pfcmd, weights, labels = ['wAto0(r) wAto1(b)', 'wBto0(r) wBto1(b)', 'wCto0(r) wCto1(b)']):
    wOuts, wPFC2MDs, wMD2PFCs, wMD2PFCMults, wJrecs, MDpreTraces = weights
    xticks = [0, 1000, 2000]
    # plot output weights evolution
    pfcmd.fig3, axes = plt.subplots(4,3)#, sharex=True) #, sharey=True)
    # pfcmd.fig3.set_figheight = pltu.twocolumnwidth
    # pfcmd.fig3.set_figwidth = pltu.twocolumnwidth
    pfcmd.fig3.set_size_inches([9,7])
    p = pfcmd.Nsub
    for pi, PFC in enumerate(['PFC cue 1', 'PFC cue 2', 'PFC cue 3']):
        ax = axes[0,pi]
        ax.plot(wOuts[:,0, p*pi:p*pi+5],'tab:red', linewidth= pltu.linewidth)
        ax.plot(wOuts[:,1, p*pi:p*pi+5],'tab:blue', linewidth= pltu.linewidth)
        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        if pi == 0: pltu.axes_labels(ax,'','to Out-0 & 1 (r,b)')
        ax.set_title(PFC)

    for pi, PFC in enumerate(['PFC cue 1', 'PFC cue 2', 'PFC cue 3']):
        ax = axes[1,pi]
        ax.plot(wPFC2MDs[:,0, p*pi:p*pi+5],'tab:red', linewidth= pltu.linewidth)
        ax.plot(wPFC2MDs[:,1, p*pi:p*pi+5],'tab:blue', linewidth= pltu.linewidth)
        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        if pi == 0: pltu.axes_labels(ax,'','to MD-0(r) 1(b)')

        ax = axes[2,pi]
        ax.plot(wMD2PFCs[:,p*pi:p*pi+5, 0],'tab:red', linewidth= pltu.linewidth)
        ax.plot(wMD2PFCs[:,p*pi:p*pi+5, 1],'tab:blue', linewidth= pltu.linewidth)
        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        if pi == 0: pltu.axes_labels(ax,'','from MD-0(r) 1(b)')

        # plot PFC to MD pre Traces
        ax = axes[3,pi]
        ax.plot(MDpreTraces[:,p*pi:p*pi+5], linewidth = pltu.linewidth)
        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        pltu.axes_labels(ax,'Trials','pre')


    # axes[0,0].plot(wOuts[:,0,:5],'tab:red', linewidth= pltu.linewidth)
    # axes[0,0].plot(wOuts[:,1,:5],'tab:red', linewidth= pltu.linewidth)
    # pltu.beautify_plot(axes[0,0],x0min=False,y0min=False)
    # pltu.axes_labels(axes[0,0],'Trials','wAto0(r) wAto1(b)')
    # axes[0,1].plot(wOuts[:,0,pfcmd.Nsub:pfcmd.Nsub+5],'tab:red', linewidth= pltu.linewidth)
    # axes[0,1].plot(wOuts[:,1,pfcmd.Nsub:pfcmd.Nsub+5],'tab:red', linewidth= pltu.linewidth)
    # pltu.beautify_plot(axes[0,1],x0min=False,y0min=False)
    # pltu.axes_labels(axes[0,1],'Trials','wBto0(r) wBto1(b)')
    # axes[0,2].plot(wOuts[:,0,pfcmd.Nsub*2:pfcmd.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    # axes[0,2].plot(wOuts[:,1,pfcmd.Nsub*2:pfcmd.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    # pltu.beautify_plot(axes[0,2],x0min=False,y0min=False)
    # pltu.axes_labels(axes[0,2],'Trials','wCto0(r) wCto1(b)')
    # # pfcmd.fig3.tight_layout()

    # if pfcmd.MDlearn:
    #     # plot PFC2MD weights evolution
    #     # pfcmd.fig3 = plt.figure(
    #                     # figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
    #                     # facecolor='w')
    #     axes[1,0].plot(wPFC2MDs[:,0,:5],'tab:red', linewidth= pltu.linewidth)
    #     axes[1,0].plot(wPFC2MDs[:,0,pfcmd.Nsub*2:pfcmd.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    #     pltu.beautify_plot(axes[1,0],x0min=False,y0min=False)
    #     pltu.axes_labels(axes[1,0],'','A -> MD0(r) C (b)')
    #     axes[1,1].plot(wPFC2MDs[:,1,:5],'tab:red', linewidth= pltu.linewidth)
    #     axes[1,1].plot(wPFC2MDs[:,1,pfcmd.Nsub*2:pfcmd.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    #     pltu.beautify_plot(axes[1,1],x0min=False,y0min=False)
    #     pltu.axes_labels(axes[1,1],'','wA->MD1(r) C->MD1(b)')
    if pfcmd.reinforceReservoir:
        axes[1,2].plot(wJrecs[:,1,:5],'tab:red', linewidth= pltu.linewidth)
        axes[1,2].plot(wJrecs[:,-1,-5:],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[1,2],x0min=False,y0min=False)
        pltu.axes_labels(axes[1,2],'Trials','wRec1(r) wRec40(b)')

        # plot MD2PFC weights evolution
        # pfcmd.fig3 = plt.figure(
                        # figsize=(pltu.columnwidth,pltu.columnwidth), 
                        # facecolor='w')
        axes[2,0].plot(wMD2PFCs[:,:5,0],'r')
        axes[2,0].plot(wMD2PFCs[:,pfcmd.Nsub*2:pfcmd.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[2,0],x0min=False,y0min=False)
        pltu.axes_labels(axes[2,0],'Trials','MD 0->A (r) 0->C (b)')
        axes[2,1].plot(wMD2PFCMults[:,:5,0],'tab:red', linewidth= pltu.linewidth)
        axes[2,1].plot(wMD2PFCMults[:,pfcmd.Nsub*2:pfcmd.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[2,1],x0min=False,y0min=False)
        pltu.axes_labels(axes[2,1],'Trials','Mw MD0toA(r) 0->C (b)')
        # pfcmd.fig3.tight_layout()
        axes[3,0].plot(wMD2PFCs[:,:5,0],'tab:red', linewidth= pltu.linewidth)
        axes[3,0].plot(wMD2PFCs[:,pfcmd.Nsub*2:pfcmd.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[3,0],x0min=False,y0min=False)
        pltu.axes_labels(axes[3,0],'Trials','MD 1->A (r) 1->C (b)')
        axes[3,1].plot(wMD2PFCMults[:,:5,0],'tab:red', linewidth= pltu.linewidth)
        axes[3,1].plot(wMD2PFCMults[:,pfcmd.Nsub*2:pfcmd.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[3,1],x0min=False,y0min=False)
        pltu.axes_labels(axes[3,1],'Trials','Mw MD1toA(r) 1->C (b)')

    pfcmd.fig3.tight_layout()

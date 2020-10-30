# classes for training, testing, experimenting or running any simulations ...

class Train_model():
    """
    Class to hold model training procedure parameters and algorithm..
    """
    def __init__(self, parameter_list):
        """
        Initialize class with default parametes
        """

        raise NotImplementedError

    def train(self, model, Ntrain):
        """
        trains model model for Ntrain iterations
        """
        for traini in range(Ntrain):
            pass

        return model #return trained model


class Test_model():
    """
    Test the performance of a model, also plots exemplar neuronal responses for a given list of input combinations
    """

    def __init__(self, parameter_list):
        """
        Initialize class with default parametes
        """
        raise NotImplementedError

    def test(self, model, Ntest):
        """
        tests model model for Ntest iterations
        """
        for testi in range(Ntest):
            pass

        
class Experiment():
    def __init__(self, model, parameter_list):
        trainer = Train_model(parameter_list)
        tester  = Test_model(parameter_list)

        trainer.train(model, 1000)
        #plot training weights, average firing rates.
        tester.test(model, 1000)
        #plot results, performance, exemplar neuron responses for each area.


# code under investigation for usefulness:


    def plot_column(self,fig,cues,routs,MDouts,outs,ploti=0):
        print('Plotting ...')
        cols=4
        if ploti==0:
            yticks = (0,1)
            ylabels=('Cues','PFC for cueA','PFC for cueB',
                        'PFC for cueC','PFC for cueD','PFC for rest',
                        'MD 1,2','Output 1,2')
        else:
            yticks = ()
            ylabels=('','','','','','','','')
        ax = fig.add_subplot(8,cols,1+ploti)
        ax.plot(cues,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[0])
        ax = fig.add_subplot(8,cols,cols+1+ploti)
        ax.plot(routs[:,:10],linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[1])
        ax = fig.add_subplot(8,cols,cols*2+1+ploti)
        ax.plot(routs[:,self.Nsub:self.Nsub+10],
                    linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[2])
        if self.Ncues > 2:
            ax = fig.add_subplot(8,cols,cols*3+1+ploti)
            ax.plot(routs[:,self.Nsub*2:self.Nsub*2+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[3])
            ax = fig.add_subplot(8,cols,cols*4+1+ploti)
            ax.plot(routs[:,self.Nsub*3:self.Nsub*3+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[4])
            ax = fig.add_subplot(8,cols,cols*5+1+ploti)
            ax.plot(routs[:,self.Nsub*4:self.Nsub*4+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[5])
        ax = fig.add_subplot(8,cols,cols*6+1+ploti)
        ax.plot(MDouts,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[6])
        ax = fig.add_subplot(8,cols,cols*7+1+ploti)
        ax.plot(outs,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=[0,self.tsteps],yticks=yticks)
        pltu.axes_labels(ax,'time (ms)',ylabels[7])
        fig.tight_layout()
        
        if self.saveData:
            d = {}
            # 1st column of all matrices is number of time steps
            # 2nd column is number of neurons / units
            d['cues'] = cues                # tsteps x 4
            d['routs'] = routs              # tsteps x 1000
            d['MDouts'] = MDouts            # tsteps x 2
            d['outs'] = outs                # tsteps x 2
            savemat('simData'+str(ploti), d)
        
        return ax

    def performance(self,cuei,outs,errors,target):
        meanErr = np.mean(errors[-100:,:]*errors[-100:,:])
        # endout is the mean of all end 100 time points for each output
        endout = np.mean(outs[-100:,:],axis=0)
        targeti = 0 if target[0]>target[1] else 1
        non_targeti = 1 if target[0]>target[1] else 0
        ## endout for targeti output must be greater than for the other
        ##  with a margin of 50% of desired difference of 1. between the two
        #if endout[targeti] > (endout[non_targeti]+0.5): correct = 1
        #else: correct = 0
        # just store the margin of error instead of thresholding it
        correct = endout[targeti] - endout[non_targeti]
        return meanErr, correct

    def do_test(self,Ntest,MDeffect,MDCueOff,MDDelayOff,
                    cueList,cuePlot,colNum,train=True):
        NcuesTest = len(cueList)
        MSEs = np.zeros(Ntest*NcuesTest)
        corrects = np.zeros(Ntest*NcuesTest)
        wOuts = np.zeros((Ntest,self.Nout,self.Nneur))
        self.meanAct = np.zeros(shape=(self.Ncontexts*self.inpsPerContext,\
                                        self.tsteps,self.Nneur))
        for testi in range(Ntest):
            if self.plotFigs: print(('Simulating test cycle',testi))
            cues_order = self.get_cues_order(cueList)
            for cuenum,(contexti,cuei) in enumerate(cues_order):
                cue, target = self.get_cue_target(contexti,cuei)
                cues, routs, outs, MDouts, MDinps, errors = \
                    self.sim_cue(contexti,cuei,cue,target,
                            MDeffect,MDCueOff,MDDelayOff,train=train)
                MSEs[testi*NcuesTest+cuenum], corrects[testi*NcuesTest+cuenum] = \
                    self.performance(cuei,outs,errors,target)

                if cuePlot is not None:
                    if self.plotFigs and testi == 0 and contexti==cuePlot[0] and cuei==cuePlot[1]:
                        ax = self.plot_column(self.fig,cues,routs,MDouts,outs,ploti=colNum)

            if self.outExternal:
                wOuts[testi,:,:] = self.wOut

        self.meanAct /= Ntest
        if self.plotFigs and cuePlot is not None:
            ax.text(0.1,0.4,'{:1.2f}$\pm${:1.2f}'.format(np.mean(corrects),np.std(corrects)),
                        transform=ax.transAxes)
            ax.text(0.1,0.25,'{:1.2f}$\pm${:1.2f}'.format(np.mean(MSEs),np.std(MSEs)),
                        transform=ax.transAxes)

        if self.saveData:
            # 1-Dim: numCycles * 4 cues/cycle i.e. 70*4=280
            self.fileDict['corrects'+str(colNum)] = corrects
            # at each cycle, a weights matrix 2x1000:
            # weights to 2 output neurons from 1000 cue-selective neurons
            # 3-Dim: 70 (numCycles) x 2 x 1000
            self.fileDict['wOuts'+str(colNum)] = wOuts
            #savemat('simDataTrials'+str(colNum), d)
        
        return MSEs,corrects,wOuts



    def taskSwitch2(self,Nblock):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
        task1Cues = self.get_cue_list(0)
        task2Cues = self.get_cue_list(1)
        self.do_test(Nblock,self.MDeffect,True,False,
                    task1Cues,task1Cues[0],0,train=True)
        self.do_test(Nblock,self.MDeffect,False,False,
                    task2Cues,task2Cues[0],1,train=True)
        
        if self.plotFigs:
            self.fig.tight_layout()
            dirname="results/results_"+str(list(self.args.values())[0])+"_"+str(list(self.args.values())[1])+"/"
            if not os.path.exists(dirname):    os.makedirs(dirname)
            filename4=os.path.join(dirname,'fig_plasticPFC2Out_{}.png')
            self.fig.savefig(filename4.format(time.strftime("%Y%m%d-%H%M%S")),
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

    def taskSwitch3(self,Nblock,MDoff=True):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
        task1Cues = self.get_cue_list(0)
        task2Cues = self.get_cue_list(1)
        # after learning, during testing the learning rate is low, just performance tuning
        self.learning_rate /= 100.
        MSEs1,_,wOuts1 = self.do_test(Nblock,self.MDeffect,False,False,\
                            task1Cues,task1Cues[0],0,train=True)
        if MDoff:
            self.learning_rate *= 100.
            MSEs2,_,wOuts2 = self.do_test(Nblock,self.MDeffect,MDoff,False,\
                                task2Cues,task2Cues[0],1,train=True)
            self.learning_rate /= 100.
        else:
            MSEs2,_,wOuts2 = self.do_test(Nblock,self.MDeffect,MDoff,False,\
                                task2Cues,task2Cues[0],1,train=True)
        MSEs3,_,wOuts3 = self.do_test(Nblock,self.MDeffect,False,False,\
                            task1Cues,task1Cues[0],2,train=True)
        self.learning_rate *= 100.
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('results/fig_plasticPFC2Out_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

            # plot the evolution of mean squared errors over each block
            fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
            ax2 = fig2.add_subplot(111)
            ax2.plot(MSEs1,'-,r')
            #ax2.plot(MSEs2,'-,b')
            ax2.plot(MSEs3,'-,g')

            # plot the evolution of different sets of weights
            fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
            ax2 = fig2.add_subplot(231)
            ax2.plot(np.reshape(wOuts1[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(232)
            ax2.plot(np.reshape(wOuts2[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(233)
            ax2.plot(np.reshape(wOuts3[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(234)
            ax2.plot(np.reshape(wOuts1[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(235)
            ax2.plot(np.reshape(wOuts2[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(236)
            ax2.plot(np.reshape(wOuts3[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))

    def test(self,Ntest):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
            # self.fig2 = plt.figure(figsize=(pltu.columnwidth,pltu.columnwidth),
            #                     facecolor='w')
        cues = self.get_cue_list()
        
        # after learning, during testing the learning rate is low, just performance tuning
        self.learning_rate /= 100.
        
        self.do_test(Ntest,self.MDeffect,False,False,cues,(0,0),0)
        if self.plotFigs:
            axs = self.fig.get_axes() #self.fig2.add_subplot(111)
            ax = axs[0]
            # plot mean activity of each neuron for this contexti+cuei
            #  further binning 10 neurons into 1
            ax.plot(np.mean(np.reshape(\
                                np.mean(self.meanAct[0,:,:],axis=0),\
                            (self.Nneur//10,10)),axis=1),',-r')
        if self.saveData:
            self.fileDict['meanAct0'] = self.meanAct[0,:,:]
        self.do_test(Ntest,self.MDeffect,False,False,cues,(0,1),1)
        if self.plotFigs:
            # plot mean activity of each neuron for this contexti+cuei
            ax.plot(np.mean(np.reshape(\
                                np.mean(self.meanAct[1,:,:],axis=0),\
                            (self.Nneur//10,10)),axis=1),',-b')
            ax.set_xlabel('neuron #')
            ax.set_ylabel('mean rate')
        if self.saveData:
            self.fileDict['meanAct1'] = self.meanAct[1,:,:]

        if self.xorTask:
            self.do_test(Ntest,self.MDeffect,True,False,cues,(0,2),2)
            self.do_test(Ntest,self.MDeffect,True,False,cues,(0,3),3)
        else:
            self.do_test(Ntest,self.MDeffect,True,False,cues,(1,0),2)
            self.do_test(Ntest,self.MDeffect,True,False,cues,(1,1),3)
            #self.learning_rate *= 100
            ## MDeffect and MDCueOff
            #self.do_test(Ntest,self.MDeffect,True,False,cues,self.cuePlot,2)
            ## MDeffect and MDDelayOff
            ## network doesn't (shouldn't) learn this by construction.
            #self.do_test(Ntest,self.MDeffect,False,True,cues,self.cuePlot,3)
            ## back to old learning rate
            #self.learning_rate *= 100.
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('results/fig_plasticPFC2Out_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            # self.fig2.tight_layout()

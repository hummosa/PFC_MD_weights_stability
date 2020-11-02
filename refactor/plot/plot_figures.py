
class monitor():
    # logs values for a number of model parameters, with labels, and plots them
    def __init__(self, labels):
        # Get the labels of vars to follow
        self.labels = labels
        self.vars = [[] for n in range(len(labels))]
        self.Nvars = len(labels)

    def log(self, vars):
        [self.vars[n].append(vars[n]) for n in range(len(vars))]
    def plot(self, fig, pfcmd):
        xticks = [0, 1000, 2000]
        axes = fig.subplots(4,3)#, shaqrex=True) #, sharey=True)
        fig.set_size_inches([9,7])
        p = pfcmd.Nsub
        for i, label in enumerate(self.labels):
            ax = axes.flatten()[i]
            ax.plot(self.vars[i],'tab:red', linewidth= pltu.linewidth)
            ax.set_title(label)
            pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
                        

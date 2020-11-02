import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

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

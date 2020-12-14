import numpy as np

class data_generator():
    def __init__(self, local_Ntrain):
        # self.non_matches = { #cause a non-match (1.) every so many matches.
        # '90': np.array([0. if (i+1)%10!=0 else 1. for i in range(local_Ntrain) ]),
        # '75': np.array([0. if (i+1)%4!=0  else 1. for i in range(local_Ntrain)  ]),
        # '50': np.array([0. if (i+1)%2!=0  else 1. for i in range(local_Ntrain)  ]),
        # '25': np.array([1. if (i+1)%4!=0  else 0. for i in range(local_Ntrain)  ]),
        # '10': np.array([1. if (i+1)%10!=0 else 0. for i in range(local_Ntrain) ]),
        #  }

        self.non_matches = { # randomly sample non-matches (1.) with set probabilities
        '90': np.array([0. if np.random.rand()<0.9 else 1. for i in range(local_Ntrain) ]),
        '75': np.array([0. if np.random.rand()<0.7 else 1. for i in range(local_Ntrain)  ]),
        '50': np.array([0. if np.random.rand()<0.5 else 1. for i in range(local_Ntrain)  ]),
        '25': np.array([1. if np.random.rand()<0.7 else 0. for i in range(local_Ntrain)  ]),
        '10': np.array([1. if np.random.rand()<0.9 else 0. for i in range(local_Ntrain) ]),
         }

        self.task_data_gen = {
        0: self.trial_generator(self.non_matches['90']),
        1: self.trial_generator(self.non_matches['10']),
        2: self.trial_generator(self.non_matches['50']),
        3: self.trial_generator(self.non_matches['25']),
        4: self.trial_generator(self.non_matches['75']),
        }

    def trial_generator(self, non_matches):
        for non_match in non_matches:
            yield (non_match)
            
import matplotlib.pyplot as plt
# plt.get_backend()
# 'Qt5Agg'
# fig, ax = plt.subplots()
# mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
# mngr.window.setGeometry(50,100,640, 545)
    
# # note that instead of mngr = get_current_fig_manager(), we can also use fig.canvas.manager 

# geom = mngr.window.geometry()
# x,y,dx,dy = geom.getRect()

def move_figure(figh, col=1, position="top"):
    '''
    Move and resize a window to a set of standard positions on the screen.
    Possible positions are:
    top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
    '''

    mgr = figh.canvas.manager
    
    fig_h = mgr.canvas.height()
    fig_w = mgr.canvas.width()
    mgr.full_screen_toggle()  # primitive but works to get screen size
    py = mgr.canvas.height()
    px = mgr.canvas.width()

    d = 10  # width of the window border in pixels
    num_of_cols            = 6
    w_col = (px//num_of_cols) + d*2

    top = (d*4)
    bottom = py-fig_h-(d*4) 
    vertical_pos = top if position is 'top' else bottom
    mgr.window.setGeometry(d+(w_col*col), vertical_pos , fig_w, fig_h)

    # if position == "col1":
    #     # x-top-left-corner, y-top-left-corner, x-width, y-width (in pixels)
    #     mgr.window.setGeometry(d, 4*d, px - 2*d, py/2 - 4*d)
    # elif position == "bottom":
    #     mgr.window.setGeometry(d, py/2 + 5*d, px - 2*d, py/2 - 4*d)
    # elif position == "left":
    #     mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py - 4*d)
    # elif position == "right":
    #     mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py - 4*d)
    # elif position == "top-left":
    #     mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "top-right":
    #     mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "bottom-left":
    #     mgr.window.setGeometry(d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "bottom-right":
    #     mgr.window.setGeometry(px/2 + d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)

import numpy as np

from network import Network
import models


md = models.MD({})
pfc = models.PFC({})

inputs = np.array([1])
W_in = np.array([1])
W_pfc_md = np.array([1])
W_md_pfc = np.array([1])

network = Network()
network.define_inputs(len(inputs), W_in, pfc)
network.connect(pfc, md, W_pfc_md, lambda x: x)
network.connect(md, pfc, W_md_pfc, lambda x: x)

for t in range(0, 3):
    inputs = [t]
    network.step(inputs)
# network.trial_end()

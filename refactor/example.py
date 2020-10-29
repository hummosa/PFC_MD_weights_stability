from network import Network
import models


md = models.MD({})
pfc = models.PFC({})

inputs = [1]
W_in = [1]
W_pfc_md = [1]

network = Network()
network.define_inputs(len(inputs), W_in, pfc)
network.connect(pfc, md, W_pfc_md, lambda x: x)

for t in range(0, 10):
    inputs = [t]
    network.step(inputs)
network.trial_end()

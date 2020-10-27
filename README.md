# Role of MD in the PFC-MD-OFC circuit

This work extends the [repository](https://github.com/adityagilra/PFC_MD_weights_stability) by Aditya Gilra to further examine the role of MD in shaping the behavior of PFC circuits.

reservoir_PFC.py trains the model over 4 blocks and saves the weights.

test_reservoir_PFC.py is a copy of the same file that loads the weights and disables learning at the MD-PFC and tests the circuit for 4 blocks.


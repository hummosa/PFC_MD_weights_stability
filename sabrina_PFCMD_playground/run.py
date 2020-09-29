import numpy as np
import matplotlib.pyplot as plt

from PFC import PFC


input_A = [1, 0]
input_B = [0, 1]
AB_BA_response = 1.0
AA_BB_response = -1.0

trial_types = [(input_A, input_A, AA_BB_response), (input_A, input_B, AB_BA_response),
               (input_B, input_A, AB_BA_response), (input_B, input_B, AA_BB_response)]

# Run the example XOR task from Miconi paper.
errors = [[], [], [], []]

# -- Create the model.
pfc = PFC()

# -- Run trials.
n_trials = 20
for i in range(n_trials):
    print(f'Running trial {i+1}/{n_trials}')
    rand_i = np.random.randint(len(trial_types))
    # rand_i = 0

    trial = trial_types[rand_i]
    error = pfc.run_trial(trial[0], trial[1], trial[2])
    errors[rand_i].append(error)
    print(f'i: {rand_i}, error: {error}')

plt.plot(errors[0], color='g', marker='o', label='AA')
plt.plot(errors[1], color='b', marker='o', label='AB')
plt.plot(errors[2], color='r', marker='o', label='BA')
plt.plot(errors[3], color='m', marker='o', label='BB')
plt.legend()
# plt.show()

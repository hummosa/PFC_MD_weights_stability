This branch deals with experimentation around adding a delay between stimulus presentation and response. 
Trial is 200ms, stimulus is on for 100ms.
Depending on value set for self.response_delay: 0 uses average of output neurons activity through out the entire trial. vs specifying a number of ms to specify a duration of time, at the end of the trial, to use as a response. For ex. 50ms, would take the last 50ms of the trial and average output neuron activity over that.

The model struggles, and takes longer but eventually solves the problem. which is a surprise considering that PFC neurons should move chaotically and lose that information especially with presence of noise. 

Plan:
Look at PFC rates during trial, and also compare across trials, to understand how the model is solving this task.
Adjust MD learning dynamics, maybe increase MDrange to increase the range

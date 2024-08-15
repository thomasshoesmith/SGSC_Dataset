########################################################
# * converts a current injection dataset to spike times
# * script useful for custom dataset configs
# * currently requires 80GB system memory for DB loading
########################################################

import numpy as np
from tqdm import trange

from dataset_loader import load_gsc

# load current injection gsc dataset
train_x, train_y, validation_x, validation_y, test_x, test_y = load_gsc("/its/home/ts468/data/rawSC/rawSC_80input_updated/",
                                                                        num_frames = 20,
                                                                        shuffle=False)

def convert_currents_to_spike_times(this_x):
    tau_mem = 20.0
    dt_ms = 1.0
    vth = 1.0
    alpha = np.exp(-dt_ms / tau_mem)

    number_of_trials = this_x.shape[0]      
    number_of_frames = this_x.shape[1]
    number_of_channels = this_x.shape[2] 

    spike_times = []

    for trial in trange(number_of_trials):

        v = np.zeros((number_of_channels))
        spike_times_per_trial = []

        for timestep in range(number_of_frames):
            v[:] = (v * alpha) + this_x[trial][timestep]
            spikes_out = v > vth
            v[spikes_out] = 0

            spike_indexes = np.where(spikes_out == True)[0]

            for spike_index in spike_indexes:
                spike_times_per_trial.append((spike_index, timestep))

        spike_times.append(np.array(spike_times_per_trial, 
                                    dtype = ([('neuron', np.uint16), 
                                            ('time', np.uint16)])))
        
    return np.array(spike_times, dtype = object)

print("converting training dataset")
np.save("data/training_x_spikes.npy", convert_currents_to_spike_times(train_x))
np.save("data/training_y_spikes.npy", np.array(train_y, dtype = np.uint8))

print("converting testing dataset")
np.save("data/testing_x_spikes.npy", convert_currents_to_spike_times(test_x))
np.save("data/testing_y_spikes.npy", np.array(test_y, dtype = np.uint8))

print("converting validation dataset")
np.save("data/validation_x_spikes.npy", convert_currents_to_spike_times(validation_x))
np.save("data/validation_y_spikes.npy", np.array(validation_y, dtype = np.uint8))
import math
import numpy as np


def get_relative_entropy(firing_rate0, firing_rate1):
    param_m = 24
    param_lambda = 250
    bias = 1/(2*param_m*param_lambda)
    relative_entropy = []
    assert len(firing_rate0) == len(firing_rate1)
    count_valid_firing_rate_data = 0
    for i in range(len(firing_rate0)):
        if np.isnan(firing_rate0[i]) or np.isnan(firing_rate1[i]):
            pass
        else:
            if firing_rate0[i] > bias:
                relative_entropy_neuron_i = firing_rate0[i]*math.log((firing_rate0[i] - bias)/(firing_rate1[i] + bias)) \
                                            - firing_rate0[i] + firing_rate1[i]
                relative_entropy.append(relative_entropy_neuron_i)
            else:
                relative_entropy.append(firing_rate1[i])
            count_valid_firing_rate_data += 1

    return np.sum(relative_entropy)/count_valid_firing_rate_data


def get_l1_distance(firing_rate0, firing_rate1):
    l1_distance = []
    assert len(firing_rate0) == len(firing_rate1)
    count_valid_firing_rate_data = 0
    for i in range(len(firing_rate0)):
        if np.isnan(firing_rate0[i]) or np.isnan(firing_rate1[i]):
            pass
        else:
            l1_distance_neuron_i = abs(firing_rate0[i] - firing_rate1[i])
            l1_distance.append(l1_distance_neuron_i)
        count_valid_firing_rate_data += 1

    return np.sum(l1_distance)/len(firing_rate1)


def get_average_search_time(reaction_times, baseline):
    return np.nanmean(reaction_times) - baseline




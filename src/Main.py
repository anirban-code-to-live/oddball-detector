import numpy as np
from scipy.stats.mstats import gmean
import pandas as pd
from src import LineFit as lf
from src import GammaFit as gf


if __name__ == '__main__':
    print('Welcome to the world of visual neuroscience!')
    firing_rate_data = pd.read_csv('../data/02_data_visual_neuroscience_firingrates.csv')
    # print(firing_rate_data.shape)
    search_time_data = pd.read_csv('../data/02_data_visual_neuroscience_searchtimes.csv')
    # print(search_time_data.shape)
    # print(firing_rate_data.values[0:2, :])
    # print(search_time_data.values[0:2, :])

    # First part - Fit straight line
    line_fit = lf.LineFit(search_time_data, firing_rate_data)
    average_search_times = line_fit.find_average_search_time()
    relative_entropy_data, l1_distance_data = line_fit.calculate_entropy_and_l1_distance()
    line_fit.plot_search_vs_l1_distance()
    line_fit.plot_search_vs_entropy()
    ratio_am_gm_search_entropy, ratio_am_gm_search_l1_distance = line_fit.measure_am_gm_spread()

    # Second part - Fit Gamma Distribution
    gamma_fitter = gf.GammaDistributionFitter(search_time_data)
    gamma_fitter.randomly_select_groups()
    gamma_fitter.measure_mean_stddev_random_groups()
    gamma_fitter.plot_mean_vs_stddev()
    gamma_fitter.find_shape_parameter()
    gamma_fitter.find_rate_parameter_and_kolmogorov_statistic()




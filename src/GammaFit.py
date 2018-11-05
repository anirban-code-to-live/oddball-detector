import random
import numpy as np
import math
from matplotlib import pyplot as plt
from random import shuffle
import itertools
from scipy import stats


class GammaDistributionFitter:

    def __init__(self, search_time_data):
        self._search_time_data = search_time_data
        self._groups_count = search_time_data.shape[1]
        self._random_groups = None
        self._mean_list = []
        self._stddev_list = []
        self._shape = None
        self._rate = None

    def randomly_select_groups(self):
        random_number_list = []
        while len(random_number_list) < self._groups_count//2:
            random_number = random.randint(0, self._groups_count-1)
            if random_number not in random_number_list:
                random_number_list.append(random_number)
        print('Random groups :: ', random_number_list)
        self._random_groups = random_number_list

    def measure_mean_stddev_random_groups(self):
        for index in self._random_groups:
            search_time_col_i = np.array(self._search_time_data.values[:, index][2:]).astype(np.float)
            mean = np.nanmean(search_time_col_i)
            stddev = math.sqrt(np.nanvar(search_time_col_i))
            self._mean_list.append(mean)
            self._stddev_list.append(stddev)
        print('Means :: ', self._mean_list)
        print('Standard Deviations :: ', self._stddev_list)

    def plot_mean_vs_stddev(self):
        ax = plt.subplot(111)
        plt.xlabel('Mean')
        plt.ylabel('Standard Deviation')
        ax.scatter(self._mean_list, self._stddev_list, color='r')
        plt.savefig('../output_plots/gamma_stddev_mean.png')
        plt.close()

    def find_shape_parameter(self):
        params_line = np.polyfit(self._mean_list, self._stddev_list, deg=1, full=True)
        self._shape = 1/math.pow(params_line[0][0], 2)
        # print('Slope :: ', params_line[0][0])
        # print('Intercept :: ', params_line[0][1])
        print('Shape parameter :: ', self._shape)

    def find_rate_parameter_and_kolmogorov_statistic(self):
        left_out_groups = []
        mean_list = []
        variance_list = []
        cdf_points = []
        for index in range(self._groups_count):
            if index not in self._random_groups:
                left_out_groups.append(index)
        # print('Left out groups to be used in rate parameters :: ', left_out_groups)
        for index in left_out_groups:
            search_time_col_i = np.array(self._search_time_data.values[:, index][2:]).astype(np.float)
            clean_search_time_col_i = [time for time in search_time_col_i if str(time) != 'nan']
            shuffle(clean_search_time_col_i)
            randomized_search_times = clean_search_time_col_i[0:len(clean_search_time_col_i)//2]
            cdf_points.append(clean_search_time_col_i[len(clean_search_time_col_i)//2:len(clean_search_time_col_i)])
            mean = np.nanmean(randomized_search_times)
            variance = np.nanvar(randomized_search_times)
            mean_list.append(mean)
            variance_list.append(variance)

        # print(len(mean_list))
        # print(len(variance_list))

        params_line = np.polyfit(mean_list, variance_list, deg=1, full=True)
        self._rate = 1/params_line[0][0]
        print('Rate parameter :: ', self._rate)

        cdf_points = list(itertools.chain.from_iterable(cdf_points))
        sorted_cdf_points = np.sort(cdf_points)
        # Plot empirical gamma distribution
        y_cdf_empirical = np.arange(len(sorted_cdf_points)) / float(len(sorted_cdf_points) - 1)
        plt.plot(sorted_cdf_points, y_cdf_empirical)

        # # Plot gamma distribution
        x_gamma = np.linspace(0, sorted_cdf_points[-1], 200)
        y_gamma = stats.gamma.cdf(x_gamma, a=self._shape, scale=1/self._rate)
        plt.plot(x_gamma, y_gamma, color='r')
        plt.savefig('../output_plots/gamma_distribution.png')
        plt.close()

        y_pdf = stats.gamma.rvs(size=len(cdf_points), a=self._shape, scale=1 / self._rate)
        ks_test = stats.ks_2samp(sorted_cdf_points, y_pdf)
        print('Kolmogorov statistic :: ', ks_test)

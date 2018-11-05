import numpy as np
from src import Utility
from scipy.stats.mstats import gmean
from matplotlib import pyplot as plt


class LineFit:

    def __init__(self, search_time_data, firing_rate_data):
        self._search_time_data = search_time_data
        self._firing_rate_data = firing_rate_data
        self._average_search_times = []
        self._relative_entropy_data = []
        self._l1_distance_data = []
        self._inverse_search_times = []
        self._ratio_am_gm_search_entropy = None
        self._ratio_am_gm_search_l1_distance = None

    def find_average_search_time(self):
        len_search_data = self._search_time_data.shape[1]
        print('Size of search data list :: ' + str(len_search_data))
        for i in range(len_search_data):
            search_time_col_i = np.array(self._search_time_data.values[:, i][2:]).astype(np.float)
            # print(search_time_col_i[2:])
            average_search_time = Utility.get_average_search_time(search_time_col_i, 328)
            self._average_search_times.append(average_search_time)
            self._inverse_search_times = [1000 / search_time for search_time in self._average_search_times]
        return self._average_search_times

    def calculate_entropy_and_l1_distance(self):
        set_count = 4
        column_count_per_set = 6
        for i in range(set_count):
            if i != 3:
                for j in range(column_count_per_set // 2):
                    column_index = i * column_count_per_set + 2 * j
                    # print(column_index)
                    firing_rate_0 = np.array(self._firing_rate_data.values[:, column_index][2:]).astype(np.float)
                    firing_rate_1 = np.array(self._firing_rate_data.values[:, column_index + 1][2:]).astype(np.float)
                    relative_entropy_ij = Utility.get_relative_entropy(firing_rate_0, firing_rate_1)
                    self._relative_entropy_data.append(relative_entropy_ij)
                    l1_distance_ij = Utility.get_l1_distance(firing_rate_0, firing_rate_1)
                    self._l1_distance_data.append(l1_distance_ij)

                    relative_entropy_ji = Utility.get_relative_entropy(firing_rate_1, firing_rate_0)
                    self._relative_entropy_data.append(relative_entropy_ji)
                    l1_distance_ji = Utility.get_l1_distance(firing_rate_1, firing_rate_0)
                    self._l1_distance_data.append(l1_distance_ji)
            else:
                for j in range(3):
                    column_index = i * column_count_per_set + 2 * j
                    firing_rate_0 = np.array(self._firing_rate_data.values[:, column_index][2:]).astype(np.float)
                    firing_rate_1 = np.array(self._firing_rate_data.values[:, column_index + 2][2:]).astype(np.float)
                    relative_entropy_ij_1 = Utility.get_relative_entropy(firing_rate_0, firing_rate_1)
                    l1_distance_ij_1 = Utility.get_l1_distance(firing_rate_0, firing_rate_1)
                    relative_entropy_ji_1 = Utility.get_relative_entropy(firing_rate_1, firing_rate_0)
                    l1_distance_ji_1 = Utility.get_l1_distance(firing_rate_1, firing_rate_0)

                    firing_rate_0 = np.array(self._firing_rate_data.values[:, column_index][2:]).astype(np.float)
                    firing_rate_1 = np.array(self._firing_rate_data.values[:, column_index + 3][2:]).astype(np.float)
                    relative_entropy_ij_2 = Utility.get_relative_entropy(firing_rate_0, firing_rate_1)
                    l1_distance_ij_2 = Utility.get_l1_distance(firing_rate_0, firing_rate_1)
                    relative_entropy_ji_2 = Utility.get_relative_entropy(firing_rate_1, firing_rate_0)
                    l1_distance_ji_2 = Utility.get_l1_distance(firing_rate_1, firing_rate_0)

                    firing_rate_0 = np.array(self._firing_rate_data.values[:, column_index + 1][2:]).astype(np.float)
                    firing_rate_1 = np.array(self._firing_rate_data.values[:, column_index + 2][2:]).astype(np.float)
                    relative_entropy_ij_3 = Utility.get_relative_entropy(firing_rate_0, firing_rate_1)
                    l1_distance_ij_3 = Utility.get_l1_distance(firing_rate_0, firing_rate_1)
                    relative_entropy_ji_3 = Utility.get_relative_entropy(firing_rate_1, firing_rate_0)
                    l1_distance_ji_3 = Utility.get_l1_distance(firing_rate_1, firing_rate_0)

                    firing_rate_0 = np.array(self._firing_rate_data.values[:, column_index + 1][2:]).astype(np.float)
                    firing_rate_1 = np.array(self._firing_rate_data.values[:, column_index + 3][2:]).astype(np.float)
                    relative_entropy_ij_4 = Utility.get_relative_entropy(firing_rate_0, firing_rate_1)
                    l1_distance_ij_4 = Utility.get_l1_distance(firing_rate_0, firing_rate_1)
                    relative_entropy_ji_4 = Utility.get_relative_entropy(firing_rate_1, firing_rate_0)
                    l1_distance_ji_4 = Utility.get_l1_distance(firing_rate_1, firing_rate_0)

                    relative_entropy_ij = np.mean([relative_entropy_ij_1, relative_entropy_ij_2, relative_entropy_ij_3,
                                                   relative_entropy_ij_4])
                    l1_distance_ij = np.mean([l1_distance_ij_1, l1_distance_ij_2, l1_distance_ij_3, l1_distance_ij_4])
                    self._relative_entropy_data.append(relative_entropy_ij)
                    self._l1_distance_data.append(l1_distance_ij)

                    relative_entropy_ji = np.mean([relative_entropy_ji_1, relative_entropy_ji_2, relative_entropy_ji_3,
                                                   relative_entropy_ji_4])
                    l1_distance_ji = np.mean([l1_distance_ji_1, l1_distance_ji_2, l1_distance_ji_3, l1_distance_ji_4])
                    self._relative_entropy_data.append(relative_entropy_ji)
                    self._l1_distance_data.append(l1_distance_ji)

        print('Size of relative entropy list :: ' + str(len(self._relative_entropy_data)))
        # print(self._relative_entropy_data)
        print('Size of L1 distance list :: ' + str(len(self._l1_distance_data)))
        # print(self._l1_distance_data)
        return self._relative_entropy_data, self._l1_distance_data

    @staticmethod
    def _fit_straight_line_through_origin(x, y):
        x = x[:,np.newaxis]
        a, residuals, _, _ = np.linalg.lstsq(x, y, rcond=None)
        # print(a)
        # print(residuals)
        return a, residuals

    def plot_search_vs_entropy(self):
        ax = plt.subplot(111)
        plt.xlabel('Relative Entropy distance')
        plt.gca().set_ylabel(r'$s^{-1}$')
        ax.scatter(self._relative_entropy_data, self._inverse_search_times, c='red')
        slope, residual_error = LineFit._fit_straight_line_through_origin(np.array(self._relative_entropy_data),
                                                             np.array(self._inverse_search_times))
        print('Slope for relative entropy vs. inverse search time curve :: ', slope[0])
        print('Residual error for the straight line fit for relative entropy :: ', residual_error[0])
        ax.plot(self._relative_entropy_data, slope*self._relative_entropy_data)
        plt.savefig('../output_plots/relative_entropy.png')
        plt.close()
        # plt.show()

    def plot_search_vs_l1_distance(self):
        ax = plt.subplot(111)
        plt.xlabel('L1 distance')
        plt.gca().set_ylabel(r'$s^{-1}$')
        ax.scatter(self._l1_distance_data, self._inverse_search_times, c='red')
        slope, residual_error = LineFit._fit_straight_line_through_origin(np.array(self._l1_distance_data),
                                                             np.array(self._inverse_search_times))
        print('Slope for l1 distance vs. inverse search time curve :: ', slope[0])
        print('Residual error for the straight line fit for l1 distance :: ', residual_error[0])
        ax.plot(self._l1_distance_data, slope * self._l1_distance_data)
        plt.savefig('../output_plots/l1_distance.png')
        plt.close()
        # plt.show()

    def measure_am_gm_spread(self):
        product_search_entropy = np.multiply(self._average_search_times, self._relative_entropy_data)
        # print(len(product_search_entropy))
        product_search_l1_distance = np.multiply(self._average_search_times, self._l1_distance_data)
        # print(len(product_search_l1_distance))

        AM_product_search_entropy = np.mean(product_search_entropy)
        GM_product_search_entropy = gmean(product_search_entropy)
        print('Arithmetic mean for search * relative entropy :: ' + str(AM_product_search_entropy))
        print('Geometric mean for search * relative entropy :: ' + str(GM_product_search_entropy))

        AM_product_search_l1_distance = np.mean(product_search_l1_distance)
        GM_product_search_l1_distance = gmean(product_search_l1_distance)
        print('Arithmetic mean for search * L1 distance :: ' + str(AM_product_search_l1_distance))
        print('Geometric mean for search * L1 distance :: ' + str(GM_product_search_l1_distance))

        self._ratio_am_gm_search_entropy = AM_product_search_entropy / GM_product_search_entropy
        print('Ratio of AM and GM for search * relative entropy :: ' + str(self._ratio_am_gm_search_entropy))

        self._ratio_am_gm_search_l1_distance = AM_product_search_l1_distance / GM_product_search_l1_distance
        print('Ratio of AM and GM for search * L1 distance :: ' + str(self._ratio_am_gm_search_l1_distance))

        return self._ratio_am_gm_search_entropy, self._ratio_am_gm_search_l1_distance

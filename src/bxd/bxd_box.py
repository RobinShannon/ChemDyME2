import numpy as np
import src.bxd.bxd_bound as bound
from copy import deepcopy
import src.utility.bxd_plotter as plt
import os

class BXDBox:

    def __init__(self, lower, upper, type, dir = None):
        self.upper = upper
        self.lower = lower
        self.type = type
        # store all s values
        self.data = []
        self.top_data = []
        self.top = 0
        self.bot_data = []
        self.bot = 0
        self.eq_population = 0
        self.eq_population_err = 0
        self.gibbs = 0
        self.gibbs_err = 0
        self.last_hit = 'lower'
        self.milestoning_count = 0
        self.upper_non_milestoning_count = 0
        self.lower_non_milestoning_count = 0
        self.decorrelation_count = 0
        self.points_in_box = 0
        self.min_segment = np.inf
        self.max_segment = 0
        self.upper_rates_file = None
        self.upper_milestoning_rates_file = None
        self.lower_rates_file = None
        self.lower_milestoning_rates_file = None
        self.data_file = None
        self.hit_file = None
        self.dir = dir

    def reset(self, type, active):
        self.type = type
        self.active = active
        self.hits = 0
        self.stuck_count = 0
        self.transparent = False
        self.step_since_hit = 0
        self.rates = []
        self.average_rate = 0
        self.rate_error = 0
        self.invisible = False
        self.s_point = 0
        self.upper.reset()
        self.lower.reset()
        self.last_hit = 'lower'
        self.milestoning_count = 0
        self.decorrelation_count = 0
        self.decorrelation_time = 0


    def get_s_extremes(self, b, eps):
        self.top_data = []
        self.bot_data = []
        data = [d[1] for d in self.data]
        hist, edges = np.histogram(data, bins=b)
        cumulative_probability = 0
        cumulative_probability2 = 0
        limit = 0
        limit2 = 0
        for h in range(0, len(hist)):
            cumulative_probability += hist[h] / len(data)
            if cumulative_probability > eps:
                limit = h
                break
        for i,h in enumerate(hist):
            cumulative_probability2 += h / len(data)
            if cumulative_probability2 > (1 - eps):
                limit2 = i
                break
        if limit == 0:
            limit = len(hist) - 2
        for d in self.data:
            if d[1] > edges[limit] and d[1] <= edges[limit + 1]:
                self.top_data.append(d[0])
        self.top = np.mean(self.top_data, axis=0)
        for d in self.data:
            if d[1] >= edges[limit2] and d[1] < edges[limit2 + 1]:
                self.bot_data.append(d[0])
        self.bot = np.mean(self.bot_data, axis=0)

    def get_modified_box_data(self):
        modified_data = []
        for d in self.data:
            normalisation_factor = (float(self.eq_population) / float(len(self.data)))
            ar = [d[1], normalisation_factor, d[3]]
            modified_data.append(ar)
        return modified_data

    def get_full_histogram(self, boxes=10, data_frequency=1):
        data1 = self.data[0::data_frequency]
        d = np.asarray([np.fromstring(d[0].replace('[', '').replace(']', ''), dtype=float, sep=' ') for d in data1])
        proj = np.asarray([float(d[2]) for d in data1])
        edge = (max(proj) - min(proj)) / boxes
        edges = np.arange(min(proj), max(proj),edge).tolist()
        energy = np.asarray([float(d[3]) for d in data1])
        sub_bound_list = self.get_sub_bounds(boxes)
        hist = [0] * boxes
        energies = []
        for j in range(0, boxes):
            temp_ene = []
            for ene,da in zip(energy,d):
                try:
                    if not (sub_bound_list[j+1].hit(da,"up")) and not (sub_bound_list[j].hit(da,"down")):
                        hist[j] += 1
                        temp_ene.append(float(ene))
                except:
                    pass
            try:
                temp_ene = np.asarray(temp_ene)
                energies.append(np.mean(temp_ene))
            except:
                energies.append(0)
        return edges, hist, energies

    def get_sub_bounds(self, boxes):
        # Get difference between upper and lower boundaries
        n_diff = np.subtract(self.upper.n,self.lower.n)
        d_diff = self.upper.d - self.lower.d

        # now divide this difference by the number of boxes
        n_increment = np.true_divide(n_diff, boxes)
        d_increment = d_diff / boxes

        # create a set of "boxes" new bounds divide the space between the upper and lower bounds,
        # these bounds all intersect at the same point in space

        bounds = []
        for i in range(0,boxes):
            new_n = self.lower.n + i * n_increment
            new_d = self.lower.d + i * d_increment
            b = bound.BXDBound(new_n,new_d)
            bounds.append(b)
        bounds.append(deepcopy(self.upper))
        return bounds

    def read_box_data(self, path):
        path += '/box_data.txt'
        file = open(path, 'r')
        for line in file.readlines():
            line = line.rstrip('\n')
            line = line.split('\t')
            if float(line[2]) >= 0:
                self.data.append(line)

    def open_box(self):
        os.makedirs(self.dir, exist_ok=True)
        self.upper_rates_file = open(self.dir + '/upper_rates.txt', 'a')
        self.upper_milestoning_rates_file = open(self.dir + '/upper_milestoning.txt', 'a')
        self.lower_rates_file = open(self.dir + '/lower_rates.txt', 'a')
        self.lower_milestoning_rates_file = open(self.dir + '/lower_milestoning.txt', 'a')
        self.data_file = open(self.dir + '/box_data.txt', 'a')
        self.milestoning_count = 0
        self.upper_non_milestoning_count = 0
        self.lower_non_milestoning_count = 0

    def close_box(self, path='None'):
        self.upper_rates_file.close()
        self.upper_milestoning_rates_file.close()
        self.lower_rates_file.close()
        self.lower_milestoning_rates_file.close()
        for d in self.data:
            for s in d:
                self.data_file.write(str(s))
                self.data_file.write('\t')
            self.data_file.write('\n')
        self.data_file.close()
        self.lower.s_point = self.data[0]
        self.upper.s_point = self.data[-1]
        if path is not None:
            fig = plt.bxd_plotter_2d(path, zoom = True, all_bounds = False)
            ar = [self.lower.get_bound_array2D(),self.upper.get_bound_array2D()]
            fig.plot_bxd_from_array(self.data, ar, save_file=True, save_root = self.dir)
            fig.animate(save_file=True, save_root = self.dir, frames = 0.1 *len(self.data))
        self.milestoning_count = 0
        self.upper_non_milestoning_count = 0
        self.lower_non_milestoning_count = 0
import numpy as np

class BXDBound:

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.hits = 0
        self.stuck_count = 0
        self.transparent = False
        self.invisible = False
        self.step_since_hit = 0
        self.rates = []
        self.average_rate = 0
        self.rate_error = 0
        self.s_point = None
        self.random_rate=0

    def reset(self):
        self.hits = 0
        self.stuck_count = 0
        self.transparent = False
        self.step_since_hit = 0
        self.rates = []
        self.average_rate = 0
        self.rate_error = 0
        self.invisible = False
        self.s_point = 0

    def get_data(self):
        return self.d, self.n, self.s_point

    def compare_distances_from_bound(self,s,old_s,direction):
        coord1 = np.vdot(s, self.n) + self.d
        coord2 = np.vdot(old_s, self.n) + self.d
        if direction == 'up' and coord1 < coord2:
            return True
        elif direction == 'down' and coord1 > coord2:
            return True
        else:
            return False

    def hit(self, s, bound):
        if self.invisible:
            return False
        coord = np.vdot(s, self.n) + self.d
        if bound == "up" and coord > 0.0001:
            if self.s_point is None:
                self.s_point = s
            return True
        elif bound == "down" and coord < -0.0001:
            if self.s_point is None:
                self.s_point = s
            return True
        else:
            return False

    def average_rates(self, milestoning, bound, path, decorrelation_limit):
        if milestoning:
            if bound == 'upper':
                path += '/upper_milestoning.txt'
            else:
                path += '/lower_milestoning.txt'
        else:
            if bound == 'upper':
                path += '/upper_rates.txt'
            else:
                path += '/lower_rates.txt'
        file = open(path, 'r')
        rates = np.loadtxt(file)
        maxi = np.max(rates)
        if maxi > 2.0 * decorrelation_limit:
            rates = rates[rates > decorrelation_limit]
        else:
            rates = rates[rates > 3]
        self.rates = 1.0 / rates
        self.average_rate = np.mean(self.rates)
        self.rate_error = np.std(self.rates) / np.sqrt(len(self.rates))

    def get_bound_array2D(self):
        return [self.d, self.n[0], self.n[1], self.s_point[0],self.s_point[1]]

    def sample_from_dist(self):
        self.random_rate = float(np.random.normal(self.average_rate,self.rate_error,1))
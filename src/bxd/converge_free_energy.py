import numpy as np
import os
import scipy.linalg as linalg

def get_free_energy(BXD, T, boxes=1, milestoning=False, directory='Converging_Data', decorrelation_limit=5,
                    data_limit=100000000):
    """
    Reads the data in the output directory to calculate the free_energy profile
    :param T: Temperature MD was run at in K
    :param boxes: Integer DEFAULT = 1
                  NB needs renaming. Controls the resolution of the free energy profile. Each bxd box with be
                  histogrammed into "boxes" subboxes
    :param milestoning: Boolean DEFAULT = False
                        If True the milestoning rates files will be used, otherwise normal rates files will be used
    :param directory: String DEFAULT = 'Converging_Data"
                      Name of output directory to be read
    :param decorrelation_limit: Integer DEFAULT = 1
                                Only rates in excess of decorrelation_limit will be read
    :return:
    """
    # Multiply T by the gas constant in kJ/mol
    T *= (8.314 / 1000)
    for i, box in enumerate(BXD.box_list):
        temp_dir = directory + ("/box_" + str(i) + '/')
        try:
            box.upper.average_rates(milestoning, 'upper', temp_dir, decorrelation_limit)
        except:
            box.lower.average_rate = 0
        try:
            box.lower.average_rates(milestoning, 'lower', temp_dir, decorrelation_limit)
        except:
            box.lower.average_rate = 0
        try:
            box.read_box_data(temp_dir, data_limit=data_limit)
        except:
            pass

    for i in range(0, len(BXD.box_list) - 1):
        if i == 0:
            BXD.box_list[i].gibbs = 0
            BXD.box_list[i].gibbs_err = 0
        try:
            k_eq = BXD.box_list[i].upper.average_rate / BXD.box_list[i + 1].lower.average_rate
            K_eq_err = k_eq * np.sqrt((BXD.box_list[i].upper.rate_error / BXD.box_list[i].upper.average_rate) ** 2 + (
                        BXD.box_list[i + 1].lower.rate_error / BXD.box_list[i + 1].lower.average_rate) ** 2)
            try:
                delta_g = -1.0 * np.log(k_eq)
            except:
                delta_g = 0
            delta_g_err = (K_eq_err) / k_eq
            BXD.box_list[i + 1].gibbs = delta_g + BXD.box_list[i].gibbs
            BXD.box_list[i + 1].gibbs_err = delta_g_err + BXD.box_list[i].gibbs_err
        except:
            BXD.box_list[i + 1].gibbs = 0
            BXD.box_list[i + 1].gibbs_err = 0
    if boxes == 1:
        profile = []
        for i in range(0, len(BXD.box_list)):
            try:
                enedata = [float(d[3]) for d in BXD.box_list[i].data]
                ave_ene = min(np.asarray(enedata))
            except:
                ave_ene = "nan"
            profile.append((str(i), BXD.box_list[i].gibbs, BXD.box_list[i].gibbs_err, ave_ene))
        return profile
    else:
        try:
            profile = []
            total_probability = 0
            for i in range(0, len(BXD.box_list)):
                BXD.box_list[i].eq_population = np.exp(-1.0 * (BXD.box_list[i].gibbs))
                BXD.box_list[i].eq_population_err = BXD.box_list[i].eq_population * (1) * BXD.box_list[i].gibbs_err
                total_probability += BXD.box_list[i].eq_population

            for i in range(0, len(BXD.box_list)):
                BXD.box_list[i].eq_population /= total_probability
                BXD.box_list[i].eq_population_err /= total_probability
            last_s = 0
            for i in range(0, len(BXD.box_list)-1):
                s, dens = BXD.box_list[i].get_full_histogram(boxes)
                for sj in s:
                    sj -= s[0]
                box_gibbs_diff = (BXD.box_list[i+1].gibbs - BXD.box_list[i].gibbs)
                gibbs_err = (BXD.box_list[i+1].gibbs_err - BXD.box_list[i].gibbs_err)
                max_d = -1.0 * np.log(float(dens[-1]) / (float(len(BXD.box_list[i].data))))
                min_d = -1.0 * np.log(float(dens[0]) / (float(len(BXD.box_list[i].data))))
                max_d_err = np.sqrt(float(dens[-1])) / (float(len(BXD.box_list[i].data))) / -1*np.log(float(dens[-1]) / (float(len(BXD.box_list[i].data))))
                correction =  box_gibbs_diff/(max_d-min_d)
                correction_err =correction * np.sqrt((gibbs_err/box_gibbs_diff)**2 + (max_d_err/max_d)**2)
                offset = -1.0 * np.log(float(dens[0]) / (float(len(BXD.box_list[i].data))))
                for j in range(1, len(dens)):
                    d_err = np.sqrt(float(dens[j])) / (float(len(BXD.box_list[i].data)))
                    d = float(dens[j]) / (float(len(BXD.box_list[i].data)))
                    p = d
                    p_err = p * np.sqrt((d_err / d) ** 2 )
                    p_log = -1.0 * np.log(p)
                    p_log_err = (p_err) / p
                    p_corr = ((p_log-offset) * correction) + BXD.box_list[i].gibbs
                    p_corr_err = p_corr * np.sqrt((p_log_err / p_log)**2 + (correction_err/correction)**2)
                    s_path = s[j] + last_s
                    profile.append((s_path, p_corr, p_corr_err))
                last_s += s[-1]
            return profile
        except:
            return profile
            print('couldnt find histogram data for high resolution profile')


def collate_free_energy_data(BXD, prefix='Converging_Data', outfile='Combined_converging'):
    """
    Collates data from a number of different output directories with the same prefix into a new directory
    :param prefix: String DEFAULT = 'Converging_Data'
                   Prefix of input directories to be read
    :param outfile: String DEFAULT = 'Combined_converging'
                    Filename for output directory
    :return:
    """
    dir_root_list = []
    number_of_boxes = []
    for subdir, dirs, files in os.walk(os.getcwd()):
        for dir in dirs:
            if prefix in dir:
                dir_root_list.append(dir)
                number_of_boxes.append(len(next(os.walk(dir))[1]))
    # Check number of boxes is consistant among directories
    if number_of_boxes.count(number_of_boxes[0]) == len(number_of_boxes):
        boxes = number_of_boxes[0]
    else:
        boxes = min(number_of_boxes)
        print('not all directories have the same number of boxes. Check these all correspond to the same system')

    os.mkdir(outfile)
    for i in range(0, boxes):
        os.mkdir(outfile + '/box_' + str(i))
        u_rates = [(root + '/box_' + str(i) + '/upper_rates.txt') for root in dir_root_list]
        with open(outfile + '/box_' + str(i) + '/upper_rates.txt', 'w') as outfile0:
            for u in u_rates:
                try:
                    with open(u) as infile:
                        for line in infile:
                            outfile0.write(line)
                except:
                    pass
        u_m = [(root + '/box_' + str(i) + '/upper_milestoning.txt') for root in dir_root_list]
        with open(outfile + '/box_' + str(i) + '/upper_milestoning.txt', 'w') as outfile1:
            for um in u_m:
                try:
                    with open(um) as infile:
                        for line in infile:
                            outfile1.write(line)
                except:
                    pass
        l_rates = [(root + '/box_' + str(i) + '/lower_rates.txt') for root in dir_root_list]
        with open(outfile + '/box_' + str(i) + '/lower_rates.txt', 'w') as outfile2:
            for l in l_rates:
                try:
                    with open(l) as infile:
                        for line in infile:
                            outfile2.write(line)
                except:
                    pass
        l_m = [(root + '/box_' + str(i) + '/lower_milestoning.txt') for root in dir_root_list]
        with open(outfile + '/box_' + str(i) + '/lower_milestoning.txt', 'w') as outfile3:
            for lm in l_m:
                try:
                    with open(lm) as infile:
                        for line in infile:
                            outfile3.write(line)
                except:
                    pass
        data = [(root + '/box_' + str(i) + '/box_data.txt') for root in dir_root_list]
        with open(outfile + '/box_' + str(i) + '/box_data.txt', 'w') as outfile4:
            for d in data:
                try:
                    with open(d) as infile:
                        for line in infile:
                            if line.rstrip():
                                outfile4.write(line)
                except:
                    pass


def get_rates(self, milestoning = False, directory = 'Converging_Data', decorrelation_limit = 1, errors = False, n_samp = 10000):

    # Get number of boxes
    boxes = len(self.box_list)
    for i,box in enumerate(self.box_list):
        temp_dir = directory + ("/box_" + str(i) + '/')
        try:
            box.upper.average_rates(milestoning, 'upper', temp_dir, decorrelation_limit)
        except:
            box.lower.average_rate = 0
        try:
            box.lower.average_rates(milestoning, 'lower', temp_dir, decorrelation_limit)
        except:
            box.lower.average_rate = 0

    if errors:
        rates = []
        rates2= []
        for b in range(0,n_samp):
            matrix = np.zeros((boxes,boxes))
            for i in range(0, boxes):
                self.box_list[i].upper.sample_from_dist()
                self.box_list[i].lower.sample_from_dist()
            for i in range(0, boxes):
                if i == 0:
                    matrix[i][i] = - self.box_list[i].upper.random_rate
                elif i < boxes - 1:
                    matrix[i][i] = - self.box_list[i].upper.random_rate - self.box_list[i].lower.random_rate
                if i > 0:
                    matrix[i][i - 1] = self.box_list[i].lower.random_rate
                    matrix[i - 1][i] = self.box_list[i - 1].upper.random_rate
                if i < boxes - 2:
                    matrix[i + 1][i] = self.box_list[i + 1].lower.random_rate
                if i < boxes - 1:
                    matrix[i][i + 1] = self.box_list[i].upper.random_rate
            eig= linalg.eigvals(matrix)
            rate = (eig[-2])
            rate2= (eig[-1])
            rates.append(rate)
            rates2.append(rate2)
        return [np.mean(rates),np.std(rates)]
    else:
        matrix = np.zeros((boxes, boxes))
        for i in range(0, boxes):
            self.box_list[i].upper.sample_from_dist()
            self.box_list[i].lower.sample_from_dist()
        for i in range(0, boxes):
            if i == 0:
                matrix[i][i] = - self.box_list[i].upper.random_rate
            elif i < boxes-1:
                matrix[i][i] = - self.box_list[i].upper.random_rate - self.box_list[i].lower.random_rate
            if i > 0:
                matrix[i][i-1] = self.box_list[i].lower.random_rate
                matrix[i-1][i] = self.box_list[i - 1].upper.random_rate
            if i < boxes - 2:
                matrix[i+1][i] = self.box_list[i + 1].lower.random_rate
            if i < boxes - 1:
                matrix[i][i+1] = self.box_list[i].upper.random_rate
        eig = linalg.eigvals(matrix)
        return [eig[-2],0]
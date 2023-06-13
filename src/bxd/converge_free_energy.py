import numpy as np
def get_free_energy(BXD, T, boxes=1, milestoning=False, directory='Converging_Data', decorrelation_limit=1,
                    data_frequency=1):
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
            box.read_box_data(temp_dir,BXD.progress_metric)
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
            for i in range(0, len(BXD.box_list)):
                s, dens, ene = BXD.box_list[i].get_full_histogram(boxes, data_frequency)
                for sj in s:
                    sj -= s[0]
                for j in range(0, len(dens)):
                    d_err = np.sqrt(float(dens[j])) / (float(len(BXD.box_list[i].data)) / data_frequency)
                    d = float(dens[j]) / (float(len(BXD.box_list[i].data)) / data_frequency)
                    p = d * BXD.box_list[i].eq_population
                    p_err = p * np.sqrt(
                        (d_err / d) ** 2 + (BXD.box_list[i].eq_population_err / BXD.box_list[i].eq_population) ** 2)
                    p_log = -1.0 * np.log(p)
                    p_log_err = (p_err) / p
                    s_path = s[j] + last_s
                    profile.append((s_path, p_log, p_log_err, ene[j]))
                last_s += s[-1]
            return profile
        except:
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

def get_overall_rates(BXD, directory, milestoning=True, decorrelation_limit=1):
    bxs = len(BXD.box_list)
    transition_array = np.zeros((bxs,bxs))
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
            box.read_box_data(temp_dir)
        except:
            pass
    for i in range(0, i < bxs - 1):
        lowerloss = BXD.box_list[i].lower.average_rate
        lowergain = BXD.box_list[i - 1].upper.average_rate if i > 0 else 0
        upperloss = BXD.box_list[i].upper.average_rate
        uppergain = BXD.box_list[i + 1].lower.average_rate if i < bxs - 1 else 0
        transition_array[i][i] =uppergain + lowergain - upperloss - lowerloss
        if i > 0:
            transition_array[i][i - 1] = lowerloss
            transition_array[i - 1][i] = lowergain
        if i < bxs - 1:
            transition_array[i][i+1] = upperloss
            transition_array[i + 1][i] = uppergain
        eigval, eigvec = np.linalg.eig(transition_array)
        f = open('output.txt', 'w')
        f.write(eigval)
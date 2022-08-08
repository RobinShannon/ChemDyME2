import copy

#from src.Calculators.ScineCalculator import SparrowCalculator as SP

class calculator_manager:
    def __init__(self, trajectory = None, low = None, high = None, single_point = None,
                 calc_hindered_rotors = False, calc_BXDE_DOS = False, multi_level = True ):
        self.multi_level = multi_level
        self.low = low
        self.low_original = copy.deepcopy(self.low.__dict__)
        self.trajectory = trajectory
        self.trajectory_original = copy.deepcopy(self.trajectory.__dict__)
        self.high = high
        self.high_original = copy.deepcopy(self.high.__dict__)
        self.single_point = single_point
        self.single_point_original = copy.deepcopy(self.single_point.__dict__)
        self.calc_hindered_rotors = calc_hindered_rotors
        self.calc_BXDE_DOS = calc_BXDE_DOS

    def set_calculator(self, mol, level = 'low'):
        if level == 'trajectory':
            self.trajectory.__dict__ = copy.deepcopy(self.trajectory_original)
            mol._calc = self.trajectory
        if level == 'low':
            self.low.__dict__ = copy.deepcopy(self.low_original)
            mol.calc = self.low
            #mol._calc.reinitialize(mol.copy())
        if level == 'high':
            self.high.__dict__ = copy.deepcopy(self.high_original)
            mol.calc = self.high
            #mol._calc.reinitialize(mol.copy())
        if level == 'single':
            self.single_point.__dict__ = copy.deepcopy(self.single_point_original)
            mol.calc = self.single_point
            #mol._calc.reinitialize(mol.copy())

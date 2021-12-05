from src.Calculators.ScineCalculator import SparrowCalculator as SP

class Calculator_manager:
    def __init__(self, trajectory = SP(), low = SP(), high = SP(), single_point = SP(),
                 calc_hindered_rotors = False, calc_BXDE_DOS = False, multi_level = True ):
        self.multi_level = multi_level
        self.low = low
        self.trajectory = trajectory
        self.high = high
        self.single_point = single_point
        self.calc_hindered_rotors = calc_hindered_rotors
        self.calc_BXDE_DOS = calc_BXDE_DOS

    def set_calculator(self, mol, level = 'low'):
        if level == 'trajectory':
            mol._calc = self.trajectory
            mol._calc.reinitialize(mol.copy())
        if level == 'low':
            mol.calc = self.low
            mol._calc.reinitialize(mol.copy())
        if level == 'high':
            mol.calc = self.high
            mol._calc.reinitialize(mol.copy())
        if level == 'single':
            mol.calc = self.single_point
            mol._calc.reinitialize(mol.copy())

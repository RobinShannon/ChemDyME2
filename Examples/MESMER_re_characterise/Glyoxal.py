import src.mechanism_generation.calculator_manager as cm
from src.mechanism_generation.mol_types import ts, well, vdw
import sys
from ase.io import read
from src.Calculators.XtbCalculator import XTB
from src.Calculators.GaussianCalculator import Gaussian

def refine_mol(dir):
    mol = read(str(dir)+'/min.log')
    low =XTB(method="GFN2xTB", electronic_temperature=1000)
    high = Gaussian(
        nprocshared=1,
        label='Gauss',
        method='M062x',
        basis='6-31+G**',
        mult=int(2),
        scf='qc'
    )
    if is_ts != True:
        if is_vdw == True:
            calculator_manager = cm.calculator_manager(trajectory = low,low=low, high=high, single_point=low, calc_hindered_rotors=False, multi_level=False)
            sp = vdw.vdw(mol, calculator_manager, dir = dir)
            sp.write_hindered_rotors(mol, partial=True)
        else:
            calculator_manager = cm.calculator_manager(trajectory = low,low=low, high=high, single_point=low, calc_hindered_rotors=False, multi_level=False)
            sp = well.well(mol, calculator_manager, dir = dir)
            sp.write_hindered_rotors(mol, partial=True)
    else:
        calculator_manager = cm.calculator_manager(trajectory = low, low=low, high=high, single_point=low, calc_hindered_rotors=False, multi_level=False)
        sp = ts.ts(mol, calculator_manager, dir = dir)
        sp.write_hindered_rotors(mol, partial=True)

#ts = bool(sys.argv[2])
#vdw = bool(sys.argv[3])
#refine_mol(sys.argv[1])

is_ts = True
is_vdw = False
refine_mol('TS3')
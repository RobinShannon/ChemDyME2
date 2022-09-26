
import src.molecular_dynamics.md_Integrator as MD
import src.mechanism_generation.reaction_crtieria as RC
import src.master_equation.MESMER as ME
import src.mechanism_generation.mechanism_main as Mech
import src.mechanism_generation.calculator_manager as CM
#from src.Calculators.QcoreCalculator import QcoreCalculator as QC
from src.Calculators.XtbCalculator import XTB
from src.Calculators.DFTBCalculator import Dftb
from ase.io import read,write
from ase.optimize import BFGS as BFGS
from ase.neb import NEB
from sella import Sella, Constraints, IRC
import numpy as np

min_scan = []
mol = read('scan.log', index=':')
del mol[-1]
for i,frame in enumerate(mol):
    previous = 100
    next = 100
    if i > 0:
        previous= np.amax(mol[i-1].get_potential_energy())
    if i < len(mol)-1:
        next = np.amax(mol[i+1].get_potential_energy())
    current = np.amax(frame.get_potential_energy())
    if  current < previous and current < next:
        min_scan.append(frame.copy())
        print(current)

write('min_scan.xyz', min_scan)

mol = read('CH3OCO(OO).xyz')

md = MD.Langevin(mol, temperature=298, timestep=0.5, friction=0.01)

reaction_criteria = RC.NunezMartinez(mol)

master_eq = ME.MasterEq()

calculator_manager = CM.Calculator_manager(trajectory = XTB(method="GFN2-xTB"), low = XTB(method="GFN2-xTB"), high = XTB(method="GFN2-xTB"), single_point = XTB(method="GFN2-xTB"), calc_hindered_rotors=True)

Mechanism = Mech.ReactionNetwork(mol,reaction_criteria,master_eq,calculator_manager,md, bimolecular_start=False)
Mechanism.run()
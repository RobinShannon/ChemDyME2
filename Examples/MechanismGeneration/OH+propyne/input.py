
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
mol = read('scanForm.log', index=':')
del mol[-1]
old_dist = 0
for i,frame in enumerate(mol):
    dist = frame.get_distance(1,4)
    if dist > old_dist + 0.05:
        min_scan.append(mol[i].copy())
        print(mol[i].get_potential_energy())
    old_dist = dist

write('min_scan.xyz', min_scan)

mol = read('OH+Benzene.xyz')

md = MD.Langevin(mol, temperature=298, timestep=0.5, friction=0.01)

reaction_criteria = RC.NunezMartinez(mol)

master_eq = ME.MasterEq()

calculator_manager = CM.calculator_manager(trajectory = XTB(method="GFN2-xTB"), low = XTB(method="GFN2-xTB"), high = XTB(method="GFN2-xTB"), single_point = XTB(method="GFN2-xTB"), calc_hindered_rotors=True)

Mechanism = Mech.ReactionNetwork(mol,reaction_criteria,master_eq,calculator_manager,md, bimolecular_start=True, start_frag_indicies=[[0,1,2,3,4,5,6,7,8,9,10,11],[12,13]])
Mechanism.run()
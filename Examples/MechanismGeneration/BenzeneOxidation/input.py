import src.molecular_dynamics.md_Integrator as MD
import src.mechanism_generation.reaction_crtieria as RC
import src.master_equation.MESMER as ME
import src.mechanism_generation.mechanism_main as Mech
import src.mechanism_generation.calculator_manager as CM
#fromsrc.Calculators.QcoreCalculator import QcoreCalculator as QC
#from src.Calculators.DFTBCalculator import Dftb
from src.Calculators.ScineCalculator import SparrowCalculator as SP
from src.Calculators.GaussianCalculator import Gaussian
from ase.io import read,write
from ase.optimize import BFGS as BFGS
from ase.neb import NEB
from sella import Sella, Constraints, IRC
import numpy as np

mol = read('OH+Benzene.xyz')

md = MD.Langevin(mol, temperature=298, timestep=0.5, friction=0.01)

reaction_criteria = RC.NunezMartinez(mol)

master_eq = ME.MasterEq(start_mol='c1ccccc1')

calculator_manager = CM.calculator_manager(trajectory = SP(), low = SP(), high = SP(), single_point =Gaussian(
        nprocshared=1,
        label='Gauss',
        method='M062x',
        basis='6-31+G**',
        mult=int(2),
        #extra='NoSymm int=coarse',
        scf='qc'
    ), calc_hindered_rotors=True)

Mechanism = Mech.ReactionNetwork(mol,reaction_criteria,master_eq,calculator_manager,md, bimolecular_start=True, start_frag_indicies=[[0,1,2,3,4,5,6,7,8,9,10,11],[12,13]], bimolecular_bath={"O=O":1E5})
Mechanism.run()
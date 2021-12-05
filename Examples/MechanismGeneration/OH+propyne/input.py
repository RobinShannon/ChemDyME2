
import src.MolecularDynamics.MDIntegrator as MD
import src.MechanismGeneration.ReactionCriteria as RC
import src.MasterEquation.MESMER as ME
import src.MechanismGeneration.MechanismMain as Mech
import src.MechanismGeneration.Calculator_manager as CM

from ase.io import read
from ase.optimize import BFGS

# Get starting coordinates
mol = read('CH3OCO(OO).xyz')

md = MD.Langevin(mol, temperature=298, timestep=0.5, friction=0.01)

reaction_criteria = RC.NunezMartinez(mol)

master_eq = ME.MasterEq()

calculator_manager = CM.Calculator_manager(calc_hindered_rotors=True)

Mechanism = Mech.ReactionNetwork(mol,reaction_criteria,master_eq,calculator_manager,md, bimolecular_start=False)
Mechanism.run()
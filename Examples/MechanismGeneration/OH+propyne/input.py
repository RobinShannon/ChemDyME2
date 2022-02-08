
import src.MolecularDynamics.MDIntegrator as MD
import src.MechanismGeneration.ReactionCriteria as RC
import src.MasterEquation.MESMER as ME
import src.MechanismGeneration.MechanismMain as Mech
import src.MechanismGeneration.Calculator_manager as CM
#from src.Calculators.QcoreCalculator import QcoreCalculator as QC
from src.Calculators.XtbCalculator import XTB
from src.Calculators.DFTBCalculator import Dftb

from ase.io import read,write
from ase.optimize import BFGS as BFGS
from ase.neb import NEB
from sella import Sella, Constraints, IRC

path = read('new_path.xyz', index=':')

path[34].set_calculator(Dftb(label='lab',
                            atoms=path[34],
                            Hamiltonian_Filling = 'Fermi{',
                            Hamiltonian_Filling_Temperature = 0.006,
                            Hamiltonian_SCC='Yes',
                            Hamiltonian_SCCTolerance = 1.00E-5,
                            Hamiltonian_MaxSCCIterations = 100,
                            Hamiltonian_MaxAngularMomentum_='',
                            Hamiltonian_MaxAngularMomentum_O='"p"',
                            Hamiltonian_MaxAngularMomentum_C='"p"',
                            Hamiltonian_MaxAngularMomentum_H='"s"',
                            ))
dyn=Sella(path[34], internal=True)
dyn.run()
TSenergy = path[34].get_potential_energy()
irc=IRC(path[34], trajectory='irc_rev.traj', dx=0.01, eta=1e-4, gamma=0.4)
irc.run(fmax=0.1, steps=25, direction='reverse')
irc=IRC(path[34], trajectory='irc_for.traj', dx=0.01, eta=1e-4, gamma=0.4)
irc.run(fmax=0.1, steps=25, direction='forward')
rev= read('irc_rev.traj', ':')
forward= read('irc_for.traj', ':')
rev.reverse()
path = rev + forward
write('irc.xyz',path)
path = read('new_path.xyz', index=':')
path[21].set_calculator(XTB(method="GFN1xTB", electronic_temperature=300))
dyn=BFGS(path[21])
dyn.run()
Renergy = path[21].get_potential_energy()

path[-1].set_calculator(XTB(method="GFN1xTB", electronic_temperature=300))
dyn=BFGS(path[-1])
dyn.run()
Penergy = path[-1].get_potential_energy()

images = [path[21].copy()]
images += [path[21].copy() for i in range(30)]
images += [path[-1].copy()]
for i in images:
    i.set_calculator(XTB(method="GFN1xTB", electronic_temperature=300))
neb = NEB(images, climb=True)
# Interpolate linearly the potisions of the three middle images:
neb.interpolate()
optimizer = BFGS(neb)
optimizer.run(fmax=0.04, steps=100)
write('NEB.xyz',images)
for i in images:
    print(str((i.get_potential_energy()-Renergy)*96))
# Get starting coordinates
print('barrier = ' + str(TSenergy-Renergy))
mol = read('CH3OCO(OO).xyz')

md = MD.Langevin(mol, temperature=298, timestep=0.5, friction=0.01)

reaction_criteria = RC.NunezMartinez(mol)

master_eq = ME.MasterEq()

calculator_manager = CM.Calculator_manager(trajectory = XTB(method="GFN2-xTB"), low = XTB(method="GFN2-xTB"), high = QC(), single_point = QC(), calc_hindered_rotors=True)

Mechanism = Mech.ReactionNetwork(mol,reaction_criteria,master_eq,calculator_manager,md, bimolecular_start=False)
Mechanism.run()
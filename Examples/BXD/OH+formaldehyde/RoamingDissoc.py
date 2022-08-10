try:
    import src.bxd.collective_variable as CV
except:
    pass
from src.Calculators.NNCalculator import NNCalculator
import src.utility.tools as Tl
from ase.io import read
from ase.io import write
from ase.md.langevin import Langevin
from ase import units
from ase.md import velocitydistribution as vd


narupa_mol = read('NN_Comp.xyz')
narupa_mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-1510000', atoms=narupa_mol))
for run in range(0, 1000):
    # The general set-up here is identical to the adaptive run to ensure the converging run samples the same path
    #narupa_mol = read('water', index=0)
    temp = read('NN_Comp.xyz')
    narupa_mol.set_positions(temp.get_positions())
    vd.MaxwellBoltzmannDistribution(narupa_mol, temperature_K=500, force_temp=True)
    dyn = Langevin(narupa_mol, 1 * units.fs, 800*units.kB, 0.01)
    traj = []
    for i in range(0,1000):
        dyn.run(20)
        name = Tl.getSMILES(narupa_mol, False)
        print(name)
        traj.append(narupa_mol.copy())
    name = Tl.getSMILES(narupa_mol, False)
    if len(name) > 1 and name[0] == 'C=O':
        write('DisocTrajs/Run' +str(run)+'.xyz', traj)
    else:
        write('ReactTrajs/Run' +str(run)+'.xyz', traj)


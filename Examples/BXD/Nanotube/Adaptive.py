from src.Calculators.OpenMM import OpenMMCalculator
import src.bxd.collective_variable as CV
import src.bxd.ProgressMetric as PM
import src.bxd.path as Path
import src.molecular_dynamics.md_Integrator as MD
import src.bxd.dimensionality_reduction as DR
import src.bxd.bxd_constraint as BXD
import src.molecular_dynamics.trajectory as Traj
from ase.io import read,write
import src.molecular_dynamics.md_logger as lg
from ase.optimize import BFGS

narupa_path = read('NanoTraj4.xyz', index =':')
print(str(len(narupa_path)))
narupa_mol =  read('NanoTraj4.xyz', index ='0')
narupa_mol.set_calculator(OpenMMCalculator('Nano.xml', narupa_mol))
narupa_path[0] = narupa_mol.copy()
dyn = BFGS(narupa_mol,maxstep=100)
try:
   dyn.run(1e-2, 100)
except:
   pass
print(str(narupa_mol.get_potential_energy()))


pcs = 2
dim_red = DR.DimensionalityReduction(narupa_path, number=pcs, c_only=True)
dim_red.print_pcs('PCpruned')
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=250)

path = Path.Path(narupa_path, collective_var, stride=1, max_distance_from_path=1)
progress = PM.Curve(collective_var, path, max_nodes_skiped=2)

narupa_mol.set_positions(narupa_path[0].get_positions())
md = MD.Langevin(narupa_mol, temperature=1000, friction=1, timestep=0.25)

loggers = []
lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].s)) +'\n'
tf1 = lambda var: var.mdsteps % 50000 == 0
log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
loggers.append(log1)



bxd_manager = BXD.Adaptive(progress, epsilon=0.995, adaptive_steps=150000, fix_to_path=True)
bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd_manager], md, loggers = loggers)
bxd_trajectory.run_trajectory()
bxd_manager.print_bounds('bounds_out.txt')


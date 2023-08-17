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

narupa_path = read('NanoTraj4.xyz', index =':')
narupa_mol =  read('NanoTraj4.xyz', index ='0')

pcs = 2
dim_red = DR.DimensionalityReduction(narupa_path, number=pcs, c_only=True)
dim_red.print_pcs('PCpruned')
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=250)

path = Path.Path(narupa_path, collective_var, stride=1, max_distance_from_path=20)
progress = PM.Curve(collective_var, path, max_nodes_skiped=1)

narupa_mol.set_calculator(OpenMMCalculator('Nano.xml', narupa_mol))
md = MD.Langevin(narupa_mol, temperature=2000, friction=10, timestep=1)

logfile = open('log.txt', 'w')
loggers = []
lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].s) /(var.bxd_list[0].progress_metric.full_distance)) +'\n'
tf1 = lambda var: var.mdsteps % 10000 == 0
log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
loggers.append(log1)
file = 'geom.xyz'
lf3 = lambda var: str(write(file, var.mol, append=True))
tf3 = lambda var: var.mdsteps % 10000 == 0
log3 = lg.MDLogger(logging_function=lf3, triggering_function=tf3)
loggers.append(log3)


bxd_manager = BXD.Adaptive(progress, epsilon=0.999, adaptive_steps=1000, fix_to_path=True)
bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd_manager], md, loggers = loggers)
bxd_trajectory.run_trajectory()
bxd_manager.print_bounds('bounds_out.txt')


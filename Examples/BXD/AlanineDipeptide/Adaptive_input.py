from src.Calculators.OpenMM import OpenMMCalculator
import src.bxd.collective_variable as CV
import src.bxd.ProgressMetric as PM
import src.bxd.path as Path
import src.molecular_dynamics.md_Integrator as MD
import src.bxd.dimensionality_reduction as DR
import src.bxd.bxd_constraint as BXD
import src.molecular_dynamics.trajectory as Traj
import src.molecular_dynamics.md_logger as lg
from ase.io import read
from ase.optimize import BFGS


# The general set-up here is identical to the adaptive run to ensure the converging run samples the same path
narupa_mol = read('2ala.xyz', index=0)
narupa_path = read('2ala.xyz', index=':')
narupa_end =  read('2ala.xyz', index=-1)
narupa_mol.set_calculator(OpenMMCalculator('sys.xml', narupa_mol))
pcs = 2
dim_red = DR.DimensionalityReduction(narupa_path, number=pcs, subset=True, start_ind = [4,14], end_ind = [10,17], ignore_h=False)
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=10)
path = Path.Path(narupa_path, collective_var, stride=1, max_distance_from_path=0.5)
progress = PM.Curve(collective_var, path, max_nodes_skiped=1)
md = MD.Langevin(narupa_mol, temperature=298, timestep=1, friction=0.5)
loggers = []
lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].s) /var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].progress_metric.end))
tf1 = lambda var: var.mdsteps % 100 == 0
log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
loggers.append(log1)
lf2 = lambda var: "hit!!\t" + str(var.bxd_list[0].bound_hit) + "\tmdstep\t=\t"+str(var.mdsteps)
tf2 = lambda var: var.bxd_list[0].inversion
log2 = lg.MDLogger( logging_function=lf2, triggering_function=tf2)
loggers.append(log2)
bxd_manager = BXD.Adaptive(progress, epsilon=0.97, adaptive_steps=10000, fix_to_path = True)
bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd_manager], md, loggers = loggers)
bxd_trajectory.run_trajectory()
bxd_manager.print_bounds()
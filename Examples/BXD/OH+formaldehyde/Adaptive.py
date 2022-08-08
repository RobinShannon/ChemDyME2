import src.bxd.collective_variable as CV
import src.bxd.ProgressMetric as PM
import src.bxd.path as Path
import src.molecular_dynamics.md_Integrator as MD
import src.bxd.dimensionality_reduction as DR
import src.bxd.bxd_constraint as BXD
import src.molecular_dynamics.trajectory as Traj
import src.molecular_dynamics.md_logger as lg
from ase.io import read, write
from src.Calculators.NNCalculator import NNCalculator


narupa_mol = read('start.xyz')
narupa_end = read('end.xyz')
narupa_path = read('new.xyz', index='::5')
narupa_path.insert(0,narupa_mol)
narupa_path.append(narupa_end)

pcs = 2
dim_red = DR.DimensionalityReduction(narupa_path, number=pcs)
dim_red.print_pcs('PC')
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=50)
narupa_mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-660000', atoms=narupa_mol))
path = Path.Path(narupa_path, collective_var,  stride=1, max_distance_from_path=0.75)
progress = PM.Curve(collective_var, path,  max_nodes_skiped=3)
#progress = PM.Line(narupa_mol, collective_var, narupa_end)
md = MD.Langevin(narupa_mol, temperature=298, friction=10, timestep=0.1)

loggers = []
lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.proj /var.bxd_list[0].progress_metric.full_distance)
tf1 = lambda var: var.mdsteps % 100 == 0
log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
loggers.append(log1)
lf3 = lambda var: str(write('geom6.xyz', var.mol, append=True))
tf3 = lambda var: var.mdsteps % 10 == 0 and var.bxd_list[0].progress_metric.proj /var.bxd_list[0].progress_metric.full_distance > 0.35 and var.bxd_list[0].progress_metric.proj /var.bxd_list[0].progress_metric.full_distance < 0.8
log3 = lg.MDLogger(logging_function=lf3, triggering_function=tf3, outpath=None)
loggers.append(log3)

#bxd_manager = bxd.Converging(progress, bound_hits=50,read_from_file=True, bound_file="bounds_out.txt", decorrelation_limit = 100, box_data_print_freqency=100)
bxd_manager = BXD.Adaptive(progress, epsilon=0.99, adaptive_steps=25000, fix_to_path=True)
bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd_manager], md, loggers = loggers)
bxd_trajectory.run_trajectory()
bxd_manager.print_bounds('bounds_out.txt')
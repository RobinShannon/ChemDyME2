import src.BXD.CollectiveVariable as CV
import src.BXD.ProgressMetric as PM
import src.BXD.Path as Path
import src.MolecularDynamics.MDIntegrator as MD
import src.BXD.DimensionalityReduction as DR
import src.BXD.BXDConstraint as BXD
import src.MolecularDynamics.Trajectory as Traj
import src.MolecularDynamics.MDLogger as lg
from ase.io import read, write
import ase
import sys
from ase.optimize import BFGS
from src.Calculators.XtbCalculator import XTB

narupa_mol = read('NEB.xyz', index='0')
narupa_mol.set_calculator(XTB(method="GFN1xTB", electronic_temperature=300))
narupa_end = read('NEB.xyz', index='-1')
narupa_path = read('NEB.xyz', index=':')

pcs = 2
dim_red = DR.DimensionalityReduction(narupa_path, subset=True,start_ind=[-1,1,3,6],end_ind=[0,2,4,9], number=pcs)
dim_red.print_pcs('PC')
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=7)

path = Path.Path(narupa_path, collective_var,  stride=2, max_distance_from_path=0.1)
progress = PM.Curve(collective_var, path,  max_nodes_skiped=3)
#progress = PM.Line(narupa_mol, collective_var, narupa_end)
md = MD.Langevin(narupa_mol, temperature=100, friction=0.01, timestep=0.1)

loggers = []
lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].s) /var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].progress_metric.end))
tf1 = lambda var: var.mdsteps % 100 == 0
log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
loggers.append(log1)
lf2 = lambda var: "hit!!\t" + str(var.bxd_list[0].bound_hit) + "\tmdstep\t=\t"+str(var.mdsteps)
tf2 = lambda var: var.bxd_list[0].inversion
log2 = lg.MDLogger( logging_function=lf2, triggering_function=tf2)
loggers.append(log2)

#bxd_manager = BXD.Converging(progress, bound_hits=50,read_from_file=True, bound_file="bounds_out.txt", decorrelation_limit = 100, box_data_print_freqency=100)
bxd_manager = BXD.Adaptive(progress, epsilon=0.999, adaptive_steps=5000, fix_to_path=True)
bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd_manager], md, loggers = loggers)
bxd_trajectory.run_trajectory()
bxd_manager.print_bounds('bounds_out.txt')
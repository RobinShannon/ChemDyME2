import src.bxd.collective_variable as CV
import src.bxd.ProgressMetric as PM
import src.bxd.path as Path
import src.molecular_dynamics.md_Integrator as MD
import src.bxd.dimensionality_reduction as DR
import src.bxd.bxd_constraint as BXD
import src.molecular_dynamics.trajectory as Traj
import src.molecular_dynamics.md_logger as lg
from ase.io import read, write
import src.Calculators.ScineCalculator as SP

narupa_mol = read('NEB.xyz', index='0')
narupa_mol.set_calculator(SP.SparrowCalculator(method='AM1'))
narupa_end = read('NEB.xyz', index='-1')
narupa_path = read('NEB.xyz', index=':')

pcs = 2
#dim_red = DR.DimensionalityReduction(narupa_path, subset=True,start_ind=[-1,6],end_ind=[0,9], number=pcs)
dim_red = DR.DimensionalityReduction(narupa_path, subset=True,start_ind=[-1,1,3,6],end_ind=[0,2,4,9], number=pcs)
dim_red.print_pcs('PC')
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=8)

path = Path.Path(narupa_path, collective_var,  stride=2, max_distance_from_path=1)
progress = PM.Curve(collective_var, path,  max_nodes_skiped=3)
#progress = PM.Line(narupa_mol, collective_var, narupa_end)
md = MD.Langevin(narupa_mol, temperature=600, friction=10, timestep=0.1)
logfile = open('log.txt', 'w')
loggers = []
lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.proj) + "\tdistance from path\t=\t" +str(var.bxd_list[0].progress_metric.distance_from_path)
tf1 = lambda var: var.bxd_list[0].inversion == True
log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
loggers.append(log1)
lf2 = lambda var: 'steps\t=\t' + str(var.mdsteps) + '\tupper hits\t=\t'+str(var.bxd_list[0].box_list[var.bxd_list[0].box].upper.hits) + "\tlower hits\t=\t" +str(var.bxd_list[0].box_list[var.bxd_list[0].box].lower.hits)
tf2 = lambda var: var.bxd_list[0].inversion == True
log2 = lg.MDLogger( logging_function=lf2, triggering_function=tf2)
loggers.append(log2)
names = ['COCO[O]','[CH2]OCOO']

#bxd_manager = bxd.Converging(progress, bound_hits=50,read_from_file=True, bound_file="bounds_out.txt", decorrelation_limit = 100, box_data_print_freqency=100)
bxd_manager = BXD.Converging(progress_metric=progress, bound_hits=1, plot_box_data=True)
bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd_manager], md, loggers = loggers,  allowed_species=names)
bxd_trajectory.run_trajectory()

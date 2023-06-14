import src.bxd.collective_variable as CV
import src.bxd.ProgressMetric as PM
import src.bxd.path as Path
import src.molecular_dynamics.md_Integrator as MD
import src.bxd.dimensionality_reduction as DR
import src.bxd.bxd_constraint as BXD
import src.bxd.converge_free_energy as FE
import src.molecular_dynamics.trajectory as Traj
import src.molecular_dynamics.md_logger as lg
from ase.io import read, write
import src.Calculators.ScineCalculator as SP
from datetime import date

f = open("free_energy_summary.txt", "a")

narupa_mol = read('NEB.xyz', index='0')
narupa_mol.set_calculator(SP.SparrowCalculator(method='AM1'))
narupa_end = read('NEB.xyz', index='-1')
narupa_path = read('NEB.xyz', index=':')

pcs = 2
#dim_red = DR.DimensionalityReduction(narupa_path, subset=True,start_ind=[-1,6],end_ind=[0,9], number=pcs)
dim_red = DR.DimensionalityReduction(narupa_path, subset=True,start_ind=[-1,1,3,6],end_ind=[0,2,4,9], number=pcs)
dim_red.print_pcs('PC')
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=7)

path = Path.Path(narupa_path, collective_var,  stride=2, max_distance_from_path=1)
progress = PM.Curve(collective_var, path,  max_nodes_skiped=3)
#bxd_manager = bxd.Converging(progress, bound_hits=50,read_from_file=True, bound_file="bounds_out.txt", decorrelation_limit = 100, box_data_print_freqency=100)
bxd_manager = BXD.Converging(progress_metric=progress, bound_hits=1)
free_energy = FE.get_free_energy(bxd_manager,800,milestoning=True, boxes=10)
f.write('Free Energy profile: Resolution = ' + (str(5)) + ' Date ' + str(date.today())+'\n\n')
for fe in free_energy:
    f.write(str(fe[0])+'\t'+str(fe[1])+'\n')
rate = FE.get_rates(bxd_manager,milestoning=True,directory='Converging_Data',errors = True)
f.write('Rate coefficient' + '\n\n')
f.write(str(rate))
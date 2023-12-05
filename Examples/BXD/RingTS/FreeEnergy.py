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
from src.Calculators.OpenMM import OpenMMCalculator
from datetime import date

f = open("free_energy_summary.txt", "a")

narupa_path = read('NanoTraj4.xyz', index =':')
print(str(len(narupa_path)))
narupa_mol =  read('NanoTraj4.xyz', index ='0')

pcs = 2
dim_red = DR.DimensionalityReduction(narupa_path, number=pcs, c_only=True)
dim_red.print_pcs('PCpruned')
collective_var = CV.PrincipalCoordinates(dim_red.pc_list, number_of_elements=250)

path = Path.Path(narupa_path, collective_var, stride=1, max_distance_from_path=2)
progress = PM.Curve(collective_var, path, max_nodes_skiped=2)

narupa_mol.set_calculator(OpenMMCalculator('Nano.xml', narupa_mol))

#bxd_manager = bxd.Converging(progress, bound_hits=50,read_from_file=True, bound_file="bounds_out.txt", decorrelation_limit = 100, box_data_print_freqency=100)
bxd_manager = BXD.Converging(progress_metric=progress, bound_hits=1)
free_energy = FE.get_free_energy(bxd_manager,1000,milestoning=True, boxes=1, decorrelation_limit=5)
f.write('Free Energy profile: Decorrelation = ' + (str(5)) + ' Date ' + str(date.today())+'\n\n')
for fe in free_energy:
    f.write(str(fe[0])+'\t'+str(fe[1])+'\n')
free_energy = FE.get_free_energy(bxd_manager,1000,milestoning=True, boxes=1, decorrelation_limit=20)
f.write('Free Energy profile: Decorrelation = ' + (str(20)) + ' Date\n\n')
for fe in free_energy:
    f.write(str(fe[0])+'\t'+str(fe[1])+'\n')

free_energy = FE.get_free_energy(bxd_manager,1000,milestoning=True, boxes=1, decorrelation_limit=100)
f.write('Free Energy profile: Decorrelation = ' + (str(100)) + ' Date\n\n')
for fe in free_energy:
    f.write(str(fe[0])+'\t'+str(fe[1])+'\n')
rate = FE.get_rates(bxd_manager,milestoning=True,directory='Converging_Data',errors = True)
f.write('Rate coefficient' + '\n\n')
f.write(str(rate))
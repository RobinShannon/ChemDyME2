import src.mechanism_generation.mol_types.species as Species
import src.utility.connectivity_tools as CT
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import os
from ase.io import write
import gc

# Class to store stable stationary points of a reaction and perform geometry optimisations and energy refinements
class Reaction:
    """
    Class designed to store the stationary points and barrier information for a particular reactive event.
    """
    def __init__(self,reac, prod, trajectory, calculator, core = 1):
        self.trajectory = trajectory.criteria.atom_list
        self.ts_points = trajectory.criteria.transition_mol
        self.calculator = calculator
        self.reac = reac
        self.TS = Species.TS(self.ts_points[0], self.calculator, reac.combined_mol, prod.combined_mol)
        self.prod = prod
        self.activationEnergy = 0
        self.events_forward = 0
        self.events_reverse = 0
        self.calculator = calculator
        self.path_maxima = []
        self.path_minima = []
        self.path_energies = []
        self.path_structures = []
        self.name = self.reac.smiles + '___' + self.prod.smiles
        self.reverse_name = self.prod.smiles + '___' + self.reac.smiles
        self.found_TS = False
        self.core = core

    def __eq__(self, other):
        if self.name is other.name:
            return True
        elif self.name is other.reverse_name:
            return True
        else:
            return False

    def characterise(self, bimolecular_traj):
        self.found_TS = self.check_TS()
        self.get_mep(bimolecular_traj)
        self.examine_mep()
        for i in range(1,len(self.ts_points)-1):
            self.TS = Species.TS(self.ts_points[i], self.calculator)
            if self.check_TS():
                break

        if not self.barrierless and not self.check_TS():
            for ind in self.path_maxima:
                self.TS = Species.TS(self.path_structures[ind], self.calculator)
                if self.check_TS():
                    break

    def check_TS(self):
        if self.TS.real_saddle:
            if self.TS.rmol_name == self.reac.smiles and self.TS.pmol_name == self.prod.smiles:
                return True
            elif self.TS.rmol_name == self.prod.smiles and self.TS.pmol_name == self.reac.smiles:
                return True
            else:
                return False
        else:
            return False

    def get_mep(self, bimolecular_traj = False):
        try:
            self.calculator.set_calculator(self.TS.mol, 'low')
            if bimolecular_traj:
                self.path_structures = self.TS.mol._calc.minimise_bspline('Raw/Low/' + str(self.core) + '/Path/', self.reac.combined_mol, self.prod.combined_mol)
            else:
                self.path_structures = self.TS.mol._calc.minimise_bspline('Raw/Low/' + str(self.core) + '/Path/', self.reac.mol, self.prod.mol)
            for ps in self.path_structures:
                self.calculator.set_calculator(ps, 'low')
                self.path_energies.append(ps.get_potential_energy())
        except:
            self.optimise_dynamic_path(bimolecular_traj)

    def optimise_dynamic_path(self,bimolecular_traj):
        """
        Takes a list of atom objects from a trajectory and performs constrained minimisations along this path keeping
        fixed any bonds that changes over the course of the reaction. This can then be used as a guess path for subsequent
        calculations
        :param trajectory: List of atoms objects representing a reactive trajectory
        :return: List of atoms objects with the partially minimised trajectory
        """
        # extract start and end points along the trajectory
        if bimolecular_traj:
            reactant = self.reac.combined_mol
        else:
            reactant = self.reac.combined_mol
        product = self.prod.mol
        traj = self.trajectory[::10]
        changed_bonds = CT.get_changed_bonds(reactant, product)
        for i in traj:
            mol = i.copy()
            self.calculator.set_calculator(mol, 'low')
            c = FixAtoms(changed_bonds)
            mol.set_constraint(c)
            min = BFGS(mol)
            try:
                min.run(fmax=0.1, steps=50)
            except:
                min.run(fmax=0.1, steps=1)
            del mol.constraints
            self.path_structures.append(mol)
            self.path_energies.append(mol.get_potential_energy())

    def examine_mep(self):
        """
        Looks at a minimum energy path to determine whether there is more than one minima and hence more than one
        reaction along it.
        :param MEP:
        :return:
        """
        for i in range(1,len(self.path_energies)-2):
            if self.path_energies[i] > self.path_energies[i-1] and self.path_energies[i] > self.path_energies[i+1]:
                self.path_maxima.append(i)
            elif self.path_energies[i] < self.path_energies[i-1] and self.path_energies[i] < self.path_energies[i+1]:
                self.path_minima.append(i)
        if len(self.path_maxima) == 0:
            self.barrierless = True

    def print_to_file(self):
        base_path = '/Network/' + str(self.reac.smiles) + '/'
        os.makedirs(base_path, exist_ok=True)
        prod_path = base_path + str(self.prod.smiles)
        TS_path = prod_path + 'TS/'
        os.makedirs(TS_path, exist_ok=True)
        data_path = prod_path + 'data/'
        os.makedirs(data_path, exist_ok=True) + '/'
        write(prod_path +'geometry.xyz', self.prod.mol)
        write(data_path + 'trajectory.xyz', self.path_structures)
        write(TS_path+'TS_geom.xyz', self.TS.mol)
        del self.trajectory
        del self.path_structures
        gc.collect()







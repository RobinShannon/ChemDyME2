import src.mechanism_generation.mol_types.ts as ts
import src.utility.connectivity_tools as CT
from ase.optimize import BFGS
from ase.constraints import FixBondLengths
import os
from ase.io import write,read
import gc
from ase.neb import NEB
try:
    import MESMER_API.src.meReaction as me_reac
except:
    pass

# Class to store stable stationary points of a reaction and perform geometry optimisations and energy refinements
class Reaction:
    """
    Class designed to store the stationary points and barrier information for a particular reactive event.
    """
    def __init__(self,reac, prod, trajectory, calculator, core = 1):
        self.reac = reac
        self.prod = prod
        self.name = self.reac[-1].name + '___' + self.prod[-1].name
        self.calculator = calculator
        if not trajectory == None:
            self.trajectory = trajectory.criteria.atom_list
            self.ts_points = trajectory.criteria.transition_mol
            self.ts = ts.ts(self.ts_points[-1], self.calculator, 'TS', reac[-1].mol.copy(), prod[-1].mol.copy(),
                            name='TS' + self.name)
        self.barrierless = False
        self.activationEnergy = 0
        self.events_forward = 0
        self.events_reverse = 0
        self.path_maxima = []
        self.path_minima = []
        self.path_energies = []
        self.path_structures = []
        self.reverse_name = self.prod[-1].name + '___' + self.reac[-1].name
        self.found_TS = False
        self.cml = ''
        self.core = core

    def copy(self):
        new_reac = []
        for r in self.reac:
            new_reac.append(r.copy())
        new_prod = []
        for p in self.prod:
            new_prod.append(p.copy())
        new = Reaction(new_reac,new_prod, None, None)
        new.calculator = None
        new.ts = self.ts.copy()
        new.ts_points = []
        return new

    def __eq__(self, other):
        if self.name == other.name:
            return True
        elif self.name is other.reverse_name:
            return True
        else:
            return False

    def characterise(self, bimolecular_traj):
        self.found_TS = self.check_TS()
        self.get_mep(bimolecular_traj)
        self.examine_mep()

        if not self.found_TS:
            for i in range(1,len(self.path_maxima)-1):
                try:
                    ts_guess = ts.ts(self.path_maxima[i], self.calculator, 'TS', self.reac[-1].mol.copy(), self.prod[-1].mol.copy(), name='TS'+self.name)
                    if self.check_TS():
                        self.ts = ts_guess
                        break
                except:
                    pass


    def check_TS(self):
        if self.ts.real_saddle:
            if self.ts.rmol_name == self.reac[-1].smiles and self.ts.pmol_name == self.prod[-1].smiles:
                return True
            elif self.ts.rmol_name == self.prod[-1].smiles and self.ts.pmol_name == self.reac[-1].smiles:
                return True
            else:
                return False
        else:
            return False

    def get_mep(self, bimolecular_traj = False):
        try:
            self.calculator.set_calculator(self.ts.mol, 'low')
            self.path_structures = self.ts.mol._calc.minimise_bspline('Raw/Low/' + str(self.core) + '/Path/', self.reac[-1].mol, self.prod[-1].mol)
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
        reactant = self.reac[-1].mol.copy()
        product = self.prod[-1].mol.copy()
        traj_1 = self.trajectory[:-250:1000]
        traj_2 = self.trajectory[-250:-140:4]
        traj_3 =  self.trajectory[-140::20]
        traj = traj_1+traj_2+traj_3
        changed_bonds = CT.get_changed_bonds(reactant, product)
        for i in traj:
            mol = i.copy()
            self.calculator.set_calculator(mol, 'low')
            c = FixBondLengths(changed_bonds)
            mol.set_constraint(c)
            min = BFGS(mol)
            try:
                min.run(fmax=0.1, steps=100)
            except:
                pass
            del mol.constraints
            self.path_structures.append(mol)
            self.path_energies.append(mol.get_potential_energy())
        neb = NEB(self.path_structures, climb=True, allow_shared_calculator=True, remove_rotation_and_translation=True, method="eb")
        optimizer = BFGS(neb)
        try:
            optimizer.run(fmax=0.04, steps = 50)
            self.path_structures=[]
            self.path_energies=[]
            for n in neb.images:
                self.calculator.set_calculator(n, 'low')
                self.path_structures.append(n.copy())
                try:
                    self.path_energies.append(n.get_potential_energy())
                except:
                    last = self.path_energies[-1]
                    self.path_energies.append(last)
        except:
            pass


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
            self.activationEnergy = (max(self.path_energies)-self.path_energies[0])*96.48


    def print_to_file(self):
        base_path = 'Network/' + str(self.reac[0].name)
        os.makedirs(base_path, exist_ok=True)
        if len(self.reac) > 2 :
            bi_path = base_path + '/bimolecular_' + str(self.reac[0].name)
            os.makedirs(bi_path, exist_ok=True)
            prod_path = bi_path + '/' +  str(self.prod[0].name)
        else:
            uni_path = base_path + '/unimolecular_' + str(self.reac[0].name)
            os.makedirs(uni_path, exist_ok=True)
            prod_path = uni_path + '/' + str(self.prod[0].name)
        os.makedirs(prod_path, exist_ok=True)
        TS_path = prod_path + '/TS'
        os.makedirs(TS_path, exist_ok=True)
        data_path = prod_path + '/data'
        os.makedirs(data_path, exist_ok=True)
        write(prod_path +'/geometry.xyz', self.prod[0].mol)
        write(data_path + '/optimised_trajectory.xyz', self.path_structures)
        write(data_path + '/trajectory.xyz', self.trajectory)
        write(TS_path+'/TS_geom.xyz', self.ts.mol)
        write(TS_path+'/IRC.xyz',self.ts.IRC)
        with open(data_path + '/energies.txt', 'w') as f:
            for ene in self.path_energies:
                f.write(str(ene)+'\n')
        del self.trajectory
        del self.path_structures
        gc.collect()

    def write_cml(self):
        ea = 0
        reacs = []
        reacs.append(self.reac[0].name)
        if len(self.reac) > 2:
            reacs.append(self.reac[1].name)
            trans = None
            if self.check_TS():
                ea = self.activationEnergy
            else:
                self.barrierless = True
        else:
            trans = self.ts.name
            self.barrierless = False
        prods = []
        prods.append(self.prod[0].name)
        if len(self.prod) >2:
            prods.append(self.prod[1].name)
        mes_reac = me_reac.meReaction(reacs,prods,name=self.name,ts=trans, ea = ea)
        self.cml = mes_reac.cml










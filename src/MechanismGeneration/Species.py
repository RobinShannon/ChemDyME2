import src.Utility.ConnectivityTools as CT
import src.Utility.Tools as TL
from ase.constraints import FixInternals, FixAtoms, FixBondLength
from ase.optimize import BFGS
import math
import MESMER_API.src.meMolecule as me_mol
from ase.io import write
from abc import abstractmethod
import os


class Species:
    def __init__(self, mol, calculator, dir):
        self.dir =dir
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        self.mol = mol
        self.combined_mol = mol.copy()
        self.vibs = []
        self.calculator = calculator
        self.energy = {}
        self.zpe = 0.0
        self.sprint_coordinates = []
        self.smiles = ''
        self.node_visits = 0
        self.core = 1
        self.hinderance_potentials = []
        self.hinderance_trajectories=[]
        self.hinderance_indexes = []
        self.rotor_indexes=[]
        self.bonds_to_add = []
        self.vdw = False

    def characterise(self, mol, bimolecular = False):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        try:
            self.calculator.set_calculator(mol, 'low')
            self.optimise( 'Low/', mol)
            try:
                self.conformer_search(mol)
            except:
                self.bonds_to_add = []
                self.conformer_search(mol)
            self.calculator.set_calculator(mol, 'high')
            if self.calculator.multi_level == True:
                self.optimise('High/' , mol)
            if len(self.mol.get_masses())>1:
                self.get_frequencies('High/', bimolecular,mol)
            if bimolecular:
                self.energy['bimolecular_high'] = mol.get_potential_energy()
            else:
                self.energy['high'] = mol.get_potential_energy()
            self.calculator.set_calculator(mol, 'single')
            if bimolecular:
                self.energy['bimolecular_single'] = mol.get_potential_energy()
            else:
                self.energy['single'] = mol.get_potential_energy()
            if self.calculator.calc_hindered_rotors:
                self.get_hindered_rotors(mol)
            if self.calculator.calc_BXDE_DOS:
                self.get_BXDE_DOS()
            os.chdir(current_dir)
        except:
            os.chdir(current_dir)

    @abstractmethod
    def optimise(self, path, mol):
        pass

    @abstractmethod
    def get_frequencies(self):
        pass

    @abstractmethod
    def get_bxde_dos(self):
        pass

    def conformer_search(self, mol):
        rotatable_bonds = CT.get_rotatable_bonds(mol, self.bonds_to_add)
        conformer_energies = []
        conformers = []
        found_new_min = True
        dyn = BFGS(mol)
        dyn.run(fmax=0.01, steps = 100)
        ene = mol.get_potential_energy()
        conformer_energies.append(ene)
        self.rotor_indexes = rotatable_bonds
        while found_new_min == True:
            found_new_min = False
            for b in rotatable_bonds:
                for i in range(0,12):
                    conformer = mol.copy()
                    constraints1 =[]
                    if self.vdw == False:
                        for bond in self.bonds_to_add:
                            constraints1.append(FixBondLength(bond[0], bond[1]))
                    conformer.set_constraint(constraints1)
                    conformer.rotate_dihedral(b[0], b[1], b[2], b[3], angle = 30)
                    conf = conformer.copy()
                    self.calculator.set_calculator(conf, 'low')
                    constraints = []
                    if self.vdw == False:
                        for bond in self.bonds_to_add:
                            constraints.append(FixBondLength(bond[0], bond[1]))
                    conf.set_constraint(constraints)
                    dyn = BFGS(conf)
                    dyn.run(fmax=0.01, steps = 100)
                    ene = conf.get_potential_energy()
                    found = False
                    for c in conformer_energies:
                        if math.isclose(ene, c, rel_tol=1e-4):
                            found = True
                    if not found:
                        conformer_energies.append(ene)
                        conformers.append(conf.copy())
                        if min(conformer_energies) == ene:
                            mol = conf
                            found_new_min = True
                            break
        write('Conformers.xyz', conformers)


    def get_hindered_rotors(self,mol, rigid=False):
        rotatable_bonds = CT.get_rotatable_bonds(mol,self.bonds_to_add)
        self.rotor_indexes = rotatable_bonds
        for count ,b in enumerate(rotatable_bonds):
            hmol = mol.copy()
            self.calculator.set_calculator(hmol, 'low')
            hinderance_potential = []
            hinderance_traj = []
            for i in range(0,36):
                hmol.rotate_dihedral(b[0], b[1], b[2], b[3], angle = 10)
                del hmol.constraints
                constraints = []
                dihedral = [hmol.get_dihedral(*b), b]
                constraints.append(FixInternals(dihedrals_deg=[dihedral]))
                if self.vdw == False:
                    for bond in self.bonds_to_add:
                        constraints.append(FixBondLength(bond[0], bond[1]))
                hmol.set_constraint(constraints)
                if not rigid:
                    dyn = BFGS(hmol)
                    dyn.run(fmax = 0.05, steps = 100)
                self.calculator.set_calculator(hmol, 'high')
                hinderance_potential.append(hmol.get_potential_energy())
                hinderance_traj.append(hmol.copy())

            self.hinderance_potentials.append(hinderance_potential)
            self.hinderance_trajectories.append(hinderance_traj)
            self.hinderance_indexes.append([b[1],b[2]])
            write('hindered_rotor' +str(count)+ '.xyz', hinderance_traj)



class TS(Species):

    def __init__(self, mol, calculator, reac, prod, dir):
        super(TS, self).__init__(mol, calculator, dir)
        self.imaginary_frequency = 0.0
        self.TS_correct = False
        self.rmol = reac
        self.pmol = prod
        self.rmol_name = ""
        self.pmol_name = ""
        self.IRC = []
        self.hessian =[]
        self.real_saddle = False
        write( str(self.dir) + '/reac.xyz', self.rmol)
        write(str(self.dir) + '/prod.xyz', self.pmol)
        write(str(self.dir) + '/tsguess.xyz', self.mol)
        try:
            self.bonds_to_add = CT.get_changed_bonds(self.rmol, self.pmol)
            self.pre_optimise()
        except:
            pass
        self.characterise(self.mol)

    def pre_optimise(self):
        self.calculator.set_calculator(self.mol, 'low')
        del self.mol.constraints
        constraints = []
        for bond in self.bonds_to_add:
            constraints.append(FixBondLength(bond[0], bond[1]))
        self.mol.set_constraint(constraints)
        dyn = BFGS(self.mol)
        dyn.run(fmax=0.05, steps=100)
        del self.mol.constraints

    def optimise(self, path, mol):
        self.mol, self.rmol, self.pmol, irc_for, irc_rev = mol._calc.minimise_ts(path = path, atoms=mol, ratoms= self.rmol, patoms= self.pmol)
        try:
            self.rmol_name = TL.getSMILES(self.rmol,False)
            self.pmol_name = TL.getSMILES(self.pmol,False)
        except:
            pass

    def get_frequencies(self, path, bimolecular, mol, TS=True):
        self.vibs, self.zpe, self.imaginary_frequency, self.hessian = mol._calc.get_frequencies(path, mol, TS=True)


    def write_cml(self):
        data = {}
        data['zpe'] = self.energy['single']
        data['vibFreqs'] = self.vibs
        data['name'] = self.smiles
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['imaginary_frequency'] = self.imaginary_frequency
        data['hessian'] = self.hessian
        mes_mol = me_mol.meMolecule(self.mol, role = 'ts', **data)
        mes_mol.write_cml(self.dir + '/mes.xml')

class Stable(Species):

    def __init__(self, mol, calculator, dir = 'Raw'):
        super(Stable, self).__init__(mol, calculator, dir)
        self.calculator.set_calculator(self.mol, 'low')
        smiles = TL.getSMILES(self.mol, True)
        self.combined_mol = mol.copy()
        self.calculator.set_calculator(self.combined_mol, 'low')
        self.smiles = smiles[0]
        self.hessian = []
        if len(smiles) > 1:
            self.bimolecular = True
            self.mol = TL.getMolFromSmile(smiles[0])
            self.fragment_two = TL.getMolFromSmile(smiles[1])
            self.characterise(self.fragment_two, bimolecular=True)
            self.bi_smiles = smiles[1]
        self.characterise(self.mol)

    def optimise(self, path, mol):
        mol._calc.minimise_stable(path=path, atoms=mol)

    def get_frequencies(self, path, bimolecular, mol):
        if bimolecular:
            self.bimolecular_vibs, self.bimolecular_zpe, imag, hess = mol._calc.get_frequencies(path, mol, bimolecular=True)
        else:
            self.vibs, self.zpe, imag, self.hessian = mol._calc.get_frequencies(path, mol)

    def write_cml(self):
        data = {}
        data['zpe'] = self.energy['single']
        data['vibFreqs'] = self.vibs
        data['name'] = self.smiles
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', **data)
        mes_mol.write_cml(self.dir + '/mes.xml')


class vdw(Species):

    def __init__(self, mol, calculator, dir = 'Raw'):
        super(vdw, self).__init__(mol, calculator, dir)
        self.vdw = True
        self.calculator.set_calculator(self.mol, 'low')
        smiles = TL.getSMILES(self.mol, True, True)
        self.hessian =[]
        self.combined_mol = mol.copy()
        self.calculator.set_calculator(self.combined_mol, 'low')
        self.smiles = 'vdw_' + str(smiles[0]) + '_' + str(smiles[1])
        self.bonds_to_add = CT.get_hbond_idxs(self.mol)
        self.characterise(self.mol)

    def optimise(self, path, mol):
        mol._calc.minimise_stable(path=path, atoms=mol)

    def get_frequencies(self, path, bimolecular, mol):
        self.vibs, self.zpe, imag, self.hessian = mol._calc.get_frequencies(path, mol)

    def write_cml(self):
        data = {}
        data['zpe'] = self.energy['single']
        data['vibFreqs'] = self.vibs
        data['name'] = self.smiles
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', **data)
        mes_mol.write_cml(self.dir + '/mes.xml')


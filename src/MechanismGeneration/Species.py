import src.Utility.ConnectivityTools as CT
import src.Utility.Tools as TL
from ase.constraints import FixInternals, FixAtoms, FixBondLength, FixBondLengths
from ase.optimize.sciopt import SciPyFminBFGS as BFGS
import math
try:
    import MESMER_API.src.meMolecule as me_mol
except:
    pass
from ase.io import write
from abc import abstractmethod
import copy
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

    def characterise(self, bimolecular = False):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        self.calculator.set_calculator(self.mol, 'low')
        self.optimise('Low/', self.mol)
        #self.conformer_search(self.mol)
        #if self.calculator.calc_hindered_rotors:
            #self.get_hindered_rotors(self.mol)
        self.calculator.set_calculator(self.mol, 'high')
        if self.calculator.multi_level == True:
            self.optimise('High/', self.mol)
            if len(self.mol.get_masses()) > 1:
                self.calculator.set_calculator(self.mol, 'high')
                self.get_frequencies('High/', bimolecular, self.mol)
            self.calculator.set_calculator(self.mol, 'high')
        else:
            self.calculator.set_calculator(self.mol, 'low')
        if bimolecular:
            self.energy['bimolecular_high'] = self.mol.get_potential_energy() * 96.4869
        else:
            self.energy['high'] = self.mol.get_potential_energy() * 96.4869
        self.calculator.set_calculator(self.mol, 'single')
        if bimolecular:
            self.energy['bimolecular_single'] = self.mol.get_potential_energy() * 96.4869
        else:
            try:
                self.energy['single'] = self.mol.get_potential_energy() * 96.4869
            except:
                self.energy['single'] = 0.0
        if self.calculator.calc_BXDE_DOS:
            self.get_BXDE_DOS()
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
        rotatable_bonds, coId = CT.get_rotatable_bonds(mol,self.bonds_to_add)
        conformer_energies = []
        conformers = []
        found_new_min = True
        self.calculator.set_calculator(mol, 'low')
        ene = mol.get_potential_energy()
        conformer_energies.append(ene)
        self.rotor_indexes = rotatable_bonds
        distances = []
        for bond in self.bonds_to_add:
            dist = mol.get_distance(bond[0],bond[1])
            distances.append(dist)
        while found_new_min == True:
            found_new_min = False
            for b, co in zip(rotatable_bonds, coId):
                conformer = mol.copy()
                dihed = conformer.get_dihedral(*b)
                for i in range(0,73):
                    print(str(i))
                    del conformer.constraints
                    conformer.set_dihedral(b[0], b[1], b[2], b[3], dihed, indices=co)
                    constraints1 =[]
                    constraints1.append(FixBondLengths(self.bonds_to_add, bondlengths = distances))
                    dihedral = [dihed, b]
                    constraints1.append(FixInternals(dihedrals_deg=[dihedral]))
                    conformer.set_constraint(constraints1)
                    self.calculator.set_calculator(conformer, 'low')
                    dyn = BFGS(conformer)
                    try:
                        dyn.run(fmax = 0.05, steps = 75)
                    except:
                        pass
                    conf = conformer.copy()
                    del conf.constraints
                    self.calculator.set_calculator(conf, 'low')
                    constraints = []
                    constraints.append(FixBondLengths(self.bonds_to_add))
                    conf.set_constraint(constraints)
                    dyn = BFGS(conf)
                    dyn.run(fmax=0.01, steps = 1)
                    ene = conf.get_potential_energy()
                    dihed += 5.
                    found = False
                    for c in conformer_energies:
                        if math.isclose(ene, c, rel_tol=1e-4):
                            found = True
                    if not found:
                        conformer_energies.append(ene)
                        conformers.append(conf.copy())
                        if min(conformer_energies) == ene:
                            self.mol = conf.copy()
                            mol = conf.copy()
                            found_new_min = True
                    if found_new_min:
                        break
                if found_new_min:
                    break


            write('Conformers.xyz', conformers)



    def get_hindered_rotors(self,mol, rigid=False):
        rotatable_bonds, coId = CT.get_rotatable_bonds(mol,self.bonds_to_add)
        self.rotor_indexes = rotatable_bonds
        distances = []
        self.calculator.set_calculator(mol, 'high')
        high_min = mol.get_potential_energy()*96.49
        for bond in self.bonds_to_add:
            dist = mol.get_distance(bond[0],bond[1])
            distances.append(dist)
        for count,(b,co) in enumerate(zip(rotatable_bonds,coId)):
            hmol = mol.copy()
            dihed = hmol.get_dihedral(*b)
            self.calculator.set_calculator(hmol, 'low')
            hinderance_potential = []
            hinderance_traj = []
            for i in range(0,73):
                hmol.set_dihedral(b[0], b[1], b[2], b[3], dihed, indices=co)
                del hmol.constraints
                constraints = []
                dihedral = [dihed, b]
                constraints.append(FixInternals(dihedrals_deg=[dihedral]))
                constraints.append(FixBondLengths(self.bonds_to_add, bondlengths = distances))
                hmol.set_constraint(constraints)
                if not rigid:
                    dyn = BFGS(hmol)
                    try:
                        dyn.run(fmax = 0.01, steps = 75)
                    except:
                        pass
                hinderance_potential.append(hmol.get_potential_energy()*96.4869)
                hinderance_traj.append(hmol.copy())
                dihed += 5.

            max_value = max(hinderance_potential)
            max_index = hinderance_potential.index(max_value)
            max_structure = hinderance_traj[max_index].copy()
            self.calculator.set_calculator(max_structure, 'high')
            high_barrier = max_structure.get_potential_energy() * 96.4869 - high_min
            low_barrier = hinderance_potential[max_index] - hinderance_potential[0]
            try:
                ratio = high_barrier / low_barrier
            except:
                ratio = 1
            min = copy.deepcopy(hinderance_potential[0])
            for i in range(0,len(hinderance_potential)):
                hinderance_potential[i] -= min
                hinderance_potential[i] *= ratio


            self.hinderance_potentials.append(hinderance_potential)
            self.hinderance_trajectories.append(hinderance_traj)
            self.hinderance_indexes.append([b[1],b[2]])
            write('hindered_rotor' +str(count)+ '.xyz', hinderance_traj)



class TS(Species):

    def __init__(self, mol, calculator, reac, prod, dir, name=''):
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
        self.name = name
        write( str(self.dir) + '/reac.xyz', self.rmol)
        write(str(self.dir) + '/prod.xyz', self.pmol)
        write(str(self.dir) + '/tsguess.xyz', self.mol)
        try:
            self.bonds_to_add = CT.get_changed_bonds(self.rmol, self.pmol)
            self.pre_optimise()
        except:
            pass
        self.characterise()

    def pre_optimise(self):
        self.calculator.set_calculator(self.mol, 'low')
        del self.mol.constraints
        constraints = []
        constraints.append(FixBondLengths(self.bonds_to_add))
        self.mol.set_constraint(constraints)
        dyn = BFGS(self.mol)
        dyn.run(fmax=0.05, steps=25)
        del self.mol.constraints

    def optimise(self, path, mol):
        self.rmol, self.pmol, irc_for, irc_rev = mol._calc.minimise_ts(path = path, atoms=mol, ratoms= self.rmol, patoms= self.pmol)
        self.mol = mol.copy()
        try:
            self.rmol_name = TL.getSMILES(self.rmol,False)
            self.pmol_name = TL.getSMILES(self.pmol,False)
        except:
            pass

    def get_frequencies(self, path, bimolecular, mol, TS=True):
        self.vibs, self.zpe, self.imaginary_frequency, self.hessian = mol._calc.get_frequencies(path, mol, TS=True)


    def write_cml(self):
        data = {}
        data['zpe'] = self.energy['single'] + float(self.zpe)
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['imaginary_frequency'] = self.imaginary_frequency
        data['hessian'] = self.hessian
        data['newBonds'] = self.bonds_to_add
        mes_mol = me_mol.meMolecule(self.mol, role = 'ts', **data)
        mes_mol.write_cml(self.dir + '/mes.xml')

class Stable(Species):

    def __init__(self, mol, calculator, dir = 'Raw', name=''):
        super(Stable, self).__init__(mol, calculator, dir)
        self.calculator.set_calculator(self.mol, 'low')
        smiles = TL.getSMILES(self.mol, True)
        self.combined_mol = mol.copy()
        self.calculator.set_calculator(self.combined_mol, 'low')
        self.smiles = smiles[0]
        if name == '':
            name = self.smiles
        self.name = name
        self.hessian = []
        write(str(self.dir) + '/reac.xyz', self.mol)
        if len(smiles) > 1:
            self.bimolecular = True
            self.mol = TL.getMolFromSmile(smiles[0])
            self.fragment_two = TL.getMolFromSmile(smiles[1])
            self.characterise( bimolecular=True)
            self.bi_smiles = smiles[1]
        self.characterise()

    def optimise(self, path, mol):
        mol._calc.minimise_stable(path=path, atoms=mol)
        self.mol = mol.copy()

    def get_frequencies(self, path, bimolecular, mol):
        if bimolecular:
            self.bimolecular_vibs, self.bimolecular_zpe, imag, hess = mol._calc.get_frequencies(path, mol, bimolecular=True)
        else:
            self.vibs, self.zpe, imag, self.hessian = mol._calc.get_frequencies(path, mol)

    def write_cml(self):
        data = {}
        data['zpe'] = self.energy['single'] + float(self.zpe)
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', **data)
        mes_mol.write_cml(self.dir + '/mes.xml')


class vdw(Species):

    def __init__(self, mol, calculator, dir = 'Raw', name = ''):
        super(vdw, self).__init__(mol, calculator, dir)
        self.vdw = True
        self.calculator.set_calculator(self.mol, 'low')
        smiles = TL.getSMILES(self.mol, True, True)
        if name == '':
            name = smiles
        self.name = name
        self.hessian =[]
        self.combined_mol = mol.copy()
        self.calculator.set_calculator(self.combined_mol, 'low')
        self.smiles = 'vdw_' + str(smiles[0]) + '_' + str(smiles[1])
        self.bonds_to_add = CT.get_hbond_idxs(self.mol)
        self.characterise()

    def optimise(self, path, mol):
        mol._calc.minimise_stable(path=path, atoms=mol)
        self.mol = mol.copy()

    def get_frequencies(self, path, bimolecular, mol):
        self.vibs, self.zpe, imag, self.hessian = mol._calc.get_frequencies(path, mol)

    def write_cml(self):
        data = {}
        data['zpe'] = self.energy['single'] + float(self.zpe)
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        data['newBonds'] = self.bonds_to_add
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', **data)
        mes_mol.write_cml(self.dir + '/mes.xml')


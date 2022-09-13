import src.utility.connectivity_tools as CT
import src.utility.tools as TL
from ase.constraints import FixInternals, FixAtoms, FixBondLength, FixBondLengths
from ase.optimize.sciopt import SciPyFminBFGS as BFGS
from ase.io import read,write
import math
try:
    import MESMER_API.src.meMolecule as me_mol
except:
    pass
from ase.io import write
from abc import abstractmethod
import copy
import os
from sella import Sella
from src.mechanism_generation.mol_types.species import species
import numpy as np
import os

class ts(species):

    def __init__(self, mol, calculator,dir, reac = None, prod=None, name=''):
        super(ts, self).__init__(mol, calculator, dir)
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
        self.fragments = CT.get_fragments(mol)
        write(str(self.dir) + '/tsguess.xyz', self.mol)
        if reac != None and prod != None:
            write(str(self.dir) + '/reac.xyz', self.rmol)
            write(str(self.dir) + '/prod.xyz', self.pmol)
            self.bonds_to_add = CT.get_changed_bonds(self.rmol, self.pmol)
            self.pre_optimise()
        else:
            self.bonds_to_add = CT.get_hbond_idxs(self.mol, self.fragments)

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

    def get_frequencies(self, path, mol):
        self.vibs, self.zpe, self.imaginary_frequency, self.hessian = mol._calc.get_frequencies(path, mol, TS=True)


    def write_cml(self, coupled = False):
        data = {}
        data['zpe'] = self.energy['single'] + float(self.zpe)
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedAngles'] = self.hinderance_angles
        data['hinderedBonds'] = self.hinderance_indexes
        data['imaginary_frequency'] = self.imaginary_frequency
        data['hessian'] = self.hessian
        data['newBonds'] = self.bonds_to_add
        mes_mol = me_mol.meMolecule(self.mol, role = 'ts', coupled = coupled, **data)
        mes_mol.write_cml('mes.xml')

    def conformer_search(self, mol, directory='conformers'):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        os.makedirs(directory, exist_ok=True)
        os.chdir(directory)
        rotatable_bonds, coId = CT.get_rotatable_bonds(mol,self.bonds_to_add)
        conformer_energies = []
        conformers = []
        found_new_min = True
        self.calculator.set_calculator(mol, 'low')
        ene = mol.get_potential_energy()
        conformer_energies.append(ene)
        conformers.append(mol.copy())
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
                for i in range(0,5):
                    print(str(i))
                    del conformer.constraints
                    conformer.set_dihedral(b[0], b[1], b[2], b[3], dihed, indices=co)
                    constraints1 =[]
                    constraints1.append(FixBondLengths(self.bonds_to_add, bondlengths = distances))
                    dihedral = [dihed, b]
                    constraints1.append(FixInternals(dihedrals_deg=[dihedral]))
                    conformer.set_constraint(constraints1)
                    self.calculator.set_calculator(conformer, 'low')
                    dyn = Sella(conformer)
                    try:
                        dyn.run(fmax = 0.05, steps = 10)
                    except:
                        pass
                    conf = conformer.copy()
                    del conf.constraints
                    self.calculator.set_calculator(conf, 'low')
                    dyn = Sella(conf)
                    dyn.run(fmax=0.01, steps = 30)
                    ene = conf.get_potential_energy()
                    dihed += 72.
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
            write('minimum_conformer.xyz', self.mol)
            np.savetxt("sample.txt", conformer_energies, delimiter ="\n")
            os.chdir(current_dir)

    def get_hindered_rotors(self,mol, rigid=False, increment = 30, directory="hindered_rotor"):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        os.makedirs(directory, exist_ok=True)
        os.chdir(directory)
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
            rng = 360.0 / float(increment)
            for i in range(0,rng):
                hmol.set_dihedral(b[0], b[1], b[2], b[3], dihed, indices=co)
                del hmol.constraints
                constraints = []
                dihedral = [dihed, b]
                constraints.append(FixInternals(dihedrals_deg=[dihedral]))
                constraints.append(FixBondLengths(self.bonds_to_add, bondlengths = distances))
                hmol.set_constraint(constraints)
                if not rigid:
                    dyn = Sella(hmol)
                    try:
                        dyn.run(fmax = 0.01, steps = 75)
                    except:
                        pass
                hinderance_potential.append(hmol.get_potential_energy()*96.4869)
                hinderance_traj.append(hmol.copy())
                dihed += float(increment)

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
            write('hindered_rotor' + str(count) + '.xyz', hinderance_traj)
            np.savetxt("hindered_rotor_energies" +str(count)+ ".txt", self.hinderance_potentials, delimiter ="\n")


    def write_hindered_rotors(self,mol, rigid=False, increment = 10, directory="hindered_rotor", partial = False):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        os.makedirs(directory, exist_ok=True)
        os.chdir(directory)
        rotatable_bonds, coId = CT.get_rotatable_bonds(mol,self.bonds_to_add)
        self.rotor_indexes = rotatable_bonds
        distances = []
        for bond in self.bonds_to_add:
            dist = mol.get_distance(bond[0],bond[1])
            distances.append(dist)

        for count,(b,co) in enumerate(zip(rotatable_bonds,coId)):
            os.makedirs('Hind' +str(count), exist_ok=True)
            hmol = mol.copy()
            dihed = hmol.get_dihedral(*b)
            self.calculator.set_calculator(hmol, 'low')
            hinderance_potential = []
            hinderance_traj = []
            rng = 360.0 / float(increment)
            for i in range(0,int(rng)):
                hmol.set_dihedral(b[0], b[1], b[2], b[3], dihed, indices=co)
                dihed = hmol.get_dihedral(*b)
                del hmol.constraints
                constraints = []
                dihedral = [dihed, b]
                constraints.append(FixInternals(dihedrals_deg=[dihedral]))
                constraints.append(FixBondLengths(self.bonds_to_add, bondlengths = distances))
                hmol.set_constraint(constraints)
                if partial:
                    try:
                        opt = BFGS(hmol)
                        opt.run(steps=2)
                    except:
                        pass
                dihed += float(increment)
                self.calculator.set_calculator(hmol, 'high')
                hmol._calc.minimise_ts_write(dihedral=b, path = "Hind"+str(count), title="H" +str(i), atoms= hmol, rigid=rigid)
                self.calculator.set_calculator(hmol, 'low')
                hinderance_traj.append(hmol.copy())
            write('test'+str(count)+'.xyz', hinderance_traj)
        os.chdir(current_dir)

    def write_conformers(self,mol, rigid=False, increment = 60, partial=False, directory="conformers"):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        os.makedirs(directory, exist_ok=True)
        os.chdir(directory)
        rotatable_bonds, coId = CT.get_rotatable_bonds(mol,self.bonds_to_add)
        self.rotor_indexes = rotatable_bonds
        distances = []
        for bond in self.bonds_to_add:
            dist = mol.get_distance(bond[0],bond[1])
            distances.append(dist)

        for count,(b,co) in enumerate(zip(rotatable_bonds,coId)):
            os.makedirs('Hind' +str(count), exist_ok=True)
            hmol = mol.copy()
            dihed = hmol.get_dihedral(*b)
            self.calculator.set_calculator(hmol, 'low')
            rng = 360.0 / float(increment)
            for i in range(0,int(rng)):
                hmol.set_dihedral(b[0], b[1], b[2], b[3], dihed, indices=co)
                del hmol.constraints
                constraints =[]
                constraints.append(FixInternals(dihedrals_deg=[dihedral]))
                constraints.append(FixBondLengths(self.bonds_to_add, bondlengths=distances))
                hmol.set_constraint(constraints)
                if partial:
                    try:
                        opt = BFGS(hmol)
                        opt.run(steps=5)
                    except:
                        pass
                dihed += float(increment)
                hmol._calc.minimise_ts_write( path = "Hind"+str(count), title="H" +str(i), atoms= hmol)
        os.chdir(current_dir)


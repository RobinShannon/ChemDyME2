import src.utility.connectivity_tools as CT
from ase.constraints import FixInternals, FixBondLengths
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
import numpy as np
import glob
from ase.io import read
import src.utility.tools as tl
from ase.units import kJ, mol, invcm
import pickle

class species:
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
        self.hinderance_angles = []
        self.rotor_indexes= []
        self.bonds_to_add = []
        self.vdw = False
        self.bimolecular = False

    def __eq__(self, other):
        if self.name == other.name:
            return True
        else:
            return False


    def characterise(self, bimolecular = False):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        if bimolecular:
            self.calculator.set_calculator(self.fragment_two, 'low')
            self.optimise('Low/', self.fragment_two)
            self.calculator.set_calculator(self.fragment_two, 'high')
            if self.calculator.multi_level == True:
                self.optimise('High/', self.fragment_two)
                if len(self.mol.get_masses()) > 1:
                    self.calculator.set_calculator(self.fragment_two, 'high')
                    self.get_frequencies('High/', self.fragment_two)
                self.calculator.set_calculator(self.fragment_two, 'high')
            else:
                self.calculator.set_calculator(self.fragment_two, 'low')
                if len(self.mol.get_masses()) > 1:
                    self.calculator.set_calculator(self.fragment_two, 'low')
                    self.get_frequencies('High/', self.fragment_two)
                self.calculator.set_calculator(self.fragment_two, 'low')
        else:
            self.calculator.set_calculator(self.mol, 'low')
            self.optimise('Low/', self.mol)
            self.calculator.set_calculator(self.mol, 'high')
            if self.calculator.multi_level == True:
                self.optimise('High/', self.mol)
                if len(self.mol.get_masses()) > 1:
                    self.calculator.set_calculator(self.mol, 'high')
                    self.get_frequencies('High/', self.mol)
                self.calculator.set_calculator(self.mol, 'high')
            else:
                self.calculator.set_calculator(self.mol, 'low')
                if len(self.mol.get_masses()) > 1:
                    self.calculator.set_calculator(self.mol, 'low')
                    self.get_frequencies('High/', self.mol)
                self.calculator.set_calculator(self.mol, 'low')

        if bimolecular:
            self.energy['bimolecular_high'] = self.fragment_two.get_potential_energy() * mol / kJ
        else:
            self.energy['high'] = self.mol.get_potential_energy() * mol / kJ
        self.calculator.set_calculator(self.mol, 'single')
        if bimolecular:
            self.energy['bimolecular_single'] = self.fragment_two.get_potential_energy() * mol / kJ
        else:
            try:
                self.energy['single'] = self.mol.get_potential_energy() * mol / kJ
            except:
                self.energy['single'] = 0.0
        os.chdir(current_dir)

    @abstractmethod
    def optimise(self, path, mol):
        pass

    @abstractmethod
    def save_object(self, directory, filename = 'mol.pkl'):
        current_dir = os.getcwd()
        os.chdir(directory)
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        os.chdir(current_dir)

    @abstractmethod
    def get_frequencies(self):
        pass

    @abstractmethod
    def get_bxde_dos(self):
        pass


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
                for i in range(0,36):
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
                        dyn.run(fmax = 0.05, steps = 20)
                    except:
                        pass
                    conf = conformer.copy()
                    del conf.constraints
                    self.calculator.set_calculator(conf, 'low')
                    dyn = BFGS(conf)
                    dyn.run(fmax=0.01, steps = 60)
                    ene = conf.get_potential_energy()
                    dihed += 10.
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
            rnge = 360 / int(increment)
            for i in range(0,int(rnge)):
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
                        dyn.run(fmax = 0.01, steps = 2)
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

    def write_hindered_rotors(self, mol, rigid=False, increment=10, directory="hindered_rotor", partial=False):
        current_dir = os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        os.makedirs(directory, exist_ok=True)
        os.chdir(directory)
        rotatable_bonds, coId = CT.get_rotatable_bonds(mol, self.bonds_to_add, combine=False)
        self.rotor_indexes = rotatable_bonds
        distances = []
        for bond in self.bonds_to_add:
            dist = mol.get_distance(bond[0], bond[1])
            distances.append(dist)

        for count, (b, co) in enumerate(zip(rotatable_bonds, coId)):
            os.makedirs('Hind' + str(count), exist_ok=True)
            hmol = mol.copy()
            dihed = hmol.get_dihedral(*b)
            self.calculator.set_calculator(hmol, 'low')
            hinderance_potential = []
            hinderance_traj = []
            rng = 360.0 / float(increment)
            for i in range(0, int(rng)):
                hmol.set_dihedral(b[0], b[1], b[2], b[3], dihed, indices=co)
                dihed = hmol.get_dihedral(*b)
                del hmol.constraints
                constraints = []
                dihedral = [dihed, b]
                constraints.append(FixInternals(dihedrals_deg=[dihedral]))
                constraints.append(FixBondLengths(self.bonds_to_add, bondlengths=distances))
                hmol.set_constraint(constraints)
                if partial:
                    try:
                        opt = BFGS(hmol)
                        opt.run(steps=2)
                    except:
                        pass
                dihed += float(increment)
                self.calculator.set_calculator(hmol, 'high')
                hmol._calc.minimise_stable_write(dihedral=b, path="Hind" + str(count), title="H" + str(i), atoms=hmol, rigid=rigid)
                self.calculator.set_calculator(hmol, 'low')
                hinderance_traj.append(hmol.copy())
            write('test' + str(count) + '.xyz', hinderance_traj)
        os.chdir(current_dir)

    def write_conformers(self,mol, rigid=False, increment = 60, directory="conformers"):
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
                dihed += float(increment)
                hmol._calc.minimise_stable_write( path = "Hind"+str(count), title="H" +str(i), atoms= hmol)


    def read_conformer_files(self, path):
        os.chdir(path)
        os.chdir('conformers')
        rotors =len([f for f in os.listdir(".") if os.path.isdir(f)])
        min_ene = np.inf
        conformers =[]
        ene_list = []
        minimum = None
        for i in range(0,rotors):
            os.chdir("Hind" + str(i))
            steps = len([f for f in os.listdir(".") if f.endswith('.log')])
            names = []
            for j in range(0,steps):
                mol = read("H" +str(j)+ ".log")
                ene = mol.get_potential_energy()
                names.append(tl.getSMILES(mol,False,False))
                if ene < min_ene and j > 0 and names[j] == names[0]:
                    min_ene = ene
                    minimum = mol.copy()
                if ene not in ene_list and j > 0 and names[j] == names[0]:
                    ene_list.append(ene)
                    conformers.append(mol.copy())
            os.chdir('../')
        write("conformers.xyz", conformers)
        np.savetxt("energies.txt", ene_list, delimiter="\n")
        os.chdir('../')
        write("min.xyz", minimum)


    def read_hindered_files(self, path, index=-1):
        os.chdir(path)
        os.chdir('hindered_rotor')
        rotors =len([f for f in os.listdir(".") if os.path.isdir(f)])
        for i in range(0,rotors):
            os.chdir("Hind" + str(i))
            steps = len([f for f in os.listdir(".") if f.endswith('.log')])
            hinderance_potential = []
            hinderance_traj = []
            hinderance_angles = []
            dihedral = tl.read_mod_redundant('H0.com')
            for j in range(0,steps):
                hind_mol = read("H" +str(j)+ ".log", index=index)
                try:
                    ene = hind_mol.get_potential_energy()
                except:
                    ene = hinderance_potential[-1]
                if j == 0:
                    baseline =  ene * mol / kJ
                hinderance_potential.append((ene * mol / kJ)-baseline)
                hinderance_traj.append(hind_mol.copy())
                hinderance_angles.append(hind_mol.get_dihedral(int(dihedral[0])-1,int(dihedral[1])-1,int(dihedral[2])-1,int(dihedral[3])-1))
            self.hinderance_trajectories.append(hinderance_traj)
            self.hinderance_potentials.append(hinderance_potential)
            self.hinderance_angles.append(hinderance_angles)
            self.hinderance_indexes.append([dihedral[1],dihedral[2]])
            write('traj.xyz',hinderance_traj)
            np.savetxt("energies.txt", hinderance_potential, delimiter="\n")
            np.savetxt("angles.txt", hinderance_angles, delimiter="\n")
            os.chdir('../')
        os.chdir('../')


    def write_multi_dimensional_torsion(self,mol, increment = 18, directory="hindered_rotor", rotors_to_exclude = None, rigid = False, ts =False, all_bonds = False, combine=True, explicit_bond=None):
        current_dir = os.getcwd()
        print(str(rigid))
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)
        os.makedirs(directory, exist_ok=True)
        os.chdir(directory)
        if explicit_bond != None:
            self.bonds_to_add.append(explicit_bond)
        rotatable_bonds, coId = CT.get_rotatable_bonds(mol,self.bonds_to_add, combine=combine)
        self.rotor_indexes = rotatable_bonds
        distances = []
        for bond in self.bonds_to_add:
            dist = mol.get_distance(bond[0],bond[1])
            distances.append(dist)
            print(str(self.bonds_to_add))
        if rotors_to_exclude != None:
            del rotatable_bonds[rotors_to_exclude]
            del coId[rotors_to_exclude]
        diheds=[]
        for b in rotatable_bonds:
            dihed = mol.get_dihedral(*b)
            diheds.append(dihed)
            print(str(rotatable_bonds))
        if all_bonds:
            bonds_to_fix = ['All']
        else:
            bonds_to_fix = self.bonds_to_add
        os.makedirs('MultiHind', exist_ok=True)
        hmol = mol.copy()
        self.calculator.set_calculator(hmol, 'high')
        rng = 360.0 / float(increment)
        if len(rotatable_bonds) > 2:
            for i in range(0, int(rng)):
                for j in range(0, int(rng)):
                    for l in range(0, int(rng)):
                        hmol.set_dihedral(rotatable_bonds[0][0], rotatable_bonds[0][1], rotatable_bonds[0][2],
                                          rotatable_bonds[0][3], diheds[0], indices=coId[0])
                        hmol.set_dihedral(rotatable_bonds[1][0], rotatable_bonds[1][1], rotatable_bonds[1][2],
                                          rotatable_bonds[1][3], diheds[1], indices=coId[1])
                        hmol.set_dihedral(rotatable_bonds[2][0], rotatable_bonds[2][1], rotatable_bonds[2][2],
                                          rotatable_bonds[2][3], diheds[2], indices=coId[2])
                        if ts:
                            hmol._calc.minimise_ts_write(dihedral=rotatable_bonds,fixed_bonds=bonds_to_fix, path="MultiHind",
                                                     title="H" + str(i) + '_' + str(j) + '_' + str(l), atoms=hmol,
                                                     rigid=rigid)
                        else:
                            hmol._calc.minimise_stable_write(dihedral=rotatable_bonds, path="MultiHind",
                                                     title="H" + str(i) + '_' + str(j) + '_' + str(l), atoms=hmol,
                                                     rigid=rigid)
                        diheds[2] += float(increment)
                    diheds[1] += float(increment)
                diheds[0] += float(increment)
        elif len(rotatable_bonds) == 2:
            for j in range(0, int(rng)):
                for l in range(0, int(rng)):
                    hmol.set_dihedral(rotatable_bonds[0][0], rotatable_bonds[0][1], rotatable_bonds[0][2],
                                      rotatable_bonds[0][3], diheds[0], indices=coId[0])
                    hmol.set_dihedral(rotatable_bonds[1][0], rotatable_bonds[1][1], rotatable_bonds[1][2],
                                      rotatable_bonds[1][3], diheds[1], indices=coId[1])
                    if ts:
                        hmol._calc.minimise_ts_write(dihedral=rotatable_bonds, fixed_bonds=bonds_to_fix, path="MultiHind",
                                                 title="H" + str(j) + '_' + str(l), atoms=hmol,
                                                 rigid=rigid)
                    else:
                        hmol._calc.minimise_stable_write(dihedral=rotatable_bonds, path="MultiHind",
                                                     title="H" + str(j) + '_' + str(l), atoms=hmol,
                                                     rigid=rigid)
                    diheds[1] += float(increment)
                diheds[0] += float(increment)

        os.chdir(current_dir)

    def read_multi_dimensional_torsion2D(self,path, f_coeffs = 5, index=-1,smooth=5000000, combine=True, explicit_bond=None, cutoff=np.inf):
        os.chdir(path)
        os.chdir('hindered_rotor')
        os.chdir('MultiHind')
        nmol = read("H0_0.log")
        baseline = nmol.get_potential_energy()
        steps = np.sqrt(len([f for f in os.listdir(".") if f.endswith('.log')]))
        rot_array_2D = []
        ene_arr_1D =[]
        angle_arr_1D = []
        dihedrals = tl.read_mod_redundant2d('H0_0.com')
        print(dihedrals)
        for i in range(0,int(steps)):
            arr = []
            traj =[]
            angle = []
            for j in range(0,int(steps)):
                try:
                    hmol = read("H" + str(i) + '_' + str(j) + ".log", index=index)
                    ene = (hmol.get_potential_energy() - baseline) / (invcm)
                except:
                    print("error getting energy for file H" + str(i) + '_' + str(j) + ".log")
                    hmol = read("H" + str(i) + '_' + str(j) + ".log", index=0)
                    if not (i<2 or j<2):
                        ene = ene_arr_1D[-1] + 2 * (ene_arr_1D[-1] - ene_arr_1D[-2])
                    else:
                        ene = (hmol.get_potential_energy() - baseline) / (invcm)

                #if not tl.check_gaussian("H" + str(i) + '_' + str(j) + ".log") or ene < 0:
                    #ene = ene_arr_1D[-1]

                if ene > cutoff:
                    ene = ene_arr_1D[-1]+3*(ene_arr_1D[-1] - ene_arr_1D[-2])

                if j > 1 and np.abs((ene - ene_arr_1D[-1]))  > np.abs((smooth*(ene_arr_1D[-1]-ene_arr_1D[-2]))):
                    if (ene_arr_1D[-1]-ene_arr_1D[-2]) == 0:
                        print('passing')
                        pass
                    else:

                        ene = ene_arr_1D[-1]+3*(ene_arr_1D[-1] - ene_arr_1D[-2])
                        print('new E = ' + str(ene))
                        print('old E = ' + str(ene_arr_1D[-1]))

                #if j == 0 and i > 1 and np.abs((ene - ene_arr_1D[-int(steps)]))  > np.abs((smooth*(ene_arr_1D[-int(steps)]-ene_arr_1D[-2*int(steps)]))):
                    #if (ene_arr_1D[-int(steps)]-ene_arr_1D[-2*int(steps)]) == 0:
                        #pass
                    #else:
                        #ene = ene_arr_1D[-int(steps)] + 0.5 * (ene_arr_1D[-int(steps)] - ene_arr_1D[-int(steps)+1])

                arr.append(ene)
                ene_arr_1D.append(ene)
                a =[]
                a.append(np.radians(hmol.get_dihedral(int(dihedrals[0][0]) - 1, int(dihedrals[0][1]) - 1, int(dihedrals[0][2]) - 1,int(dihedrals[0][3]) - 1)))
                a.append(np.radians(hmol.get_dihedral(int(dihedrals[1][0]) - 1, int(dihedrals[1][1]) - 1, int(dihedrals[1][2]) - 1,int(dihedrals[1][3]) - 1)))
                angle.append(a)
                angle_arr_1D.append(a)
                traj.append(hmol.copy())
            #ene_arr_1D.append(arr[0])
            #angle_arr_1D.append(angle[0])
            write('../T'+str(i)+'.xyz',traj)
            rot_array_2D.append(arr)
        #for i in range(0,int(steps+1)):
            #ene_arr_1D.append(ene_arr_1D[i])
            #angle_arr_1D.append(angle_arr_1D[i])
        print(np.sqrt(len(ene_arr_1D)))
        min_chi = np.inf
        min_check =[]
        min_coeffs = []
        min_cos = []
        chi_surface2d = []
        for i in range(1,f_coeffs):
            chi_surface = []
            for j in range(1,f_coeffs):
                coeffs=tl.fitFourier2D(ene_arr_1D, angle_arr_1D, [i,j])
                check = []
                chi = 0
                for a, e in zip(angle_arr_1D, ene_arr_1D):
                    fit_ene = tl.Fourier2D(coeffs, a, [i,j])
                    chi += abs(fit_ene - e) / (80)
                    check.append([fit_ene, e])
                if chi < min_chi:
                    min_chi = chi
                    min_check = check
                    min_coeffs = coeffs
                    min_cos = [i,j]
                chi_surface.append(chi)
                print(str(chi))
                print(str(i) + ' ' + str(j))
            chi_surface2d.append(chi_surface)


        os.chdir('../')
        np.savetxt('multi1.txt', rot_array_2D, delimiter='\t')
        np.savetxt('chi_surface.txt', chi_surface2d, delimiter='\t')
        np.savetxt('coeffs1.txt', min_coeffs[0][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('angles.txt', angle_arr_1D, delimiter=' ', fmt='%4.4f')
        np.savetxt('comparison.txt', min_check, delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs2.txt', min_coeffs[1][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs3.txt', min_coeffs[2][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs4.txt', min_coeffs[3][:], delimiter=' ', fmt='%4.4f')
        os.chdir('../')

    def read_multi_dimensional_torsion3D(self,path, f_coeffs = 6, index=-1, sin=[True,True,True], cos=[True,True,True], limit=np.inf, smooth =100000000):
        os.chdir(path)
        os.chdir('hindered_rotor')
        os.chdir('MultiHind')
        min_mol = None
        min_idxs = []
        min_ene = np.inf
        steps = np.cbrt(len([f for f in os.listdir(".") if f.endswith('.log')]))
        print(steps)
        ene_arr_1D_temp =[]
        angle_arr_1D = []
        ene_arr_2Ds = []
        dihedrals = tl.read_mod_redundant3d('H0_0_0.com')
        hmol = read("H0_0_0.log", index=index)
        a1 = np.radians(hmol.get_dihedral(int(dihedrals[0][0]) - 1, int(dihedrals[0][1]) - 1, int(dihedrals[0][2]) - 1,
                              int(dihedrals[0][3]) - 1))
        a2 = np.radians(hmol.get_dihedral(int(dihedrals[1][0]) - 1, int(dihedrals[1][1]) - 1, int(dihedrals[1][2]) - 1,
                              int(dihedrals[1][3]) - 1))
        a3 = np.radians(hmol.get_dihedral(int(dihedrals[2][0]) - 1, int(dihedrals[2][1]) - 1, int(dihedrals[2][2]) - 1,
                              int(dihedrals[2][3]) - 1))
        print(dihedrals)
        for i in range(0,int(steps)):
            ene_arr_2D = []
            for j in range(0,int(steps)):
                arr = []
                for k in range(0,int(steps)):
                    try:
                        hmol = read("H" + str(i) + '_' + str(j) + '_' + str(k)+ ".log", index=index)
                        ene = (hmol.get_potential_energy()) / (invcm)
                    except:
                        try:
                            hmol = read("H" + str(i) + '_' + str(j) + '_' + str(k)+ ".log", index=-2)
                            ene = (hmol.get_potential_energy()) / (invcm)
                        except:
                            print("error getting energy for file H" + str(i) + '_' + str(j) + '_' + str(k)+ ".log")
                            ene = ene_arr_1D_temp[-1]
                    if ene- min_ene > limit:
                        ene = ene_arr_1D_temp[-1]

                    if k > 1 and np.abs((ene - ene_arr_1D_temp[-1])) > np.abs((smooth * (ene_arr_1D_temp[-1] - ene_arr_1D_temp[-2]))):
                        if (ene_arr_1D_temp[-1] - ene_arr_1D_temp[-2]) == 0:
                            print('passing')
                            pass
                        else:

                            ene = ene_arr_1D_temp[-1] + 3 * (ene_arr_1D_temp[-1] - ene_arr_1D_temp[-2])
                            print('new E = ' + str(ene))
                            print('old E = ' + str(ene_arr_1D_temp[-1]))
                    ene_arr_1D_temp.append(ene)
                    if ene < min_ene:
                        min_ene = ene
                        min_mol = hmol.copy()
                        min_idxs = [i,j,k]
                    a = []
                    arr.append(ene)
                    a.append(np.radians(
                        hmol.get_dihedral(int(dihedrals[0][0]) - 1, int(dihedrals[0][1]) - 1, int(dihedrals[0][2]) - 1,
                                          int(dihedrals[0][3]) - 1)))
                    a.append(np.radians(
                        hmol.get_dihedral(int(dihedrals[1][0]) - 1, int(dihedrals[1][1]) - 1, int(dihedrals[1][2]) - 1,
                                          int(dihedrals[1][3]) - 1)))
                    a.append(np.radians(
                        hmol.get_dihedral(int(dihedrals[2][0]) - 1, int(dihedrals[2][1]) - 1, int(dihedrals[2][2]) - 1,
                                          int(dihedrals[2][3]) - 1)))
                    angle_arr_1D.append(a)
                ene_arr_2D.append(arr)
            ene_arr_2Ds.append(ene_arr_2D)
        ene_arr_1D = [x - min_ene for x in ene_arr_1D_temp]

        min_chi = np.inf
        min_check =[]
        min_cos = []
        min_coeffs = []
        for i in range(1,f_coeffs):
            for j in range(1,f_coeffs):
                for k in range(1,f_coeffs):
                    coeffs=tl.fitFourier3D(ene_arr_1D, angle_arr_1D, [i,j,k])
                    check = []
                    chi = 0
                    for a, e in zip(angle_arr_1D, ene_arr_1D):
                        fit_ene = tl.Fourier3D(coeffs, a, [i,j,k], sin, cos)
                        chi += abs(fit_ene - e) / (80)
                        check.append([fit_ene, e])
                    if chi < min_chi:
                        min_chi = chi
                        min_check = check
                        min_cos = [i,j,k]
                        min_coeffs = coeffs

                    print(str(chi))
                    print(str(i)+' '+str(j)+' '+str(k))
        os.chdir('../')
        print('minimum chi = ' + str(min_chi))
        print('idxs = ' + str(min_cos) )
        write('newMin.xyz', min_mol)
        array_2D = np.asarray(ene_arr_2Ds)
        arr_reshaped = array_2D.reshape(array_2D.shape[0], -1)
        np.savetxt('full_array.txt', arr_reshaped, delimiter=' ', fmt='%4.4f')
        for i,ar in enumerate(ene_arr_2Ds):
            np.savetxt('array'+str(i)+'.txt', ar, delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs1.txt', min_coeffs[0][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('angles.txt', angle_arr_1D, delimiter=' ', fmt='%4.4f')
        np.savetxt('comparison.txt', min_check, delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs2.txt', min_coeffs[1][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs3.txt', min_coeffs[2][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs4.txt', min_coeffs[3][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs5.txt', min_coeffs[4][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs6.txt', min_coeffs[5][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs7.txt', min_coeffs[6][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs8.txt', min_coeffs[7][:], delimiter=' ', fmt='%4.4f')
        os.chdir('../')

    def read_multi_dimensional_torsion3D_2(self,path, f_coeffs = 6, index=-1, sin=[True,True,True], cos=[True,True,True], limit=np.inf):
        os.chdir(path)
        os.chdir('hindered_rotor')
        os.chdir('MultiHind')
        min_mol = None
        min_idxs = []
        min_ene = np.inf
        steps = np.cbrt(len([f for f in os.listdir(".") if f.endswith('.log')]))
        print(steps)
        ene_arr_1D_temp =[]
        angle_arr_1D = []
        ene_arr_2Ds = []
        dihedrals = tl.read_mod_redundant3d('H0_0_0.com')
        hmol = read("H0_0_0.log", index=-1)
        baseline = (hmol.get_potential_energy()) / (invcm)
        print(dihedrals)
        for i in range(0,int(steps)):
            ene_arr_2D = []
            for j in range(0,int(steps)):
                arr = []
                for k in range(0,int(steps)):
                    try:
                        try:
                            hmol = read("H" + str(i) + '_' + str(j) + '_' + str(k)+ ".log", index=index)
                        except:
                            hmol = read("H" + str(i) + '_' + str(j) + '_' + str(k) + ".log", index=-1)
                        ene = (hmol.get_potential_energy()) / (invcm)
                    except:
                        try:
                            print("minor error getting energy for file H" + str(i) + '_' + str(j) + '_' + str(k) + ".log")
                            hmol = read("H" + str(i) + '_' + str(j) + '_' + str(k)+ ".log", index=0)
                            ene = ene_arr_1D_temp[-1]
                        except:
                            print("error getting energy for file H" + str(i) + '_' + str(j) + '_' + str(k)+ ".log")
                            ene = ene_arr_1D_temp[-1]
                    if ene- min_ene > limit:
                        ene = ene_arr_1D_temp[-1]
                    ene_arr_1D_temp.append(ene)
                    if ene < min_ene:
                        min_ene = ene
                        min_mol = hmol.copy()
                        min_idxs = [i,j,k]
                    a = []
                    arr.append(ene)
                    a.append(np.radians(
                        hmol.get_dihedral(int(dihedrals[0][0]) - 1, int(dihedrals[0][1]) - 1, int(dihedrals[0][2]) - 1,
                                          int(dihedrals[0][3]) - 1)))
                    a.append(np.radians(
                        hmol.get_dihedral(int(dihedrals[1][0]) - 1, int(dihedrals[1][1]) - 1, int(dihedrals[1][2]) - 1,
                                          int(dihedrals[1][3]) - 1)))
                    a.append(np.radians(
                        hmol.get_dihedral(int(dihedrals[2][0]) - 1, int(dihedrals[2][1]) - 1, int(dihedrals[2][2]) - 1,
                                          int(dihedrals[2][3]) - 1)))
                    angle_arr_1D.append(a)
                ene_arr_2D.append(arr)
            ene_arr_2Ds.append(ene_arr_2D)
        ene_arr_1D = [x - min_ene for x in ene_arr_1D_temp]

        min_chi = np.inf
        min_check =[]
        min_cos = []
        min_coeffs = []
        coeffs=tl.fitFourier3D(ene_arr_1D, angle_arr_1D, [f_coeffs,f_coeffs,f_coeffs])
        check = []
        chi = 0
        for a, e in zip(angle_arr_1D, ene_arr_1D):
            fit_ene = tl.Fourier3D(coeffs, a, [f_coeffs,f_coeffs,f_coeffs], sin, cos)
            chi += abs(fit_ene - e) / (80)
            check.append([fit_ene, e])
        if chi < min_chi:
            min_chi = chi
            min_check = check
            min_cos = [f_coeffs,f_coeffs,f_coeffs]
            min_coeffs = coeffs

        print(str(chi))
        os.chdir('../')
        print('minimum chi = ' + str(min_chi))
        print('idxs = ' + str(min_cos) )
        write('newMin.xyz', min_mol)
        array_2D = np.asarray(ene_arr_2Ds)
        arr_reshaped = array_2D.reshape(array_2D.shape[0], -1)
        np.savetxt('full_array.txt', arr_reshaped, delimiter=' ', fmt='%4.4f')
        for i,ar in enumerate(ene_arr_2Ds):
            np.savetxt('array'+str(i)+'.txt', ar, delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs1.txt', min_coeffs[0][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('angles.txt', angle_arr_1D, delimiter=' ', fmt='%4.4f')
        np.savetxt('comparison.txt', min_check, delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs2.txt', min_coeffs[1][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs3.txt', min_coeffs[2][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs4.txt', min_coeffs[3][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs5.txt', min_coeffs[4][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs6.txt', min_coeffs[5][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs7.txt', min_coeffs[6][:], delimiter=' ', fmt='%4.4f')
        np.savetxt('coeffs8.txt', min_coeffs[7][:], delimiter=' ', fmt='%4.4f')
        os.chdir('../')
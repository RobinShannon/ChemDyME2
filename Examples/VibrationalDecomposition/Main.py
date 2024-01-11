import src.bxd.collective_variable as CV
from ase.constraints import Hookean
import scipy
import src.bxd.ProgressMetric as PM
import src.molecular_dynamics.md_Integrator as MD
import src.bxd.bxd_constraint as BXD
import src.molecular_dynamics.trajectory as Traj
import src.molecular_dynamics.md_logger as lg
import src.mechanism_generation.reaction_crtieria as RC
import src.utility.tools as Tl
from copy import deepcopy
import numpy as np
from ase.io import read
from ase.optimize import BFGS as bfgs
from ase.vibrations import Vibrations
from ase.io import write
from src.Calculators.NNCalculator import NNCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import sys
from ase.vibrations import Vibrations
from ase.md.verlet import VelocityVerlet
from ase import units
import random
from sella import Sella
from ase.optimize import BFGS
import math
from ase.md import MDLogger
import os
import copy
from ase.constraints import FixAtoms


def convert_hessian_to_cartesian(Hess, m):
    masses = np.tile(m, (len(m), 1))
    reduced_mass = 1 / masses ** 0.5
    H = np.multiply(Hess, reduced_mass.T)
    sum_of_rows = H.sum(axis=0)
    normalized_array = H / sum_of_rows[np.newaxis, :]
    return normalized_array


def generate_displacements(mol, disp, rand_dis, seccond_order):
    copy = mol.copy()
    atoms = len(mol.get_atomic_numbers())
    positions = copy.get_positions()
    traj = []
    for i in range(0, atoms):
        for j in range(0, 3):
            positions[i][j] += disp
            copy.set_positions(positions)
            traj.append(copy.copy())
            positions[i][j] -= disp
            positions[i][j] -= disp
            copy.set_positions(positions)
            traj.append(copy.copy())
            positions[i][j] += disp
    if rand_dis:
        for i in range(0, atoms):
            for j in range(0, 3):
                r_i = random.randint(0, atoms - 1)
                r_j = random.randint(0, 2)
                positions[r_i][r_j] += disp
                positions[i][j] += disp
                copy.set_positions(positions)
                traj.append(copy.copy())
                positions[i][j] -= disp
                positions[i][j] -= disp
                copy.set_positions(positions)
                traj.append(copy.copy())
                positions[i][j] += disp
                positions[r_i][r_j] -= disp
    if seccond_order:
        for i in range(0, atoms):
            for j in range(0, 3):
                for i_2 in range(0, atoms):
                    for j_2 in range(0, 3):
                        positions[i][j] += disp
                        positions[i_2][j_2] += disp
                        copy.set_positions(positions)
                        traj.append(copy.copy())
                        positions[i_2][j_2] -= disp
                        positions[i_2][j_2] -= disp
                        copy.set_positions(positions)
                        traj.append(copy.copy())
                        positions[i_2][j_2] += disp
                        positions[i][j] -= disp
    return traj


def get_rot_tran(coord_true, coord_pred):
    """
    Given two matrices, return a rotation and translation matrix to move
    pred coords onto true coords.
    Largely based on SVDSuperimposer implementation from BioPython with some tweaks.
    """
    centroid_pred = np.sum(coord_pred, axis=0) / coord_pred.shape[0]
    centroid_true = np.sum(coord_true, axis=0) / coord_true.shape[0]

    p_prime = coord_pred - centroid_pred
    q_prime = coord_true - centroid_true

    W = np.dot(q_prime.T, p_prime)
    U, S, Vt = np.linalg.svd(W)

    V = Vt.T

    rot = np.dot(V, U.T)
    det = np.linalg.det(rot)

    # The determinant is needed to detect whether we need a right-hand coordinate system or not
    # This basically means we just have to flip the Z-axis
    if det < 0:
        Vt[:, 2] = -Vt[:, 2]
        V = Vt.T
        rot = np.dot(V, U.T)

    model_coords_rotated = np.dot(coord_pred, rot)

    return rot, model_coords_rotated


narupa_mol = read('MethylFormate/NN_TS.xyz', index=0)
narupa_mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-540000', atoms=narupa_mol))


f4 = open("vibresults4.txt", 'a')


prodfile = open('product_file.txt', 'a')
for run in range(0, 1000):
    # The general set-up here is identical to the adaptive run to ensure the converging run samples the same path
    # narupa_mol = read('water', index=0)
    temp = read('MethylFormate/NN_TS.xyz', index="0")
    dyn = Sella(temp, internal=True)
    try:
        dyn.run(1e-2, 1000)
    except:
        pass
    narupa_mol.set_positions(temp.get_positions())
    velocities = []
    MaxwellBoltzmannDistribution(narupa_mol, temperature_K=0.000000000001, force_temp=True)
    print(str(narupa_mol.get_kinetic_energy()))
    constraints = []
    constraints.append(FixAtoms(indices=[5, 6, 8]))
    narupa_mol.set_constraint(constraints)
    dyn = VelocityVerlet(narupa_mol, 0.1 * units.fs)
    dyn.attach(MDLogger(dyn, narupa_mol, 'md.log', header=False, stress=False,
                        peratom=False, mode="a"), interval=10)
    dyn.run(10000)
    del narupa_mol.constraints
    print(str(narupa_mol.get_total_energy()))
    for i in range(0, 4000):
        dyn.run(5)
        velocities.append(narupa_mol.get_velocities())
        print(str(narupa_mol.get_total_energy()))

    name = Tl.getSMILES(narupa_mol, False)
    print('Product ' + str(name) + ' found')
    prodfile.write('Product ' + str(name) + ' found\n')
    prodfile.flush()
    if (len(name) == 2 and name[1] == "O") or (len(name) == 3 and name[2] == "O"):

        narupa_mol1 = narupa_mol.copy()
        narupa_mol2 = narupa_mol.copy()

        del (narupa_mol2[-1])
        del (narupa_mol2[-1])
        del (narupa_mol2[-2])

        del (narupa_mol1[0])
        del (narupa_mol1[0])
        del (narupa_mol1[0])
        del (narupa_mol1[0])
        del (narupa_mol1[0])
        del (narupa_mol1[0])
        del (narupa_mol1[1])

        mod_vels = []
        mod_vels2 = []

        for vel in velocities:
            a = deepcopy(vel)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 1, 0)

            b = deepcopy(vel)
            b = np.delete(b, -1, 0)
            b = np.delete(b, -1, 0)
            b = np.delete(b, -2, 0)

            mod_vels.append(a)
            mod_vels2.append(b)

        # get ref mol

        ref = read('water.xyz')
        Hess = np.load('water.npy')

        ref2 = read('co.xyz')
        Hess2 = np.load('co.npy')

        masses = ((np.tile(narupa_mol1.get_masses(), (3, 1))).transpose()).flatten()
        # Hess = convert_hessian_to_cartesian(H,masses)
        Linv = np.linalg.inv(Hess)

        masses2 = ((np.tile(narupa_mol2.get_masses(), (3, 1))).transpose()).flatten()
        # Hess2 = convert_hessian_to_cartesian(H2,masses2)
        Linv2 = np.linalg.inv(Hess2)
        check_vels = Hess2


        vibs = [0] * 9
        overall_K_e = 0
        time_profile1 = []
        for vel in mod_vels:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            temp_vibs = [0] * 9
            for i in range(0, 9):
                kinetic = 0
                for j in range(0, 9):
                    kinetic += ((masses[j] / 2) * (Hess[j, i] * Q_dot[i]) ** 2)
                vibs[i] += kinetic
                temp_vibs[i] = kinetic
            time_profile1.append(temp_vibs)
        overall_K_e /= len(mod_vels)
        vibsav = [v / len(mod_vels) for v in vibs]

        overall_K_e2 = 0
        vibs2 = [0] * 21
        time_profile2 = []
        for vel in mod_vels2:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2, Q_dot2)
            overall_K_e2 += 0.5 * np.dot(masses2 * q_dot, q_dot)
            temp_vibs2 = [0] * 21
            for i in range(0, 21):
                kinetic = 0
                for j in range(0, 21):
                    kinetic += ((masses2[j] / 2) * (Hess2[j, i] * Q_dot2[i]) ** 2)
                vibs2[i] += kinetic
                temp_vibs2[i] = kinetic
            time_profile2.append(temp_vibs2)
        vibs2av = [v / len(mod_vels) for v in vibs2]
        overall_K_e2 /= len(mod_vels)


        i = 0
        while i < 5000:
            if not os.path.exists('Time_profiles/time_profile' + str(i) + '.txt'):
                with open('Time_profiles/time_profile' + str(i) + '.txt', 'w') as tp:
                    for time in time_profile1:
                        for t in time:
                            tp.write(str(t) + '\t')
                        tp.write('\n')
                break
            else:
                i += 1
        i = 0
        while i < 5000:
            if not os.path.exists('Time_profiles/time_profile_co' + str(i) + '.txt'):
                with open('Time_profiles/time_profile_co' + str(i) + '.txt', 'w') as tp:
                    for time in time_profile2:
                        for t in time:
                            tp.write(str(t) + '\t')
                        tp.write('\n')
                break
            else:
                i += 1


        vibs = [0] * 9
        overall_K_e = 0
        for vel in mod_vels[2000:]:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            for i in range(0, 9):
                kinetic = 0
                for j in range(0, 9):
                    kinetic += ((masses[j] / 2) * (Hess[j, i] * Q_dot[i]) ** 2)
                vibs[i] += kinetic
        overall_K_e /= 2000
        vibsav = [v / 2000 for v in vibs]

        overall_K_e2 = 0
        vibs2 = [0] * 21
        for vel in mod_vels2[2000:]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2, Q_dot2)
            overall_K_e2 += 0.5 * np.dot(masses2 * q_dot, q_dot)
            for i in range(0, 21):
                kinetic = 0
                for j in range(0, 21):
                    kinetic += ((masses2[j] / 2) * (Hess2[j, i] * Q_dot2[i]) ** 2)
                vibs2[i] += kinetic
        vibs2av = [v / 2000 for v in vibs2]
        overall_K_e2 /= 2000
        f3.write("Vib E (H20)  =\t")
        for v in vibsav:
            f3.write(str(v) + '\t')
        f3.write("Vib E (co)  = \t")
        for v in vibs2av:
            f3.write(str(v) + '\t')
        f3.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f3.write("\n")
        f3.flush()

        vibs = [0] * 9
        overall_K_e = 0
        for vel in mod_vels[1000:]:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            for i in range(0, 9):
                kinetic = 0
                for j in range(0, 9):
                    kinetic += ((masses[j] / 2) * (Hess[j, i] * Q_dot[i]) ** 2)
                vibs[i] += kinetic
        overall_K_e /= 3000
        vibsav = [v / 3000 for v in vibs]

        overall_K_e2 = 0
        vibs2 = [0] * 21
        for vel in mod_vels2[1000:]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2, Q_dot2)
            overall_K_e2 += 0.5 * np.dot(masses2 * q_dot, q_dot)
            for i in range(0, 21):
                kinetic = 0
                for j in range(0, 21):
                    kinetic += ((masses2[j] / 2) * (Hess2[j, i] * Q_dot2[i]) ** 2)
                vibs2[i] += kinetic
        vibs2av = [v / 3000 for v in vibs2]
        overall_K_e2 /= 3000
        f4.write("Vib E (H20)  =\t")
        for v in vibsav:
            f4.write(str(v) + '\t')
        f4.write("Vib E (co)  = \t")
        for v in vibs2av:
            f4.write(str(v) + '\t')
        f4.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f4.write("\n")
        f4.flush()

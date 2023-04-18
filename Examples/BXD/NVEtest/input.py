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


narupa_mol = read('Start.xyz', index=0)
narupa_mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-580000', atoms=narupa_mol))
f = open("vibresults.txt", 'a')
f2 = open("vibresults2.txt", 'a')
f3 = open("vibresults3.txt", 'a')
f4 = open("vibresults4.txt", 'a')
prodfile = open('product_file.txt', 'a')
for run in range(0, 1000):
    # The general set-up here is identical to the adaptive run to ensure the converging run samples the same path
    # narupa_mol = read('water', index=0)
    temp = read('Start.xyz', index="0")
    narupa_mol.set_positions(temp.get_positions())
    ##dyn = Langevin(narupa_mol, .5 * units.fs, 291, 1)
    # dyn.run(1000)
    pcs = 2
    collective_variable_therm = CV.Distances(narupa_mol, [[6, 3]])
    progress_metric_therm = PM.Line(collective_variable_therm, [10], [11])
    bxd_therm = BXD.Fixed(progress_metric_therm)
    md_therm = MD.VelocityVerlet(narupa_mol, temperature=500, timestep=0.1)
    bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd_therm], md_therm)
    bxd_trajectory.run_trajectory(max_steps=10000)

    narupa_mol.set_positions(bxd_trajectory.mol.get_positions())
    narupa_mol.set_velocities(bxd_trajectory.mol.get_velocities())
    collective_variable = CV.Distances(narupa_mol, [[6, 3]])
    progress_metric = PM.Line(collective_variable, [0], [1.70])
    bxd = BXD.Fixed(progress_metric)
    loggers = []
    lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t' + str(
        var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].s) / var.bxd_list[
            0].progress_metric.project_point_on_path(var.bxd_list[0].progress_metric.end))
    tf1 = lambda var: var.mdsteps % 100 == 0
    log1 = lg.MDLogger(logging_function=lf1, triggering_function=tf1)
    loggers.append(log1)
    lf2 = lambda var: "hit!!\t" + str(var.bxd_list[0].bound_hit) + "\tmdstep\t=\t" + str(var.mdsteps)
    tf2 = lambda var: var.bxd_list[0].inversion
    log2 = lg.MDLogger(logging_function=lf2, triggering_function=tf2)
    loggers.append(log2)
    try:
        os.remove(str(sys.argv[1]))
        os.remove('g' + str(sys.argv[1]))
    except:
        pass
    file = str('test.txt')
    md = MD.VelocityVerlet(narupa_mol, temperature=500, timestep=0.1)
    reaction_criteria = RC.NunezMartinez(narupa_mol, consistant_hit_steps=5, relaxation_steps=1)
    bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd], md, loggers=loggers, criteria=reaction_criteria, reactive=True,
                                     maxwell_boltzman=False)
    bxd_trajectory.run_trajectory(max_steps=10000000)
    ind = 0
    while ind < 5000:
        if not os.path.exists('TSs/structure' + str(ind) + '.xyz'):
            write('TSs/structure' + str(ind) + '.xyz', bxd_trajectory.mol)
            break
        else:
            ind += 1

    narupa_mol = bxd_trajectory.mol
    dyn = VelocityVerlet(bxd_trajectory.mol, 0.1 * units.fs)
    velocities = loggers[-1].lst[-1000:-1]
    txt_log = loggers[-2].lst
    for i in range(0, 4000):
        dyn.run(5)
        velocities.append(narupa_mol.get_velocities())
        out = '0\t' + str(narupa_mol.get_total_energy()) + '\t' + str(narupa_mol.get_potential_energy()) + '\t' + str(
            narupa_mol.get_kinetic_energy()) + '\n'
        txt_log.append(out)
        write(str(sys.argv[1]), narupa_mol, append=True)

    l = 0
    while l < 1000:
        if not os.path.exists('Logs/log' + str(l) + '.txt'):
            with open('Logs/log' + str(l) + '.txt', 'w') as tp:
                for lo in txt_log:
                    tp.write(str(lo))
            break
        else:
            l += 1

    name = Tl.getSMILES(narupa_mol, False)
    print('Product ' + str(name) + ' found')
    prodfile.write('Product ' + str(name) + ' found\n')
    prodfile.flush()
    if len(name) == 2 and name[1] == "O":

        narupa_mol1 = narupa_mol.copy()
        narupa_mol2 = narupa_mol.copy()
        narupa_mol3 = narupa_mol.copy()
        narupa_mol4 = narupa_mol.copy()

        del (narupa_mol2[-1])
        del (narupa_mol2[-1])
        del (narupa_mol2[-3])

        del (narupa_mol1[0])
        del (narupa_mol1[0])
        del (narupa_mol1[0])
        del (narupa_mol1[1])
        del (narupa_mol1[1])

        del (narupa_mol3[-1])
        del (narupa_mol3[-1])

        del (narupa_mol4[0])
        del (narupa_mol4[0])
        del (narupa_mol4[0])
        del (narupa_mol4[0])
        del (narupa_mol4[0])
        del (narupa_mol4[0])

        mod_vels = []
        mod_vels2 = []
        mod_vels3 = []
        mod_vels4 = []

        for vel in velocities:
            a = deepcopy(vel)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 0, 0)
            a = np.delete(a, 1, 0)
            a = np.delete(a, 1, 0)

            b = deepcopy(vel)
            b = np.delete(b, -1, 0)
            b = np.delete(b, -1, 0)
            b = np.delete(b, -3, 0)

            d = deepcopy(vel)
            d = np.delete(d, 0, 0)
            d = np.delete(d, 0, 0)
            d = np.delete(d, 0, 0)
            d = np.delete(d, 0, 0)
            d = np.delete(d, 0, 0)
            d = np.delete(d, 0, 0)

            c = deepcopy(vel)
            c = np.delete(c, -1, 0)
            c = np.delete(c, -1, 0)

            mod_vels.append(a)
            mod_vels2.append(b)
            mod_vels3.append(c)
            mod_vels4.append(d)

        # get ref mol

        ref = read('water.xyz')
        Hess = np.load('water.npy')

        ref2 = read('HCOCO_3removed.xyz')
        Hess2 = np.load('HCOCO.npy')

        ref3 = read('gly.xyz')
        Hess3 = np.load('gly.npy')

        ref4 = read('hydroxyl.xyz')
        Hess4 = np.load('hydroxyl.npy')

        masses = ((np.tile(narupa_mol1.get_masses(), (3, 1))).transpose()).flatten()
        # Hess = convert_hessian_to_cartesian(H,masses)
        Linv = np.linalg.inv(Hess)

        masses2 = ((np.tile(narupa_mol2.get_masses(), (3, 1))).transpose()).flatten()
        # Hess2 = convert_hessian_to_cartesian(H2,masses2)
        Linv2 = np.linalg.inv(Hess2)
        check_vels = Hess2

        masses3 = ((np.tile(narupa_mol3.get_masses(), (3, 1))).transpose()).flatten()
        # Hess2 = convert_hessian_to_cartesian(H2,masses2)
        Linv3 = np.linalg.inv(Hess3)
        check_vels = Hess3

        masses4 = ((np.tile(narupa_mol4.get_masses(), (3, 1))).transpose()).flatten()
        # Hess2 = convert_hessian_to_cartesian(H2,masses2)
        Linv4 = np.linalg.inv(Hess4)
        check_vels = Hess4

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
        vibs2 = [0] * 15
        time_profile2 = []
        for vel in mod_vels2:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2, Q_dot2)
            overall_K_e2 += 0.5 * np.dot(masses2 * q_dot, q_dot)
            temp_vibs2 = [0] * 15
            for i in range(0, 15):
                kinetic = 0
                for j in range(0, 15):
                    kinetic += ((masses2[j] / 2) * (Hess2[j, i] * Q_dot2[i]) ** 2)
                vibs2[i] += kinetic
                temp_vibs2[i] = kinetic
            time_profile2.append(temp_vibs2)
        vibs2av = [v / len(mod_vels) for v in vibs2]
        overall_K_e2 /= len(mod_vels)
        f.write("Vib E (H20)  =\t")
        for v in vibsav:
            f.write(str(v) + '\t')
        f.write("Vib E (co)  = \t")
        for v in vibs2av:
            f.write(str(v) + '\t')
        f.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f.write("\n")
        f.flush()

        overall_K_e3 = 0
        vibs3 = [0] * 18
        time_profile3 = []
        for vel in mod_vels3:
            q_dot = vel.flatten()
            Q_dot3 = np.matmul(Linv3, q_dot)
            check_vels = np.matmul(Hess3, Q_dot3)
            overall_K_e3 += 0.5 * np.dot(masses3 * q_dot, q_dot)
            temp_vibs3 = [0] * 18
            for i in range(0, 18):
                kinetic = 0
                for j in range(0, 18):
                    kinetic += ((masses3[j] / 2) * (Hess3[j, i] * Q_dot3[i]) ** 2)
                vibs3[i] += kinetic
                temp_vibs3[i] = kinetic
            time_profile3.append(temp_vibs3)
        vibs3av = [v / len(mod_vels) for v in vibs3]
        overall_K_e3 /= len(mod_vels)

        overall_K_e4 = 0
        vibs4 = [0] * 6
        time_profile4 = []
        for vel in mod_vels4:
            q_dot = vel.flatten()
            Q_dot4 = np.matmul(Linv4, q_dot)
            check_vels = np.matmul(Hess4, Q_dot4)
            overall_K_e4 += 0.5 * np.dot(masses4 * q_dot, q_dot)
            temp_vibs4 = [0] * 6
            for i in range(0, 6):
                kinetic = 0
                for j in range(0, 6):
                    kinetic += ((masses4[j] / 2) * (Hess4[j, i] * Q_dot4[i]) ** 2)
                vibs4[i] += kinetic
                temp_vibs4[i] = kinetic
            time_profile4.append(temp_vibs4)
        vibs4av = [v / len(mod_vels) for v in vibs4]
        overall_K_e4 /= len(mod_vels)

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

        i = 0
        while i < 5000:
            if not os.path.exists('Time_profiles/time_profile_R1' + str(i) + '.txt'):
                with open('Time_profiles/time_profile_R1' + str(i) + '.txt', 'w') as tp:
                    for time in time_profile3:
                        for t in time:
                            tp.write(str(t) + '\t')
                        tp.write('\n')
                break
            else:
                i += 1

        i = 0
        while i < 5000:
            if not os.path.exists('Time_profiles/time_profile_R2' + str(i) + '.txt'):
                with open('Time_profiles/time_profile_R2' + str(i) + '.txt', 'w') as tp:
                    for time in time_profile4:
                        for t in time:
                            tp.write(str(t) + '\t')
                        tp.write('\n')
                break
            else:
                i += 1

        vibs = [0] * 9
        overall_K_e = 0
        for vel in mod_vels[3000:]:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            for i in range(0, 9):
                kinetic = 0
                for j in range(0, 9):
                    kinetic += ((masses[j] / 2) * (Hess[j, i] * Q_dot[i]) ** 2)
                vibs[i] += kinetic
        overall_K_e /= 1000
        vibsav = [v / 1000 for v in vibs]

        overall_K_e2 = 0
        vibs2 = [0] * 15
        for vel in mod_vels2[3000:]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2, Q_dot2)
            overall_K_e2 += 0.5 * np.dot(masses2 * q_dot, q_dot)
            for i in range(0, 15):
                kinetic = 0
                for j in range(0, 15):
                    kinetic += ((masses2[j] / 2) * (Hess2[j, i] * Q_dot2[i]) ** 2)
                vibs2[i] += kinetic
        vibs2av = [v / 1000 for v in vibs2]
        overall_K_e2 /= 1000
        f2.write("Vib E (H20)  =\t")
        for v in vibsav:
            f2.write(str(v) + '\t')
        f2.write("Vib E (co)  = \t")
        for v in vibs2av:
            f2.write(str(v) + '\t')
        f2.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f2.write("\n")
        f2.flush()

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
        vibs2 = [0] * 15
        for vel in mod_vels2[2000:]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2, Q_dot2)
            overall_K_e2 += 0.5 * np.dot(masses2 * q_dot, q_dot)
            for i in range(0, 15):
                kinetic = 0
                for j in range(0, 15):
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
        vibs2 = [0] * 15
        for vel in mod_vels2[1000:]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2, Q_dot2)
            overall_K_e2 += 0.5 * np.dot(masses2 * q_dot, q_dot)
            for i in range(0, 15):
                kinetic = 0
                for j in range(0, 15):
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

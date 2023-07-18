try:
    import src.bxd.collective_variable as CV
except:
    pass
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
from ase.vibrations import Vibrations
from ase.md.verlet import VelocityVerlet
from ase import units
import random
from sella import Sella
from ase.optimize import BFGS

def convert_hessian_to_cartesian(Hess, m):
    masses = np.tile(m,(len(m),1))
    reduced_mass = 1/masses**0.5
    H = np.multiply(Hess,reduced_mass.T)
    sum_of_rows = H.sum(axis=0)
    normalized_array = H / sum_of_rows[np.newaxis, :]
    return normalized_array


def generate_displacements(mol, disp, rand_dis, seccond_order):
    copy = mol.copy()
    atoms = len(mol.get_atomic_numbers())
    positions = copy.get_positions()
    traj = []
    for i in range(0,atoms):
        for j in range(0,3):
            positions[i][j] += disp
            copy.set_positions(positions)
            traj.append(copy.copy())
            positions[i][j] -= disp
            positions[i][j] -= disp
            copy.set_positions(positions)
            traj.append(copy.copy())
            positions[i][j] += disp
    if rand_dis:
        for i in range(0,atoms):
            for j in range(0,3):
                r_i = random.randint(0,atoms-1)
                r_j = random.randint(0,2)
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
        for i in range(0,atoms):
            for j in range(0,3):
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

path_s1 = (read('FormF/Scan3.log',index=":"))
path_s2 =(read('FormF/Scan3_2.log',index=":"))
path_s3 = (read('FormF/Scan_3_3.log',index=":"))
path_s = path_s1 +path_s2+path_s3




for i, mol in enumerate(path_s):
    list = generate_displacements(mol.copy(), 0.05, rand_dis=True, seccond_order=False)
    write('FormF/Traj' + str(i) + '.xyz', list)



#ts.set_calculator(NNCalculator(checkpoint='best_model.ckpt-880000', atoms=ts))
#f = ts.get_forces()
# Set up a Sella Dynamics object
#dyn = Sella(ts, internal = True)
#try:
#    dyn.run(1e-3, 1000)
#except:
#    pass
#ts_ene = ts.get_potential_energy()*96.58
#write('MFTS.xyz', ts)
#ts.set_positions(GlyIRC[-1].get_positions())
#dyn = BFGS(ts)
#try:
#    dyn.run(1e-3, 1000)
#except:
#    pass
#comp_ene = ts.get_potential_energy()*96.58
#diff = ts_ene - comp_ene
#write('MFComp.xyz', ts)

#ts.set_positions(GlyIRC[47].get_positions())
#dyn = BFGS(ts)
#try:
#    dyn.run(1e-3, 1000)
#except:
#    pass
#comp_ene = ts.get_potential_energy()*96.58
#diff = ts_ene - comp_ene
#write('MFComp2.xyz', ts)
#narupa_mol = read('Start.xyz', index=0)
narupa_mol = read('MFGeoms/Start.xyz')
narupa_mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-410000', atoms=narupa_mol))
f =open("vibresults.txt", 'a')
f2 =open("vibresults2.txt", 'a')
f3 =open("vibresults3.txt", 'a')
f4 =open("vibresults4.txt", 'a')
for run in range(0, 1000):
    # The general set-up here is identical to the adaptive run to ensure the converging run samples the same path
    #narupa_mol = read('water', index=0)
    temp = read('MFGeoms/Start.xyz')
    narupa_mol.set_positions(temp.get_positions())
    ##dyn = Langevin(narupa_mol, .5 * units.fs, 291, 1)
    #dyn.run(1000)
    pcs = 2
    #collective_variable = CV.Distances(narupa_mol, [[1,3]])
    collective_variable = CV.Distances(narupa_mol, [[8,2]])
    progress_metric = PM.Line(collective_variable, [0], [1.8])

    #collective_variable = CV.COM(narupa_mol, [0,1,2,3,4,5], [6,7])
    #progress_metric = PM.Line(collective_variable, [0], 0.5)

    bxd = BXD.Fixed(progress_metric)
    loggers = []
    lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].s) /var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].progress_metric.end))
    tf1 = lambda var: var.mdsteps % 1000 == 0
    log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
    loggers.append(log1)
    file = 'geom.xyz'
    gfile = open('gtemp.xyz', 'a')
    lf3 = lambda var: str(write(file,var.mol, append=True))
    tf3 = lambda var: var.mdsteps % 100 == 0
    log3 = lg.MDLogger( logging_function=lf3, triggering_function=tf3, outpath=gfile)
    loggers.append(log3)


    md = MD.Langevin(narupa_mol, temperature=500, timestep=0.5, friction=0.01)
    reaction_criteria = RC.NunezMartinez(narupa_mol, consistant_hit_steps = 10, relaxation_steps = 1)
    bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd], md, loggers = loggers, criteria = reaction_criteria, reactive=True)
    bxd_trajectory.run_trajectory(max_steps=10000000)

    narupa_mol = bxd_trajectory.mol
    dyn = VelocityVerlet(bxd_trajectory.mol, 0.05 * units.fs)

    velocities = []

    for i in range(0,1000):
        dyn.run(200)
        velocities.append(narupa_mol.get_velocities())
        name = Tl.getSMILES(narupa_mol, False)
        print(name)
    write('prod.xyz',narupa_mol)

    name = Tl.getSMILES(narupa_mol, False)
    if len(name) == 2 and name[1] == "O":

        dist1 = narupa_mol.get_distance(3,1)
        dist2 = narupa_mol.get_distance(3,5)
        narupa_mol2 = narupa_mol.copy()

        del(narupa_mol[0])
        del(narupa_mol[1])


        del(narupa_mol2[3])
        del(narupa_mol2[3])

        mod_vels = []
        mod_vels2= []

        if dist1 < dist2:
            del (narupa_mol[3])
            del (narupa_mol2[1])
            for vel in velocities:
                a = deepcopy(vel)
                a = np.delete(a,0,0)
                a = np.delete(a,1,0)
                a = np.delete(a, 3, 0)

                b = deepcopy(vel)
                b = np.delete(b,3,0)
                b = np.delete(b, 3, 0)
                b = np.delete(b, 1, 0)
                mod_vels.append(a)
                mod_vels2.append(b)
        else:
            del (narupa_mol[0])
            del (narupa_mol2[3])
            for vel in velocities:
                a = deepcopy(vel)
                a = np.delete(a,0,0)
                a = np.delete(a, 0, 0)
                a = np.delete(a, 0, 0)

                b = deepcopy(vel)
                b = np.delete(b,3,0)
                b = np.delete(b, 3, 0)
                b = np.delete(b, 3, 0)
                mod_vels.append(a)
                mod_vels2.append(b)
                mod_vels.append(a)


        # get ref mol
        if dist2 < dist1:
            ref = read('Water_5added.xyz')
            H = np.loadtxt('Water_5added.txt', dtype=float)
        else:
            ref = read('Water_1added.xyz')
            H = np.loadtxt('Water_1added.txt', dtype=float)

        if dist2 < dist1:
            ref2=read('HCO_5removed.xyz')
            H2 = np.loadtxt('HCO_5removed.txt', dtype=float)
        else:
            ref2 = read('HCO_1removed.xyz')
            H2 = np.loadtxt('HCO_1removed.txt', dtype=float)



        masses = ((np.tile(narupa_mol.get_masses(), (3, 1))).transpose()).flatten()
        Hess = convert_hessian_to_cartesian(H,masses)
        Linv = np.linalg.inv(Hess)


        masses2 = ((np.tile(narupa_mol2.get_masses(), (3, 1))).transpose()).flatten()
        Hess2 = convert_hessian_to_cartesian(H2,masses2)
        Linv2 = np.linalg.inv(Hess2)
        check_vels = Hess2


        vibs = [0]*9
        overall_K_e = 0
        for vel in mod_vels:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses[j]/2) * (Hess[j,i] * Q_dot[i])**2)
                vibs[i] += kinetic
        overall_K_e /= len(mod_vels)
        vibsav = [v / len(mod_vels) for v in vibs]

        overall_K_e2 = 0
        vibs2 = [0]*9
        for vel in mod_vels2:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2,Q_dot2)
            overall_K_e2 += 0.5 *np.dot(masses2 * q_dot,q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses2[j]) * (Hess2[j,i] * Q_dot2[i])**2)
                vibs2[i] += kinetic * 0.5
        vibs2av = [v /len(mod_vels) for v in vibs2]
        overall_K_e2 /= len(mod_vels)
        f.write("Vib E (H20)  =\t" )
        for v in vibsav:
            f.write(str(v) + '\t')
        f.write("Vib E (co)  = \t")
        for v in vibs2av:
            f.write(str(v) + '\t')
        f.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f.write("\n")
        f.flush()

        vibs = [0]*9
        overall_K_e = 0
        for vel in mod_vels[:4500]:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses[j]/2) * (Hess[j,i] * Q_dot[i])**2)
                vibs[i] += kinetic
        overall_K_e /= 4500
        vibsav = [v / 4500 for v in vibs]


        overall_K_e2 = 0
        vibs2 = [0]*9
        for vel in mod_vels2[:4500]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2,Q_dot2)
            overall_K_e2 += 0.5 *np.dot(masses2 * q_dot,q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses2[j]) * (Hess2[j,i] * Q_dot2[i])**2)
                vibs2[i] += kinetic * 0.5
        vibs2av = [v / 4500 for v in vibs2]
        overall_K_e2 /= 4500
        f2.write("Vib E (H20)  =\t" )
        for v in vibsav:
            f2.write(str(v) + '\t')
        f2.write("Vib E (co)  = \t")
        for v in vibs2av:
            f2.write(str(v) + '\t')
        f2.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f2.write("\n")
        f2.flush()

        vibs = [0]*9
        overall_K_e = 0
        for vel in mod_vels[:3000]:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses[j]/2) * (Hess[j,i] * Q_dot[i])**2)
                vibs[i] += kinetic
        overall_K_e /= 3000
        vibsav = [v / 3000 for v in vibs]


        overall_K_e2 = 0
        vibs2 = [0]*9
        for vel in mod_vels2[:3000]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2,Q_dot2)
            overall_K_e2 += 0.5 *np.dot(masses2 * q_dot,q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses2[j]) * (Hess2[j,i] * Q_dot2[i])**2)
                vibs2[i] += kinetic * 0.5
        vibs2av = [v / 3000 for v in vibs2]
        overall_K_e2 /= 3000
        f3.write("Vib E (H20)  =\t" )
        for v in vibsav:
            f3.write(str(v) + '\t')
        f3.write("Vib E (co)  = \t")
        for v in vibs2av:
            f3.write(str(v) + '\t')
        f3.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f3.write("\n")
        f3.flush()

        vibs = [0]*9
        overall_K_e = 0
        for vel in mod_vels[:1500]:
            q_dot = vel.flatten()
            Q_dot = np.dot(Linv, q_dot)
            overall_K_e += 0.5 * np.dot(masses * q_dot, q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses[j]/2) * (Hess[j,i] * Q_dot[i])**2)
                vibs[i] += kinetic
        overall_K_e /= 1500
        vibsav = [v / 1500 for v in vibs]


        overall_K_e2 = 0
        vibs2 = [0]*9
        for vel in mod_vels2[:1500]:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2,Q_dot2)
            overall_K_e2 += 0.5 *np.dot(masses2 * q_dot,q_dot)
            for i in range(0,9):
                kinetic = 0
                for j in range(0,9):
                    kinetic += ((masses2[j]) * (Hess2[j,i] * Q_dot2[i])**2)
                vibs2[i] += kinetic * 0.5
        vibs2av = [v / 1500 for v in vibs2]
        overall_K_e2 /= 1500
        f4.write("Vib E (H20)  =\t" )
        for v in vibsav:
            f4.write(str(v) + '\t')
        f4.write("Vib E (co)  = \t")
        for v in vibs2av:
            f4.write(str(v) + '\t')
        f4.write("total kinetic energies  = \t" + str(overall_K_e) + "\t" + str(overall_K_e2))
        f4.write("\n")
        f4.flush()
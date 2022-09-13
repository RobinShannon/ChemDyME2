import numpy

try:
    import src.bxd.collective_variable as CV
except:
    pass
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
from ase.vibrations import Vibrations
from ase.md.verlet import VelocityVerlet
from ase import units
import random
from sella import Sella
from ase.optimize import BFGS
import math
import copy
#from src.Calculators.XtbCalculator import XTB
import src.utility.connectivity_tools as CT


def get_moments_of_inertia(mol):
    """Get the moments of inertia along the principal axes.

    The three principal moments of inertia are computed from the
    eigenvalues of the symmetric inertial tensor. Periodic boundary
    conditions are ignored. Units of the moments of inertia are
    amu*angstrom**2.
    """
    com = mol.get_center_of_mass()
    positions = mol.get_positions()
    positions -= com  # translate center of mass to origin
    masses = mol.get_masses()

    # Initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(mol)):
        x, y, z = positions[i]
        m = masses[i]

        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    Itensor = np.array([[I11, I12, I13],
                        [I12, I22, I23],
                        [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(Itensor)
    absevec = abs(evecs)
    sum_of_rows = absevec.sum(axis=0)
    evecs = evecs / sum_of_rows[np.newaxis, :]
    is_normal = np.dot(evecs[:,0],evecs[:,1])
    is_normal = np.dot(evecs[:,0],evecs[:,2])
    is_normal = np.dot(evecs[:,1],evecs[:,2])
    return evecs, Itensor

def get_translational_vectors(mol):
    T = np.zeros((3,len(mol)*3))
    masses = mol.get_masses()
    for i in range(len(mol)):
        m = masses[i]
        T[0][(i*3)] = math.sqrt(m)
        T[1][(i*3)+1] = math.sqrt(m)
        T[2][(i * 3) + 2] = math.sqrt(m)
    row_sums = T.sum(axis=1)
    T_normalised = T / row_sums[:, np.newaxis]
    return T_normalised

def get_rotational_vectors(mol, X):
    com = mol.get_center_of_mass()
    positions = mol.get_positions()
    positions -= com  # translate center of mass to origin
    Rot = np.zeros((3,len(mol)*3))
    R = np.asarray(positions)
    Px = np.dot(R,X[0,:])
    Py = np.dot(R,X[1,:])
    Pz = np.dot(R,X[2,:])
    masses = mol.get_masses()
    for i in range(len(mol)):
        m = math.sqrt(masses[i])
        for j in range(0,3):
            Rot[0][i * 3 + j] = ((Py[i] * X[j][2]) - (Pz[i] * X[j][1])) * m
            Rot[1][i * 3 + j] = ((Pz[i] * X[j][0]) - (Px[i] * X[j][2])) * m
            Rot[2][i * 3 + j] = ((Px[i] * X[j][1]) - (Py[i] * X[j][0])) * m
    RotGS = gram_schmidt_columns(Rot.T).T
    Rot[2,:] = RotGS[2,:]
    absR = abs(Rot)
    row_sums = absR.sum(axis=1)
    Rot_normalised = Rot / row_sums[:, np.newaxis]
    is_normal = np.dot(Rot[0,:],Rot[1,:])
    is_normal = np.dot(Rot[0, :], Rot[2, :])
    is_normal = np.dot(Rot[1, :], Rot[2, :])
    return Rot_normalised

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q

def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            #print "i =", i, ", projection vector =", proj_vec
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
            #print "i =", i, ", temporary vector =", temp_vec
        Y.append(temp_vec)
    return Y

def convert_hessian_to_cartesian(Hess, m):
    masses = np.tile(m,(len(m),1))
    reduced_mass = 1/masses**0.5
    H = np.multiply(Hess,reduced_mass.T)
    absH = abs(H)
    sum_of_rows = absH.sum(axis=0)
    normalized_array = H / sum_of_rows[np.newaxis,:]
    return normalized_array

def convert_hessian_to_cartesian2(Hess, m):
    masses = np.tile(m,(3,1))
    reduced_mass = 1/masses**0.5
    H = np.multiply(Hess,reduced_mass.T)
    absH = abs(H)
    sum_of_rows = absH.sum(axis=0)
    normalized_array = H / sum_of_rows[np.newaxis,:]
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
                positions[i][j] -=  disp
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

mol = read('MFGeoms/[CH2]OC=O_1.xyz')


mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-1620000', atoms=mol))
dyn = BFGS(mol)
dyn.run(1e-7, 1000)
vib = Vibrations(mol)
vib.clean()
vib.run()
vib.summary()
vib.clean()
L = vib.modes
T = get_translational_vectors(mol)
X,I = get_moments_of_inertia(mol)
R = get_rotational_vectors(mol, X.T)

L[0:3,:] = copy.deepcopy(T)
L[3:6,:] = copy.deepcopy(R)
new = L.T
is_normal = np.dot(new[:,3],new[:,0])
is_normal = np.dot(new[:,3],new[:,1])
is_normal = np.dot(new[:,3],new[:,2])
is_normal = np.dot(new[:,3],new[:,4])
is_normal = np.dot(new[:,3],new[:,5])
is_normal = np.dot(new[:,3],new[:,6])
is_normal = np.dot(new[:,3],new[:,7])
is_normal = np.dot(new[:,3],new[:,8])
newGS = (gram_schmidt_columns(L.T))
new[:,6:9] = copy.deepcopy(newGS[:,6:9])
is_normal = np.dot(new[:,3],new[:,4])
is_normal = np.dot(new[:,3],new[:,5])
is_normal = np.dot(new[:,3],new[:,6])
is_normal = np.dot(new[:,3],new[:,7])
is_normal = np.dot(new[:,3],new[:,8])
#new_converted = L.T
masses = ((np.tile(mol.get_masses(), (3, 1))).transpose()).flatten()
new_converted = convert_hessian_to_cartesian(new,masses)
is_normal = np.dot(new[:,3],new[:,4])
is_normal = np.dot(new[:,3],new[:,5])
is_normal = np.dot(new[:,3],new[:,6])
is_normal = np.dot(new[:,3],new[:,7])
is_normal = np.dot(new[:,3],new[:,8])

np.save('MFGeoms/CO[CH2]OC=O_1_GS.npy', new_converted)
mode1 = new_converted[:,3].reshape(3,3)
mode1_traj= []
com = mol.get_center_of_mass()
positions = mol.get_positions()
positions -= com
mol.set_positions(positions)
for i in range(0,30):
    mode1_traj.append(mol.copy())
    mol.set_positions(mol.get_positions() + 0.1 * mode1)
write('WaterModes/t1.xyz', mode1_traj)

mode2 = new_converted[:,4].reshape(3,3)
mode2_traj= []
mol.set_positions(positions)
for i in range(0,30):
    mode2_traj.append(mol.copy())
    mol.set_positions(mol.get_positions() + 0.1 * mode2)
write('WaterModes/t2.xyz', mode2_traj)

mode3 = new_converted[:,5].reshape(3,3)
mode3_traj= []
mol.set_positions(positions)
for i in range(0,30):
    mode3_traj.append(mol.copy())
    mol.set_positions(mol.get_positions() + 0.1 * mode3)
write('WaterModes/t3.xyz', mode3_traj)

mode4 = new_converted[:,6].reshape(3,3)
mode4_traj= []
mol.set_positions(positions)
for i in range(0,30):
    mode4_traj.append(mol.copy())
    mol.set_positions(mol.get_positions() + 0.1 * mode4)
write('WaterModes/t4.xyz', mode4_traj)

mode5 = new_converted[:,7].reshape(3,3)
mode5_traj= []
mol.set_positions(positions)
for i in range(0,30):
    mode5_traj.append(mol.copy())
    mol.set_positions(mol.get_positions() + 0.1 * mode5)
write('WaterModes/t5.xyz', mode5_traj)

mode6 = new_converted[:,8].reshape(3,3)
mode6_traj= []
mol.set_positions(positions)
for i in range(0,30):
    mode6_traj.append(mol.copy())
    mol.set_positions(mol.get_positions() + 0.1 * mode6)
write('WaterModes/t6.xyz', mode6_traj)



GlyIRC = read('GlyoxalGeoms/GlyoxalIRC.log',':')
write('GlyoxalGeoms/GlyoxalPaths.xyz',GlyIRC)
narupa_mol = GlyIRC[0].copy()
narupa_mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-1620000', atoms=narupa_mol))
baseline = narupa_mol.get_potential_energy()
for i in GlyIRC:
    narupa_mol.set_positions(i.get_positions())
    ene = narupa_mol.get_potential_energy()
    print(str((ene-baseline)*96.48))


narupa_mol.set_positions(GlyIRC[0].get_positions())
# Set up a Sella Dynamics object
dyn = Sella(narupa_mol, internal = True)
try:
    dyn.run(1e-2, 1000)
except:
    pass
ts_ene = narupa_mol.get_potential_energy()*96.58
write('GlyoxalGeoms/NN_TS.xyz', narupa_mol)
narupa_mol.set_positions(GlyIRC[43].get_positions())
dyn = BFGS(narupa_mol)
try:
    dyn.run(1e-2, 1000)
except:
    pass
comp_ene = narupa_mol.get_potential_energy()*96.58
diff = ts_ene - comp_ene
write('GlyoxalGeoms/NN_Comp.xyz', narupa_mol)

narupa_mol.set_positions(GlyIRC[-1].get_positions())
dyn = BFGS(narupa_mol)
try:
    dyn.run(1e-2, 1000)
except:
    pass
comp_ene = narupa_mol.get_potential_energy()*96.58
diff = ts_ene - comp_ene
write('GlyoxalGeoms/NNComp2.xyz', narupa_mol)

f =open("vibresults.txt", 'a')
f2 =open("vibresults2.txt", 'a')
f3 =open("vibresults3.txt", 'a')
f4 =open("vibresults4.txt", 'a')
for run in range(0, 1000):
    # The general set-up here is identical to the adaptive run to ensure the converging run samples the same path
    #narupa_mol = read('water', index=0)
    temp = read('GlyoxalGeoms/glyoxal.xyz')
    narupa_mol.set_positions(temp.get_positions())
    ##dyn = Langevin(narupa_mol, .5 * units.fs, 291, 1)
    #dyn.run(1000)
    pcs = 2
    #collective_variable = CV.Distances(narupa_mol, [[1,3]])
    collective_variable = CV.Distances(narupa_mol, [[5,6]])
    progress_metric = PM.Line(collective_variable, [0], [1.7])

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
    tf3 = lambda var: var.mdsteps % 1 == 0
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
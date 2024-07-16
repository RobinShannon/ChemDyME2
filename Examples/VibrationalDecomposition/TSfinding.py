import numpy


import src.bxd.collective_variable as CV
from ase.constraints import FixAtoms,FixInternals
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
from src.Calculators.ScineCalculator import  SparrowCalculator as SP
from ase.vibrations import Vibrations
from ase.md.verlet import VelocityVerlet
from ase import units
#import random
from sella import Sella
from ase.optimize import BFGS
import math
import os
import copy
#from src.Calculators.XtbCalculator import XTB as xtb
import src.utility.connectivity_tools as CT


def superimpose_points(A, B):
    """
    Superimpose set of points A onto set of points B using translation and rotation.

    Parameters:
    A (numpy.ndarray): N x 3 array of points to be transformed.
    B (numpy.ndarray): N x 3 array of reference points.

    Returns:
    A_transformed (numpy.ndarray): N x 3 array of transformed points A.
    """
    # Centering the data by subtracting the centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the covariance matrix
    H = A_centered.T @ B_centered

    # Compute SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R_opt = Vt.T @ U.T

    # Special case when the determinant of the rotation matrix is -1
    if np.linalg.det(R_opt) < 0:
        Vt[2, :] *= -1
        R_opt = Vt.T @ U.T

    # Apply the rotation
    A_rotated = A_centered @ R_opt

    # Translate the points
    A_transformed = A_rotated + centroid_B

    return A_transformed

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
    evecs = evecs.T
    absE = (evecs*evecs)
    row_sums = np.sqrt(absE.sum(axis=1))
    evec_normalised = evecs / row_sums[:, np.newaxis]
    return evec_normalised, Itensor

def get_translational_vectors(mol):
    T = np.zeros((3,len(mol)*3))
    masses = mol.get_masses()
    for i in range(len(mol)):
        m = masses[i]
        T[0][(i*3)] = m
        T[1][(i*3)+1] = m
        T[2][(i * 3) + 2] = m
    absT = (T * T)
    row_sums = np.sqrt(absT.sum(axis=1))
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
    absR = (Rot*Rot)
    row_sums = np.sqrt(absR.sum(axis=1))
    Rot_normalised = Rot / row_sums[:, np.newaxis]
    is_norm = np.dot((Rot_normalised[0, :]), Rot_normalised[1, :])
    is_norm = np.dot((Rot_normalised[0, :]), Rot_normalised[2, :])
    is_norm = np.dot((Rot_normalised[1, :]), Rot_normalised[2, :])
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
            proj_vec = (np.dot(X[i], inY) / np.dot(inY, inY))*inY
            #print "i =", i, ", projection vector =", proj_vec
            temp_vec -= proj_vec

            #print "i =", i, ", temporary vector =", temp_vec
        temp_vec = np.divide(temp_vec, np.sqrt(np.dot(temp_vec, temp_vec)))
        Y.append(temp_vec)
    return np.asarray(Y)

def convert_hessian_to_cartesian(Hess, m):
    masses = np.tile(m,(len(m),1))
    reduced_mass = 1/masses**0.5
    H = np.multiply(Hess,reduced_mass.T)
    absH = abs(H)
    sum_of_rows = absH.sum(axis=0)
    print(sum_of_rows)
    normalized_array = H / sum_of_rows[np.newaxis,:]
    return normalized_array

def convert_hessian_to_MW(Hess, m):
    masses = np.tile(m,(len(m),1))
    reduced_mass = masses**0.5
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

TS=True
complex=False
dir = 'MethylFormate'
prefix = 'TS'
if not complex:
    mol = read(str(dir)+'/'+str(prefix)+'_NN.xyz')
else:
    mol = read(str(dir)+'/'+str(prefix)+'.xyz')
if not TS and not complex:
    mol2= read(str(dir)+'/'+str(prefix)+'proj.xyz')
mol.set_calculator(NNCalculator(checkpoint='best_model.ckpt-540000', atoms=mol))


# Set up a Sella Dynamics object
#dyn = Sella(mol, internal=True)
#try:
#    dyn.run(1e-4, 1000)
#except:
#    pass
#ts_ene = mol.get_potential_energy()*96.58

#reac = read('MethylFormate/Start.xyz')
#mol.set_positions(reac.get_positions())

#comp_ene = mol.get_potential_energy()*96.58
#diff = ts_ene - comp_ene
#constraints = []
#constraints.append(FixAtoms(indices=[0, 3]))
#mol.set_constraint(constraints)
if TS:
    dyn = Sella(mol, internal=True)
    try:
        dyn.run(1e-4, 1000)
    except:
       pass

elif complex:
    vec =mol.get_positions()[3]-mol.get_positions()[0]
    pos = mol.get_positions()
    pos[3] += 2*vec
    pos[4] += 2*vec
    mol.set_positions(pos)
    constraints = []
    constraints.append(FixAtoms(indices=[0,4]))
    mol.set_constraint(constraints)
    dyn = BFGS(mol)
    try:
        dyn.run(1e-6, 20)
    except:
        pass
    del mol.constraints
else:
    dyn = BFGS(mol)
    try:
        dyn.run(1e-6, 100)
    except:
        pass
    pos = superimpose_points(mol.get_positions(),mol2.get_positions())
    mol.set_positions(pos)
#del mol.constraints
print(str(mol.get_potential_energy()*96.58))
write(str(dir)+'/'+str(prefix)+'_NN.xyz', mol)
vib = Vibrations(mol)
vib.clean()
vib.run()
vib.summary()
vib.clean()
L = vib.modes
L2 = vib.get_mode(0)
T = get_translational_vectors(mol)
X,I = get_moments_of_inertia(mol)
norms = []
min_i = 10000.0
ith = 0
R = get_rotational_vectors(mol, X)
if TS:
    L=np.roll(L, -1, axis=0)
if complex:
    L = np.roll(L, -1, axis=0)
    L = np.roll(L, -1, axis=0)
    h1 = np.load('FormH/H22.npy')
    h2 = np.load('FormH/HCO2.npy')
    R1 = h2[3]
    h1 = np.pad(h1, [(9, 0), (9, 0)], mode='constant')
    h2 = np.pad(h2, [(0, 6), (0, 6)], mode='constant')

min = 0
for i in range(0,L.T.shape[0]):
    for j in range(0, L.T.shape[0]):
        if i != j:
            is_norm = np.dot((L.T[i, :]), L.T[j, :])
            if is_norm > min:
                min = is_norm

print(str(is_norm))
np.save(str(dir)+'/'+str(prefix)+'_ASE.npy', L.T)



L[0:3,:] = copy.deepcopy(T)
#L[3:6,:] = copy.deepcopy(R)
L[3,:] = copy.deepcopy(R[0,:])
L[4,:] = copy.deepcopy(R[2,:])

#new[10:15,:] = L[6:11,:]
if complex:
    L[6:9,:]=copy.deepcopy(h2[6:9,:])
    L[9, :] = copy.deepcopy(h1[14, :])

new = L
newGS = (gs(L))
norm = np.dot((new[0, :]), new[0, :])
new[5:,:] = copy.deepcopy(newGS[5:,:])
min = 0
for i in range(0,new.shape[0]):
    for j in range(0, new.shape[0]):
        if i != j:
            is_norm = np.dot(new[i, :], new[j, :])
            if abs(is_norm) > abs(min):
                min = is_norm

print(str(is_norm))

np.save(str(dir)+'/'+str(prefix)+'2.npy', new)
masses = ((np.tile(mol.get_masses(), (3, 1))).transpose()).flatten()

new_converted = convert_hessian_to_cartesian(new.T,masses)
#new_converted= (gs(new_converted.T))
np.save(str(dir)+'/'+str(prefix)+'corrected.npy', new_converted)
for i in range(0, new_converted.shape[0]):
    mode = new[:,i].reshape(int(new.shape[0]/3),3)
    mode_traj= []
    com = mol.get_center_of_mass()
    positions = mol.get_positions()
    positions -= com
    mol.set_positions(positions)
    for j in range(0,30):
        com = mol.get_center_of_mass()
        positions = mol.get_positions()
        positions -= com
        mol.set_positions(positions)
        mode_traj.append(mol.copy())
        mol.set_positions(mol.get_positions() + 0.1 * mode)
    for j in range(0, 30):
        com = mol.get_center_of_mass()
        positions = mol.get_positions()
        positions -= com
        mol.set_positions(positions)
        mode_traj.append(mol.copy())
        mol.set_positions(mol.get_positions() - 0.1 * mode)
    write('Modes/t' +str(i) +'.xyz', mode_traj)





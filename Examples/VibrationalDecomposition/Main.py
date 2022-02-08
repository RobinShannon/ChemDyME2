
import src.BXD.CollectiveVariable as CV
import src.BXD.ProgressMetric as PM
import src.MolecularDynamics.MDIntegrator as MD
import src.BXD.BXDConstraint as BXD
import src.MolecularDynamics.Trajectory as Traj
import src.MolecularDynamics.MDLogger as lg
import src.MechanismGeneration.ReactionCriteria as RC
import src.Utility.ConnectivityTools as CT
import src.Utility.Tools as Tl
from copy import deepcopy
import numpy as np
from ase.io import read
from ase.optimize import BFGS as bfgs
from ase.md.verlet import VelocityVerlet
import ase.md.velocitydistribution as vd
import ase.units as units
from ase.vibrations import Vibrations
from ase.io import write
from sella import Sella, IRC
from src.Calculators.XtbCalculator import XTB
from ase.constraints import FixInternals, FixAtoms, FixBondLength, FixBondLengths

def convert_hessian_to_cartesian(Hess, m):
    masses = np.tile(m,(len(m),1))
    reduced_mass = 1/masses**0.5
    H = np.multiply(Hess,reduced_mass.T)
    sum_of_rows = H.sum(axis=0)
    normalized_array = H / sum_of_rows[np.newaxis, :]
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


React1 = read('R1.xyz')
React1.set_calculator((XTB(method="GFN1xTB", electronic_temperature=300)))
dyn = bfgs(React1)
dyn.run(fmax=0.01, steps=75)
React2 = read('R2.xyz')
React2.set_calculator((XTB(method="GFN1xTB", electronic_temperature=300)))
dyn = bfgs(React2)
dyn.run(fmax=0.01, steps=75)
Prod1 = read('P1.xyz')
Prod1.set_calculator((XTB(method="GFN1xTB", electronic_temperature=300)))
dyn = bfgs(Prod1)
dyn.run(fmax=0.01, steps=75)
Prod2 = read('P2.xyz')
Prod2.set_calculator((XTB(method="GFN1xTB", electronic_temperature=300)))
dyn = bfgs(Prod2)
dyn.run(fmax=0.05, steps=75)

ReactE =  (React1.get_potential_energy() + React2.get_potential_energy()) - (Prod1.get_potential_energy() + Prod2.get_potential_energy())


f =open("vibresults.txt", 'a')
for run in range(0, 1000):
    # The general set-up here is identical to the adaptive run to ensure the converging run samples the same path
    #narupa_mol = read('water', index=0)
    narupa_mol = read('glyoxal.xyz', index=0)
    narupa_mol.set_calculator((XTB(method="GFN1xTB", electronic_temperature=1500)))
    ##dyn = Langevin(narupa_mol, .5 * units.fs, 291, 1)
    #dyn.run(1000)
    pcs = 2
    collective_variable = CV.COM(narupa_mol, [0,1,2,3,4,5], [6,7])
    progress_metric = PM.Line(collective_variable, [0], 3.)
    bxd = BXD.Fixed(progress_metric)
    loggers = []
    lf1 = lambda var: 'box\t=\t' + str(var.bxd_list[0].box) + '\tprogress\t=\t'+str(var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].s) /var.bxd_list[0].progress_metric.project_point_on_path(var.bxd_list[0].progress_metric.end))
    tf1 = lambda var: var.mdsteps % 1000 == 0
    log1 = lg.MDLogger( logging_function=lf1, triggering_function=tf1)
    loggers.append(log1)
    #lf2 = lambda var: "hit!!\t" + str(var.bxd_list[0].bound_hit) + "\tmdstep\t=\t"+str(var.mdsteps)
    #tf2 = lambda var: var.bxd_list[0].inversion
    #log2 = lg.MDLogger( logging_function=lf2, triggering_function=tf2)
    #loggers.append(log2)
    #lf3 = lambda var: str(write('geom.xyz',var.mol, append=True))
    #tf3 = lambda var: var.mdsteps % 10 == 0
    #log3 = lg.MDLogger( logging_function=lf3, triggering_function=tf3, outpath='Geom.xyz')
    #loggers.append(log3)
    #lf4 = lambda var: var.md_integrator.current_velocities
    #tf4 = lambda var: var.criteria.counter > 250
    #log4 = lg.MDLogger( logging_function=lf4, triggering_function=tf4, write_to_list=True)
    #loggers.append(log4)


    md = MD.Langevin(narupa_mol, temperature=298, timestep=0.25, friction=0.002)
    reaction_criteria = RC.NunezMartinez(narupa_mol, consistant_hit_steps = 50, relaxation_steps = 1)
    bxd_trajectory = Traj.Trajectory(narupa_mol, [bxd], md, loggers = loggers,criteria = reaction_criteria, reactive=True)
    bxd_trajectory.run_trajectory()

    narupa_mol = bxd_trajectory.mol

    dyn = VelocityVerlet(narupa_mol, dt=0.25 * units.fs)
    velocities = []
    for i in range(0,400):
        dyn.run(10)
        velocities.append(narupa_mol.get_velocities())

    narupa_mol = bxd_trajectory.mol
    name = Tl.getSMILES(narupa_mol, False)
    if len(name) >1 and name[1] == "O":

        dist1 = narupa_mol.get_distance(6,5)
        dist2 = narupa_mol.get_distance(6,3)
        narupa_mol2 = narupa_mol.copy()

        del(narupa_mol[0])
        del(narupa_mol[0])
        del(narupa_mol[0])
        del(narupa_mol[1])

        del(narupa_mol2[7])
        del(narupa_mol2[6])

        mod_vels = []
        mod_vels2= []

        if dist1 < dist2:
            del (narupa_mol[0])
            del (narupa_mol2[5])
            for vel in velocities:
                a = deepcopy(vel)
                a = np.delete(a,0,0)
                a = np.delete(a, 0, 0)
                a = np.delete(a, 0, 0)
                a = np.delete(a,1,0)
                a = np.delete(a, 0, 0)

                b = deepcopy(vel)
                b = np.delete(b,7,0)
                b = np.delete(b, 6, 0)
                b = np.delete(b, 5, 0)
                mod_vels.append(a)
                mod_vels2.append(b)
        else:
            del (narupa_mol[1])
            del (narupa_mol2[3])
            for vel in velocities:
                a = deepcopy(vel)
                a = np.delete(a,0,0)
                a = np.delete(a, 0, 0)
                a = np.delete(a, 0, 0)
                a = np.delete(a,1,0)
                a = np.delete(a, 1, 0)

                b = deepcopy(vel)
                b = np.delete(b,7,0)
                b = np.delete(b, 6, 0)
                b = np.delete(b, 3, 0)
                mod_vels.append(a)
                mod_vels2.append(b)
                mod_vels.append(a)


        # get ref mol
        ref = read('water.xyz')
        ref.set_calculator((XTB(method="GFN1xTB", electronic_temperature=1500)))
        dyn = bfgs(ref)
        dyn.run(fmax=0.01)
        if dist2 < dist1:
            ref2=read('HCOCO_3removed.xyz')
        else:
            ref2 = read('HCOCO_5removed.xyz')
        ref2.set_calculator((XTB(method="GFN1xTB", electronic_temperature=1500)))
        dyn = bfgs(ref2)
        dyn.run(fmax=0.01)
        # Translate to COM frame

        vib = Vibrations(ref)
        vib.run()
        vib.summary()
        H = vib.modes.T
        masses = ((np.tile(narupa_mol.get_masses(), (3, 1))).transpose()).flatten()
        Hess = convert_hessian_to_cartesian(H,masses)
        vib.clean()
        Linv = np.linalg.inv(Hess)

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


        vib2 = Vibrations(ref2)
        vib2.run()
        vib2.summary()
        masses2 = ((np.tile(narupa_mol2.get_masses(), (3, 1))).transpose()).flatten()
        H2 = vib2.modes.T
        Hess2 = convert_hessian_to_cartesian(H2,masses2)
        Linv2 = np.linalg.inv(Hess2)
        check_vels = Hess2
        vib.clean()

        overall_K_e2 = 0
        vibs2 = [0]*15
        for vel in mod_vels2:
            q_dot = vel.flatten()
            Q_dot2 = np.matmul(Linv2, q_dot)
            check_vels = np.matmul(Hess2,Q_dot2)
            overall_K_e2 += 0.5 *np.dot(masses2 * q_dot,q_dot)
            for i in range(0,15):
                kinetic = 0
                for j in range(0,15):
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
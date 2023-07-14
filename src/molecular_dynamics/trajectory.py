import numpy as np
import sys
from ase.md import velocitydistribution as vd
import src.mechanism_generation.reaction_crtieria as RC
import src.utility.tools as Tl
import src.molecular_dynamics.md_logger as Log
from typing import Optional
from ase.io import write




class Trajectory:
    """
    Controls the running of a bxd trajectory. This class interfaces with a a list of one or more bxd object to track
    whether or not a boundary has been hit and then uses an attached md_integrator object to propagate the dynamics.
    In the case of a reactive trajectory this class consult an attatched reaction criteria object to look for reactions
    :param mol: ASE atoms object
    :param bxd_list: A  list bxd constraint objects containing the details of the bxd bounds etc
    :param md_integrator: An MDintegrator object controlling the propagation of the trajectory
    :param loggers: OPTIONAL: A list of loggers controlling output printing
    :param reactive=False: Bool, determines whether to track for reactive events
    :param criteria: OPTIONAL: Reaction_criteria object which tracks for reactive events in the case of reactive = True
    """

    def __init__(self, mol, bxd_list, md_integrator, loggers: Optional[Log.MDLogger] = [Log.MDLogger()],
                 criteria: Optional[RC.ReactionCriteria] = None, reactive=False, maxwell_boltzman = True, allowed_species = None):
        self.bxd_list = bxd_list
        self.md_integrator = md_integrator
        self.mol = mol.copy()
        self.mol._calc = mol.get_calculator()
        initial_temperature = md_integrator.temperature
        if maxwell_boltzman:
            vd.MaxwellBoltzmannDistribution(self.mol, temperature_K= initial_temperature, force_temp=True)
            vd.Stationary(self.mol,True)
        self.md_integrator.current_velocities = self.mol.get_velocities()
        self.md_integrator.half_step_velocity = self.mol.get_velocities()
        self.hit = False
        self.reactive = reactive
        self.loggers = loggers
        self.criteria = criteria
        self.mdsteps = 1
        if allowed_species != None:
            self.allowed_species = allowed_species
            self.check_molecular_species = True
        else:
            self.check_molecular_species = False



    def run_trajectory(self, max_steps=np.inf):
        """
        Runs a bxd trajectory until either the attached BXDconstraint indicates sufficient information has been
        obtained, a reactive event occurs, or the max_steps parameter is exceeded
        :param max_steps: DEFAULT np.inf: Maximum number of steps in MD trajectory
        """


        # Then loop through bxd objects and set each one up
        for bxd in self.bxd_list:
            bxd.initialise_files()

        traj = []

        # Get forces from atoms
        forces = self.mol.get_forces()

        # Want to make sure the trajectory doesnt try to perform and bxd inversion on the first MD step. This shouldnt
        # happen. While first_run = True, we will override the bxd inversion.
        first_run = True

        # Set up boolean for while loop to determine whether the trajectory loop should keep going and then track number
        # of md steps
        keep_going = True
        self.mdsteps = 0
        bxd_complete = False

        # Run MD trajectory for specified number of steps or until bxd reaches its end point
        while keep_going:

            if self.mdsteps % 100 == 0 and self.check_molecular_species:
                temp_mol = self.mol.copy()
                temp_mol._calc = self.mol.get_calculator()
                name = Tl.getSMILES(temp_mol, opt=True)
                if str(name) not in self.allowed_species:
                    return None


            del_phi = []
            # update each bxd constraint with the current geometry and determine whether a bxd inversion is neccessary
            for bxd in self.bxd_list:
                if bxd.connected_BXD is None or bxd.connected_BXD.active is True:
                    bxd.update(self.mol)
                    if bxd.inversion and not first_run:
                        self.mol.set_positions(self.md_integrator.old_positions)
                        if bxd.bound_hit != 'path':
                            del_phi.append(bxd.del_constraint(self.mol))
                        if bxd.progress_metric.reflect_back_to_path():
                            del_phi.append(bxd.path_del_constraint(self.mol))
                        self.md_integrator.constrain(del_phi)
                    if bxd.inversion and first_run:
                        sys.stderr.write(" bxd bound hit on first MD step, possibly due a small rounding error but check "
                                        "input")

            # Now we have gone through the first inversion section we can set first_run to false
            first_run = False


            # Now get the md object to propagate the dynamics according to the standard Velocity Verlet / Langevin
            # procedure:
            # 1. md_step_pos: Get the half step velocity v(t + 1/2 * delta_t) and then new positions x(t + delta_t)
            # 2. Get forces at new positions
            # 3. md_step_vel : Get the  new velocities v(t + delta_t)
            self.md_integrator.md_step_pos(forces, self.mol)

            try:
                forces = self.mol.get_forces()
            except:
                pass
            self.md_integrator.md_step_vel(forces, self.mol)

            traj.append(self.mol.copy())

            if self.loggers != None:
                for log in self.loggers:
                    log.write_log(self)

            if self.reactive:
                self.criteria.update(self.mol)

            # Check whether bxd has gathered all the info it needs, if so signal that the trajectory should stop
            for bxd in self.bxd_list:
                if bxd.complete():
                    bxd_complete = True
                    for bxd in self.bxd_list:
                        bxd.close('temp',self.mol)
            if bxd_complete or self.mdsteps > max_steps or self.criteria is not None and self.criteria.complete:
                keep_going = False
                if self.criteria is not None:
                    return self.criteria.product_geom
                else:
                    return self.mol.copy()

            self.mdsteps += 1


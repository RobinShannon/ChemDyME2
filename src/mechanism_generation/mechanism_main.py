import os
import src.MechanismGeneration.reaction as rxn
import src.molecular_dynamics.trajectory as Traj
import src.bxd.collective_variable as CV
import src.bxd.ProgressMetric as PM
import src.bxd.bxd_constraint as BXD
import gc

import src.MechanismGeneration.species.species as Species


class ReactionNetwork:

    def __init__(self, starting_species, reaction_criteria, master_equation, calculators, md_integrator,
                 adaptive_bxd_steps = 5000, bxd_epsilon = 0.95, bimolecular_start = False, bimolecular_bath = {},
                 cores = 1, iterations = 4, reaction_trials = 5, start_frag_indicies = [], max_com_seperation = 1.7, path = os.getcwd()):

        self.start = starting_species
        self.master_equation = master_equation
        self.calculators = calculators
        self.md_integrator = md_integrator
        self.adaptive_bxd_steps = adaptive_bxd_steps
        self.bxd_epsilon = bxd_epsilon
        self.bimolecular_start = bimolecular_start
        self.cores = cores
        self.iterations = iterations
        self.path = path
        self.max_com_seperation = max_com_seperation
        self.restart = False
        self.mechanism_run_time = 0.0
        self.reaction_criteria = reaction_criteria
        self.reaction_trials = reaction_trials
        self.network = []
        self.species = []
        self.start_frag_indicies = start_frag_indicies
        self.bimolecular_bath = bimolecular_bath
        # Check whether there is a directory for putting calculation data in. If not create it
        if not os.path.exists(self.path + '/Raw'):
            os.mkdir(self.path + '/Raw')


    def run(self):

        # Create Molecule object for the current reactant
        reac = Species.Stable(self.start, self.calculators)
        reac.node_visits = 1
        # Open files for saving summary
        mainsumfile = open(('mainSummary.txt'), "a")


        while self.master_equation.time < self.master_equation.max_time:
            for itr in range(0, self.reaction_trials):
                self.run_single_generation(reac,bimolecular=self.bimolecular_start)
            if self.bimolecular_bath:
                lifetime = self.run_KME(dummy=True)
                for item in self.bimolecular_bath.items():
                    if isinstance(item[1],float and 1/item[1]) < lifetime:
                        reac.add_fragment()
                        self.run_single_generation(reac.combined_mol,bimolecular=True,bath=item[0])
            while reac.node_visits >= 1 :
                self.run_KME(dummy = False)
                for sp in self.species:
                    if sp.smiles == self.master_equation.current_node:
                        reac = sp
            reac.node_visits += 1




    def run_single_generation(self, reactant, bimolecular=False, bath=None):
        # Reinitialise the reaction criteria to the geometry of "reactant" and set up default logger and MD integrator
        if bimolecular:
            self.reaction_criteria.reinitialise(reactant.combined_mol)
        else:
            self.reaction_criteria.reinitialise(reactant.mol)
        log = None
        # Set up the first bxd object as as bxd in energy
        bxd_list = []
        collective_variable1 = CV.Energy(reactant.mol)
        if bimolecular:
            progress_metric1 = PM.ProgressMetric(collective_variable1, reactant.combined_mol, [100000])
        else:
            progress_metric1 = PM.ProgressMetric(collective_variable1, reactant.mol, [100000])
        bxd1 = BXD.Adaptive(progress_metric1, fix_to_path=False, adaptive_steps = self.adaptive_bxd_steps, epsilon = self.bxd_epsilon)
        bxd_list.append(bxd1)
        # If there are multiple fragments then set up a second bxd object pulling the moieties together
        if bimolecular:
            #frag1, frag2 = reactant.get_fragment_indicies()
            collective_variable2 = CV.COM(reactant.combined_mol, self.start_frag_indicies[0], self.start_frag_indicies[1])
            progress_metric2 = PM.Line(collective_variable2, [0], [self.max_com_seperation])
            bxd2 = BXD.Fixed(progress_metric2)
            bxd1.connected_BXD = bxd2
            bxd_list.append(bxd2)
            traj = Traj.Trajectory(reactant.combined_mol, bxd_list, self.md_integrator, loggers = log,criteria = self.reaction_criteria, reactive=True)
        else:
            traj = Traj.Trajectory(reactant.mol, bxd_list, self.md_integrator, loggers=log, criteria=self.reaction_criteria, reactive=True)
        product_geometry = traj.run_trajectory()
        product = Species.Stable(product_geometry, self.calculators)
        for s in self.species:
            if s.smiles is product.smiles:
                break
        r = rxn.Reaction(reactant, product, traj, self.calculators)
        for reaction in self.network:
            if r == reaction:
                break
        r.characterise(bimolecular_traj=bimolecular)
        if not r.found_TS and not r.barrierless and len(r.path_minima) > 0:
            r1,r2 = r.split_reaction(r)
            self.network.append(r1)
            self.species.append(r.prod)
            self.network.append(r2)
            self.species.append(r2.prod)
        else:
            self.network.append(r)
            self.species.append(r.prod)
        del traj
        gc.collect()



    def run_KME(self,dummy=False):
        time, new_node = self.master_equation.run_stochastic_transition()
        if not dummy:
            self.master_equation.current_node = new_node
            self.mechanism_run_time += time
        return time
















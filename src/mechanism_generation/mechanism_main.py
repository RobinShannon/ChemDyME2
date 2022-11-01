import os
import src.mechanism_generation.reaction as rxn
import src.molecular_dynamics.trajectory as Traj
import src.bxd.collective_variable as CV
import src.bxd.ProgressMetric as PM
import src.bxd.bxd_constraint as BXD
import src.utility.connectivity_tools as CT
import src.utility.tools as Tl
from ase import Atoms
import copy
from ase.io import write
import pickle
import gc

import src.mechanism_generation.mol_types.well as stable



class ReactionNetwork:

    def __init__(self, starting_species, reaction_criteria, master_equation, calculators, md_integrator,
                 adaptive_bxd_steps = 5000, bxd_epsilon = 0.95, bimolecular_start = False, bimolecular_bath = {},
                 cores = 1, iterations = 4, reaction_trials = 4, start_frag_indicies = [], max_com_seperation = 1.7, path = os.getcwd()):

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
        if not os.path.exists(self.path + '/Network'):
            os.mkdir(self.path + '/Network')

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['master_equation'] #don't pickle this
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self.somedevicehandlethatcannotbepickled=GetDeviceHandle1()

    def run(self):

        # Create Molecule object for the current reactant
        reacs = []
        if self.bimolecular_start:
            bimolecular, reactants = CT.split_mol(self.start)
            reacs.append(stable.well(reactants[0], self.calculators))
            reacs.append(stable.well(reactants[1], self.calculators))
            reacs.append(stable.well(self.start, self.calculators))
            baseline = reacs[0].energy['single'] + float(reacs[0].zpe) + reacs[1].energy['single'] + float(reacs[1].zpe)
            reacs[0].energy['co'] = reacs[1].energy['single'] + float(reacs[1].zpe)
            reacs[0].energy['baseline'] = baseline
        else:
            reacs.append(stable.well(self.start, self.calculators))
            baseline = reacs[0].energy['single'] + float(reacs[0].zpe)
            reacs[0].energy['baseline'] = baseline
        reacs[0].node_visits = 1
        # Open files for saving summary
        mainsumfile = open(('mainSummary.txt'), "a")


        while self.master_equation.time < self.master_equation.max_time:
            for itr in range(0, self.reaction_trials):
                self.run_single_generation(reacs,bimolecular=self.bimolecular_start)
            if self.bimolecular_bath:
                lifetime = self.run_KME(dummy=True)
                for item in self.bimolecular_bath.items():
                    if isinstance(item[1],float and 1/item[1]) < lifetime:
                        bi_reacs = []
                        bath = Tl.getMolFromSmile(item)
                        combine_geometry = CT.get_bi_xyz(item, reacs[0])
                        atoms = reacs[0].mol.get_chemical_symbols() + bath.get_chemical_symbols()
                        combined_mol = Atoms(atoms,combine_geometry)
                        bi_reacs.append(reacs[0].copy())
                        bi_reacs.append(stable.well(bath, self.calculators))
                        bi_reacs.append(stable.well(combined_mol, self.calculators))
                        baseline = bi_reacs[0].energy['single'] + float(bi_reacs[0].zpe) + bi_reacs[1].energy['single'] + float(
                            bi_reacs[1].zpe)
                        reacs[0].energy['bath'] = bi_reacs[1].energy['single'] + float(bi_reacs[1].zpe)
                        reacs[0].energy['baseline'] = baseline
                        self.run_single_generation(bi_reacs.mol,bimolecular=True,bath=item[0])
            while reacs[0].node_visits >= 1 :
                self.run_KME(dummy = False)
                for sp in self.species:
                    if sp.name == self.master_equation.current_node:
                        sp.calculator = self.calculators
                        reacs = [sp.copy()]
                        sp.calculator = None
            reacs[0].node_visits += 1
            reacs[0].calculator = self.calculators
            self.bimolecular_start = False
            #with open('mech.pkl', 'wb') as outp:  # Overwrites any existing file.
                #pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)



    def run_single_generation(self, reactant, bimolecular=False, bath=None):
        # Reinitialise the reaction criteria to the geometry of "reactant" and set up default logger and MD integrator

        if bimolecular:
            self.md_integrator.reinitialise(reactant[2].mol.copy())
            self.calculators.set_calculator(reactant[2].mol, 'trajectory')
            self.reaction_criteria.reinitialise(reactant[2].mol)
        else:
            self.md_integrator.reinitialise(reactant[0].mol.copy())
            self.calculators.set_calculator(reactant[0].mol, 'trajectory')
            self.reaction_criteria.reinitialise(reactant[0].mol)
        log = None
        # Set up the first bxd object as as bxd in energy
        bxd_list = []
        if bimolecular:
            collective_variable1 = CV.Energy(reactant[2].mol)
            progress_metric1 = PM.ProgressMetric(collective_variable1, [reactant[2].mol.get_potential_energy()], [100000])
        else:
            collective_variable1 = CV.Energy(reactant[0].mol)
            progress_metric1 = PM.ProgressMetric(collective_variable1, [reactant[0].mol.get_potential_energy()], [100000])
        bxd1 = BXD.Adaptive(progress_metric1, fix_to_path=False, adaptive_steps = self.adaptive_bxd_steps, epsilon = self.bxd_epsilon)
        bxd_list.append(bxd1)
        # If there are multiple fragments then set up a second bxd object pulling the moieties together
        if bimolecular:
            #frag1, frag2 = reactant.get_fragment_indicies()
            collective_variable2 = CV.COM(reactant[2].mol, self.start_frag_indicies[0], self.start_frag_indicies[1])
            progress_metric2 = PM.Line(collective_variable2, [0], [self.max_com_seperation])
            bxd2 = BXD.Fixed(progress_metric2)
            bxd1.connected_BXD = bxd2
            bxd_list.append(bxd2)
            traj = Traj.Trajectory(reactant[2].mol, bxd_list, self.md_integrator, loggers=log, criteria=self.reaction_criteria, reactive=True)
        else:
            traj = Traj.Trajectory(reactant[0].mol, bxd_list, self.md_integrator, loggers=log, criteria=self.reaction_criteria, reactive=True)
        product_geometry = traj.run_trajectory()
        prods = []
        bimolecular_product, products = CT.split_mol(product_geometry)
        if bimolecular_product:
            prods.append(stable.well(products[0].copy(), self.calculators))
            prods.append(stable.well(products[1].copy(), self.calculators))
            prods.append(stable.well(product_geometry.copy(), self.calculators))
            prods[0].energy['co'] = prods[1].energy['single'] + float(prods[1].zpe)
        else:
            prods.append(stable.well(product_geometry.copy(), self.calculators))

        prods[0].energy['baseline'] = reactant[0].energy['baseline']
        if 'co' in reactant[0].energy and not bimolecular:
            prods[0].energy['baseline'] += reactant[0].energy['co']

        for s in self.species:
            if s == prods[0]:
                return
        r = rxn.Reaction(reactant, prods, traj, self.calculators)
        for reaction in self.network:
            if r == reaction:
                return
        r.characterise(bimolecular_traj=bimolecular)
        r.print_to_file()

        self.network.append(r)
        if r.reac[0] not in self.species:
            self.master_equation.add_molecule(r.reac[0])
            self.species.append(reactant[0].copy())
            if bimolecular:
                self.master_equation.add_molecule(r.reac[1].copy(), bi = True)
                self.species.append(reactant[1].copy())

        self.species.append(r.prod[0].copy())
        self.master_equation.add_molecule(r.prod[0])
        if bimolecular_product:
            self.species.append(r.prod[1].copy())
            self.master_equation.add_molecule(r.prod[1], bi = True)
        self.master_equation.add_reaction(r)
        del traj
        gc.collect()

    def run_KME(self,dummy=False):
        run_properly = False
        while not run_properly:
            run_properly = self.master_equation.run_stochastic_transition()
        if not dummy:
            self.master_equation.current_node = self.master_equation.prodName
            self.mechanism_run_time += self.master_equation.time

        return self.master_equation.time
















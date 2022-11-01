from abc import ABCMeta, abstractmethod
import src.utility.connectivity_tools as CT
import numpy as np
import src.utility.tools as TL



# Class to control connectivity maps / to determine  whether transitions have occured
class ReactionCriteria:

    def __init__(self, consistant_hit_steps = 25, relaxation_steps = 100):
        self.criteria_met = False
        self.transition_mol = []
        self.consistant_hit_steps = consistant_hit_steps
        self.relaxation_steps = relaxation_steps
        self.counter = 0
        self.complete = False
        self.atom_list = []
        self.product_geom = None


    def update(self, mol):
        self.check_criteria(mol)
        if self.criteria_met and self.counter == 0:
            self.transition_mol.append(mol.copy())
        if self.criteria_met or self.counter >= self.consistant_hit_steps:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter > self.consistant_hit_steps + self.relaxation_steps:
            self.product_geom = mol.copy()
            self.complete = self.check_product(mol)

        self.atom_list.append(mol.copy())



    @abstractmethod
    def reinitialise(self, mol):
        pass

    @abstractmethod
    def check_criteria(self, mol):
        pass

    @abstractmethod
    def check_product(self, mol):
        pass

class NunezMartinez(ReactionCriteria):

    def __init__(self,mol, consistant_hit_steps = 25, relaxation_steps = 125):
        super(NunezMartinez, self).__init__(consistant_hit_steps, relaxation_steps)
        self.dRef = CT.refBonds(mol)
        self.C = CT.bondMatrix(self.dRef, mol)
        self.transitionIndices = np.zeros(3)

    def check_criteria(self, mol):
        self.criteria_met = False
        size = len(mol.get_positions())
        # Loop over all atoms
        for i in range(0,size):
            bond = 0
            nonbond = 1000
            a_1  = 0
            a_2 = 0
            for j in range(0,size):
                # Get distances between all atoms bonded to i
                if self.C[i][j] == 1:
                    newbond = max(bond,mol.get_distance(i,j)/self.dRef[i][j])
                    #Store Index corresponding to current largest bond
                    if newbond > bond:
                        a_1 = j
                    bond = newbond
                elif self.C[i][j] == 0:
                    newnonbond = min(nonbond, mol.get_distance(i,j)/self.dRef[i][j])
                    if newnonbond < nonbond:
                        a_2 = j
                    nonbond = newnonbond
            if bond > nonbond:
                self.criteria_met = True
                self.transitionIndices = [a_1, i, a_2]
                break
        for i in range(0, size):
            for j in range(0, size):
                # Get distances between all atoms bonded to i
                if self.C[i][j] == 0:
                    length = mol.get_distance(i, j)
                    # Store Index corresponding to current largest bond
                    if length < (self.dRef[i][j] * 0.8):
                        self.criteria_met = True
                        self.transitionIndices = [i, j]
                        break

    def reinitialise(self, mol):
        self.dRef = CT.refBonds(mol)
        self.C = CT.bondMatrix(self.dRef, mol)
        self.transitionIdices = np.zeros(3)
        self.atom_list = []
        super(NunezMartinez, self).__init__(self.consistant_hit_steps, self.relaxation_steps)

    def ReactionType(self, mol):
        oldbonds = np.count_nonzero(self.C)
        self.dRef = CT.refBonds(mol)
        self.C = CT.bondMatrix(self.dRef, mol)
        newbonds = np.count_nonzero(self.C)
        if oldbonds > newbonds:
            reacType = 'Dissociation'
        if oldbonds < newbonds:
            reacType = 'Association'
        if oldbonds == newbonds:
            reacType = 'Isomerisation'
        return reacType

    def check_product(self, mol):
        c_mol = mol.copy()
        c_mol._calc = mol.get_calculator()
        s_mol = self.atom_list[0].copy()
        s_mol._calc = mol.get_calculator()
        smiles1 = TL.getSMILES(c_mol, True)
        smiles2 = TL.getSMILES(s_mol, True)
        if len(smiles1) > 2:
            self.look_for_initial_disociation(mol)
            return True
        if smiles1 == smiles2:
            self.counter = 0
            return False
        return True

    def look_for_initial_disociation(self,mol):
        found_product = False
        i = 0
        while i < 15:
            c_mol = self.atom_list[- int(i * 0.1 * self.relaxation_steps)].copy()
            c_mol._calc = mol.get_calculator()
            smiles1 = TL.getSMILES(c_mol, True)
            if len(smiles1) == 2:
                self.product_geom = self.atom_list[- int(i * 0.1 * self.relaxation_steps)].copy()
                return True
            i += 1
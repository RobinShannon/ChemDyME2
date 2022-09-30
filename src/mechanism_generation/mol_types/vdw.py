import src.utility.connectivity_tools as CT
import src.utility.tools as TL
from ase.constraints import FixInternals, FixAtoms, FixBondLength, FixBondLengths
from ase.optimize.sciopt import SciPyFminBFGS as BFGS
import math
try:
    import MESMER_API.src.meMolecule as me_mol
except:
    pass
from ase.io import write
from abc import abstractmethod
import copy
import os
import sella
from src.mechanism_generation.mol_types.species import species

class vdw(species):

    def __init__(self, mol, calculator, dir = 'Raw', name = ''):
        super(vdw, self).__init__(mol, calculator, dir)
        self.vdw = True
        self.calculator.set_calculator(self.mol, 'low')
        smiles = TL.getSMILES(self.mol, True, True)
        if name == '':
            name = smiles
        self.name = name
        self.hessian =[]
        self.combined_mol = mol.copy()
        self.calculator.set_calculator(self.combined_mol, 'low')
        self.smiles = 'vdw_' + str(smiles[0]) + '_' + str(smiles[1])
        self.fragments = CT.get_fragments(mol)
        self.bonds_to_add = CT.get_hbond_idxs(self.mol, self.fragments, is_vdw=True)

    def optimise(self, path, mol):
        mol._calc.minimise_stable(path=path, atoms=mol)
        self.mol = mol.copy()

    def get_frequencies(self, path, mol):
        self.vibs, self.zpe, imag, self.hessian = mol._calc.get_frequencies(path, mol)

    def write_cml(self, coupled = False):
        data = {}
        data['zpe'] = self.energy['single'] + float(self.zpe)
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        data['newBonds'] = self.bonds_to_add
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', coupled=coupled, **data)
        mes_mol.write_cml(self.dir + '/mes.xml')

import src.Utility.connectivity_tools as CT
import src.Utility.Tools as TL
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

class well(species):

    def __init__(self, mol, calculator, dir = 'Raw', name=''):
        super(well, self).__init__(mol, calculator, dir)
        self.calculator.set_calculator(self.mol, 'low')
        smiles = TL.getSMILES(self.mol, False, False)
        self.combined_mol = mol.copy()
        self.calculator.set_calculator(self.combined_mol, 'low')
        self.smiles = smiles[0]
        if name == '':
            name = self.smiles
        self.name = name
        self.hessian = []
        write(str(self.dir) + '/reac.xyz', self.mol)
        if len(smiles) > 1:
            self.bimolecular = True
            self.mol = TL.getMolFromSmile(smiles[0])
            self.fragment_two = TL.getMolFromSmile(smiles[1])
            self.characterise( bimolecular=True)
            self.bi_smiles = smiles[1]

    def optimise(self, path, mol):
        mol._calc.minimise_stable(path=path, atoms=mol)
        self.mol = mol.copy()

    def get_frequencies(self, path, mol):
        if len(mol) ==2:
            self.bimolecular_vibs, self.bimolecular_zpe, imag, hess = mol._calc.get_frequencies(path, mol, bimolecular=True)
        else:
            self.vibs, self.zpe, imag, self.hessian = mol._calc.get_frequencies(path, mol)

    def write_cml(self):
        data = {}
        data['zpe'] = self.energy['single'] + float(self.zpe)
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', **data)
        mes_mol.write_cml(self.dir + '/mes.xml')
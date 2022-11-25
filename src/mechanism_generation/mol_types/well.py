
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

class well(species):

    def __init__(self, mol, calculator, dir = 'Raw', name=''):
        super(well, self).__init__(mol, calculator, dir)
        self.calculator.set_calculator(self.mol, 'low')
        smiles = TL.getSMILES(self.mol, False, False)
        self.calculator.set_calculator(self.combined_mol, 'low')
        if len(smiles) > 1:
            self.smiles = smiles[0] + 'bi' + smiles[1]
        else:
            self.smiles = smiles[0]
        if name == '':
            name = self.smiles
        self.name = name
        self.hessian = []
        write(str(self.dir) + '/reac.xyz', self.mol)
        self.characterise(bimolecular=False)

    def copy(self):
        new = well(self.mol.copy(),self.calculator,dir = self.dir, name = self.name)
        new.mol = self.mol.copy()
        new.calculator = None
        new.energy = self.energy
        return new

    def optimise(self, path, mol):
        mol._calc.minimise_stable(path=path, atoms=mol)
        self.mol = mol.copy()

    def get_frequencies(self, path, mol):
        if len(mol) ==2:
            self.vibs, self.zpe, imag, hess = mol._calc.get_frequencies(path, mol, bimolecular=True)
        else:
            self.vibs, self.zpe, imag, self.hessian = mol._calc.get_frequencies(path, mol)

    def write_cml(self, coupled = False, zero_energy = False):
        data = {}
        data['zpe'] = self.energy['single'] + float(self.zpe)
        if 'baseline' in self.energy:
            if 'co' in self.energy:
                data['zpe'] = (self.energy['single'] + float(self.zpe) + self.energy['co']) - self.energy['baseline']
            else:
                data['zpe'] = (self.energy['single'] + float(self.zpe)) - self.energy['baseline']
        if zero_energy:
            data['zpe'] = 0
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedAngles'] = self.hinderance_angles
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', coupled = coupled, **data)
        self.cml = mes_mol.cml
        mes_mol.write_cml('mes.xml')

    def write_bi_cml(self, coupled = False):
        data = {}
        data['zpe'] = (self.energy['single'] + float(self.zpe))
        if 'baseline' in self.energy:
            if 'co' in self.energy:
                data['zpe'] = (self.energy['single'] + float(self.zpe) + self.energy['co']) - self.energy['baseline']
            else:
                data['zpe'] = (self.energy['single'] + float(self.zpe)) - self.energy['baseline']
        data['vibFreqs'] = self.vibs
        data['name'] = self.name
        data['hinderedRotors'] = self.hinderance_potentials
        data['hinderedAngles'] = self.hinderance_angles
        data['hinderedBonds'] = self.hinderance_indexes
        data['hessian'] = self.hessian
        mes_mol = me_mol.meMolecule(self.mol, role = 'modeled', coupled = coupled, **data)
        self.cml = mes_mol.cml
        mes_mol.write_cml('mes.xml')
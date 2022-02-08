import codex
import qcore
from typing import Optional, Collection
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import os
import src.Utility.Tools as tl

EV_PER_HARTREE = 27.2114
ANG_PER_BOHR = 0.529177

class QcoreCalculator(Calculator):
    """
    Simple implementation of an ASE calculator for Sparrow.

    Parameters:
        method :  The electronic structure method to use in calculations.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms: Optional[Atoms] = None, method='xtb', basis=None, triplet=False, **kwargs):
        super().__init__(**kwargs)
        self.atoms = atoms
        self.method = method
        self.basis = basis
        self.triplet = triplet
        if atoms is None:
            self.has_atoms = False
        if method == 'xtb':
            self.method_dict = {"kind": "xtb", "model": 'GFN0', "details" : {}}
        else:
            self.method_dict = {"kind": "dft", "xc": 'b3lyp', "ao": '3-21G',"details" : {}}

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties=('energy', 'forces'),
                  system_changes=all_changes):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError('No ASE atoms supplied to calculator, and no ASE atoms supplied with initialisation.')
        if not self.has_atoms:
            sym = atoms.get_chemical_symbols()
            is_O = len(sym) == 1 and sym[0] == 'O'
            is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
            s = sum(atoms.get_atomic_numbers())
            if s % 2 != 0:
                self.spin_mult = 2
                self.unrestricted = True
            elif is_O or is_OO or self.triplet:
                self.spin_mult = 3
                self.unrestricted = False
            else:
                self.spin_mult = 1
                self.unrestricted = False
        self.has_atoms = False
        self._calculate_qcore(atoms, properties)


    def _calculate_qcore(self, atoms: Atoms, properties: Collection[str]):

        qcore_input = {
            "molecule": {
                "geometry": atoms.get_positions()*1.889,
                "atomic_numbers": atoms.get_atomic_numbers(),
                "charge": float(0.0),
                "multiplicity": self.spin_mult,
            },
            "method": self.method_dict,
            "result_contract": {"wavefunction": "all"},
            "result_type": 'gradient'
        }


        try:
            result = qcore.run(qcore_input, ncores=int(4))
        except Exception as exc:
            pass

        if 'energy' in properties:
            self.results['energy'] = result.energy * EV_PER_HARTREE
        if 'forces' in properties:
            self.results['forces'] = - result.gradient * EV_PER_HARTREE / ANG_PER_BOHR

    def minimise_ts(self,path=os.getcwd(), atoms=None, ratoms=None, patoms=None, constraints = []):
        if atoms is None:
            atoms = self.atoms
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        os.remove('temp.xyz')
        os.chdir(current_dir)
        sym = atoms.get_chemical_symbols()
        is_O = len(sym) == 1 and sym[0] == 'O'
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_O or is_OO or self.triplet:
            self.spin_mult = 3
            self.unrestricted = False
        else:
            self.spin_mult = 1
            self.unrestricted = False
        init_mol = {"geometry": atoms.get_positions(),
                "atomic_numbers": atoms.get_atomic_numbers(),
                "charge": float(0.0),
                "multiplicity": self.spin_mult}

        opt = codex.models.optimization.OptimizationInput(initial_molecule = init_mol, method = self.method_dict)
        try:
            result = qcore.run(opt, ncores=int(4))
        except Exception as exc:
            pass
        atoms.set_positions(result.final_molecule.geometry / 1.889)
        rmol = atoms.copy()
        pmol= atoms.copy()
        irc_for = []
        irc_rev = []
        modes = get_normal_modes(atoms)
        positive_displacement = atoms.get_positions() + modes
        negative_displacement = atoms.get_positions() - modes
        rmol.set_positions(positive_displacement)
        pmol.set_positions(negative_displacement)
        return rmol, pmol, irc_for, irc_rev

    def minimise_stable(self, path = os.getcwd(), atoms: Optional[Atoms] = None):

        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        sym = atoms.get_chemical_symbols()
        is_O = len(sym) == 1 and sym[0] == 'O'
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_O or is_OO or self.triplet:
            self.spin_mult = 3
            self.unrestricted = False
        else:
            self.spin_mult = 1
            self.unrestricted = False
        init_mol = {"geometry": atoms.get_positions()*1.889,
                "atomic_numbers": atoms.get_atomic_numbers(),
                "charge": float(0.0),
                "multiplicity": self.spin_mult}

        opt = codex.models.optimization.OptimizationInput(initial_molecule = init_mol, method = self.method_dict)
        try:
            result = qcore.run(opt, ncores=int(1))
        except Exception as exc:
            pass
        atoms.set_positions(result.final_molecule.geometry/1.889)
        os.chdir(current_dir)

    def get_frequencies(self, path=os.getcwd(), atoms = None, bimolecular = False, TS = False):
        if atoms is None:
            atoms = self.atoms
        current_dir = os.getcwd()
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        sym = atoms.get_chemical_symbols()
        is_O = len(sym) == 1 and sym[0] == 'O'
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = True
        elif is_O or is_OO or self.triplet:
            self.spin_mult = 3
            self.unrestricted = False
        else:
            self.spin_mult = 1
            self.unrestricted = False
        qcore_input = {
            "molecule": {
                "geometry": atoms.get_positions()*1.889,
                "atomic_numbers": atoms.get_atomic_numbers(),
                "charge": float(0.0),
                "multiplicity": self.spin_mult,
            },
            "method": self.method_dict,
            "result_contract": {"wavefunction": "all"},
            "result_type": 'hessian'
        }
        try:
            result = qcore.run(qcore_input, ncores=int(1))
        except Exception as exc:
            pass
        viblist = result.extras['frequencies']
        hessian = result.hessian
        freqs, zpe = tl.getVibString(viblist, bimolecular, TS)
        os.chdir(current_dir)
        imaginary_frequency = 0
        if TS:
            imaginary_frequency = freqs[0]
            del freqs[0]
        return freqs, zpe, imaginary_frequency, hessian

def get_normal_modes(self, atoms):
    if atoms is None:
        atoms = self.atoms
    sym = atoms.get_chemical_symbols()
    is_O = len(sym) == 1 and sym[0] == 'O'
    is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
    s = sum(atoms.get_atomic_numbers())
    if s % 2 != 0:
        self.spin_mult = 2
        self.unrestricted = True
    elif is_O or is_OO or self.triplet:
        self.spin_mult = 3
        self.unrestricted = False
    else:
        self.spin_mult = 1
        self.unrestricted = False
    qcore_input = {
        "molecule": {
            "geometry": atoms.get_positions() * 1.889,
            "atomic_numbers": atoms.get_atomic_numbers(),
            "charge": float(0.0),
            "multiplicity": self.spin_mult,
        },
        "method": self.method_dict,
        "result_contract": {"wavefunction": "all"},
        "result_type": 'hessian'
    }
    try:
        result = qcore.run(qcore_input, ncores=int(1))
    except Exception as exc:
        pass
    modes= result.extras['normal_modes']
    mode = modes[0,:]
    mode = mode.reshape(3,9)
    mode *=1.889 *10
    return mode

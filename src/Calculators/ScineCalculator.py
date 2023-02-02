# Copyright (c) Intangible Realities Lab, University Of Bristol. All rights reserved.
# Licensed under the GPL. See License.txt in the project root for license information.
from typing import Optional, Collection
import os
import numpy as np
import copy
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import scine_utilities as su
import scine_sparrow
from ase.vibrations import Vibrations
import src.utility.tools as tl
from ase.io import write, read
import scine_readuct
import time

EV_PER_HARTREE = 27.2114
ANG_PER_BOHR = 0.529177

class SparrowCalculator(Calculator):
    """
    Simple implementation of an ASE calculator for Sparrow.

    Parameters:
        method :  The electronic structure method to use in calculations.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms: Optional[Atoms] = None, method='AM1', triplet=False, **kwargs):
        super().__init__(**kwargs)
        self.atoms = atoms
        self.method = method
        self.triplet = triplet
        #self.calc =  Calculation(method = self.method)
        if atoms is None:
            self.has_atoms = False

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties=('energy', 'forces'),
                  system_changes=all_changes):
        manager = su.core.ModuleManager()
        self.calc = manager.get('calculator', self.method)
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
                self.unrestricted = 'unrestricted'
            elif is_O or is_OO or self.triplet:
                self.spin_mult = 3
                self.unrestricted = 'restricted'
            else:
                self.spin_mult = 1
                self.unrestricted = 'restricted'

            self.structure = su.AtomCollection()
            elements = []
            for s in sym:
                if s =='H':
                    ele = su.ElementType.H
                elif s =='O':
                    ele = su.ElementType.O
                elif s == 'C':
                    ele = su.ElementType.C
                elements.append(ele)
            self.structure.elements = elements
            self.calc.settings['spin_multiplicity'] = self.spin_mult
            self.calc.settings['spin_mode'] = self.unrestricted
            self.calc.settings['electronic_temperature'] =0.0
            log = su.core.Log()
            self.calc.log = log
            self.calc.set_required_properties([su.Property.Energy,
                                                su.Property.Gradients])
        self.has_atoms = False
        self._calculate_sparrow(atoms, properties)


    def reinitialize(self, atoms):
        self.atoms = atoms
        self.results = {}


    def _calculate_sparrow(self, atoms: Atoms, properties: Collection[str]):
        self.structure.positions = copy.deepcopy(atoms.positions) * su.BOHR_PER_ANGSTROM
        self.calc.structure = copy.deepcopy(self.structure)
        res =self.calc.calculate()
        if np.isnan(res.energy):
            res = self.calc.calculate()
        if 'energy' in properties:
            self.results['energy'] = res.energy * EV_PER_HARTREE
        if 'forces' in properties:
            self.results['forces'] = -res.gradients * EV_PER_HARTREE / ANG_PER_BOHR
        return

    def minimise_stable(self,path = os.getcwd(), atoms: Optional[Atoms] = None):
        if atoms is None:
            atoms = self.atoms
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        sym = atoms.get_chemical_symbols()
        if len(sym) == 1:
            return
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = 'unrestricted'
        elif is_OO:
            self.spin_mult = 3
            self.unrestricted = 'restricted'
        else:
            self.spin_mult = 1
            self.unrestricted = 'restricted'
        atoms.write('temp.xyz')
        system1 = su.core.load_system_into_calculator('temp.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, spin_mode=self.unrestricted, spin_multiplicity=self.spin_mult)
        systems = {}
        systems['reac'] = system1
        try:
            systems, success = scine_readuct.run_opt_task(systems, ['reac'], output = ['reac_opt'], optimizer ='bfgs', stop_on_error = False)
            atoms.set_positions(systems['reac_opt'].positions * ANG_PER_BOHR)
        except:
            print('error with readuct opt')
            pass
        os.remove('temp.xyz')
        os.chdir(current_dir)

    def close(self):
        self.calc = None

    def minimise_ts(self,path=os.getcwd(), atoms=None, ratoms=None, patoms=None):
        if atoms is None:
            atoms = self.atoms
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        sym = atoms.get_chemical_symbols()
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = 'unrestricted'
        elif is_OO:
            self.spin_mult = 3
            self.unrestricted = 'restricted'
        else:
            self.spin_mult = 1
            self.unrestricted = 'restricted'
        if atoms is None:
            atoms = self.atoms
        write('temp.xyz',atoms)
        rmol=atoms.copy()
        pmol=atoms.copy()
        irc_for = atoms.copy()
        irc_rev = atoms.copy()
        system1 = su.core.load_system_into_calculator('temp.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, spin_mode=self.unrestricted, spin_multiplicity=self.spin_mult)
        systems = {}
        systems['reac'] = system1
        try:
            systems, success = scine_readuct.run_tsopt_task(systems, ['reac'], output= ['ts_opt'], optimizer='ef',  stop_on_error = False)
            if not success:
                systems, success = scine_readuct.run_tsopt_task(systems, ['reac'], output=['ts_opt'], optimizer='bofill',stop_on_error=False)
            if not success:
                systems, success = scine_readuct.run_tsopt_task(systems, ['reac'], output=['ts_opt'], optimizer='dimer',
                                                                stop_on_error=False)
            atoms.set_positions(systems['ts_opt'].positions * ANG_PER_BOHR)
            systems, success = scine_readuct.run_irc_task(systems, ['ts_opt'], output=['forward','reverse'], convergence_max_iterations =5000, stop_on_error = False)
            rmol.set_positions(systems['forward'].positions * ANG_PER_BOHR)
            pmol.set_positions(systems['reverse'].positions * ANG_PER_BOHR)
            irc_for = read('forward/forward.irc.forward.trj.xyz', '::100')
            irc_rev = read('reverse/reverse.irc.backward.trj.xyz', '::100')
        except:
            pass
        os.remove('temp.xyz')
        os.chdir(current_dir)
        return rmol, pmol, irc_for, irc_rev

    def minimise_bspline(self,path, reac, prod ):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        sym = reac.get_chemical_symbols()
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(reac.get_atomic_numbers())
        if s % 2 != 0:
            self.spin_mult = 2
            self.unrestricted = 'unrestricted'
        elif is_OO:
            self.spin_mult = 3
            self.unrestricted = 'restricted'
        else:
            self.spin_mult = 1
            self.unrestricted = 'restricted'

        write('reac.xyz',reac)
        write('prod.xyz', prod)

        system1 = su.core.load_system_into_calculator('reac.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, spin_mode=self.unrestricted, spin_multiplicity=self.spin_mult)
        system2 = su.core.load_system_into_calculator('prod.xyz', self.method, program='Sparrow',
                                            molecular_charge=0, spin_mode=self.unrestricted,
                                            spin_multiplicity=self.spin_mult)

        systems = {}
        systems['reac'] = system1
        systems['prod'] = system2
        try:
            systems, success = scine_readuct.run_bspline_task(systems, ['reac','prod'], output = ['spline'],  num_integration_points=20, num_control_points=10,  num_structures = 60)
        except:
            try:
                systems, success = scine_readuct.run_bspline_task(systems, ['reac','prod'], output = ['spline'],  num_integration_points=10, num_control_points=5,  num_structures = 60)
            except:
                pass
        try:
            spline_traj = read('spline/spline_optimized.xyz', index=':')
        except:
            print('spline_failed')
            spline_traj = read('spline/spline_interpolated.xyz', index=':')
        os.remove('spline/spline_optimized.xyz')
        os.remove('spline/spline_interpolated.xyz')
        os.remove('reac.xyz')
        os.remove('prod.xyz')
        try:
            os.chdir(current_dir)
        except:
            pass
        return spline_traj

    def get_frequencies(self, path=os.getcwd(), atoms = None, bimolecular = False, TS = False):
        if atoms is None:
            atoms = self.atoms
        current_dir = os.getcwd()
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        vib = Vibrations(atoms)
        vib.clean()
        vib.run()
        viblist = vib.get_frequencies()
        freqs, zpe = tl.getVibString(viblist, bimolecular, TS)
        vib.clean()
        os.chdir(current_dir)
        hessian = []
        imaginary_frequency = 0
        if TS:
            imaginary_frequency = freqs[0]
            del freqs[0]
        return freqs, zpe, imaginary_frequency, hessian

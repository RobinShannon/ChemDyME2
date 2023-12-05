# This file is part of xtb.
#
# Copyright (C) 2020 Sebastian Ehlert
#
# xtb is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# xtb is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with xtb.  If not, see <https://www.gnu.org/licenses/>.
"""`ASE calculator <https://wiki.fysik.dtu.dk/ase/>`_ implementation
for the ``xtb`` program.

This module provides the basic single point calculator implementation
to integrate the ``xtb`` API into existing ASE workflows.

Supported properties by this calculator are:

- energy (free_energy)
- forces
- stress (GFN0-xTB only)
- dipole
- charges

Example
-------
>>> from ase.build import molecule
>>> from xtb.ase.calculator import XTB
>>> atoms = molecule('H2O')
>>> atoms.calc = XTB(method="GFN2-xTB")
>>> atoms.get_potential_energy()
-137.9677758730299
>>> atoms.get_forces()
[[ 1.30837706e-16  1.07043680e-15 -7.49514699e-01]
 [-1.05862195e-16 -1.53501989e-01  3.74757349e-01]
 [-2.49755108e-17  1.53501989e-01  3.74757349e-01]]

Supported keywords are

======================== ============ ============================================
 Keyword                  Default      Description
======================== ============ ============================================
 method                   "GFN2-xTB"   Underlying method for energy and forces
 accuracy                 1.0          Numerical accuracy of the calculation
 electronic_temperature   300.0        Electronic temperatur for TB methods
 max_iterations           250          Iterations for self-consistent evaluation
 solvent                  "none"       GBSA implicit solvent model
 cache_api                True         Reuse generate API objects (recommended)
======================== ============ ============================================
"""

from typing import List, Optional

from xtb.utils import get_method, get_solvent
from xtb.libxtb import VERBOSITY_MUTED
from xtb.interface import Calculator, XTBException
import ase.calculators.calculator as ase_calc
from ase.atoms import Atoms
from ase.units import Hartree, Bohr
import os
from ase.optimize.bfgs import BFGS
from ase.io import read
from sella import Sella, Constraints, IRC
from ase.vibrations import Vibrations
import src.utility.tools as tl

class XTB(ase_calc.Calculator):
    """ASE calculator for xtb related methods.

    The XTB class can access all methods exposed by the ``xtb`` API.
    """

    implemented_properties = [
        "energy",
        "forces",
        "charges",
        "dipole",
        "stress",
    ]

    default_parameters = {
        "method": "GFN2-xTB",
        "accuracy": 1.0,
        "max_iterations": 250,
        "electronic_temperature": 1500.0,
        "solvent": "None",
        "cache_api": True,
    }

    _res = None
    _xtb = None

    def __init__(
        self, atoms: Optional[Atoms] = None, **kwargs,
    ):
        """Construct the xtb base calculator object."""

        ase_calc.Calculator.__init__(self, atoms=atoms, **kwargs)

    def set(self, **kwargs) -> dict:
        """Set new parameters to xtb"""

        changed_parameters = ase_calc.Calculator.set(self, **kwargs)

        self._check_parameters(changed_parameters)

        # Always reset the calculation if parameters change
        if changed_parameters:
            self.reset()

        # If the method is changed, invalidate the cached calculator as well
        if "method" in changed_parameters:
            self._xtb = None
            self._res = None

        # Minor changes can be updated in the API calculator directly
        if self._xtb is not None:
            if "accuracy" in changed_parameters:
                self._xtb.set_accuracy(self.parameters.accuracy)

            if "electronic_temperature" in changed_parameters:
                self._xtb.set_electronic_temperature(
                    self.parameters.electronic_temperature
                )

            if "max_iterations" in changed_parameters:
                self._xtb.set_max_iterations(self.parameters.max_iterations)

            if "solvent" in changed_parameters:
                self._xtb.set_solvent(get_solvent(self.parameters.solvent))

        return changed_parameters

    def _check_parameters(self, parameters: dict) -> None:
        """Verifiy provided parameters are valid"""

        if "method" in parameters and get_method(parameters["method"]) is None:
            raise ase_calc.InputError(
                "Invalid method {} provided".format(parameters["method"])
            )

    def reset(self) -> None:
        """Clear all information from old calculation"""
        ase_calc.Calculator.reset(self)

        if not self.parameters.cache_api:
            self._xtb = None
            self._res = None

    def _check_api_calculator(self, system_changes: List[str]) -> None:
        """Check state of API calculator and reset if necessary"""

        # Changes in positions and cell parameters can use a normal update
        _reset = system_changes.copy()
        if "positions" in _reset:
            _reset.remove("positions")
        if "cell" in _reset:
            _reset.remove("cell")

        # Invalidate cached calculator and results object
        if _reset:
            self._xtb = None
            self._res = None
        else:
            if system_changes and self._xtb is not None:
                try:
                    _cell = self.atoms.cell
                    self._xtb.update(
                        self.atoms.positions / Bohr, _cell / Bohr,
                    )
                # An exception in this part means the geometry is bad,
                # still we will give a complete reset a try as well
                except XTBException:
                    self._xtb = None
                    self._res = None

    def _create_api_calculator(self) -> Calculator:
        """Create a new API calculator object"""

        _method = get_method(self.parameters.method)
        if _method is None:
            raise ase_calc.InputError(
                "Invalid method {} provided".format(self.parameters.method)
            )

        try:
            _cell = self.atoms.cell
            _periodic = self.atoms.pbc
            _charge = self.atoms.get_initial_charges().sum()
            _uhf = int(self.atoms.get_initial_magnetic_moments().sum().round())

            calc = Calculator(
                _method,
                self.atoms.numbers,
                self.atoms.positions / Bohr,
                _charge,
                _uhf,
                _cell / Bohr,
                _periodic,
            )
            calc.set_verbosity(VERBOSITY_MUTED)
            calc.set_accuracy(self.parameters.accuracy)
            calc.set_electronic_temperature(self.parameters.electronic_temperature)
            calc.set_max_iterations(self.parameters.max_iterations)
            calc.set_solvent(get_solvent(self.parameters.solvent))

        except XTBException:
            raise ase_calc.InputError("Cannot construct calculator for xtb")

        return calc

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = ase_calc.all_changes,
    ) -> None:
        """Perform actual calculation with by calling the xtb API"""

        if not properties:
            properties = ["energy"]
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)

        self._check_api_calculator(system_changes)

        if self._xtb is None:
            self._xtb = self._create_api_calculator()

        try:
            self._res = self._xtb.singlepoint(self._res)
        except XTBException:
            raise ase_calc.CalculationFailed("xtb could not evaluate input")

        # Check if a wavefunction object is present in results
        _wfn = self._res.get_number_of_orbitals() > 0

        # These properties are garanteed to exist for all implemented calculators
        self.results["energy"] = self._res.get_energy() * Hartree
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -self._res.get_gradient() * Hartree / Bohr
        self.results["dipole"] = self._res.get_dipole() * Bohr
        # stress tensor is only returned for periodic systems
        if self.atoms.pbc.any():
            _stress = self._res.get_virial() * Hartree / self.atoms.get_volume()
            self.results["stress"] = _stress.flat[[0, 4, 8, 5, 2, 1]]
        # Not all xtb calculators provide access to partial charges yet,
        # this is mainly an issue for the GFN-FF calculator
        if _wfn:
            self.results["charges"] = self._res.get_charges()


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

    def minimise_stable(self,path = os.getcwd(), atoms: Optional[Atoms] = None):
        if atoms is None:
            atoms = self.atoms
        dyn = BFGS(atoms)
        dyn.run(fmax=0.05, steps=50)

    def minimise_ts(self,path=os.getcwd(), atoms=None, ratoms=None, patoms=None):
        if atoms is None:
            atoms = self.atoms
        dyn = Sella(atoms)
        dyn.run(fmax=0.05, steps=50)
        ts = atoms.copy()
        ts2 = atoms.copy()
        ts._calc = atoms.get_calculator()
        ts2._calc = atoms.get_calculator()
        opt = IRC(ts, trajectory='irc.traj', dx=0.1, eta=1e-4, gamma=0.4)
        try:
            opt = IRC(ts, trajectory='irc.traj', dx=0.01, eta=1e-4, gamma=0.4)
            opt.run(fmax=0.1, steps=15, direction='forward')
            ircf = read('irc.traj',index=':')
            patoms = ircf[-1]
        except:
            ircf = []
            patoms = atoms.copy()
        try:
            opt = IRC(ts2, trajectory='irc.traj', dx=0.01, eta=1e-4, gamma=0.4)
            opt.run(fmax=0.1, steps=15, direction='reverse')
            ircr = read('irc.traj',index=':')
            ratoms = ircr[-1]
        except:
            ircr = []
            ratoms = atoms.copy()

        return ratoms,patoms,ircr,ircf
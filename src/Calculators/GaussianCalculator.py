import os
import copy
from collections.abc import Iterable
from shutil import which
from typing import Dict, Optional
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
from pathlib import Path
from shutil import copyfile
import re
import src.utility.tools as tl
import re
import string

class GaussianDynamics:
    calctype = 'optimizer'
    delete = ['force']
    keyword: Optional[str] = None
    special_keywords: Dict[str, str] = dict()

    def __init__(self, atoms, calc=None):
        self.atoms = atoms
        if calc is not None:
            self.calc = calc
        else:
            if self.atoms.calc is None:
                raise ValueError("{} requires a valid Gaussian calculator "
                                 "object!".format(self.__class__.__name__))

            self.calc = self.atoms.calc

    def todict(self):
        return {'type': self.calctype,
                'optimizer': self.__class__.__name__}

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

    def set_keywords(self, kwargs):
        args = kwargs.pop(self.keyword, [])
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))

        kwargs[self.keyword] = args

    def run(self, **kwargs):
        calc_old = self.atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)
        self.set_keywords(kwargs)

        self.calc.set(**kwargs)
        self.atoms.calc = self.calc

        try:
            self.atoms.get_potential_energy()
        except OSError:
            converged = False
        else:
            converged = True

        atoms = read(self.calc.label + '.log')
        self.atoms.cell = atoms.cell
        self.atoms.positions = atoms.positions

        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            self.atoms.calc = calc_old

        return converged


class GaussianOptimizer(GaussianDynamics):
    keyword = 'opt'
    special_keywords = {
        'fmax': '{}',
        'steps': 'maxcycle={}',
    }


class GaussianIRC(GaussianDynamics):
    keyword = 'irc'
    special_keywords = {
        'direction': '{}',
        'steps': 'maxpoints={}',
    }


class Gaussian(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'dipole']
    command = 'GAUSSIAN < PREFIX.com > PREFIX.log'
    discard_results_on_any_change = True

    def __init__(self, *args, label='', **kwargs):
        label = label + 'gaussian'
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)

    def calculate(self, *args, **kwargs):
        gaussians = ('g16', 'g09', 'g03')
        if 'GAUSSIAN' in self.command:
            for gau in gaussians:
                if which(gau):
                    self.command = self.command.replace('GAUSSIAN', gau)
                    break
            else:
                #raise EnvironmentError('Missing Gaussian executable {}'
                                       #.format(gaussians))
                pass
        try:
            FileIOCalculator.calculate(self, *args, **kwargs)
        except:
            print('Gaussian error')
            i = 0
            while Path('gauserror'+str(i)+'.log').exists():
                i += 1
            copyfile(self.label + '.log', 'gauserror'+str(i)+'.log')

    def reinitialize(self, atoms):
        self.atoms = atoms
        self.results = {}
        sym = atoms.get_chemical_symbols()
        is_O = len(sym) == 1 and sym[0] == 'O'
        is_OO = len(sym) == 2 and sym[0] == 'O' and sym[1] == 'O'
        s = sum(atoms.get_atomic_numbers())
        if s % 2 != 0:
            self.parameters['mult'] = 2
        elif is_O or is_OO:
            self.parameters['mult'] = 3
        else:
            self.parameters['mult'] = 1
        print('Reinitialising gaussian calculator. Multiplicity = ' + str(self.parameters['mult']))


    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write(self.label + '.com', atoms, properties=properties,
              format='gaussian-in', **self.parameters)

    def read_results(self):
        output = read(self.label + '.log', format='gaussian-out')
        try:
            self.calc = output.calc
            self.results = output.calc.results
        except:
            print('Gaussian error')
            i = 0
            while Path('gauserror'+str(i)+'.log').exists():
                i += 1
            copyfile(self.label + '.log', 'gauserror'+str(i)+'.log')


    # Method(s) defined in the old calculator, added here for
    # backwards compatibility
    def clean(self):
        for suffix in ['.com', '.chk', '.log']:
            try:
                os.remove(os.path.join(self.directory, self.label + suffix))
            except OSError:
                pass

    def get_version(self):
        raise NotImplementedError  # not sure how to do this yet

    def minimise_stable(self, path = os.getcwd(), atoms: Optional[Atoms] = None):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        opt = GaussianOptimizer(atoms, self)
        opt.run(steps=100, opt='calcall, cartesian')
        os.chdir(current_dir)

    def minimise_stable_write(self, dihedral=None, path=os.getcwd(), title="gauss", atoms: Optional[Atoms] = None, rigid = False):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        print(str(dihedral))
        if dihedral != None:
            mod = self.get_modred_lines(dihedral,None)
            if rigid:
                write(str(title) + '.com', atoms, format='gaussian-in', addsec=str(mod), **self.parameters)
            else:
                write(str(title) + '.com', atoms, format='gaussian-in', extra='opt=(calcall, modredundant, tight) int=ultrafine',addsec=str(mod), **self.parameters)
            f=open(str(title) + '.com','r')
            lines = f.readlines()
            pop_point = -3 - len(dihedral)
            lines.pop(pop_point)
            f.close()
            f=open(str(title) + '.com','w')
            f.writelines(lines)
            f.close()
        else:
            if rigid:
                write(str(title) + '.com', atoms, format='gaussian-in', **self.parameters)
            else:
                write(str(title) + '.com', atoms, format='gaussian-in', extra='opt=(calcall)', **self.parameters)
        os.chdir(current_dir)

    def projected_frequency(self, path=os.getcwd(), title="gauss", atoms: Optional[Atoms] = None):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        write(str(title) + '.com', atoms, format='gaussian-in', extra='frequency=(projected,anharmonic)', **self.parameters)
        os.chdir(current_dir)

    def minimise_ts_write(self, dihedral=None, fixed_bonds = None, path=os.getcwd(), title="gauss", atoms: Optional[Atoms] = None, rigid=False):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        print(str(rigid))

        if dihedral != None:
            mod = self.get_modred_lines(dihedral,fixed_bonds,title)
            if not rigid and fixed_bonds != None:
                write(str(title) + '.com', atoms, chk=str(title)+'.chk', parallel=False, format='gaussian-in',extra='opt=(calcall,modredundant)', addsec=str(mod), **self.parameters)
            elif not rigid:
                write(str(title) + '.com', atoms, parallel=False, format='gaussian-in', extra='opt=(calcall,ts,noeigentest, modredundant,tight) int=ultrafine', addsec=str(mod), **self.parameters)
            else:
                write(str(title) + '.com', atoms, parallel=False, format='gaussian-in', addsec=str(mod), **self.parameters)
            f=open(str(title) + '.com','r')
            lines = f.readlines()
            try:
                pop_point = -4 - (len(dihedral)+len(fixed_bonds))
                print('pop point = ' +str(pop_point))
            except:
                pop_point = -4 - len(dihedral)
                print('pop point = ' + str(pop_point))
            if fixed_bonds != None:
                pop_point -= 7
            lines.pop(pop_point)
            f.close()
            f=open(str(title) + '.com','w')
            f.writelines(lines)
            f.close()
        else:
            if not rigid:
                write(str(title) + '.com', atoms, format='gaussian-in', extra='opt=(calcall,ts,noeigentest)', **self.parameters)
            else:
                write(str(title) + '.com', atoms, format='gaussian-in', **self.parameters)
        os.chdir(current_dir)


    def minimise_ts(self,path=os.getcwd(), atoms=None, ratoms=None, patoms=None):
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        opt = GaussianOptimizer(ratoms, self)
        converged = False
        try:
            string = self.get_additional_lines(atoms,patoms)
            converged = opt.run(steps=100, opt='calcall, qst3, noeigentest', addsec=string)
        except:
            pass
        if converged == False:
            opt = GaussianOptimizer(atoms, self)
            converged = opt.run(steps=100, opt='calcall, ts, noeigentest')
        os.chdir(current_dir)
        return  ratoms, patoms, [], []

    def get_frequencies(self, path=os.getcwd(), atoms = None, bimolecular = False, TS = False):
        imaginary_frequency = 0
        current_dir = os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        if TS:
            freqs,zpe,imaginary_frequency,hessian,correct = self.read_ts_vibs()
        else:
            freqs,zpe,hessian = self.read_vibs()
        os.chdir(current_dir)
        return freqs, zpe, imaginary_frequency, hessian

    def read_vibs(self, file=None):
        vibs = []
        zpe = 0
        if file == None:
            file = self.label + '.log'
        inp = open(file, "r")
        for line in inp:
            if re.search("Frequencies", line):
                l = line.split()
                vibs.append(float(l[2]))
                zpe += float(l[2])
                try:
                    vibs.append(float(l[3]))
                    zpe += float(float(l[3]))
                except:
                    pass
                try:
                    vibs.append(float(l[4]))
                    zpe += float(float(l[4]))
                except:
                    pass
        zpe *= 0.00012
        zpe /= 2
        inp.close()
        try:
            with open(str(self.label) + '.log', "r") as inp2:
                data = inp2.read().replace('\n','')
                pattern = r'NImag=0\\\\(.*?)\\\\'
                substring = re.search(pattern, data).group(1)
                for elem in string.whitespace:
                   substring = substring.replace(elem, '')
                hessian = substring.split(",")
        except:
            hessian =[]
        return vibs, zpe, hessian

    def read_ts_vibs(self):
        vibs = []
        zpe = 0
        inp = open(str(self.label) + '.log', "r")
        correct = True
        for line in inp:
            if re.search("Frequencies", line):
                l = line.split()
                try:
                    vibs.append(float(l[2]))
                    zpe += float(l[2])
                except:
                    pass
                try:
                    vibs.append(float(l[3]))
                    zpe += float(float(l[3]))
                except:
                    pass
                try:
                    vibs.append(float(l[4]))
                    zpe += float(float(l[4]))
                except:
                    pass
            if re.search("Error termination", line):
                print('Error')
                return 0
        print('here are the vibs')
        print(str(vibs))
        if vibs[0] > -250:
            print("GaussianTS has no imaginary Frequency")
            correct = False
        if vibs[1] < 0:
            print("GaussianTS has more than 1 imaginary Frequency")
            correct = False
        print('read vibs')
        print(str(vibs))
        zpe -= vibs[0]
        imaginaryFreq = abs((vibs[0]))
        vibs.pop(0)
        zpe *= 0.012
        zpe /= 2
        inp.close()
        try:
            with open(str(self.label) + '.log', "r") as inp2:
                data = inp2.read().replace('\n','')
                pattern =r'NImag=1\\\\(.*?)\\\\'
                substring = re.search(pattern, data).group(1)
                for elem in string.whitespace:
                   substring = substring.replace(elem, '')
                hessian = substring.split(",")
        except:
            hessian =[]
        print('finished reading TS vibs')
        return vibs, zpe, imaginaryFreq,hessian, correct

    def get_additional_lines(self,ts,p):
        string1 = "Prod\n\n0 "+str(2)+"\n"
        xyz1 = tl.convertMolToGauss(p)
        string2 = "TS\n\n0 "+str(2)+"\n"
        xyz2 = tl.convertMolToGauss(ts)
        return string1+xyz1+string2+xyz2

    def get_modred_lines(self,dihedral,bonds,title):
        string = ""
        print(str(bonds))
        if isinstance(dihedral[0], list):
            for d in dihedral:
                string += 'D ' + str(d[0] + 1) + " " + str(d[1] + 1) + " " + str(d[2] + 1) + " " + str(
                    d[3] + 1) + " F\n"
        else:
            string += 'D ' + str(dihedral[0] + 1) + " " + str(dihedral[1] + 1) + " " + str(dihedral[2] + 1) + " " + str(
                dihedral[3] + 1) + " F\n"
        if bonds != None and isinstance(bonds[0], str):
            string += 'B * * F\n'
        elif bonds!= None and isinstance(bonds[0], tuple):
            for b in bonds:
                string += 'B ' + str(b[0] + 1) + " " + str(b[1] + 1) + " F\n"
        elif bonds != None:
            string += 'B ' + str(bonds[0][0] + 1) + " " + str(bonds[0][1] + 1) +  " F\n"

        string += '\n--Link1--\n'
        string += '%Chk='+str(title)
        string += '\n%NoSave\n# M062X/6-31+G** Geom=Check Guess=Read opt=(ts, calcall,noeigentest)\n\n'
        string += 'Title\n\n0 2\n'

        return string
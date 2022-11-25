from subprocess import Popen, PIPE
import os
import src.master_equation.io as io

# Class to run master equation calculation
class MasterEq:

    def __init__(self, start_mol, max_time = 1, temperature = 298, pressure = 1):

        self.newSpeciesFound = False
        self.max_time = max_time
        self.temperature = temperature
        self.pressure = pressure
        self.time = 0.0
        self.ene = 0
        self.prodName = 'none'
        self.visitedList = []
        self.current_node = ''
        self.xml = io.writeTemplate(start_mol=start_mol)
        io.write_to_file(self.xml)
        try:
            self.MESCommand = os.environ['CHEMDYME_ME_PATH']
        except:
            self.MESCommand = '/Users/chmrsh/Documents/MacMESMER/mesmer'

    def run_stochastic_transition(self, dummy=False):
        p = Popen([self.MESCommand,'file.xml'], stdout=PIPE, stderr=PIPE )
        stdout, stderr = p.communicate()
        out = stderr.decode("utf-8")
        lines = str(out).split('\n')
        words = lines[len(lines)-5].split(' ')
        try:
            self.ene = float(words[1])
            words = lines[len(lines)-4].split(' ')
            self.time = float(words[1])
            words = lines[len(lines)-3].split(' ')
            self.prodName = words[1]
            self.visitedList.append(self.prodName)
            if not dummy:
                io.update_starting_population(self.xml,self.ene,self.prodName)
            return True
        except:
            return False

    def add_molecule(self, mol, bi = False):
        mol.write_cml(zero_energy = bi)
        io.add_molecule(self.xml,mol.cml)


    def add_reaction(self, reac):
        reac.write_cml()
        io.add_reaction(self.xml, reac.cml)



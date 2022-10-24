from subprocess import Popen, PIPE
import os
import src.master_equation.io as io

# Class to run master equation calculation
class MasterEq:

    def __init__(self, max_time = 1, temperature = 298, pressure = 1 ):

        self.newSpeciesFound = False
        self.max_time = max_time
        self.temperature = temperature
        self.pressure = pressure
        self.time = 0.0
        self.ene = 0
        self.prodName = 'none'
        self.visitedList = []
        self.xml = io.writeTemplate(start_mol='ccc1')
        io.write_to_file(self.xml)
        try:
            self.MESCommand = os.environ['CHEMDYME_ME_PATH']
        except:
            self.MESCommand = 'mesmer'

    def runTillReac(self, args2):
        p = Popen([self.MESCommand,args2], stdout=PIPE, stderr=PIPE )
        stdout, stderr = p.communicate()
        out = stderr.decode("utf-8")
        lines = str(out).split('\n')
        words = lines[len(lines)-5].split(' ')
        self.ene = float(words[1])
        words = lines[len(lines)-4].split(' ')
        self.time = float(words[1])
        words = lines[len(lines)-3].split(' ')
        self.prodName = words[1]

    def add_molecule(self, mol):
        mol.write_cml()
        io.add_molecule(self.xml,mol.cml)


    def add_reaction(self, reac):
        reac.write_cml()
        io.add_molecule(self.xml, reac.cml)



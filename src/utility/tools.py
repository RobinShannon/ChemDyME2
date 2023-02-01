from ase import Atoms
import numpy as np
try:
    from openbabel import openbabel, pybel
except:
    import openbabel, pybel
from ase.optimize import BFGS

def convertMolToGauss(mol):
    atoms = mol.get_atomic_numbers()
    cart = mol.get_positions()
    # Create open babel molecule BABMol
    BABmol = openbabel.OBMol()
    for i in range(0,atoms.size):
        a = BABmol.NewAtom()
        a.SetAtomicNum(int(atoms[i]))
        a.SetVector(float(cart[i,0]), float(cart[i,1]), float(cart[i,2]))

    # Assign bonds and fill out angles and torsions
    BABmol.ConnectTheDots()
    BABmol.FindAngles()
    BABmol.FindTorsions()
    BABmol.PerceiveBondOrders()
    BABmol.SetTitle('')

    #Create converter object to convert from XYZ to cml
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "gjf")
    gjf = (obConversion.WriteString(BABmol))
    # Just get molecular coordinates in gaussian form, dont trust obabel to get the right spin multiplicity
    gjf = gjf.split('\n')[5:]
    return '\n'.join(gjf)

def fitFourier2D(energies,angles,coeffs):
    cs = []
    c11 = []
    c22 = []
    c33 = []
    c44 = []
    for i in range(0,coeffs[0]):
        for j in range(0,coeffs[1]):
            k = 4
            if i == 0 and j == 0:
                k = 1
            elif i == 0 or j == 0:
                k = 2
            c1 = 0
            c2 = 0
            c3 = 0
            c4 = 0
            for e,a in zip(energies,angles):
                c1 += e * np.cos(i * a[0]) * np.cos(j* a[1])
                c2 += e * np.cos(i * a[0]) * np.sin(j * a[1])
                c3 += e * np.sin(i * a[0]) * np.cos(j * a[1])
                c4 += e * np.sin(i * a[0]) * np.sin(j * a[1])

            c1 *= k / len(energies)
            c2 *= k / len(energies)
            c3 *= k / len(energies)
            c4 *= k / len(energies)
            c11.append(c1)
            c22.append(c2)
            c33.append(c3)
            c44.append(c4)
    cs.append(c11)
    cs.append(c22)
    cs.append(c33)
    cs.append(c44)
    return cs

def fitFourier2Dsimp(energies,angles,coeffs):
    cs = []
    c11 = []
    c22 = []
    c33 = []
    c44 = []
    size = int(np.sqrt(len(energies))-2)
    W1 = [4]*int(size)
    W1[1::2] = [2] * int((size/2))
    W1.append(1)
    W1.insert(0,1)
    print(str(W1))

    W = []
    for i in range(0,size+2):
        for j in range(0,size+2):
            W.append(W1[i]*W1[j])
    print(str(W))

    for i in range(0,coeffs):
        for j in range(0,coeffs):
            k = 4
            if i == 0 and j == 0:
                k = 1
            elif i == 0 or j == 0:
                k = 2
            c1 = 0
            c2 = 0
            c3 = 0
            c4 = 0
            for e,a,w in zip(energies,angles,W):
                c1 += e * np.cos(i*a[0]) * np.cos(j*a[1]) * w
                c2 += e * np.cos(i * a[0]) * np.sin(j * a[1]) * w
                c3 += e * np.sin(i * a[0]) * np.cos(j * a[1]) * w
                c4 += e * np.sin(i * a[0]) * np.sin(j * a[1]) * w
            c1 *= (k) / ( 9 * (np.sqrt(len(energies)) - 1)*(np.sqrt(len(energies))-1))
            c2 *= (k) / ( 9 * (np.sqrt(len(energies)) - 1)*(np.sqrt(len(energies))-1))
            c3 *= (k) / ( 9 * (np.sqrt(len(energies)) - 1)*(np.sqrt(len(energies))-1))
            c4 *= (k) / ( 9 * (np.sqrt(len(energies)) - 1)*(np.sqrt(len(energies))-1))
            c11.append(c1)
            c22.append(c2)
            c33.append(c3)
            c44.append(c4)
    cs.append(c11)
    cs.append(c22)
    cs.append(c33)
    cs.append(c44)
    return cs


def Fourier2D(coeffs, angles,number_of_c):
    pot = 0
    for i in range(0, number_of_c[0]):
        for j in range(0,number_of_c[1]):
            twoDIndex = (i * number_of_c[1]) + j
            pot += (coeffs[0][twoDIndex] * np.cos(i * angles[0]) * np.cos(j * angles[1]))
            pot += (coeffs[1][twoDIndex] * np.cos(i * angles[0]) * np.sin(j * angles[1]))
            pot += (coeffs[2][twoDIndex] * np.sin(i * angles[0]) * np.cos(j * angles[1]))
            pot += (coeffs[3][twoDIndex] * np.sin(i * angles[0]) * np.sin(j * angles[1]))
    return pot

def Fourier3D(coeffs, angles,number_of_c):
    pot = 0
    for i in range(0, number_of_c[0]):
        for j in range(0,number_of_c[1]):
            for k in range(0,number_of_c[2]):
                threeDIndex = (i * number_of_c[2]*number_of_c[1]) + ( j * number_of_c[2] ) + k
                pot += (coeffs[0][threeDIndex] * np.cos(i * angles[0]) * np.cos(j * angles[1]) + np.cos(k * angles[2]))
                if i < number_of_c[0]/2 and j < number_of_c[1]/2 and k < number_of_c[2]/2:
                    pot += (coeffs[1][threeDIndex] * np.cos(i * angles[0]) * np.cos(j * angles[1]) + np.sin(k * angles[2]))
                    pot += (coeffs[2][threeDIndex] * np.cos(i * angles[0]) * np.sin(j * angles[1]) + np.cos(k * angles[2]))
                    pot += (coeffs[3][threeDIndex] * np.cos(i * angles[0]) * np.sin(j * angles[1]) + np.sin(k * angles[2]))
                    pot += (coeffs[4][threeDIndex] * np.sin(i * angles[0]) * np.cos(j * angles[1]) + np.cos(k * angles[2]))
                    pot += (coeffs[5][threeDIndex] * np.sin(i * angles[0]) * np.cos(j * angles[1]) + np.sin(k * angles[2]))
                    pot += (coeffs[6][threeDIndex] * np.sin(i * angles[0]) * np.sin(j * angles[1]) + np.cos(k * angles[2]))
                    pot += (coeffs[7][threeDIndex] * np.sin(i * angles[0]) * np.sin(j * angles[1]) + np.sin(k * angles[2]))

    return pot

def fitFourier3D(energies,angles,coeffs):
    cs = []
    c11 = []
    c22 = []
    c33 = []
    c44 = []
    c55 = []
    c66 = []
    c77 = []
    c88 = []
    for i in range(0,coeffs[0]):
        for j in range(0,coeffs[1]):
            for k in range(0,coeffs[2]):
                l = 8
                if i == 0 and j == 0 and k == 0:
                    l = 1
                elif (i == 0 and j == 0) or (i == 0 and k == 0) or (j == 0 and k == 0):
                    l = 2
                elif (i == 0 ) or (k == 0) or (j == 0):
                    l = 4
                c1 = 0
                c2 = 0
                c3 = 0
                c4 = 0
                c5 = 0
                c6 = 0
                c7 = 0
                c8 = 0
                for e,a in zip(energies,angles):
                    c1 += e * np.cos(i * a[0]) * np.cos(j* a[1])  * np.cos( k * a[2])
                    c2 += e * np.cos(i * a[0]) * np.cos(j * a[1]) * np.sin( k * a[2])
                    c3 += e * np.cos(i * a[0]) * np.sin(j * a[1]) * np.cos( k * a[2])
                    c4 += e * np.cos(i * a[0]) * np.sin(j * a[1]) * np.sin( k * a[2])
                    c5 += e * np.sin(i * a[0]) * np.cos(j* a[1])  * np.cos( k * a[2])
                    c6 += e * np.sin(i * a[0]) * np.cos(j * a[1]) * np.sin( k * a[2])
                    c7 += e * np.sin(i * a[0]) * np.sin(j * a[1]) * np.cos( k * a[2])
                    c8 += e * np.sin(i * a[0]) * np.sin(j * a[1]) * np.sin( k * a[2])

                c1 *= l / (len(energies))
                c2 *= l / (len(energies))
                c3 *= l / (len(energies))
                c4 *= l / (len(energies))
                c5 *= l / (len(energies))
                c6 *= l / (len(energies))
                c7 *= l / (len(energies))
                c8 *= l / (len(energies))
                c11.append(c1)
                c22.append(c2)
                c33.append(c3)
                c44.append(c4)
                c55.append(c5)
                c66.append(c6)
                c77.append(c7)
                c88.append(c8)
    cs.append(c11)
    cs.append(c22)
    cs.append(c33)
    cs.append(c44)
    cs.append(c55)
    cs.append(c66)
    cs.append(c77)
    cs.append(c88)
    return cs




# Function takes a molecule in ASE format, converts it into an OBmol and then returns a SMILES string as a name
def getSMILES(mol, opt, partialOpt = False):
    if opt:
        min = BFGS(mol)
        if partialOpt:
            try:
                min.run(fmax=0.25, steps=5)
            except:
                min.run(fmax=0.25, steps=1)
        else:
            try:
                min.run(fmax=0.1, steps=100)
            except:
                min.run(fmax=0.1, steps=1)

    # Get list of atomic numbers and cartesian coords from ASEmol
    atoms = mol.get_atomic_numbers()
    cart = mol.get_positions()

    # Create open babel molecule BABMol
    BABmol = openbabel.OBMol()
    for i in range(0,atoms.size):
        a = BABmol.NewAtom()
        a.SetAtomicNum(int(atoms[i]))
        a.SetVector(float(cart[i,0]), float(cart[i,1]), float(cart[i,2]))

    # Assign bonds and fill out angles and torsions
    BABmol.ConnectTheDots()
    BABmol.FindAngles()
    BABmol.FindTorsions()
    BABmol.PerceiveBondOrders()

    #Create converter object to convert from XYZ to smiles
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "can")
    name = (obConversion.WriteString(BABmol))

    # Convert "." to "____" to clearly differentiate between mol_types
    name = name.replace('/' , '')
    # These options make trans / cis isomers indistinguishable and ignore chirality
    name = name.replace('\/' , '')
    name = name.replace('@' , '')
    name = name.strip('\n\t')
    name = name.split('.')
    return name

# Function takes a molecule in ASE format, converts it into an OBmol and then returns a CML stream
def getCML(ASEmol, name):

    # Get list of atomic numbers and cartesian coords from ASEmol
    atoms = ASEmol.get_atomic_numbers()
    cart = ASEmol.get_positions()

    # Create open babel molecule BABMol
    BABmol = openbabel.OBMol()
    for i in range(0,atoms.size):
        a = BABmol.NewAtom()
        a.SetAtomicNum(int(atoms[i]))
        a.SetVector(float(cart[i,0]), float(cart[i,1]), float(cart[i,2]))

    # Assign bonds and fill out angles and torsions
    BABmol.ConnectTheDots()
    BABmol.FindAngles()
    BABmol.FindTorsions()
    BABmol.PerceiveBondOrders()
    BABmol.SetTitle('xxxx')

    #Create converter object to convert from XYZ to cml
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "cml")
    cml = (obConversion.WriteString(BABmol))
    cml = cml.replace('xxxx', name)

    return cml

def read_mod_redundant(file):
    with open(file,'r') as f:
        lines = f.readlines()
        last_line = lines[-3]
        last_line=last_line.strip("FD\n")
        last_line=last_line.split(' ')
        last_line.pop(0)
        last_line.pop(-1)
        return last_line

def read_mod_redundant2d(file):
    diheds = []
    with open(file,'r') as f:
        lines = f.readlines()
        last_line = lines[-3]
        last_line=last_line.strip("FD\n")
        last_line=last_line.split(' ')
        last_line.pop(0)
        last_line.pop(-1)
        seccond_last = lines[-4]
        seccond_last=seccond_last.strip("FD\n")
        seccond_last=seccond_last.split(' ')
        seccond_last.pop(0)
        seccond_last.pop(-1)
        diheds.append(seccond_last)
        diheds.append(last_line)
    return diheds

def read_mod_redundant3d(file):
    diheds = []
    with open(file,'r') as f:
        lines = f.readlines()
        last_line = lines[-3]
        last_line=last_line.strip("FD\n")
        last_line=last_line.split(' ')
        last_line.pop(0)
        last_line.pop(-1)
        seccond_last = lines[-4]
        seccond_last=seccond_last.strip("FD\n")
        seccond_last=seccond_last.split(' ')
        seccond_last.pop(0)
        seccond_last.pop(-1)
        third_last = lines[-5]
        third_last=third_last.strip("FD\n")
        third_last=third_last.split(' ')
        third_last.pop(0)
        third_last.pop(-1)
        diheds.append(third_last)
        diheds.append(seccond_last)
        diheds.append(last_line)
    return diheds

def getSpinMult(mol, name, trip = False):
    #Babel incorrectly guessing spin multiplicity from trajectory snapshots
    #If molecule is O2 or O assume triplet rather than singlet
    if name == '[O][O]' or name == '[O]':
        spinMult = 3
    else:
    # else count electrons to determine whether singlet or doublet
        atom_num = mol.get_atomic_numbers()
        s = sum(atom_num)
        if s % 2 == 0 and trip:
            spinMult = 3
        elif s % 2 == 0:
            spinMult = 1
        else:
            spinMult = 2
    return spinMult

def getMolFromSmile(smile):

    # Create OBabel object from smiles
    smile = smile.replace("____", ".")
    mol = pybel.readstring('smi' , smile)
    mol.make3D()
    dim = len(mol.atoms)
    a = np.zeros(dim)
    b = np.zeros((dim , 3))
    i = 0
    for Atom in mol:
        a[i]= Atom.atomicnum
        b[i] = Atom.coords
        i += 1

    aseMol = Atoms(symbols=a, positions=b)
    return aseMol

# Function gets distance of current geometry from either reactant or product
def getDistAlongS(ref, mol):
    return np.linalg.norm(mol-ref)

def prettyPrint(x, path):
    f = open(path, 'w')
    for line in x.toprettyxml().split('\n'):
        if not line.strip() == '':
            f.write(line + '\n')

def printTraj(file, Mol):

    # Get symbols and positions of current molecule
    symbols = Mol.get_chemical_symbols()
    size = len(symbols)
    xyz = Mol.get_positions()

    # write xyz format to already open file
    file.write((str(size) + ' \n'))
    file.write( '\n')
    for i in range(0,size):
        file.write(( str(symbols[i]) + ' \t' + str(xyz[i][0]) + '\t' + str(xyz[i][1]) + '\t' + str(xyz[i][2]) + '\n'))


def getVibString(viblist, bi, TS):
    zpe = 0
    if bi:
        max = 4
    else:
        max = 5

    if TS:
        max += 1

    vibs = []

    if TS:
        vibs.append(viblist[0].imag)

    for i in range(0, len(viblist)):
        if i > max:
            if viblist[i].real != 0:
                vibs.append(viblist[i].real)
                zpe += viblist[i].real
            else:
                vibs.append(100.0)
                zpe += 100.0
    zpe *= 0.00012
    zpe /= 2
    return vibs,zpe







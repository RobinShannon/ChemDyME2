import numpy as np
import src.Utility.Tools as tl
from time import process_time
import util
try:
    from openbabel import openbabel, pybel
except:
    import openbabel, pybel


def center_of_mass_separation(mol, frag):
    """
    Determine the distance between the centers of mass of two molecular fragments
    :param mol:
    :param frag:
    :return:
    """
    mass1 = 0.0
    mass2 = 0.0
    masses = mol.get_masses()
    com1 = np.zeros(3)
    com2 = np.zeros(3)

    # First need total mass of each fragment
    for i in range(0,masses.size):
        if frag[0] <= i <= frag[1]:
            mass1 += masses[i]
        else:
            mass2 += masses[i]

    # Then determine center of mass co-ordinates
    for i in range(0,masses.size):
        if frag[0] <= i <= frag[1]:
            com1[0] += masses[i] * mol.get_positions()[i,0]
            com1[1] += masses[i] * mol.get_positions()[i,1]
            com1[2] += masses[i] * mol.get_positions()[i,2]
        else:
            com2[0] += masses[i] * mol.get_positions()[i,0]
            com2[1] += masses[i] * mol.get_positions()[i,1]
            com2[2] += masses[i] * mol.get_positions()[i,2]

    com1 /= mass1
    com2 /= mass2

    # Finally calculate the distance between COM1 and COM2
    com_separation = np.sqrt( ((com1[0] - com2[0]) ** 2) + ((com1[1] - com2[1]) ** 2) + ((com1[2] - com2[2]) ** 2))
    return com_separation

def getCOMonly(mol):
    mass = 0.0
    COM = np.zeros(3)
    masses = mol.get_masses()
    # First need total mass of each fragment
    for i in range(0,masses.size):
        mass += masses[i]

    # Then determine center of mass co-ordinates
    for i in range(0,masses.size):
        COM[0] += masses[i] * mol.get_positions()[i,0]
        COM[1] += masses[i] * mol.get_positions()[i,1]
        COM[2] += masses[i] * mol.get_positions()[i,2]

    COM /= mass

    return COM

# Method to return derivative of COM seperation via chain rule
# Needs double CHECKING
def getCOMdel(Mol, frag):
    mass1 = 0.0
    mass2 = 0.0
    masses = Mol.get_masses()
    COM1 = np.zeros(3)
    COM2 = np.zeros(3)
    #First need total mass of each fragment
    for i in range(0,masses.size):
        if i >= frag[0] and i <= frag[1]:
            mass1 += masses[i]
        else :
            mass2 += masses[i]
    #Then determine center of mass co-ordinates
    for i in range(0,masses.size):
        if i >= frag[0] and i <= frag[1]:
            COM1[0] += masses[i] * Mol.get_positions()[i,0]
            COM1[1] += masses[i] * Mol.get_positions()[i,1]
            COM1[2] += masses[i] * Mol.get_positions()[i,2]
        else:
            COM2[0] += masses[i] * Mol.get_positions()[i,0]
            COM2[1] += masses[i] * Mol.get_positions()[i,1]
            COM2[2] += masses[i] * Mol.get_positions()[i,2]

    COM1 /= mass1
    COM2 /= mass2

    # Finally calculate the distance between COM1 and COM2
    COMdist = np.sqrt( ((COM1[0] - COM2[0]) ** 2) + ((COM1[1] - COM2[1]) ** 2) + ((COM1[2] - COM2[2]) ** 2))

    # Now need the derivative component wise
    constraint = np.zeros(Mol.get_positions().shape)
    for i in range(0,masses.size):
        for j in range(0,3):
            constraint[i][j] = 1 / ( 2 * COMdist)
            constraint[i][j] *= 2 * (COM1[j] - COM2[j])
            if i >= frag[0] and i <= frag[1]:
                constraint[i][j] *= -masses[i] / mass1
            else:
                constraint[i][j] *= masses[i] / mass2
    return constraint

# Set up a reference matrix for ideal bond length between any two atoms in the system
# Maps species types onto a grid of stored ideal bond distances stored in the global variables module
def refBonds(mol):
    dict = {'CC' : 1.4, 'CH' : 1.2, 'HC' : 1.2, 'CO' : 1.6, 'OC' : 1.6, 'OH' : 1.2, 'HO' : 1.2, 'OO' : 1.6, 'HH' : 1.0, 'CF' : 1.4, 'FC' : 1.4, 'OF' : 1.4, 'FO' : 1.4, 'HF' : 1.1, 'FH' : 1.1, 'FF' : 1.4 }
    size =len(mol.get_positions())
    symbols = mol.get_chemical_symbols()
    dRef = np.zeros((size,size))
    for i in range(0 ,size) :
        for j in range(0, size) :
            sp = symbols[i] + symbols[j]
            dRef[i][j] = dict[sp]
    return dRef

def bondMatrix(dRef,mol):
    size =len(mol.get_positions())
    C = np.zeros((size,size))
    dratio = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            C[i][j] = mol.get_distance(i,j)
            if i != j:
                dratio[i][j] = C[i][j] / dRef[i][j]
            if i == j:
                C[i][j] = 2
            elif C[i][j] < dRef[i][j]:
                C[i][j] = 1.0
            else:
                C[i][j] = 0.0
    return C

def get_changed_bonds(mol1, mol2):
    r = refBonds(mol1)
    C1 = bondMatrix(r, mol1)
    C2 = bondMatrix(r, mol2)
    indicies = []
    size =len(mol1.get_positions())
    for i in range(1,size):
        for j in range(0,i):
            if C1[i][j] != C2[i][j]:
                indicies.append([i,j])
    ind2 = []
    [ind2.append(item) for item in indicies if item not in ind2]
    return ind2

def get_hbond_idxs(mol):
    r = refBonds(mol)
    C = bondMatrix(r, mol)
    dref = refBonds(mol)
    size =len(mol.get_positions())
    min_dist = 100000
    for i in range(1,size):
        for j in range(0,i):
            if C[i][j] == 0 and (mol.get_distance(i,j) / dref[i][j]) < min_dist:
                min_dist = (mol.get_distance(i,j) / dref[i][j])
                indicies = [[i,j]]
    return indicies

def get_rotatable_bonds(mol, add_bonds):
    # Get list of atomic numbers and cartesian coords from ASEmol
    rotors = []
    rotatablebonds =[]
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

    temp_torsions =[]
    temp_rotatable =[]

    for bond in add_bonds:
        bond_added = BABmol.AddBond(bond[0] + 1, bond[1] + 1, 1)

    for bond in add_bonds:
        torsion = [0,0,0,0]
        atom = BABmol.GetAtom(int(bond[0]+1))
        for neighbour_atom in openbabel.OBAtomAtomIter(atom):
            bnd = atom.GetBond(neighbour_atom)
            if bnd.GetBeginAtom().GetIdx()-1 != bond[0]:
                torsion[0] = bnd.GetBeginAtom().GetIdx()-1
                torsion[1] = bnd.GetEndAtom().GetIdx()-1
            else:
                torsion[1] = bnd.GetBeginAtom().GetIdx()-1
                torsion[0] = bnd.GetEndAtom().GetIdx()-1
            if torsion[0] != bond[1]:
                break
        atom2 = BABmol.GetAtom(int(bond[1]+1))
        for neighbour_atom in openbabel.OBAtomAtomIter(atom2):
            bnd = atom2.GetBond(neighbour_atom)
            if bnd.GetBeginAtom().GetIdx()-1 != bond[1]:
                torsion[3] = bnd.GetBeginAtom().GetIdx()-1
                torsion[2] = bnd.GetEndAtom().GetIdx()-1
            else:
                torsion[2] = bnd.GetBeginAtom().GetIdx()-1
                torsion[3] = bnd.GetEndAtom().GetIdx()-1
            if torsion[3] != torsion[1] and torsion[3] != torsion[0]:
                break
        BABmol.FindRingAtomsAndBonds()
        if torsion[0] != bond[1]:
            temp_torsions.append([torsion[0], torsion[1], torsion[2], torsion[3]])
            temp_rotatable.append([torsion[1], torsion[2]])

    BABmol.FindAngles()
    BABmol.FindTorsions()
    BABmol.FindRingAtomsAndBonds()

    for tor in temp_torsions:
        bond = BABmol.GetBond(tor[1] + 1, tor[2] + 1)
        if not bond.IsInRing():
            rotors.append(tor)
            rotatablebonds.append([tor[1], tor[2]])



    for torsion in openbabel.OBMolTorsionIter(BABmol):
        bond = BABmol.GetBond(torsion[1]+1,torsion[2]+1)
        if not bond.IsInRing():
            if [torsion[1],torsion[2]] not in rotatablebonds:
                rotors.append([torsion[0],torsion[1],torsion[2],torsion[3]])
                rotatablebonds.append([torsion[1],torsion[2]])
    return rotors

def get_bi_xyz(smile1, mol):
    mol1 = tl.getMolFromSmile(smile1)
    COM1 = getCOMonly(mol1)

    #Translate COM1 to the origin
    xyz1 = mol1.get_positions()
    for i in range(0,xyz1.shape[0]):
        xyz1[i][0] -= COM1[0]
        xyz1[i][1] -= COM1[1]
        xyz1[i][2] -= COM1[2]

    # Get random point vector at 7 angstrom separation from COM1
    # Get three normally distrubted numbers
    x_y_z = np.random.normal(0,1,3)
    # normalise and multiply by sphere radius
    sum = np.sqrt(x_y_z[0]**2 + x_y_z[1]**2 + x_y_z[2]**2)
    x_y_z *= 1/sum * 7

    # Get displacement from COM2 to x_y_z
    COM2 = getCOMonly(mol)
    displace = COM2 - x_y_z
    # Modify xyz2 coords accordingly
    xyz2 = mol.get_positions()
    for i in range(0,xyz2.shape[0]):
        xyz2[i][0] -= displace[0]
        xyz2[i][1] -= displace[1]
        xyz2[i][2] -= displace[2]

    # Append xyz2 onto xyz1 and return
    xyz1 = np.append(xyz1,xyz2,axis=0)

    return xyz1

# Vectorised function to quickly get array of euclidean distances between atoms
def getDistVect(mol):
    xyz = mol.get_positions()
    size =len(mol.get_positions())
    D = np.zeros((size,size))
    D = np.sqrt(np.sum(np.square(xyz[:,np.newaxis,:] - xyz), axis=2))
    return D

def getSPRINT(xyz):
    pass

def getDistMatrix(mol,active):
    t = process_time()
    #do some stuff

    #Hack
    try:
        l = active[0].shape[0]
    except:
        l = 0
    if active == "all":
        s1 = len(mol.get_positions())
        s2 = s1*(s1+1)/2
    else:
        s2 = len(active)
    D = np.zeros((s2))
    Dind = []
    if active == "all":
        n = 0
        for i in range(0,s1):
            for j in range(0,(s1 - i)):
                Dist = mol.get_distance(i,j)
                D[n] = Dist
                Dind.append((i,j))
                n += 1
    #Hack to to read principle component in form of linear combination of atomic distances
    elif l > 2:
        dist = getDistVect(mol)
        Dind = active
        D,Dind = util.getPC(active,dist)
    elif l >1:
        for i in range(0,s2):
            D[i] = mol.get_distance(int(active[i][0]),int(active[i][1]))
            Dind.append([active[i][0],active[i][1]])
    else:
        D = mol.get_distance(int(active[0]),int(active[1]))
        Dind.append([active[0], active[1]])
    elapsed_time = process_time() - t
    #print("time to get S = " + str(elapsed_time))
    return D,Dind




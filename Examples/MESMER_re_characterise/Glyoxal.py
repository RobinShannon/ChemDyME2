import MESMER_API.src.Main as me_main
import ChemDyME2.src.MechanismGeneration.Species as species
import ChemDyME2.src.MechanismGeneration.Calculator_manager as CM
import multiprocessing

def refine_mol(name):
    me.parse_me_xml('Glyoxal.xml')
    mol = me.mols_dict[name]
    if mol.role != 'ts':
        if 'comp' in mol.name:
            calculator_manager = CM.Calculator_manager(calc_hindered_rotors=False, multi_level=False)
            sp = species.vdw(mol.ase_mol, calculator_manager, dir = str(mol.name))
            sp.write_cml()
        else:
            calculator_manager = CM.Calculator_manager(calc_hindered_rotors=False, multi_level=False)
            sp = species.Stable(mol.ase_mol, calculator_manager, dir = str(mol.name))
            sp.write_cml()
    else:
        for reac in me.reactions_dict.values():
            if reac.ts is not None and reac.ts.name == mol.name:
                r_mol = reac.reacs[0].ase_mol
                p_mol = reac.prods[0].ase_mol
                calculator_manager = CM.Calculator_manager(calc_hindered_rotors=False, multi_level=False)
                sp = species.TS(mol.ase_mol, calculator_manager, r_mol, p_mol,dir = str(mol.name))
                sp.write_cml()
                break



me = me_main.MESMER_API()
me.parse_me_xml('RingTS.xml')
number_of_mols = str(len(me.mols_dict))
values = me.mols_dict.keys()
values_list = list(values)
#p = multiprocessing.Pool(int(number_of_mols))
#results = p.map(refine_mol, values_list)
#outputs = [result for result in results]
for val in values_list:
    refine_mol(val)

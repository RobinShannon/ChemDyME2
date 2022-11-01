import lxml.etree as ET

import numpy as np
from xml.dom import minidom
from datetime import date


# write a template xml mesmer input which will populated as the mechanism generation proceeds
def writeTemplate(start_mol, temperature = 500 , pressure = 1, end_time = 1):
    ET.register_namespace('me', 'http://www.chem.leeds.ac.uk/mesmer')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')

    #Set up main me:mesmer xml tag and link to all the schemas
    mesmer = ET.Element('{http://www.chem.leeds.ac.uk/mesmer}mesmer')

    # First sub element is the title
    title = ET.SubElement(mesmer, '{http://www.chem.leeds.ac.uk/mesmer}title')
    title.text = str(start_mol) + ' ChemDyME network ' + date.today().strftime("%b-%d-%Y")

    # Molecule list
    m_ele = ET.Element('moleculeList')
    mesmer.insert(1, m_ele)

    # Reaction list
    r_ele = ET.Element('reactionList')
    mesmer.insert(2, r_ele)

    # Conditions here we assume a N2 bath gas and make a single pressure temperature pair based upon the input
    conditions = ET.SubElement(mesmer, '{http://www.chem.leeds.ac.uk/mesmer}conditions')
    bath_gas = ET.SubElement(conditions, '{http://www.chem.leeds.ac.uk/mesmer}bathGas')
    bath_gas.text = str('N2')
    pts = ET.SubElement(conditions, '{http://www.chem.leeds.ac.uk/mesmer}PTs')
    pt = ET.SubElement(pts, '{http://www.chem.leeds.ac.uk/mesmer}PTpair')
    pt.set('{http://www.chem.leeds.ac.uk/mesmer}P',str(pressure))
    pt.set('{http://www.chem.leeds.ac.uk/mesmer}T',str(temperature))
    pt.set('{http://www.chem.leeds.ac.uk/mesmer}units',"Torr")

    # initial populations also appear in the conditions element and should be set so that all the population is in the
    # inital reactant. These will be updated as mechanism generation progresses
    initial_populations = ET.SubElement(conditions, '{http://www.chem.leeds.ac.uk/mesmer}InitialPopulation')
    pop = ET.SubElement(initial_populations, '{http://www.chem.leeds.ac.uk/mesmer}molecule')
    pop.set('grain','1.0')
    pop.set('population','1.0')
    pop.set('ref',str(start_mol))

    # appropriate default model parameters for a stochastic run
    model_parameters = ET.SubElement(mesmer, '{http://www.chem.leeds.ac.uk/mesmer}modelParameters')
    stoch_trials = ET.SubElement(model_parameters, '{http://www.chem.leeds.ac.uk/mesmer}numberStochasticTrials')
    stoch_trials.text = '1'
    stoch_start_time = ET.SubElement(model_parameters, '{http://www.chem.leeds.ac.uk/mesmer}stochasticStartTime')
    stoch_start_time.text = '1E-11'
    stoch_end_time = ET.SubElement(model_parameters, '{http://www.chem.leeds.ac.uk/mesmer}stochasticEndTime')
    stoch_end_time.text = str(end_time)
    stoch_axd_limit = ET.SubElement(model_parameters, '{http://www.chem.leeds.ac.uk/mesmer}StochasticAxdLimit')
    stoch_axd_limit.text = '50'
    grain_size = ET.SubElement(model_parameters, '{http://www.chem.leeds.ac.uk/mesmer}grainSize')
    grain_size.set('units', 'cm-1')
    grain_size.text = '50'
    max_energy = ET.SubElement(model_parameters, '{http://www.chem.leeds.ac.uk/mesmer}energyAboveTheTopHill')
    max_energy.text = '25'

    # conditions section
    control = ET.SubElement(mesmer, '{http://www.chem.leeds.ac.uk/mesmer}control')
    ET.SubElement(control, '{http://www.chem.leeds.ac.uk/mesmer}printSpeciesProfile')
    ET.SubElement(control, '{http://www.chem.leeds.ac.uk/mesmer}stochasticOnePass')
    ET.SubElement(control, '{http://www.chem.leeds.ac.uk/mesmer}stochasticSimulation')
    return mesmer

def write_to_file(mesmer):
    tree = ET.ElementTree(mesmer)
    ET.indent(tree, space='\n    ')
    xml = ET.tostring(tree, encoding='unicode', pretty_print=True)
    xml = xml.splitlines()
    strip_lines = [line for line in xml if line.strip() != ""]
    mod_xml = ""
    for line in strip_lines:
        mod_xml += line + "\n"
    f = open('file.xml', "w")
    f.write(mod_xml)
    f.close()

def update_starting_population(mesmer, ene, species):
    InitialPopulation = mesmer.findall('{http://www.chem.leeds.ac.uk/mesmer}conditions')[0].findall('{http://www.chem.leeds.ac.uk/mesmer}InitialPopulation')[0]
    mol = InitialPopulation.findall('{http://www.chem.leeds.ac.uk/mesmer}molecule')[0]
    mol.set('grain',str(ene))
    mol.set('ref',str(species))
    write_to_file(mesmer)

def add_molecule(mesmer,cml_mol):
    mols = mesmer.findall("moleculeList")[0]
    mols.insert(-1,cml_mol)
    write_to_file(mesmer)

def add_reaction(mesmer,cml_reac):
    reacs = mesmer.findall("reactionList")[0]
    reacs.insert(-1,cml_reac)
    write_to_file(mesmer)

def change_parameters( param_dict, ):
    return 1
import xml.etree.cElementTree as ET

import numpy as np
from xml.dom import minidom
from datetime import date


# write a template xml mesmer input which will populated as the mechanism generation proceeds
def writeTemplate(start_mol, temperature = 500 , pressure = 1, end_time = 1):

    #Set up main me:mesmer xml tag and link to all the schemas
    mesmer = ET.Element('me:mesmer')
    mesmer.set('xmlns','http://www.xml-cml.org/schema')
    mesmer.set('xmlns:me',"http://www.chem.leeds.ac.uk/mesmer")
    mesmer.set('xmlns:xsi',"http://www.w3.org/2001/XMLSchema-instance")

    # First sub element is the title
    title = ET.SubElement(mesmer, 'me:title')
    title.txt = str(start_mol) + ' ChemDyME network ' + date.today().strftime("%b-%d-%Y")

    # Molecule list
    ET.SubElement(mesmer, 'moleculeList')

    # Reaction list
    ET.SubElement(mesmer, 'moleculeList')

    # Conditions here we assume a N2 bath gas and make a single pressure temperature pair based upon the input
    conditions = ET.SubElement(mesmer, 'me:conditions')
    bath_gas = ET.SubElement(conditions, 'me:bathGas')
    bath_gas.txt = str('N2')
    pts = ET.SubElement(conditions, 'me:PTs')
    pt = ET.SubElement(pts, 'me:PTpair')
    pt.set('me:P',str(pressure))
    pt.set('me:T',str(temperature))
    pt.set('me: units',"Torr")

    # initial populations also appear in the conditions element and should be set so that all the population is in the
    # inital reactant. These will be updated as mechanism generation progresses
    initial_populations = ET.SubElement(conditions, 'me:InitialPopulation')
    pop = ET.SubElement(conditions, 'me:molecule')
    pop.set('grain','1.0')
    pop.set('population','1.0')
    pop.set('ref',str(start_mol))

    # appropriate default model parameters for a stochastic run
    model_parameters = ET.SubElement(mesmer, 'me:modelParameters')
    stoch_trials = ET.SubElement(model_parameters, 'me:numberStochasticTrials')
    stoch_trials.txt = '1'
    stoch_start_time = ET.SubElement(model_parameters, 'me:stochasticStartTime')
    stoch_start_time.txt = '1E-11'
    stoch_end_time = ET.SubElement(model_parameters, 'me:stochasticEndTime')
    stoch_end_time.txt = str(end_time)
    stoch_axd_limit = ET.SubElement(model_parameters, 'me:StochasticAxdLimit')
    stoch_axd_limit.txt = '50'
    grain_size = ET.SubElement(model_parameters, 'me:grainLimit')
    grain_size.set('units', 'cm-1')
    grain_size.txt = '50'
    max_energy = ET.SubElement(model_parameters, 'me:energyAboveTheTopHill')
    max_energy.txt = '25'

    # conditions section
    control = ET.SubElement(mesmer, 'me:control')
    ET.SubElement(control, 'me:printSpeciesProfile')
    ET.SubElement(control, 'me:stochasticOnePass')
    ET.SubElement(control, 'me:stochasticSimulation')

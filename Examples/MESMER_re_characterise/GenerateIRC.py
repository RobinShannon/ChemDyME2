import os
import sys
from ase.io import read, write
from src.Calculators.GaussianCalculator import Gaussian
import src.mechanism_generation.calculator_manager as cm
high = Gaussian(
    nprocshared=1,
    label='Gauss',
    method='M11',
    basis='6-31+G**',
    mult=int(2),
    scf='qc'
)
calculator_manager = cm.calculator_manager(trajectory = high, low=high, high=high, single_point=high, calc_hindered_rotors=False, multi_level=False)
count = -1
i1 = []
irc = []
i1.append(0)
min_point = False
reverse = False
target = open("IRC2_M11.log", "r")
for line in target:

    if "NET REACTION COORDINATE UP TO THIS POINT" in line:
        v1=line.split()
        if not (len(i1) == 0 or abs(float(v1[8])) > abs(i1[-1])):
            reverse = True
        if reverse == False:
            i1.append(-float(v1[8]))
        else:
            i1.append(float(v1[8]))
        min_point = True
    if "Input orientation:" in line:
        if min_point:
            irc.append(read('IRC2_M11.log', index=count))
        count += 1

unsorted_dict = dict(zip(i1, irc))
sorted_dict = {key: value for key, value in sorted(unsorted_dict.items())}
irc = list(sorted_dict.values())
write('irc.xyz', irc)
for i,m in enumerate(sorted_dict.values()):
    calculator_manager.set_calculator(m,'high')
    m._calc.projected_frequency(title="point"+str(i), atoms=m)

irc1 = read('IRC1.log', index=':')
irc2 = read('IRC2_M11.log', index=':')

write('irc1.xyz', irc1)
write('irc2.xyz', irc2)

os.chdir('IRC2')
for i in range(0,56):
    target = open("point"+str(i)+".log", "r")
    for line in target:
        if "Sum of electronic and zero-point Energies=" in line:
            v1 = line.split()
            print(str(v1[6]))

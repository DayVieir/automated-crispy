#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""spectra match."""

#testando git push

# standard libraries
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
import importlib

# support
sys.path.append(r'../')
import support

Path.cwd()

%matplotlib qt5

# %% Initial definitions ======================================================
quanty_exe = Path('Quanty')

# %% General Setup ============================================================
initialSetup_dict = dict(element             = 'Co'
                         ,charge             = '2+'
                         ,symmetry           = 'Oh'
                         ,edge               = 'L2,3 (2p)'
                         ,experiment         = 'XAS'
                         )

generalSetup_dict    = dict()

hamiltonianData_dict = dict()

# %% Create generic parameter file ============================================
q = support.create_input(initialSetup_dict=initialSetup_dict,
                         generalSetup_dict=generalSetup_dict,
                         hamiltonianData_dict=hamiltonianData_dict,
                         folderpath=None,
                         filename=None,
                         magneticField=None,
                         temperature=None,
                         xLorentzian=None,
                         saveParameters2file=True)

# %% import generic parameter file ============================================
initialSetup_dict, generalSetup_dict, hamiltonianData_dict, magneticField, temperature, xLorentzian = support.load_parameter('Co2+_Oh_2p_XAS.par')

print(initialSetup_dict)
print(hamiltonianData_dict)

# %% run quanty ===============================================================
output = support.run_quanty(quanty_exe, filepath='Co2+_Oh_2p_XAS.lua')
print(output)

# %% plot spectrum ============================================================
plt.figure()

temp_data = np.loadtxt('Co2+_Oh_2p_XAS_iso.spec', skiprows=5)
plt.plot(temp_data[:, 0], temp_data[:, 2])

plt.title('This is NOT the same as in Crispy')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity')
plt.grid()

# %% fix spectrum =============================================================
x, y = support.fix_spectrum('Co2+_Oh_2p_XAS_iso.spec', xMin=q.xMin, xMax=q.xMax, xNPoints=q.xNPoints, xGaussian=0.1)

# %% plot final spectrum ======================================================
plt.figure()

plt.plot(x, y)

plt.title('This is the same as in Crispy')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity')
plt.grid()

# %% save final spectrum ======================================================
temp = np.zeros((x.shape[0], 2))
temp[:, 0] = x
temp[:, 1] = y
np.savetxt('Co2+_Oh_2p_XAS_iso.final', temp)

# %% plotting one more time ===================================================
plt.figure()

temp_data = np.loadtxt('Co2+_Oh_2p_XAS_iso.final')
plt.plot(x, y)

plt.title('This is the same as in Crispy ---- final')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity')
plt.grid()

# %% Create many hamiltonianData_dict =========================================
hamiltonianData_dict = {'Atomic': {'Initial Hamiltonian': {"F2(3d,3d)": [11.6051, 0.7, 0.8]}},
                        'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=False)

print(hamiltonianData_dict_list[0])
print(hamiltonianData_dict_list[1])
print(hamiltonianData_dict_list[2])
print(hamiltonianData_dict_list[3])

# %% We can syncronize the initial and final hamiltonian ======================
hamiltonianData_dict = {'Atomic': {'Initial Hamiltonian': {"F2(3d,3d)": [None, 0.7, 0.8]}},
                        'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=True)

print(hamiltonianData_dict_list[0])
print(hamiltonianData_dict_list[1])
print(hamiltonianData_dict_list[2])
print(hamiltonianData_dict_list[3])

# %% We can create input for each hamiltonian data ==============================
folderpath = 'inputs_A'

hamiltonianData_dict = {'Atomic': {'Initial Hamiltonian': {"F2(3d,3d)": [None, 0.7, 0.8]}},
                        'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=True)

i = 0
for hamiltonianData_dict in hamiltonianData_dict_list:
    prefix = f'input_A_{i}'
    q = support.create_input(initialSetup_dict=initialSetup_dict,
                             generalSetup_dict=generalSetup_dict,
                             hamiltonianData_dict=hamiltonianData_dict,
                             folderpath=folderpath,
                             filename=prefix,
                             magneticField=None,
                             temperature=None,
                             xLorentzian=None,
                             saveParameters2file=True)
    i += 1

# %% We can loop over temperatures ============================================
folderpath = 'inputs_B'

hamiltonianData_dict = {'Atomic': {'Initial Hamiltonian': {"F2(3d,3d)": [None, 0.7, 0.8]}},
                        'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=True)

temperature_list = [0, 10, ]

i = 0
for hamiltonianData_dict in hamiltonianData_dict_list:
    for temperature in temperature_list:
        prefix = f'input_B_{i}'
        q = support.create_input(initialSetup_dict=initialSetup_dict,
                                 generalSetup_dict=generalSetup_dict,
                                 hamiltonianData_dict=hamiltonianData_dict,
                                 folderpath=folderpath,
                                 filename=prefix,
                                 magneticField=None,
                                 temperature=temperature,
                                 xLorentzian=None,
                                 saveParameters2file=True)
        i += 1

# %% loop xLorentizian ========================================================
folderpath = 'inputs_C'

hamiltonianData_dict = {'Atomic': {'Initial Hamiltonian': {"F2(3d,3d)": [None, 0.7, 0.8]}},
                        'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=True)

xLorentzian_list = [[0.1], [1, 2], ]

i = 0
for hamiltonianData_dict in hamiltonianData_dict_list:
    for xLorentzian in xLorentzian_list:
        prefix = f'input_C_{i}'
        q = support.create_input(initialSetup_dict=initialSetup_dict,
                                 generalSetup_dict=generalSetup_dict,
                                 hamiltonianData_dict=hamiltonianData_dict,
                                 folderpath=folderpath,
                                 filename=prefix,
                                 magneticField=None,
                                 temperature=None,
                                 xLorentzian=xLorentzian,
                                 saveParameters2file=True)
        i += 1


# %% loop xGaussian ===========================================================
folderpath = Path('inputs_D')

hamiltonianData_dict = {'Atomic': {'Initial Hamiltonian': {"F2(3d,3d)": [None, 0.7, 0.8]}},
                        'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=True)

xGaussian_list = [0.5, 1.72, ]

i = 0
for hamiltonianData_dict in hamiltonianData_dict_list:
        prefix = f'input_D_{i}'
        q = support.create_input(initialSetup_dict=initialSetup_dict,
                                 generalSetup_dict=generalSetup_dict,
                                 hamiltonianData_dict=hamiltonianData_dict,
                                 folderpath=folderpath,
                                 filename=prefix,
                                 magneticField=None,
                                 temperature=None,
                                 xLorentzian=None,
                                 saveParameters2file=True)

        output = support.run_quanty(quanty_exe, filepath=folderpath/f'{prefix}.lua')

        for xGaussian in xGaussian_list:
            x, y = support.fix_spectrum(folderpath/f'{prefix}_iso.spec', xMin=q.xMin, xMax=q.xMax, xNPoints=q.xNPoints, xGaussian=xGaussian)
            temp = np.zeros((x.shape[0], 2))
            temp[:, 0] = x
            temp[:, 1] = y
            np.savetxt(folderpath/f'{prefix}_iso_broaden_{xGaussian}.final', temp)
        i += 1

# %% utimate function =========================================================
folderpath = Path('inputs_E')
prefix = 'Co_2+_Oh_'
importlib.reload(support)

initialSetup_dict = dict(element             = 'Co'
                         ,charge             = '2+'
                         ,symmetry           = 'Oh'
                         ,edge               = 'L2,3 (2p)'
                         ,experiment         = 'XAS'
                         )

generalSetup_dict    = dict()

hamiltonianData_dict = {'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=True)

magneticField_list = [0, 10, ]
temperature_list = [10, ]
xLorentzian_list = [[0.1], ]
xGaussian_list = [0.5, 1.72, ]

support.create_spectra(initialSetup_dict=initialSetup_dict,
                         generalSetup_dict=generalSetup_dict,
                         hamiltonianData_dict_list=hamiltonianData_dict_list,
                         magneticField_list=magneticField_list,
                         temperature_list=temperature_list,
                         xLorentzian_list=xLorentzian_list,
                         xGaussian_list=xGaussian_list,
                         folderpath=folderpath,
                         prefix=prefix,
                         quanty_exe=quanty_exe,
                         verbosity=True)


# %% utimate function for dichroism ===========================================
folderpath = Path('inputs_F')
prefix = 'Co_2+_Oh_'
importlib.reload(support)

initialSetup_dict = dict(element             = 'Co'
                         ,charge             = '2+'
                         ,symmetry           = 'Oh'
                         ,edge               = 'L2,3 (2p)'
                         ,experiment         = 'XAS'
                         )

generalSetup_dict    = dict(toCalculate=['Circular Dichroism', ])

hamiltonianData_dict = {'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : [1, 1.2]}}}
hamiltonianData_dict_list = support.expand_hamiltanianData(hamiltonianData_dict, synchronize=True)

magneticField_list = [0, 10, ]
temperature_list = [10, ]
xLorentzian_list = [[0.1], ]
xGaussian_list = [0.5, 1.72, ]

support.create_spectra(initialSetup_dict=initialSetup_dict,
                         generalSetup_dict=generalSetup_dict,
                         hamiltonianData_dict_list=hamiltonianData_dict_list,
                         magneticField_list=magneticField_list,
                         temperature_list=temperature_list,
                         xLorentzian_list=xLorentzian_list,
                         xGaussian_list=xGaussian_list,
                         folderpath=folderpath,
                         prefix=prefix,
                         quanty_exe=quanty_exe,
                         verbosity=True)

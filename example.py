#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""spectra generation."""

# standard libraries
import matplotlib.pyplot as plt
import sys

# spectragen
import spectragen as sg

plt.ion()  # interactive mode

# %% Example =================================================================

# initialize random calculation with default values
q = sg.calculation_file()
q.saveInput()  # file was saved as untitled.lua

# run quanty
output = sg.run_quanty(filepath_quanty='Quanty', filepath='untitled.lua')

# now we have a iso.spec file that we can load
x, y = sg.load_spectrum('untitled_iso.spec')

# plot spectrum
plt.plot(x, y)

# broaden spectrum
xb, yb = sg.broadening(x, y, xGaussian=.4)

# plot again
plt.plot(xb, yb)

# we can save the parameters used
sg.save_parameters(q, filepath='Ni_calculation')

# we can open Ni_calculation.par and edit parameters there if we want
# we can then load the parameters and run the calculation again any time
par = sg.load_parameters('Ni_calculation.par')

# run again with loaded parameters (let's change the filepath as well)
par['filepath'] = 'Ni_calculation'
q2 = sg.calculation_file(**par)
q2.saveInput()
output = sg.run_quanty(filepath_quanty='Quanty', filepath='Ni_calculation.lua')
x2, y2 = sg.load_spectrum('Ni_calculation_iso.spec')
x2b, y2b = sg.broadening(x2, y2, xGaussian=.4)
plt.plot(x2b, y2b)

# as an example, let's change something in the hamiltonian
par['hamiltonianData']['Atomic']['Initial Hamiltonian']['F2(3d,3d)'][1] = 0.65

# now let's sync this value with the final hamiltonian as well
par['hamiltonianData']['Atomic']['Final Hamiltonian']['F2(3d,3d)'][1] = 0.65

# another way of doing that would be using the sync function
par['hamiltonianData'] = sg.sync(par['hamiltonianData'])

# run again with loaded parameters and the modified hamiltonian
par['filepath'] = 'Ni_calculation2'
q3 = sg.calculation_file(**par)
q3.saveInput()
output = sg.run_quanty(filepath_quanty='Quanty', filepath='Ni_calculation2.lua')
x3, y3 = sg.load_spectrum('Ni_calculation2_iso.spec')
x3b, y3b = sg.broadening(x3, y3, xGaussian=.4)
plt.plot(x3b, y3b)

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""spectra search."""

# standard libraries
from pathlib import Path
import subprocess
import copy
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np

# backpack
sys.path.append(r'/home/dayane/Documents/carlos/06_23_20/py-backpack')
sys.path.append(r'py-backpack')
import backpack.filemanip as fmanip

# crispy
from crispy.gui.quanty import QuantyCalculation
from crispy.gui.quanty import broaden


################# Create input and parameter file
def create_input(initialSetup_dict, generalSetup_dict, hamiltonianData_dict, folderpath=None, filename=None, magneticField=None, temperature=None, xLorentzian=None, saveParameters2file=True):
    """ min xLorentzian is 0.1
        # RIXS
        q.yMin
        q.yMax
        q.yNPoints
        q.yLorentzian
        q.yGaussian

        # filename without .lua

        # q.xGaussian #(> 0)
    """

    q = QuantyCalculation(**copy.deepcopy(initialSetup_dict))

    if 'verbosity' in generalSetup_dict:
        q.verbosity = generalSetup_dict['verbosity']
    else:
        q.verbosity =  '0x0000'

    if 'denseBorder' in generalSetup_dict:
        q.denseBorder = generalSetup_dict['denseBorder']
    else:
        q.denseBorder = '2000'

    # npsis
    if 'nPsisAuto' in generalSetup_dict:
        nPsisAuto = generalSetup_dict['nPsisAuto']
    else:
        nPsisAuto = q.nPsisAuto

    if nPsisAuto == 1:
        q.nPsis = q.nPsisMax
    else:
        if 'nPsis' in generalSetup_dict:
            q.nPsis = generalSetup_dict['nPsis']
        else:
            q.nPsis = q.nPsisMax

    # nConfigurations
    if 'nConfigurations' in generalSetup_dict:
        q.nConfigurations = generalSetup_dict['nConfigurations']

    # temperature
    if temperature is not None:
        q.temperature = temperature

    # broadening
    if xLorentzian is not None:
        q.xLorentzian = copy.copy(xLorentzian)


    # magnetic Field
    if magneticField is not None:
        _updateMagneticField(q, magneticField)
    else:
        _updateMagneticField(q, q.magneticField)

    if 'k1' in generalSetup_dict:
        if 'eps11' in generalSetup_dict:
            q.eps11 = generalSetup_dict['eps11']
        else:
            pass
        _updateIncidentWaveVector(q, generalSetup_dict['k1'])
    elif 'eps11' in generalSetup_dict:
        _updateIncidentWaveVector(q, q.k1)
        _updateIncidentPolarizationVectors(q, generalSetup_dict['eps11'])


    if 'XMin' in generalSetup_dict:
        q.xMin = generalSetup_dict['XMin']
    if 'XMax' in generalSetup_dict:
        q.xMax = generalSetup_dict['XMax']
    if 'xNPoints' in generalSetup_dict:
        q.xNPoints = generalSetup_dict['xNPoints']

    if 'toCalculate' in generalSetup_dict:
        q.spectra.toCalculateChecked = generalSetup_dict['toCalculate']



    # update hamiltonian states
    if 'hamiltonianState' in generalSetup_dict:
        for item in q.hamiltonianState:
            try:
                q.hamiltonianState[item] = generalSetup_dict['hamiltonianState'][item]
            except KeyError:
                warnings.warn(f"{item} not found in generalSetup_dict['hamiltonianState'].")
            # except NameError:
            #     pass

    # fix hamiltonianData_dict
    hamiltonianData_dict2 = copy.deepcopy(hamiltonianData_dict)
    for key in hamiltonianData_dict2: # ['Atomic', 'Crystal Field', 'Magnetic Field', 'Exchange Field', '3d-Ligands Hybridization (LMCT)', '3d-Ligands Hybridization (MLCT)']
        for key2 in hamiltonianData_dict2[key]: # ['Initial Hamiltonian', 'Final Hamiltonian']
            for parameter in q.hamiltonianData[key][key2]:

                try: # double valued parameters (Atomic)
                    if hamiltonianData_dict2[key][key2][parameter][0] is None:  # if parameter not None
                        hamiltonianData_dict2[key][key2][parameter][0] = copy.deepcopy(q.hamiltonianData[key][key2][parameter][0])

                except KeyError:  # if parameter is not defined
                    hamiltonianData_dict2[key][key2][parameter] = copy.deepcopy(q.hamiltonianData[key][key2][parameter])

                except TypeError as e:    # single valued parameters
                    if hamiltonianData_dict2[key][key2][parameter] is None:
                        hamiltonianData_dict2[key][key2][parameter] = copy.deepcopy(q.hamiltonianData[key][key2][parameter])

                except IndexError as e:    # single valued parameters (if value is numpy.float64)
                    if hamiltonianData_dict2[key][key2][parameter] is None:
                        hamiltonianData_dict2[key][key2][parameter] = copy.deepcopy(q.hamiltonianData[key][key2][parameter])
    # update hamiltonianData
    q.hamiltonianData.update(hamiltonianData_dict2)


    # save input ======================================
    if folderpath is None:
        folderpath = Path.cwd()
    else:
        folderpath = Path(folderpath)
    if filename is None:
        filename = q.baseName
    q.baseName = str(folderpath/filename)
    q.saveInput()

    # save parameters ====================================
    if saveParameters2file:
        generalSetup_dict2save = dict(verbosity         = q.verbosity
                                      ,denseBorder      = q.denseBorder
                                      ,nPsisAuto        = q.nPsisAuto
                                      ,nPsis            = q.nPsis
                                      ,nConfigurations  = q.nConfigurations
                                      ,XMin             = q.xMin
                                      ,XMax             = q.xMax
                                      ,xNPoints         = q.xNPoints
                                      ,k1               = q.k1
                                      ,eps11            = q.eps11
                                      ,toCalculate      = q.spectra.toCalculateChecked
                                      ,hamiltonianState = q.hamiltonianState
                                      )

        dict2save = dict(initialSetup      = initialSetup_dict,
                         generalSetup      = generalSetup_dict2save,
                         hamiltonianData   = q.hamiltonianData,
                         magneticField     = q.magneticField,
                         temperature       = q.temperature,
                         xLorentzian       = q.xLorentzian)
        fmanip.save_obj(dict2save, q.baseName+'.par')

    return q


def _updateMagneticField(q, magneticField):

    TESLA_TO_EV = 5.788e-05

    # Normalize the current incident vector.
    k1 = np.array(q.k1)
    k1 = k1 / np.linalg.norm(k1)

    configurations = q.hamiltonianData['Magnetic Field']
    for configuration in configurations:
        parameters = configurations[configuration]
        for i, parameter in enumerate(parameters):
            value = float(magneticField * np.abs(k1[i]) * TESLA_TO_EV)
            if abs(value) == 0.0:
                    value = 0.0
            configurations[configuration][parameter] = value

    q.magneticField = magneticField


def _updateIncidentWaveVector(q, k1):

    # The k1 value should be fine; save it.
    q.k1 = k1

    # The polarization vector must be correct.
    eps11 = q.eps11

    # If the wave and polarization vectors are not perpendicular, select a
    # new perpendicular vector for the polarization.
    if np.dot(np.array(k1), np.array(eps11)) != 0:
        if k1[2] != 0 or (-k1[0] - k1[1]) != 0:
            eps11 = (k1[2], k1[2], -k1[0] - k1[1])
        else:
            eps11 = (-k1[2] - k1[1], k1[0], k1[0])

    q.eps11 = eps11

    # Generate a second, perpendicular, polarization vector to the plane
    # defined by the wave vector and the first polarization vector.
    eps12 = np.cross(np.array(eps11), np.array(k1))
    eps12 = eps12.tolist()

    q.eps12 = eps12

    # Update the magnetic field.
    _updateMagneticField(q, q.magneticField)


def _updateIncidentPolarizationVectors(q, eps11):

    k1 = q.k1
    q.eps11 = eps11

    # Generate a second, perpendicular, polarization vector to the plane
    # defined by the wave vector and the first polarization vector.
    eps12 = np.cross(np.array(eps11), np.array(k1))
    eps12 = eps12.tolist()

    q.eps12 = eps12



def load_parameter(filepath):
    """ initialSetup_dict, generalSetup_dict, hamiltonianData_dict, magneticField, temperature, xLorentzian = load_parameter(filepath)
    """
    a = copy.deepcopy(fmanip.load_obj(filepath))
    initialSetup    = a['initialSetup']
    generalSetup    = a['generalSetup']
    hamiltonianData = a['hamiltonianData']
    magneticField   = a['magneticField']
    temperature     = a['temperature']
    xLorentzian     = a['xLorentzian']
    return initialSetup, generalSetup, hamiltonianData, magneticField, temperature, xLorentzian


def run_quanty(quanty_exe, filepath):
    quanty = subprocess.Popen([f"{quanty_exe} {filepath}"], shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = quanty.stdout.read().decode("utf-8")
    error  = quanty.stderr.read().decode("utf-8")

    if error is not '':
        raise RuntimeError(f'Error while reading file: {filepath}. \n {error}')

    if 'Error while loading the script' in output:
        error = output[output.find('Error while loading the script')+len('Error while loading the script:')+1:]
        warnings.warn(f'Error while loading file: {filepath}. \n {error}')
    return output


def fix_spectrum(filepath, xMin, xMax, xNPoints, xGaussian):

    try:
        data = np.loadtxt(filepath, skiprows=5)
    except (OSError, IOError) as e:
        raise e

    x = np.linspace(xMin, xMax, xNPoints + 1)
    y = data[:, 2::2].flatten()

    fwhm = xGaussian
    xScale = np.abs(x.min() - x.max()) / x.shape[0]
    fwhm = fwhm/xScale
    y = broaden(y, fwhm, 'gaussian')

    return x, y


def expand_hamiltanianData(hamiltonianData_dict, synchronize=False):

    # calculate total number of different hamiltonianData
    t_total = 1
    if 'Atomic' in hamiltonianData_dict:
        key = 'Atomic'
        for key2 in hamiltonianData_dict[key]:  # ['Initial Hamiltonian', 'Final Hamiltonian']
            for dummy, value in hamiltonianData_dict[key][key2].items():
                t_total *= len(value)-1
    for key in hamiltonianData_dict:
        if key == 'Atomic':
            pass
        else:
            for key2 in hamiltonianData_dict[key]:  # ['Initial Hamiltonian', 'Final Hamiltonian']
                for dummy, value in hamiltonianData_dict[key][key2].items():
                    try:
                        t_total *= len(value)
                    except TypeError:
                        t_total *= 1
    hamiltonianData_dict_list = [copy.deepcopy(hamiltonianData_dict) for x in range(0, t_total)]

    a = 1
    if 'Atomic' in hamiltonianData_dict:
        key = 'Atomic'
        for key2 in hamiltonianData_dict[key]:  # ['Initial Hamiltonian', 'Final Hamiltonian']
            # expand parameters to vary
            temp_par = []
            for dummy, value in hamiltonianData_dict[key][key2].items():
                temp_par.append(value)

            g = np.zeros((len(list(hamiltonianData_dict[key][key2].keys())), t_total))
            # a = 1
            for i in range(len(temp_par)):
                temp = [y for sublist in [[x]*a for x in temp_par[i]][1:] for y in sublist]
                g[i, :] = temp*int(t_total/len(temp))
                a *= len(temp_par[i][1:])

            par_dict_list = []
            for i in range(len(g[0, :])):
                temp = dict()
                for j, key_value in enumerate(hamiltonianData_dict[key][key2].items()):
                    temp[key_value[0]] = [key_value[1][0], g[j, i]]
                par_dict_list.append(temp)

            for j, values in enumerate(par_dict_list):
                hamiltonianData_dict_list[j][key][key2].update(values)

    for key in hamiltonianData_dict:
        if key == 'Atomic':
            pass
        else:
            for key2 in hamiltonianData_dict[key]:  # ['Initial Hamiltonian', 'Final Hamiltonian']
                # expand parameters to vary
                temp_par = []
                for dummy, value in hamiltonianData_dict[key][key2].items():
                    temp_par.append(value)

                g = np.zeros((len(list(hamiltonianData_dict[key][key2].keys())), t_total))
                if len(g) == 0:  # in case there is no parameters to change
                    pass
                else:
                    # a = 1
                    for i in range(len(temp_par)):
                        temp = [y for sublist in [[x]*a for x in temp_par[i]] for y in sublist]
                        g[i, :] = temp*int(t_total/len(temp))
                        a *= len(temp_par[i])

                    par_dict_list = []
                    for i in range(len(g[0, :])):
                        temp = dict()
                        for j, key_value in enumerate(hamiltonianData_dict[key][key2].items()):
                            temp[key_value[0]] = g[j, i]
                        par_dict_list.append(temp)

                    for j, values in enumerate(par_dict_list):
                        hamiltonianData_dict_list[j][key][key2].update(values)

    if synchronize:
        index2delete = []
        for i, h in enumerate(hamiltonianData_dict_list):
            h2 = _check_sync(h)
            if h2 == -1:
                index2delete.append(i)
            else:
                hamiltonianData_dict_list[i] = copy.deepcopy(h2)

        for j, i in enumerate(index2delete):
            del hamiltonianData_dict_list[i-j]


    return hamiltonianData_dict_list


def _check_sync(hamiltonianData_dict):

    hamiltonianData_dict2 = copy.deepcopy(hamiltonianData_dict)
    for key in hamiltonianData_dict2:
        for parameter in hamiltonianData_dict2[key]['Initial Hamiltonian']:
            try:
                if hamiltonianData_dict2[key]['Initial Hamiltonian'][parameter] == hamiltonianData_dict2[key]['Final Hamiltonian'][parameter]:
                    pass
                else:
                    return -1
            except KeyError:  # in case parameter is not defined for final Hamiltonian
                try:
                    hamiltonianData_dict2[key]['Final Hamiltonian'][parameter] = copy.copy(hamiltonianData_dict2[key]['Initial Hamiltonian'][parameter])
                except KeyError:  # in case final Hamiltonian is not defined
                    hamiltonianData_dict2[key]['Final Hamiltonian'] = {}
                    hamiltonianData_dict2[key]['Final Hamiltonian'][parameter] =copy.copy(hamiltonianData_dict2[key]['Initial Hamiltonian'][parameter])

    return hamiltonianData_dict2


def create_spectra(initialSetup_dict, generalSetup_dict, hamiltonianData_dict_list, magneticField_list, temperature_list, xLorentzian_list, xGaussian_list, folderpath, quanty_exe, prefix='untitled_', verbosity=True):

    folderpath = Path(folderpath)

    n_total = len(hamiltonianData_dict_list)*len(magneticField_list)*len(temperature_list)*len(xLorentzian_list)*len(xGaussian_list)
    if verbosity:
        print(f'--- Creating {n_total} spectra ---')

    i = 0
    for h in hamiltonianData_dict_list:
        for magneticField in magneticField_list:
            for temperature in temperature_list:
                for xLorentzian in xLorentzian_list:

                    filename = prefix + str(i).zfill(len(str(n_total)))
                    q = create_input(initialSetup_dict=initialSetup_dict,
                                     generalSetup_dict=generalSetup_dict,
                                     hamiltonianData_dict=h,
                                     magneticField=magneticField,
                                     temperature=temperature,
                                     xLorentzian=xLorentzian,
                                     folderpath=folderpath,
                                     filename=filename)

                    output = run_quanty(quanty_exe, folderpath/(filename+'.lua'))

                    for xGaussian in xGaussian_list:
                        for type in q.spectra.toCalculateChecked:
                            if type == "Isotropic":
                                type_list = ['iso', ]
                            elif type == "Circular Dichroism":
                                type_list = ['r', 'l', 'cd']
                            elif type == "Linear Dichroism":
                                type_list = ['v', 'h', 'ld']
                        for type in type_list:
                            x, y = fix_spectrum(folderpath/f'{filename}_{type}.spec', xMin=q.xMin, xMax=q.xMax, xNPoints=q.xNPoints, xGaussian=xGaussian)
                            temp = np.zeros((x.shape[0], 2))
                            temp[:, 0] = x
                            temp[:, 1] = y
                            np.savetxt(folderpath/f'{filename}_{type}_broaden_{xGaussian}.final', temp)

                        if verbosity:
                            print(f'{filename}_{type}_broaden_{xGaussian} saved')
                    i += 1





########### graveyard

def create_input_OLD(generalSetup_dict, hamiltonianData_dict, filepath, temperature=None, xLorentzian=None, par_file=True):
    # if temperature is not None:
    #     generalSetup_dict['temperature']   = temperature
    # if xLorentzian is not None:
    #     generalSetup_dict['xLorentzian']   = xLorentzian

    q = QuantyCalculation(**copy.deepcopy(generalSetup_dict), temperature=temperature, xLorentzian=xLorentzian)

    # fix hamiltonianData_dict
    hamiltonianData_dict2 = copy.deepcopy(hamiltonianData_dict)
    for key in hamiltonianData_dict2: # ['Atomic', 'Crystal Field', 'Magnetic Field', 'Exchange Field', '3d-Ligands Hybridization (LMCT)', '3d-Ligands Hybridization (MLCT)']
        for key2 in hamiltonianData_dict2[key]: # ['Initial Hamiltonian', 'Final Hamiltonian']
            for parameter in q.hamiltonianData[key][key2]:

                try: # double valued parameters (Atomic)
                    if hamiltonianData_dict2[key][key2][parameter][0] is None:  # if parameter not None
                        hamiltonianData_dict2[key][key2][parameter][0] = copy.deepcopy(q.hamiltonianData[key][key2][parameter][0])

                except KeyError:  # if parameter is not defined
                    hamiltonianData_dict2[key][key2][parameter] = copy.deepcopy(q.hamiltonianData[key][key2][parameter])

                except TypeError as e:    # single valued parameters
                    if hamiltonianData_dict2[key][key2][parameter] is None:
                        hamiltonianData_dict2[key][key2][parameter] = copy.deepcopy(q.hamiltonianData[key][key2][parameter])

                except IndexError as e:    # single valued parameters (if value is numpy.float64)
                    if hamiltonianData_dict2[key][key2][parameter] is None:
                        hamiltonianData_dict2[key][key2][parameter] = copy.deepcopy(q.hamiltonianData[key][key2][parameter])

    # update hamiltonianData
    q.hamiltonianData.update(hamiltonianData_dict2)

    # update hamiltonian states
    for item in q.hamiltonianState:
        q.hamiltonianState[item] = copy.copy(generalSetup_dict['hamiltonianState'][item])

    # basename
    q.baseName = str(Path(filepath))

    # npsis
    try:
        q.nPsis = generalSetup_dict['nPsis']
        q.nPsisMax = q.nPsis
    except:
        pass

    # broadening
    try:
        q.xLorentzian = copy.copy(xLorentzian)
        # q.xGaussian = copy.copy(generalSetup_dict['xGaussian'])
    except:
        pass

    # save input
    q.saveInput()

    # fix issues probably due to quanty version
    f = open(q.baseName+'.lua', 'r')
    txt = f.read()
    f.close()
    txt = txt.replace('Verbosity(None)', 'Verbosity(0)')
    txt = txt.replace("{'DenseBorder', DenseBorder}", "")
    fmanip.save_string(txt, q.baseName+'.lua')

    # save parameters
    if par_file:
        dict2save = dict(generalSetup    = generalSetup_dict,
                         hamiltonianData = q.hamiltonianData,
                         temperature     = q.temperature,
                         xLorentzian     = q.xLorentzian)
        fmanip.save_obj(dict2save, q.baseName+'.par')

    return q


def normalize_data(x, y, x2interp, y2interp):
    # normalizing
    y = y/max(y) * max(y2interp)

    # finding max position
    i, = np.where(y2interp == max(y2interp))[0]
    x_of_max = x2interp[i]

    # shifting to match max posiion
    i, = np.where(y == max(y))[0]
    x = x - x[i] + x_of_max

    # interpolate data
    y = np.interp(x2interp, x, y)
    return y


#
# initialSetup_dict = dict(element             = 'Re'
#                          ,charge             = '6+'
#                          ,symmetry           = 'Oh'
#                          ,edge               = 'L2,3 (2p)'
#                          ,experiment         = 'XAS')
#
# generalSetup_dict = dict(nPsisAuto          = 1  # 0 or 1
#                          # ,nPsis            = 100
#                          # ,nConfigurations  = 1
#                          # ,k1               = (0, 0, 1)
#                          # ,eps11            = (0, 1, 0)
#                          # ,xNPoints         = 1000
#                          # ,XMin             = 10525
#                          # ,XMax             = 10565
#                          # ,xGaussian        = 0.1   # Does not change the input file
#                          , toCalculate = ['Isotropic', 'Circular Dichroism', 'Linear Dichoism']
#                          ,hamiltonianState   = {'Atomic' : 1,
#                                                 'Crystal Field' : 1,
#                                                 'Magnetic Field' : 1,
#                                                 'Exchange Field' : 0,
#                                                 '3d-Ligands Hybridization (LMCT)' : 0,
#                                                 '3d-Ligands Hybridization (MLCT)' : 0,
#                                                 '5d-Ligands Hybridization (LMCT)' : 0,
#                                                 '5d-Ligands Hybridization (MLCT)' : 0,}
#                          )
# hamiltonianData_dict = {'Atomic': {   'Initial Hamiltonian': {#'U(5d,5d)'  : [0],
#                                                               'F2(5d,5d)' : [None, 0.8],
#                                                               'F4(5d,5d)' : [None, 0.8],
#                                                               'ζ(5d)'     : [None, 1],
#                                                              },
#                                       'Final Hamiltonian'  : {#'U(5d,5d)'  : [0],
#                                                                'F2(5d,5d)' : [None, 0.8],
#                                                                'F4(5d,5d)' : [None, 0.8],
#                                                                'ζ(5d)'     : [None, 1],
#                                                               #'U(2p,5d)'  : [0],
#                                                                'F2(2p,5d)' : [None, 0.8],
#                                                                'G1(2p,5d)' : [None, 0.8],
#                                                                'G3(2p,5d)' : [None, 0.8],
#                                                                'ζ(5d)'     : [None, 1],
#                                                                'ζ(2p)'     : [None, 1],
#                                                                }
#                               },
#                       'Crystal Field': {'Initial Hamiltonian': {'10Dq(5d)'  : 1,  # eV
#                                                                 },
#                                         'Final Hamiltonian'  : {'10Dq(5d)'  : 1,  # eV
#                                                                }
#                                         },
#   }
#
# xLorentzian      = [0.1]
# xGaussian        = 2
# temperature = None
# magneticField = None
# filepath = 'rrrrr'
# q = create_input(initialSetup_dict, generalSetup_dict, hamiltonianData_dict,
# folderpath=None, prefix=filepath,
# magneticField=None, temperature=None,
# xLorentzian=xLorentzian,
# par_file=True)
#
# q.baseName
# quanty_exe = Path('/home/galdino/Documents/CuSb2O6/quanty/quanty_lin/Quanty')
# run_quanty(quanty_exe, filepath+'.lua')
# fix_spec(filepath='rrrrr_iso.spec', xMin=q.xMin,
#                                   xMax=q.xMax,
#                                   xNPoints=q.xNPoints,
#                                   xGaussian=xGaussian,
#                                   output='ttttt')

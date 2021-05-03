#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""spectra generation."""

# standard libraries
import matplotlib.pyplot as plt
import sys
import numpy as np

# spectragen
import spectragen as sg

# Definição de funções

def calc_var_10Dq(prefix_XAS, ten_Dq, par):
    """Calcula multipletos para cada 10Dq"""

    for num in ten_Dq:
        par['hamiltonianData']['Crystal Field']['Initial Hamiltonian']['10Dq(3d)'] = num
        par['hamiltonianData']['Crystal Field']['Final Hamiltonian']['10Dq(3d)'] = num
        arquivo = prefix_XAS+str(num)
        par['filepath'] = arquivo
        q = sg.calculation_file(**par)
        q.saveInput()
        sg.run_quanty(filepath_quanty='Quanty', filepath=arquivo+'.lua')

def plot_XAS(prefix_XAS, title_plot, suptitle_plot, Dq_lista, desloc_v, ene_i, ene_f, desloc_h, xGaussian):
    """Gráfico"""

    plt.rcParams['figure.dpi']=300
    plt.rcParams['figure.figsize'] = (6,10)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams["legend.facecolor"] = 'white'

    fig = plt.figure()

    plt.title(title_plot, size=14, fontweight='bold', y=1)
    plt.suptitle(suptitle_plot, size=18, fontweight='bold', x=0.52, y=0.91)

    #plt.annotate(' colorido: H$_A$+H$_{CF}$+H$_{LMCT}$\n --: H$_A$+H$_{CF}$+H$_{LMCT}$+U$_{dd}$', xy=(650, 1.32),
    #             bbox=dict(facecolor='None',edgecolor='k',boxstyle='square'))

    delta_v = 0
    for num in Dq_lista:
        arquivo = prefix_XAS+str(num)
        x, y = sg.load_spectrum(arquivo+'_iso.spec')
        xb, yb = sg.broadening(x, y, xGaussian)
        plt.plot(xb+desloc_h, yb+delta_v, label = str(num), linewidth=2.0)
        delta_v = delta_v+desloc_v



    plt.xlabel('Energia do fóton (eV)', fontsize=14, weight='bold')
    plt.tick_params(axis="x", labelsize=14)
    plt.ylabel('Intensidade (u.a.)', fontsize=14, weight='bold')
    plt.yticks(visible=False)
    plt.axis([ene_i,ene_f,-0.018,2.6])
    plt.legend(fontsize=10, loc=7, bbox_to_anchor=(1.14, 0.49593), title='10Dq (eV)')
    #plt.legend(fontsize=10, loc='best', title='10Dq (eV)')
    #plt.legend(fontsize=10, bbox_to_anchor=(1.2, 0.5), title='10Dq (eV)')

    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(ene_i, ene_f, 5)
    minor_ticks = np.arange(ene_i, ene_f, 2.5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor',axis='x' , alpha=1, linewidth=2, linestyle='--')
    ax.grid(which='major',axis='x', alpha=1, linewidth=2, linestyle='--')

    plt.savefig(prefix_XAS+'.png', transparent=True)
    plt.close()

def plot_XAS_norm(prefix_XAS, title_plot, suptitle_plot, Dq_lista, desloc_v, ene_i, ene_f, desloc_h, xGaussian, yene_i, yene_f):
    """Gráfico"""

    plt.rcParams['figure.dpi']=300
    plt.rcParams['figure.figsize'] = (6,10)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams["legend.facecolor"] = 'white'

    fig = plt.figure()

    plt.title(title_plot, size=14, fontweight='bold', y=1)
    plt.suptitle(suptitle_plot, size=18, fontweight='bold', x=0.52, y=0.91)

    #plt.annotate(' colorido: H$_A$+H$_{CF}$+H$_{LMCT}$\n --: H$_A$+H$_{CF}$+H$_{LMCT}$+U$_{dd}$', xy=(650, 1.32),
    #             bbox=dict(facecolor='None',edgecolor='k',boxstyle='square'))

    delta_v = 0

    #Loop para plotar todos os espectros
    for num in Dq_lista:
        arquivo = prefix_XAS+str(num)
        x, y = sg.load_spectrum(arquivo+'_iso.spec')
        xb, yb = sg.broadening(x, y, xGaussian)

        #normalizando o máximo em 1
        yb = yb/max(yb)
        plt.plot(xb+desloc_h, yb+delta_v, label = str(num), linewidth=2.0)
        delta_v = delta_v+desloc_v



    plt.xlabel('Photon Energy (eV)', fontsize=14, weight='bold')
    plt.tick_params(axis="x", labelsize=14)
    plt.ylabel('Normalized Intensity', fontsize=14, weight='bold')
    plt.yticks(visible=False)
    plt.axis([ene_i,ene_f,yene_i,yene_f])
    plt.legend(fontsize=10, loc=7, bbox_to_anchor=(1.14, 0.49593), title='10Dq (eV)')
    #plt.legend(fontsize=10, loc='best', title='10Dq (eV)')
    #plt.legend(fontsize=10, bbox_to_anchor=(1.2, 0.5), title='10Dq (eV)')

    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(ene_i, ene_f, 5)
    minor_ticks = np.arange(ene_i, ene_f, 2.5)
    ymajor_ticks = np.arange(yene_i, yene_f, desloc_v)
    ax.set_yticks(ymajor_ticks)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor',axis='x' , alpha=1, linewidth=2, linestyle='--')
    ax.grid(which='major',axis='x', alpha=1, linewidth=2, linestyle='--')
    ax.grid(which='major',axis='y', alpha=1, linewidth=2, linestyle='--')

    plt.savefig(prefix_XAS+'.png', transparent=True)
    plt.close()

#Roteiro para fazer figuras
#Passo 1: Defina elemento, valência e demais parâmetros na linha abaixo
q = sg.calculation_file(element='Ti', charge='2+', toCalculate=['Isotropic'], xLorentzian=(0.22, 0.24), hamiltonianState={'Atomic':True,
                    'Crystal Field':True,
                    '3d-Ligands Hybridization (LMCT)':False,
                    '3d-Ligands Hybridization (MLCT)':False,
                    'Magnetic Field':False,
                    'Exchange Field':False})

#Passo 2: Salve arquivo de input do Quanty
q.saveInput()  # file was saved as untitled.lua

#Passo 3: Rode o Quanty
output = sg.run_quanty(filepath_quanty='Quanty', filepath='untitled.lua')

#Passo 4: Extrair parâmetros do Quanty
# we can save the parameters used
sg.save_parameters(q, filepath='temp_calculation')
par = sg.load_parameters('temp_calculation.par')

#Passo 5: Defina valores de 10Dq
ten_Dq = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]

#Passo 6: Defina uma variável com o prefixo para os arquivos
prefix_XAS='Ti2+'

#Passo 7: Calcular todos os espectros
calc_var_10Dq(prefix_XAS, ten_Dq, par)

#Passo 8: Fazer o Gráfico
#Acertar título, subtítulo, intervalos de energia e deslocamento
titulo='Ti$^{2+}$'
subtitulo=''
ene_i=453       #Energia mínima do gráfico
ene_f=469       #Energia máxima do gráfico
shift_v=0.55    #Deslocamento vertical entre gráficos
shift_h=457.8   #Deslocamento para corrigir a escala de energia do espectro
gauss=0.3495    #Gaussiana (FWHM)
yene_i=-0.018   #Intensidade mínima do gráfico
yene_f=5.5      #Intensidade máxima do gráfico
plot_XAS_norm(prefix_XAS, subtitulo, titulo, ten_Dq, shift_v, ene_i, ene_f, shift_h, gauss, yene_i, yene_f)

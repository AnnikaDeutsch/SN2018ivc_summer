'''
.py file to hold all general use plotting and modelling functions 
'''

import numpy as np
import matplotlib.pyplot as plt
import emcee
from astropy.io import ascii
import corner
import os
from timeit import default_timer as timer
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.lines as mlines
from scipy.optimize import least_squares, curve_fit
from scipy.stats import f
from astropy.table import Table, vstack
import pandas as pd
import lmfit
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# define colors used in dark mode
lime = '#e4ffba'
lemon = '#fef6c7'
bora = '#b186ce'
sky = '#b2eaf0'
strawberry = '#ff9cff'
coral = '#feb5a4'


# Set Matplotlib style parameters for dark background
rcParams['figure.facecolor'] = 'black'  # Dark background color
rcParams['axes.facecolor'] = 'black'
rcParams['axes.edgecolor'] = sky
rcParams['axes.labelcolor'] = sky
rcParams['xtick.color'] = sky
rcParams['ytick.color'] = sky
rcParams['text.color'] = sky
rcParams['axes.titlecolor'] = sky



def plot_data(ax, sm, data, mode, scaled=False, **kwargs):
    telescope_marker_dict = {'VLA':('s', lime), 'ALMA':('o', bora), 'e-MERLIN':('d', coral)}

    for row in data:
        if mode == 'lc':
            x = row['time']
            # set marker color based on frequency
            freq = row['freq']
            colorval = sm.to_rgba(freq)
        if mode == 'sed':
            x = row['freq']
            # set marker color based on time
            time = row['time']
            # colorval = sm.to_rgba(time)

        telescope = row['telescope']
        marker = telescope_marker_dict[telescope][0]
        color = telescope_marker_dict[telescope][1]
        
        if scaled:
            flux = row['scaled_flux']
            err = row['scaled_flux_err']
        else:
            flux = row['flux']
            err = row['flux_err']

        ax.errorbar(x, flux, yerr=err, marker=marker, c=sky)
    return



def make_plot(data, mode, title='', xlabel='', ylabel='', freq_vals=np.linspace(0, 300, 300), cbar=True, scaled=False, models=None, params=None, model_names=None, plot_models=False, modelcolors=None, vline=None, vlinecolors=None, ylim=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.grid(False)

    if cbar:
        # get the scalar map, plot the data using the plot_data function
        sm = cmap_setup(mode)
        plot_data(ax, sm, data, mode, scaled=scaled)

        # set up colorbar
        if mode == 'lc':
            fig.colorbar(sm, fraction=0.046, label=r'$\nu$ [GHz]')
        elif mode == 'sed':
            fig.colorbar(sm, fraction=0.046, label='time [Days]')
    else:
        sm = None
        plot_data(ax, sm, data, mode, scaled=scaled)

    # set axis scales to log
    ax.set_yscale('log')
    ax.set_xscale('log')

    #Label axes, set axis limits etc.
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if scaled:
        ax.set_ylabel('Scaled Flux Density (mJy)')
        ax.set_title('Scaled to 3 GHz')
    else:
        ax.set_ylabel(ylabel)

    if mode == 'lc':
        x = data['time']
    elif mode == 'sed':
        x = data['freq']

    if models!=None:
        for i, (model,param,model_name,color) in enumerate(zip(models,params,model_names,modelcolors)):
            plot_model(model, param, x, ax, model_name, color, freq_vals=freq_vals, ylim=ylim)

    if vline != None:
        for i, (line, linecolor) in enumerate(zip(vline, vlinecolors)):
            ax.axvline(x=line, color=linecolor, linestyle='--')



cmap = LinearSegmentedColormap.from_list('custom_cmap', [coral, bora], N=256)
def cmap_setup(mode, cmap=cmap, min_freq=0, max_freq=300, min_time=1360, max_time=1370):
    '''
    color markers by frequency/time
    '''
    if mode == 'lc':
        freq_cmap = plt.cm.get_cmap(cmap)
        
        cNorm  = colors.Normalize(vmin=min_freq, vmax=max_freq)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        sm = scalarMap
        sm._A = []
    elif mode == 'sed':
        time_cmap = plt.cm.get_cmap(cmap)
        
        cNorm  = colors.Normalize(vmin=min_time, vmax=max_time)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        sm = scalarMap
        sm._A = []
    
    return sm   




def plot_model(model, params, x, ax, label, modelcolor, freq_vals=np.linspace(0, 300, 300), ylim=None):
    '''
    define model plotting function to be incorporated into makeplot()
    '''

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]

    fit = model(freq_vals, *params)
    ax.plot(freq_vals, fit, label=label, color=modelcolor)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.legend()
    return
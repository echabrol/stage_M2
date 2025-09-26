#Attention : les unités sont en cgs dans pRT (sauf pour les longueurs d'onde)

import matplotlib.pyplot as plt 
import numpy as np
import copy
import os
import pymultinest
import batman
import glob
import pickle
import shutil
import radvel
import sys
import corner
import json

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from petitRADTRANS.spectral_model import SpectralModel
from astropy import constants as const
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from PyAstronomy import pyasl
from built_time_series_function import compute_Vp, lower_resolution, compute_Vp_NS

#Goal : using pymultinest package to apply NS and retrieve atmospheric parameters

#1) Import the exoplanet's parameters
#2) Determine the limb darkening
#3) Loading the data
#4) Loading the interpolator
#5) Defining the parameters and the priors
#6) Defining the log-likelihood function 
#7) Run the Multinest
#8) Analyzing and saving the result

#1) Let's import the parameters required to simulate the planet's transit
###############################
# EVERYTHING IS IN SI UNITS ! #
###############################
from exoplanet_parameters import HD209_params
params = HD209_params # Set the default parameter used in the rest of the code



#2) Determine the limb dark laws based on the number of parameters
if len(params['c'])==4:
    limb_dark = 'nonlinear'                    
elif len(params['c'])==2:
    limb_dark = 'quadratic'
else:
    limb_dark = 'uniform'

# Batman's parameters
batman_params              = batman.TransitParams()     # object to store transit parameters
batman_params.t0           = 0.                         # time of inferior conjunction
batman_params.per          = params['Porb']             # orbital period
batman_params.rp           = params['Rp']/params['Rs']  # planet radius (in units of stellar radii)
batman_params.a            = params['a']/params['Rs']   # semi-major axis (in units of stellar radii)
batman_params.inc          = params['i']                # orbital inclination (in degrees)
batman_params.ecc          = params['e']                # eccentricity
batman_params.w            = params['w']                # longitude of periastron (in degrees)
batman_params.u            = params['c']                # limb darkening coefficients [u1, u2, u3, u4]
batman_params.limb_dark    = limb_dark                  # limb darkening model

# initializes model
batman_model = batman.TransitModel(batman_params, params['time_from_mid'])

# calculates light curve
light_curve = batman_model.light_curve(batman_params)
#print(light_curve)

# transit window's weight
W = (1-light_curve)
W /= W.max()

# Divide the weight by the average kimb darkening (because max limb darkening reached at mid transit is higher than 1) :
if limb_dark == 'nonlinear':
    c1,c2,c3,c4 = c
    limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4 # for non linear limb darkening only !
    W /= limb_avg

transit_weight = W



#3) Loading the synthetic data
print("Loading data...")
path = '/obs/echabrol/synthetic_data/'
hdul = fits.open(path + '/HD209_synthetic_data_HR4.fits')
synthetic_data_with_noise = hdul['FLUX'].data
synthetic_wave = hdul['WAVE'].data
SPIRou_wave = hdul['WAVE_SPIRou'].data
N = synthetic_data_with_noise.size # number of data point
print("Number of data point N=", N)
print("Size of the (simulated) data", synthetic_data_with_noise.shape)
print("Size of the associated wavelengths", synthetic_wave.shape)

#set noise for retrieval
data_sigma = np.nanstd(synthetic_data_with_noise) 



#4) Loding the interpolator
print(" ")
print("Loading the interpolator...")
with open('model_grid_HD209_HR2.pkl', 'rb') as f:
    data = pickle.load(f)

interpolator = data["interpolator"]
wavelength = data["wavelength"] 
print("Longueur du vecteur wavelength :", len(wavelength))
print("Range Tiso :", min(interpolator.grid[0]), max(interpolator.grid[0]))
print("Range log_MMR_CO :", min(interpolator.grid[1]), max(interpolator.grid[1]))
print("Range log_MMR_H2O :", min(interpolator.grid[2]), max(interpolator.grid[2]))

"""
# Test interpolator
T_test = 900        # K
log_CO_test = -3    # log10(MMR_CO)
log_H2O_test = -3    # log10(MMR_H2O)

point = np.array([T_test, log_CO_test, log_H2O_test])
spectrum = interpolator([point])[0].flatten()
print("Longueur du spectre interpolé :", len(spectrum))
"""



#5) Defining the parameters and the priors
parameters = ['Kp [km/s]', 'V0 [km/s]', 'Tiso [K]', 'CO [MMR]', 'H2O [MMR]'] 
n_params = len(parameters)
#N_obs = 27; N_order=49; N_pixel=4088

def priors(cube, ndim, nparams):
    # cube contains the quantile value for each parameter
    bounds = [  [60,120],       # Kp in km/s 
                [-10,10],       # V0 in km/s
                [1000.0,2800.0],     # Tiso in K
                [-7,-1],        # CO in MMR
                [-7,-1]         # H2O in MMR
                ]          
    # transform prior to match pymultinest definition
    bounds = np.array(bounds)
    for k in range(bounds.shape[0]):
         cube[k] = cube[k]*(bounds[k,1]-bounds[k,0]) + bounds[k,0]
    return cube



#6) Defining the log-likelihood function
def loglike(cube, ndim, nparams):
    '''
    params: (Kp, V0, Tiso, CO, H2O)
    '''
    # grab parameters
    Kp = float(cube[0]) * 1e3 # convert km/s to m/s
    V0 = float(cube[1]) * 1e3 # convert km/s to m/s
    Tiso = float(cube[2])
    CO = float(cube[3])
    H2O = float(cube[4])

    model_integrated = interpolator([[Tiso, CO, H2O]])[0]

    #Generating time series model spectra by including the Kp and V0 parameters 
    # compute planet's instant velocity
    Vp  = compute_Vp_NS(params, Kp, show=True)
    # compute the planet's RV in the Earth RF
    Vd   = V0 + Vp # planet's velocity along LOS in stellar RF
    Vtot = Vd + params['Vs'] - params['BERV'] # in Earth RF
    # shift the synthetic in Earth rest frame for each epoch and sample on data spectral bins
    synthetic = np.ones_like(synthetic_data_with_noise) # will hold the synthetic time series
    
    for obs in np.where(transit_weight != 0)[0]:
        # shift
        wave_shifted = wavelength / (1 - (Vtot[obs] / const.c.value)) # V0 > 0 -> redshift, V0 < 0 -> blueshift
        synth_shifted = interp1d(wave_shifted,model_integrated,bounds_error=False)(wavelength) 
        # compute transmission spectrum
        transmission = 1 - transit_weight[obs]*(synth_shifted**2/params['Rs']**2) # model_integrated corresponds to the apparent planet radius at a given wavelength
        # sample on data spectral bins
        synthetic[obs] = interp1d(wavelength, transmission, bounds_error=False, fill_value=1.)(SPIRou_wave[obs])

    data = synthetic_data_with_noise
    model = synthetic

    # Centering the data
    data_centered = synthetic_data_with_noise - np.mean(synthetic_data_with_noise, axis=2, keepdims=True)
    model_centered = synthetic - np.mean(synthetic, axis=2, keepdims=True)

    alpha = 1
    beta = 1
    
    # compute chi2 & log-likelihood
    chi2 = np.ma.masked_invalid( (data_centered - alpha*model_centered)*(data_centered - alpha*model_centered) / (beta*beta*data_sigma*data_sigma) ) # x*x is faster than x**2
    logL = -N/2*np.log(2*np.pi) -N*np.log(beta) - np.sum(np.log(data_sigma)) - 0.5*np.sum(chi2)

    print(f"loglike call: Kp={Kp}, V0={V0}, Tiso={Tiso}, CO={CO}, H2O={H2O}, logL={logL}")

    return logL



#7) Run Multinest
pymultinest.run(loglike,priors,n_params, 
                outputfiles_basename='result_pymultinest/result_', #préfixe pour les fichiers de sortie
                n_live_points=800, #nombre de points utilisés dans l'algorithme
                evidence_tolerance=0.5, #tolérance (0.5 par défaut)
                sampling_efficiency=0.8, #0.8 par défaut
                resume=False, #?
                verbose=True #pour afficher des infos sur l'avancement de l'exécution
                #max_iter=
                )



#8) Results 
base_output_dir = "result_pymultinest"

# Creating a new folder 
existing_runs = [d for d in os.listdir(base_output_dir) if d.startswith("run_")]
run_number = len(existing_runs) + 1
run_name = f"run_{run_number:02d}"
output_dir = os.path.join(base_output_dir, run_name)
os.makedirs(output_dir, exist_ok=True)

# Analyzing the results
analyzer = pymultinest.Analyzer(n_params=n_params, outputfiles_basename=os.path.join(base_output_dir, 'result_'))
stats = analyzer.get_stats()
samples = analyzer.get_equal_weighted_posterior()[:, :-1]  
marginals = stats["marginals"] 
 
# Printing the results
print("Résultats de l'analyse :")
for i, param in enumerate(parameters):
    stats_dict = marginals[i]
    mean = stats_dict['median']  
    sigma = stats_dict['sigma']
    print(f"{param} = {mean} ± {sigma}")

# Saving the results for analysis 
marginals_summary = {}
for i, param in enumerate(parameters):
    stats_dict = marginals[i]
    marginals_summary[param] = {
        'median': stats_dict['median'],
        'sigma': stats_dict['sigma'], 
        '1sigma_interval': stats_dict['1sigma'],
        '3sigma_interval': stats_dict['3sigma']
    }

with open(os.path.join(output_dir, "marginals.json"), "w") as f:
    json.dump(marginals_summary, f, indent=4)

np.save(os.path.join(output_dir, "samples.npy"), samples)

with open(os.path.join(output_dir, "logZ.txt"), "w") as f:
    f.write(f"logZ = {stats['global evidence']:.2f} ± {stats['global evidence error']:.2f}")

print(f"\nRésultats sauvegardés dans {output_dir}\n")
for param, values in marginals_summary.items():
    print(f"{param} = {values['median']:.3f} ± {values['sigma']:.3f}")




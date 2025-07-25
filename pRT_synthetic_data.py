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

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from petitRADTRANS.spectral_model import SpectralModel
from astropy import constants as const
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from PyAstronomy import pyasl
from built_time_series_function import compute_Vp, lower_resolution, verif_resolution

#locating the input_data folder
from petitRADTRANS.config import petitradtrans_config_parser
#print(petitradtrans_config_parser.get_input_data_path())

#########################################################
#Goal : generate high resolution spectra with petitRADTRANS and 
#       then add gaussian noise in order to create simulated data 

#1) Defining the main parameters of the exoplanetary system using "exoplanet_parameters.py"
#2) Using petitRADTRANS (pRT) to calculate a transmission 1D spectra 
#3) Applying two convolutions to reduce the resolution of the 1D spectra and to simulate the obervation's time integration 
#4) Generating time series from the 1D spectra using batman package
#4) Adding gaussian noise on the simulated time series
#5) Saving the simulated data in .fits file



#1) Let's import the parameters required to simulate the planet's transit
###############################
# EVERYTHING IS IN SI UNITS ! #
###############################
from exoplanet_parameters import HD209_params
params = HD209_params # Set the default parameter used in the rest of the code


#2) We generate synthetic spectra using pRT to simulate the exoplanet's absorption spectrum
########################################
# pRT USES CGS UNITS ! #################
# except for the wavelength boundaries #
########################################
spectral_model = SpectralModel(
    #Radtrans parameters
        pressures=np.logspace(-6, 2, 100), #default parameters
        line_species=['H2O','CO'],  
        rayleigh_species=['H2', 'He'],
        gas_continuum_contributors=['H2-H2', 'H2-He'],
        line_opacity_mode              = 'lbl',   #to have high resolution spectrum
        line_by_line_opacity_sampling  = 1,  #???
    # Model parameters
        # Planet parameters
        planet_radius       = params['Rp'] * 1e2,  # cm
        reference_pressure  = 1e-2, # ref pressure at which planet's radius & logg are defined. This is pRT default value in documentation
        reference_gravity   = 10**params['planet_logg_cgs'], # surface gravity (not logg) cgs
        star_radius         = params['Rs'] * 1e2,  # cm
        # Temperature profile parameters
        temperature=params['Teq'],  # isothermal temperature profile         
        # Mass fractions
        imposed_mass_fractions={  # these can also be arrays of the same size as pressures
            'H2O': 10**-4,
            'CO': 10**-3,
        },
        filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
            'H2': 37,
            'He': 12
        },
        # Spectral rebinning parameters
        wavelength_boundaries = [0.9,2.4] # microns
)
print("SpectralModel initialized")

# Compute and plot the petitRADTRANS model in transmission
wavelength_pRT, transit_radii = spectral_model.calculate_spectrum(
    mode='transmission',
    update_parameters=True, # update the parameters with the new values (if any)
)

wave  = wavelength_pRT.flatten() * 1e7 # wavelength in nm 
model = transit_radii[0].flatten() / 1e2 # meters, the [0] are for removing useless axis
print("pRT model calculated")

#calcul de la résolution 
res = wave[:-1] / np.diff(wave) #R_i = lambda_i / (lambda_(i+1) - lambda_i)
print(f"Resolution (mean) : R = {np.mean(res):.1f}")

"""
#Plot to check pRT 1D spectrum (wavelegth range, molecule signature...)
plt.figure(figsize=(15,7))
plt.plot(wave,model/cst.r_jup_mean)
plt.xlabel('Wavelength [nm]',fontsize=16)
plt.ylabel('Transit radius [Rjup]',fontsize=16)
plt.tick_params(axis = 'both', labelsize = 16)

plt.show()
#plt.savefig("pRT_1D_spectre6.pdf", format="pdf", bbox_inches='tight')
"""


#3) Let's apply two convolutions to lower the resolution of the 1D spectrum and to simulate integration time broadening
# lower to SPIRou resolution
model_convolved = lower_resolution(wave,model,pixel_size=1*2.28e3,nb_of_points=11)

# To check the resolution 
R = verif_resolution(wave, model_convolved, lambda_min=2250, lambda_max=2350, plot=False)
print(f"Resolution : R = {R:.0f}")

"""
#Plot to check convolved pRT 1D spectrum and compare it to the original pRT 1D spectrum
plt.figure(figsize=(15,7))
plt.plot(wave,model/cst.r_jup_mean, label='original spectrum')
plt.plot(wave,model_convolved/cst.r_jup_mean, label='convolved spectrum')
plt.xlabel('Wavelength [nm]',fontsize=16)
plt.ylabel('Transit radius [Rjup]',fontsize=16)
plt.tick_params(axis = 'both', labelsize = 16)
plt.legend()

plt.show()
#plt.savefig("pRT_1D_spectre6.pdf", format="pdf", bbox_inches='tight')
"""

# Add integration time broadening
delta_RV = np.mean(np.diff(compute_Vp(params, show=False)))
print(f'Delta RV during a single snapshot: {delta_RV:.2e} m/s')
model_integrated = lower_resolution(wave, model_convolved, pixel_size=delta_RV, nb_of_points=11)

"""
#Plot to check integrated and convolved pRT 1D spectrum and compare it to the original pRT 1D spectrum
plt.figure(figsize=(15,7))
plt.plot(wave,model/cst.r_jup_mean, label='original spectrum')
plt.plot(wave,model_convolved/cst.r_jup_mean, label='convolved spectrum')
plt.plot(wave,model_integrated/cst.r_jup_mean, label='integrated spectrum')
plt.xlabel('Wavelength [nm]',fontsize=16)
plt.ylabel('Transit radius [Rjup]',fontsize=16)
plt.tick_params(axis = 'both', labelsize = 16)
plt.legend()

plt.show()
#plt.savefig("pRT_1D_spectre6.pdf", format="pdf", bbox_inches='tight')
"""

# Determine the limb dark laws based on the number of parameters
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
# transit window's weight
W = (1-light_curve)
W /= W.max()

# Divide the weight by the average kimb darkening (because max limb darkening reached at mid transit is higher than 1) :
if limb_dark == 'nonlinear':
    c1,c2,c3,c4 = c
    limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4 # for non linear limb darkening only !
    W /= limb_avg

transit_weight = W


# compute planet's instant velocity and theoretical Kp in stellar RF taking into account eccentricity
Vp  = compute_Vp(params,show=False)

# compute the planet's RV in the Earth RF
Vd   = params['V0'] + Vp # planet's velocity along LOS in stellar RF
Vtot = Vd + params['Vs'] - params['BERV'] # in Earth RF



# shift the synthetic in Earth rest frame for each epoch and sample on data spectral bins
# Let's get t_wave to have SPIRou's wavelegnths
path = '/obs/echabrol/data/'
t_wave = fits.open(path + 'SPIRou_wavelength.fits')[0].data
#synthetic = np.ones_like(t_data) # will hold the synthetic time series
synthetic = np.ones(t_wave.shape)

for obs in np.where(transit_weight != 0)[0]:
    # shift
    wave_shifted = wave / (1 - (Vtot[obs] / const.c.value)) # V0 > 0 -> redshift, V0 < 0 -> blueshift
    synth_shifted = interp1d(wave_shifted,model_integrated,bounds_error=False)(wave) 
    # compute transmission spectrum
    transmission = 1 - transit_weight[obs]*(synth_shifted**2/params['Rs']**2) # model_integrated corresponds to the apparent planet radius at a given wavelength
    # sample on data spectral bins
    synthetic[obs] = interp1d(wave, transmission, bounds_error=False, fill_value=1.)(t_wave[obs])

print("synthetic time series simulated ")

"""
# plot the synthetic time series for a given order (2D)
order = 12
plt.figure()
plt.pcolormesh(t_wave[0,order], params['time_from_mid'], synthetic[:,order], shading='auto')
plt.xlabel('Wavelength [nm]', fontsize=16)
plt.ylabel('Time from mid transit [BJD-TDB]', fontsize=16)
c = plt.colorbar()
c.set_label('Transmission (1 - (Rp²/Rs²))', fontsize=16)
#plt.title('Synthetic time series of the planet')
plt.tick_params(axis = 'both', labelsize = 16)
c.ax.tick_params(labelsize=16)  # ici 14 est la taille de police

#plt.savefig("times_series_wn1.pdf", format="pdf", bbox_inches='tight')
plt.show()
"""

"""
# plot 1D spectrum associated to one order and one observation  
obs = 14; order = 15

plt.figure()
plt.plot(t_wave[obs,order,:], synthetic[obs,order,:])
plt.show()
"""


#4) Adding gaussian noise
std = 2e-3
gaussian_std = np.mean(std)
noise_data = np.random.normal(0,gaussian_std, t_wave.shape)
synthetic_data_with_noise = synthetic + noise_data

# Estimating the SNR associated 
snr_per_point = synthetic_data_with_noise / std
snr_mean = np.nanmean(snr_per_point)
print("SNR (mean):", snr_mean)


# plot the synthetic time series for a given order (2D) with noise added 
order = 14
plt.figure()
plt.pcolormesh(t_wave[0,order], params['time_from_mid'], synthetic_data_with_noise[:,order], shading='auto')
plt.xlabel('Wavelength [nm]', fontsize=16)
plt.ylabel('Time from mid transit [BJD-TDB]', fontsize=16)
c = plt.colorbar()
c.set_label('Transmission (1 - (Rp²/Rs²))', fontsize=16)
#plt.title('Synthetic time series of the planet')
plt.tick_params(axis = 'both', labelsize = 16)
c.ax.tick_params(labelsize=16) 

#plt.savefig("times_series_n1.pdf", format="pdf", bbox_inches='tight')
plt.show()



"""
#############################################################################
#5) Save the synthetic data in fits files
save_dir = '/obs/echabrol/synthetic_data/'

hdr = fits.Header()
hdr['AUTHOR'] = 'Estelle Chabrol'
hdr['COMMENT'] = "High resolution synthetic SPIRou times series "
hdr['MODEL'] = 'petitRADTRANS'
hdr['UNITS'] = 'nm'
hdr['SNR'] = snr_mean
hdr['NOISE_STD'] = std

primary_hdu = fits.PrimaryHDU(header=hdr)

flux = fits.ImageHDU(data=synthetic_data_with_noise, name='FLUX')
wave = fits.ImageHDU(data=wave, name='WAVE')
wave_SPIRou = fits.ImageHDU(data=t_wave, name='WAVE_SPIROU')

hdul = fits.HDUList([primary_hdu, flux, wave, wave_SPIRou])
hdul.writeto(save_dir + '/' + "HD209_synthetic_data_HR4.fits", overwrite=True)
"""

#hdu1 = fits.PrimaryHDU(data=synthetic_data_with_noise)
#hdu1.writeto(save_dir + '/' + 'HD209_synthetic_data_HR3.fits', overwrite=True)
#hdu2 = fits.PrimaryHDU(data=wave)
#hdu2.writeto(save_dir + '/' + 'HD209_synthetic_wave_HR3.fits', overwrite=True)







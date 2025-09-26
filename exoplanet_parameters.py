import matplotlib.pyplot as plt 
import numpy as np
import copy
import os
import batman
import glob
import shutil
import radvel
import sys

from astropy import constants as const
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from PyAstronomy import pyasl

#########################################################
#Goal : generate high resolution spectra with petitRADTRANS and 
#       then add gaussian noise in order to create simulated data 

################################################################################
## This is were we define all parameters used throughout the reduction
## Author : Estelle Chabrol (estelle.chabrol@obspm.fr)
## Date   : July 2025
################################################################################


#1) Set the parameters required to simulate the planet's transit
###############################
# EVERYTHING IS IN SI UNITS ! #
###############################


#############
# WASP-76 b #
#############
# Définition arbitraire : time vector, midpoint
time_axis_WASP76 = np.linspace(2459153.78228479, 2459154.02204805, 27) #27 max (27 observations)
midpoint_WASP76 = time_axis_WASP76[len(time_axis_WASP76)//2]

WASP76_params = {

    # stellar properties
    'Ms'            : 1.458 * const.M_sun.value,                                # Host star mass [kg] (Ehrenreich2020)
    'Rs'            : 1.4 * const.R_sun.value,                                  # Stellar radius (Arbitrary)
    'Ks'            : 116.02,                                                   # Host star RV amplitude [m/s] (Ehrenreich2020)
    'c'             : [0.1, 0.2],                                               # Limb darkening coefficients (EXOFAST)
    'Teff'          : 6329,                                                     # Effective Temperature (K) (Ehrenreich2020)
    'log(g)'        : 4.196,                                                    # Decimal logarithm of the surface gravity (cm/s**2) (Ehrenreich2020)
    'M/H'           : +0.366,                                                   # Metallicity index relative to the Sun (Ehrenreich2020)
    'vsini'         : 1.48e3,                                                   # stellar rotational velocity [m/s] (Ehrenreich2020)
    'Sys_Age'       : 1.816,                                                    # (Gyr) Age of the system (Ehrenreich2020)
    'distance'      : 194 * const.pc.value,                                     # system distance (m)

    # planet properties
    'Mp'            : 0.894 * const.M_jup.value,                                # Planet mass [kg] (Ehrenreich2020)
    'Rp'            : 1.854 * const.R_jup.value,                                # Planet radius (Ehrenreich2020)
    'Porb'          : 1.8098806,                                                # Planet orbital period [days] (Ehrenreich2020)
    'Teq'           : 2160,                                                     # Equilibrium (or Effective) Temperature (K) of the Planet (West2016)

    # transit ephemerides
    'midpoint'      : midpoint_WASP76,                                          # transit midpoint in BJD-TBD
    'time_vector'   : time_axis_WASP76,                                         # absolute time (BJD-TDB)
    'time_from_mid' : time_axis_WASP76 - midpoint_WASP76,                       # time from mid transit in BJD-TBD
    'BERV'          : 1,                                                        # BERV [m/s]

    # orbital properties
    'a'             : 0.05 * const.au.value,                                    # Semi Major axis (Arbitrary)
    'i'             : 89.623,                                                   # inclination (Ehrenreich2020)
    'e'             : 0,                                                        # excentricity (fixed)
    'w'             : 0,                                                        # STELLAR longitude of periapse (deg) (fixed)
    'lbda'          : 61.28,                                                    # spin-orbit angle (deg) (Ehrenreich2020)

    # system properties
    'Vs'            : -0.79e3,                                                  # GaiaDR3, [m/s] Vs < 0 means star is moving toward us. This Vsys contains the gravitational redshift & convective blueshift
}

# Add planet's theoretical Kp and logg in cgs for petitRADTRANS
#WASP76_params['Kp'] =  WASP76_params['Ks'] * WASP76_params['Ms'] / WASP76_params['Mp']      # Planet RV amplitude [m/s] manually computed
WASP76_params['planet_logg_cgs'] = np.log10(const.G.cgs.value*(WASP76_params['Mp']*1e3/(WASP76_params['Rp']*100)**2)) # planet log gravity in cgs

# Overwrite the planet's Kp with an arbitrary value for injecting the signal at a specific position in the (Kp,V0) space
WASP76_params['Kp'] = 117e3 # m/s
WASP76_params['V0'] = 0e3  # m/s



#########################################################################################################
###############
# HD 209458 b #
###############
# Définition arbitraire : time vector, midpoint
#time_axis_HD209 = np.linspace(2459153.78228479, 2459154.02204805, 27) #27 max (27 observations)
time_axis_HD209 = np.linspace(2459132.31228479, 2459132.55204805, 27) #18 max (18 observations)
midpoint_HD209 = time_axis_HD209[len(time_axis_HD209)//2]

HD209_params = {

    # stellar properties
    'Ms'            : 1.06918 * const.M_sun.value,                              # Host star mass [kg] (Rosenthal2021)
    'Rs'            : 1.19998 * const.R_sun.value,                              # Stellar radius (Rosenthal2021)
    #'Ks'            : 116.02,                                                  # Host star RV amplitude [m/s] (Ehrenreich2020)
    'c'             : [0.1, 0.2],                                               # Limb darkening coefficients (EXOFAST)
    'Teff'          : 6026,                                                     # Effective Temperature (K) (Rosenthal2021)
    'log(g)'        : 4.30739612512000,                                         # Decimal logarithm of the surface gravity (cm/s**2) (Rosenthal2021)
    'M/H'           : +0.050,                                                   # Metallicity index relative to the Sun (Stassun2019)
    'vsini'         : 4.49e3,                                                   # stellar rotational velocity [m/s] (Bonomo2017)
    'Sys_Age'       : 3.10,                                                     # (Gyr) Age of the system (Bonomo2017)
    'distance'      : 47.0 * const.pc.value,                                    # system distance (m)

    # planet properties
    'Mp'            : 0.73 * const.M_jup.value,                                 # Planet mass [kg] (Stassun2017)
    'Rp'            : 1.39 * const.R_jup.value,                                 # Planet radius (Stassun2017)
    'Porb'          : 3.52474955,                                               # Planet orbital period [days] (Kokori2023)
    'Teq'           : 1448,                                                     # Equilibrium (or Effective) Temperature (K) of the Planet (Barstow2017)

    # transit ephemerides
    'midpoint'      : midpoint_HD209,                                           # transit midpoint in BJD-TBD
    'time_vector'   : time_axis_HD209,                                          # absolute time (BJD-TDB)
    'time_from_mid' : time_axis_HD209 - midpoint_HD209,                         # time from mid transit in BJD-TBD
    'BERV'          : 24877.22685916465,                                        # BERV [m/s] (from SPIRou data : Guillaume Hebrard 31/08/2020)

    # orbital properties
    'a'             : 0.04634 * const.au.value,                                 # Semi Major axis (Rosenthal2021)
    'i'             : 86.71,                                                    # inclination (Kokori2023)
    'e'             : 0.01,                                                     # excentricity (fixed) (Rosenthal2021)
    'w'             : 0,                                                        # STELLAR longitude of periapse (deg) (fixed) (Rosenthal2021)
    'lbda'          : -5,                                                       # spin-orbit angle (deg) (Albrecht2012)

    # system properties
    'Vs'            : -15.0146,                                                 # GaiaDR2, [m/s] Vs < 0 means star is moving toward us. This Vsys contains the gravitational redshift & convective blueshift
}

# Add planet's theoretical Kp and logg in cgs for petitRADTRANS
#HD209_params['Kp'] =  HD209_params['Ks'] * HD209_params['Ms'] / HD209_params['Mp']      # Planet RV amplitude [m/s] manually computed
HD209_params['planet_logg_cgs'] = np.log10(const.G.cgs.value*(HD209_params['Mp']*1e3/(HD209_params['Rp']*100)**2)) # planet log gravity in cgs

# Overwrite the planet's Kp with an arbitrary value for injecting the signal at a specific position in the (Kp,V0) space
HD209_params['Kp'] = 84.7e3 # m/s
HD209_params['V0'] = 0e3  # m/s
 





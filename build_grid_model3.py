import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import threading 

from scipy.interpolate import RegularGridInterpolator
from astropy import constants as const
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from built_time_series_function import compute_Vp, lower_resolution
from scipy.interpolate import interp1d

# Goal : generate an interplation grid to reduce time computation of pRT spectra in the retrieval code
###############################################################

#1) Define fixed and free parameters
#2) Import the parameters required to simulate the planet's transit
#3) Define function to initialize and calculate pRT model
#4) Defining function to generate pRT model grid



#1) Define fixed and free parameters 
# free parameters
Tiso = np.linspace(500,3000,10)
CO = np.linspace(-7,-1,10)
H2O = np.linspace(-7,-1,10)



#2) Let's import the parameters required to simulate the planet's transit
###############################
# EVERYTHING IS IN SI UNITS ! #
###############################
from exoplanet_parameters import HD209_params



#3) Defining function to initialize and calculate pRT model
def initialize_pRT_model(param_Tiso, param_CO, param_H2O, params):
    """
    Function to initialize pRT model taking the free parameters 
    as inputs with the exoplanet dictionary 
    """
    spectral_model = SpectralModel(
        #Radtrans parameters
            pressures=np.logspace(-6, 2, 100),
            line_species=['H2O','CO'],
            rayleigh_species=['H2', 'He'],
            gas_continuum_contributors=['H2-H2', 'H2-He'],
            line_opacity_mode              = 'lbl',   #HRS
            line_by_line_opacity_sampling  = 1,  #default value
        # Model parameters
            # Planet parameters
            planet_radius       = params['Rp'] * 1e2,  # cm
            reference_pressure  = 1e-2, # ref pressure at which planet's radius & logg are defined. This is pRT default value in documentation
            reference_gravity   = 10**params['planet_logg_cgs'], # surface gravity (not logg) cgs
            star_radius         = params['Rs'] * 1e2,  # cm
            # Temperature profile parameters
            temperature=param_Tiso,  # isothermal temperature profile         
            # Mass fractions
            imposed_mass_fractions={  # these can also be arrays of the same size as pressures
                'H2O': 10.**(param_H2O), #10**-4,
                'CO': 10.**(param_CO)  #10**-3,
            },
            filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
                'H2': 37,
                'He': 12
            },
            # Spectral rebinning parameters
            wavelength_boundaries = [0.9,2.4] # microns
    )
    return spectral_model


def compute_pRT_model(spectral_model):
    """
    Function to compute the pRT model in transmission 
    - Spectral_model from the initialisation 
    """
    wavelength_pRT, transit_radii = spectral_model.calculate_spectrum(
        mode='transmission',
        update_parameters=True, # update the parameters with the new values (if any)
    )

    wave  = wavelength_pRT.flatten() * 1e7 # wavelength in nm 
    model = transit_radii[0].flatten() / 1e2 # meters, the [0] are for removing useless axis
    return wave, model



#4) Defining function to generate pRT model grid
def build_model_grid(Tiso_values, log_MMR_CO_values, log_MMR_H2O_values, params, output_file, Nb_threads):
    """
    Function to generate model grid
    The output is a N-D linear interpoler which provides fast computation of the transit radius for a given exoplanet.
    - free parameters
    - output_file
    - nb_threads
    """

    n_T = len(Tiso_values)
    n_CO = len(log_MMR_CO_values)
    n_H2O = len(log_MMR_H2O_values)

    # Calcul d'un modèle "dummy" pour connaître sa taille
    dummy_model = initialize_pRT_model(Tiso_values[0],log_MMR_CO_values[0],log_MMR_H2O_values[0], params)
    wave_d, model_d = compute_pRT_model(dummy_model)
    n_lambda = len(wave_d)

    # Define the grid
    grid = np.zeros((n_T,n_CO,n_H2O,n_lambda)) 

    def worker(index_tuples):
        """
        Using a worker function for the parallelisation
        Let's generate a pRT model for each combination of values of the free parameters
        """

        for i,j,k in index_tuples:
            T = Tiso_values[i]
            CO = log_MMR_CO_values[j]
            H2O = log_MMR_H2O_values[k]
            print(f"Model for Tiso={T}, log_MMR_CO={CO}, log_MMR_H2O={H2O}")
            # Generating pRT model
            model_pRT = initialize_pRT_model(T,CO,H2O,params)
            wave, model = compute_pRT_model(model_pRT)
            # Reducing spectral resolution
            model_convolved = lower_resolution(wave,model,pixel_size=1*2.28e3,nb_of_points=11)
            # Adding integration time broadening
            delta_RV = np.mean(np.diff(compute_Vp(params)))
            model_integrated = lower_resolution(wave, model_convolved, pixel_size=delta_RV, nb_of_points=11) #model_convolved remplacé par model
            grid[i,j,k,:] = model_integrated
        
    # Computing the model grid
    start_time = time.time()
    print("Starting gird computation..")

    # Starting the threads
    index_tuples = [(i,j,k) for i in range(n_T) for j in range(n_CO) for k in range(n_H2O)]

    if Nb_threads > 0:
        splits = np.array_split(index_tuples, Nb_threads)
        threads = []
        for s in splits:
            t = threading.Thread(target=worker, args=(s,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        worker(index_tuples)

    print(f"Grille de modèles construite en {time.time() - start_time:.2f} secondes.")

    # Build and save the interpolator using pickle
    print("Building and saving interpolator..")
    interpolator = RegularGridInterpolator((Tiso_values, log_MMR_CO_values, log_MMR_H2O_values), grid, bounds_error=False)

    # Create a dictionary to save main information
    grid_model = {
        'readme': "This file contains an interpolator generated from pRT grid model",
        'author': "Estelle Chabrol",
        'free_param': "Tiso, H2O, CO",
        'values_Tiso': Tiso,
        'values_CO': CO,
        'values_H2O': H2O,
        'interpolator': interpolator,
        'wavelength': wave_d
    }

    with open(output_file + ".pkl", "wb") as f:
        pickle.dump(grid_model, f)

    return interpolator


models_grid = build_model_grid(Tiso, CO, H2O, HD209_params, "model_grid_HD209_HR2", Nb_threads=7)












import numpy as np; import xarray as xr; 
import os, sys, importlib, random; 
import pandas as pd; 

import load_aqdp; 



def make_table(turb_method, indices):
    ''' Make a table of data describing the performance of 
     different a turbulent method for a set of indices.
    '''
    ens = turb_method.ensembles; # ensembles associated with method
    motion = load_aqdp.vehicle_motion(ens);  
    all_pressures = ens.get_pressure().values; # depth 
    # --- create dataframe in which to save data
    table = pd.DataFrame( \
            columns=['epsilon','noise','pressure','vert_vel'])
    # --- roll through and append one row per index
    for index in indices:
        method_output = turb_method.estimate_epsilon(index); 
        vert_vel = motion.vertical_velocity( index ).values; 
        # organize data into a dictionary
        data_2_save = { \
                'time':[ens.time[index]],
                'epsilon':[method_output[0]],
                'noise':[method_output[1]], 
                'pressure':[all_pressures[index]],
                'vert_vel':[vert_vel] }
        data_as_row = pd.DataFrame.from_dict( data=data_2_save ); 
        table = table.append( data_as_row ); 
    table = table.set_index('time'); 
    return table


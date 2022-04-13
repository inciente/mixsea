import numpy as np; import xarray as xr; 
import os, sys, importlib, random; 
import pandas as pd; 
#from parfor import parfor
import load_aqdp;



def make_table(turb_method, indices):
    ''' Make a table of data describing the performance of 
     different a turbulent method for a set of indices.
    '''
    # --- define class instances associated with turb_method
    my_ens = turb_method.ensembles; 
    my_motion = load_aqdp.vehicle_motion(my_ens);  
    all_pressures = my_ens.get_pressure().values; # depth 
    # --- create lists to save output
    epsi = []; noise = []; dpdt = []; 
    # --- roll through and append one row per index
 #   @parfor
    for index in indices:
        method_output = turb_method.estimate_epsilon(index);
        vert_vel = my_motion.vertical_velocity( index ).values; 
        # save output
        epsi.append(method_output[0]); 
        noise.append(method_output[1]);
        dpdt.append( vert_vel );
    #table = {'epsilon':epsi, 'noise':noise,
    #        'vert_vel':dpdt, 'depth':all_pressures[indices],
    #        'time':ens.time[indices],'index':indices};
    table = pd.DataFrame( data={ \
             'epsilon':epsi, 'noise':noise,
             'vert_vel':dpdt, 'depth':all_pressures[indices],
             'time':my_ens.time[indices] }, 
             index=indices)
    return table


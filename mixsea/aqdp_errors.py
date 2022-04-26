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
    orient_lists = [[], [], []];
    shak_lists = [[], [], []]; # for roll, hdg, and pitch variance
    # --- roll through and append one row per index
 #   @parfor
    for index in indices:
        # turbulence
        method_output = turb_method.estimate_epsilon(index);
        # instrument motion
        vert_vel = my_motion.vertical_velocity( index ).values; 
        unpack_orientation( my_motion, index, 
                 orient_lists, np.nanmean );
        unpack_orientation( my_motion, index, 
                 shak_lists, np.nanvar ); 
        # save output
        epsi.append(method_output[0]); 
        noise.append(method_output[1]);
        dpdt.append( vert_vel );
    table = pd.DataFrame( data={ \
             'epsilon':epsi, 'noise':noise,
             'vert_vel':dpdt, 'depth':all_pressures[indices],
             'roll':orient_lists[0], 'hdg':orient_lists[1],
             'pitch':orient_lists[2],
             'roll_var':shak_lists[0],'hdg_var':shak_lists[1],
             'pitch_var':shak_lists[2],
             'time':my_ens.time[indices] }, 
             index=indices)
    return table


def unpack_orientation( motion, index, lists, func = None ):
    # Get shaking indicators for a given index
    orientation = motion.get_orientation( index, 
            func = func, unwrap = True )
    # Save append them to lists (lists is a list of lists).
    lists[0].append( orientation['roll'] );
    lists[1].append( orientation['hdg'] ); 
    lists[2].append( orientation['pitch'] ); 
    return lists




import numpy as np; import xarray as xr; 
import os, sys, importlib, random; 
import pandas as pd; 
#from parfor import parfor
import load_aqdp;
from datetime import datetime, timedelta


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


# ---------- functions useful for gridding
def date_to_num( datelist ):
    basetime = datetime(2000,1,1,0,0,0); 
    datenums = [ (jj - basetime).total_seconds() / 3600 for jj in datelist ]; 
    return datenums

def make_t_axis( table, dt = 1 ):
    #dt is in hours
    t_0 = np.min( table['time'] ); 
    t_end = np.max( table['time'] ); 
    tnums = date_to_num( [t_0, t_end] ); 
    timevec = np.arange( np.floor( tnums[0] ), np.ceil( tnums[1] ), dt ); 
    return timevec

def find_in_grid( table, t_axis, z_axis ):
    # get soace-time location of rows in table
    timenum = date_to_num( table['time'] ); 
    depth = table['depth']; 
    # find which "bin" each measurement belongs in
    t_index = np.digitize( timenum, t_axis ); 
    z_index = np.digitize( depth, z_axis ); 
    return t_index, z_index

def grid_var( table, var ):
    # set edges of grid
    dz = 25; dt = 2; # meters and hours
    z_axis = np.arange( 850, 1150, dz ); 
    t_axis = make_t_axis( table, dt = dt ); 
    # find data within the grid
    t_ind, z_ind = find_in_grid( table, t_axis, z_axis ); 
    # create empty grid for new variable 
    gridded_var = np.zeros( [ len(z_axis)-1 , len(t_axis)-1 ] ); 
    gridded_var[gridded_var == 0] = np.nan; 

    for tt in range( 1, len(t_axis) ):
        rows_now = (t_ind == tt); 
        for zz in range( 1, len(z_axis)):
            rows_depth = (z_ind == zz ); 
            rows_cell = rows_now * rows_depth; # data from right time and depth
            vals_use = table[var][rows_cell] # isolate data
            gridded_var[ zz-1, tt-1 ] = np.nanmean( vals_use ); # save in grid
    # resize t, z, axes for plotting 
    t_axis = t_axis[1,:] - dt/2;
    z_axis = z_axis[1,:] - dt/2;
    return gridded_var, t_axis, z_axis


            

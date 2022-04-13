import sys, os
import xarray as xr; import pandas as pd; 
import numpy as np;
from datetime import datetime, timedelta
from decimal import Decimal
from scipy.stats import chi2, linregress
from scipy.optimize import curve_fit, minimize, differential_evolution
from abc import ABC, abstractmethod 
import matplotlib.pyplot as plt
import scipy.signal as signal

sys.path.append('/mnt/sda1/PhysOc/modview/modview');
import timetools, loader, mapper
sys.path.append('/mnt/sda1/PhysOc/mixsea/mixsea');
#iimport aqdpturb

''' DATA IMPORTING AND PREPROCESSING FUNCTIONS '''
data_dir = '/mnt/sda1/SciData/TTIDE/McLane/'
M5_paths = {'MMP_raw':data_dir+'TTIDE_M5_sn107_short.mat',
            'MMP_gridded':data_dir+'TTIDE_M5_sn107_gridded.mat'}
# Some information about this experiment and setup
dl_aqdp = 0.0221
std_aqdp = 0.001
data_year = 2015

# Initial data objects

# ----------- FORMATTING FUNCTIONS 
def datetime_from_yday(year,yday):
    time = np.array([datetime(year,1,1,0) + timedelta(days=kk) \
            for kk in yday]);
    return time

def grid_to_xr(grid_dict, varlist):
    # Use dimensions time and pressure to return an xr.DataArray
    valid_time = ~np.isnan(grid_dict['yday'][0]);
    p = np.squeeze( grid_dict['p']); 
    time = grid_dict['time']; 
    for kk in varlist:
        grid_dict[kk] = xr.DataArray(data=grid_dict[kk][:,valid_time], 
                dims=('pressure','time'),
                coords={'pressure':('pressure',p),
                    'time':('time',time)});
    return grid_dict

def cut_grid_var(data,limits):
    # To be used with xarray objects
    data = data.sel(time=slice(limits['t0'],limits['t1']));
    data = data.sel(pressure=slice(limits['z0'],limits['z1']));
    return data

def cut_raw_dat(v_df, limits):
    # To be used with pandas dataframes.
    data = v_df[limits['t0'],limits['t1']]
    return data

def format_e(n):
    a = '%.1E' % Decimal(n)
    return a

# --------------- GIVE FORMAT TO DATA 

def build_raw(dl=dl_aqdp):
    # get data from matfile specified above
    raw_file = loader.matfile(M5_paths['MMP_raw']);
    varlist = ['yday','dtnum','p','v1','v2','v3',
               'hdg','pitch','roll'];
    raw_dat = raw_file.read_struct('aqdp',varlist);
    # get dimensions/coordinates
    time = datetime_from_yday( data_year, raw_dat['yday'][0] )
    blank = 0.2; top = blank + dl*min( raw_dat['v1'].shape)
    l_beam = np.arange(blank, top, step=dl);
    # Store along-beam velocities as arrays in dataset
    aqdp_dat = xr.Dataset( coords={ 'time':('time',time), 
                                    'length':('length',l_beam)} );
    for beam in ['v1','v2','v3']:
        aqdp_dat[beam] = (['time','length'],raw_dat[beam]);
    for variable in ['pitch','roll','hdg','dtnum','p']:
        tseries = np.squeeze( raw_dat[variable] ); 
        if variable == 'p':
            aqdp_dat['pressure'] = ('time',tseries)
        else:
            aqdp_dat[variable] = ('time',tseries) 
  #  pressure = np.squeeze( raw_dat['p'] );  # pressure sensor on aqdp
   # aqdp_dat['pressure'] = ('time',pressure);
    aqdp_dat['yday'] = ('time',raw_dat['yday'][0]); 
    return aqdp_dat

def build_grid():
    # get data from matfile
    grid_file = loader.matfile(M5_paths['MMP_gridded']);
    # put data into dictionary
    grid_vars = ['datenum','p','th','s','u','v','yday','z'];
    grid_dat = grid_file.read_struct('MPall',grid_vars); 
    # there are nans in the time vector, so we find those
    valid_time = ~np.isnan(grid_dat['yday'][0]);
    grid_dat['time'] = datetime_from_yday(2013,grid_dat['yday'][0][valid_time]);
    grid_dat = grid_to_xr(grid_dat, ['u','v','th','s']);    
    return grid_dat

#grid_dat = build_grid();
#raw_dat = build_raw(); 

# ------------- SECONDARY COMPUTATIONS ON GRIDDED DATA
def shear_squared():
    sh2 = grid_dat['u'].differentiate(coord=pressure)**2 \
            + grid_dat['v'].differentiate(coord=pressure)**2;
    return sh2

class Ensembles:
    ''' This class takes in a dt, the raw aqdp data, creates a slow time with that
    given dt, and then offers direct access to all profiles taken within each slow 
    time step.'''
    def __init__(self, dt, raw_aqdp):
        # Take in dt in seconds
        # Take in xr.DataArray of raw aquadopp data
        self.dt = dt; 
        self.raw = raw_aqdp;
        self.timenum = self.make_slowtime(); # yday (slow time)
        self.time = datetime_from_yday(data_year,self.timenum); # (slow time)
        self.ind_start, self.ind_stop = self.lim_indices(); # ensemble limit indices
        
        self.remove_empty_ensembles(); # do some clean up 
        self.pressure = self.get_pressure(); 
        print('Will process ' + str(len(self.time)) \
                 + ' ensembles lasting ' + str(self.dt) + ' s each.')

    def make_slowtime(self):
        # dt and timenum must have the same unit (usually days)s
        timenum = self.raw['yday']; 
        dt_days = self.dt/60/60/24;
        slow_time = np.arange( start=np.floor(timenum[0]*24)/24,
                     stop=np.ceil(timenum[-1]*24)/24, step=dt_days );
        return slow_time

    def lim_indices(self):
        # Find indices where each ensemble starts or ends
        offset = (self.dt/2)/60/60/24; # seconds to days
        # self.timenum (slow_time) sets endpoints of ensembles
        ind_start = np.array( [np.searchsorted( self.raw['yday'],
            self.timenum[kk] - offset) for kk in range(len(self.timenum)) ] );
        ind_stop = np.array( [np.searchsorted( self.raw['yday'], 
            self.timenum[kk] + offset) for kk in range(len(self.timenum)) ] ); 
        return ind_start, ind_stop

    def remove_empty_ensembles(self):
        has_data = (self.ind_stop - self.ind_start) > 2;
        self.timenum = self.timenum[has_data]; 
        self.time = self.time[has_data]; 
        self.ind_start = self.ind_start[has_data]; 
        self.ind_stop = self.ind_stop[has_data];
   
    def get_compass(self, index):
        # return pitch, heading, and roll for a given ensemble
        pass

    def get_pressure(self):
        p_fast = self.raw['pressure'];
        p_slow = np.array( [ np.nanmean( p_fast[slice(self.ind_start[kk],
            self.ind_stop[kk])] ) for kk in range(len(self.time))] )
        p_slow = xr.DataArray( data=p_slow, dims=['time'],
                coords={'time':('time',self.time)} )
        return p_slow

    def see_member(self, index, beam='v1', clean=False):
        # Slice velocity data to show only profiles within a given ensemble
        vels = self.raw[beam]; 
        find_ensemble = slice( self.ind_start[index], self.ind_stop[index] ); 
        vels = vels[find_ensemble,:]; 
        if clean:
            # Remove profiles with anomalously high variability
            bad_profiles = np.nanvar( vels, axis=1 );
            bad_profiles = bad_profiles / np.mean(bad_profiles) > 5
            vels = vels[~bad_profiles,:];
        return vels

class TurbMethod(ABC):
    def __init__(self, ensemble_set, detrend, beam):
        self.ensembles = ensemble_set; 
        self.detrend = detrend; 
        self.beam = beam; 
        self.nbins = self.ensembles.raw['v1'].shape[1]; 
    
    def get_noise(self,index):
        # Take in v profiles for a given index and determine noise level
        return

    @abstractmethod
    def theory_curve(self, epsilon):
        pass

    @abstractmethod
    def observations_curve(self, index):
        pass

    @abstractmethod
    def cost_function(self, parameters, obs_curve):
        # parameters is a list [epsilon, noise]
        # this will enter scipy.optimize.minimize in estimate_epsilon
        pass

    def estimate_epsilon(self, index, save_curves=False):
        obs_curve = self.observations_curve(index);
        degfred = obs_curve.shape[0]; # number of profiles used
        obs_curve = np.mean( obs_curve, axis=0 ); # ensemble average
        # find parameters that minimize cost function
        optimal = differential_evolution( \
                func=self.cost_function, 
                bounds=[(1e-11,1e-4),(0,5e-4)], 
                args=(obs_curve, degfred));
        if optimal.success:
            parameters = optimal.x
        else:
            parameters = np.array([np.nan, np.nan])
        if save_curves:
            psi_theory = self.theory_curve( parameters[0], parameters[1]);
            return parameters, obs_curve, psi_theory
        else:
            return parameters

    def plot_obs_theory(self, ax, index, colors=['black','blue']): 
        parameters, observ, theory = self.estimate_epsilon(index, save_curves=True); 
        ens_date = self.ensembles.time[index].strftime('%d %b %H:%M');
        eps_string = format_e( parameters[0] );
        if hasattr(self, 'wvnums'):
            linobs, = ax.loglog( self.wvnums, observ, 
                    linewidth=2, color=colors[0],
                label=r'$\langle \psi \rangle_{ens}$');
            linth, = ax.loglog( self.wvnums, 
                    theory, linewidth=1.5, color=colors[1],
                    label=r'$\Psi$ for $\varepsilon=$'+eps_string);
            ax.set_xlabel('Wavenumber $k$ [cpm]'); 
            ax.set_ylabel(r'PSD [m$^{3}$ s$^{-2}$]')
        elif hasattr(self, 'l_fit'):
            linobs, = ax.plot( self.l_fit, 
                    1e4*observ, linewidth=2, color=colors[0], 
                label=r'$\langle \Lambda \rangle_{ens}$ obs');
            linth, = ax.plot( self.l_fit, 
                1e4*theory, linewidth=1.5, color=colors[1],
                label=r'$\Lambda_{2}$ for $\varepsilon=$'\
                        +eps_string);
            ax.set_xlabel('Separation $r$ [m]'); 
            ax.set_ylabel('Variance [10$^{-4}$ m$^2$ s$^{-2}$]'); 
        return linobs, linth

class Wiles(TurbMethod):
    def __init__(self, ensemble_set, detrend=False, beam='v1'):
        # set properties defined by parent class
        super(Wiles,self).__init__(ensemble_set, detrend, beam);
        # now set properties specific to this TurbMethod
        self.dl = dl_aqdp; 
        self.l = np.arange(1,self.nbins+1,1)*self.dl
        self.fit_range = [4,20]; # range over which fit is performed
        self.l_fit = self.l[self.fit_range[0]:(self.fit_range[1]+1)];
        print('The fitting range for the structure function method is '\
                + str(np.array(self.fit_range)*self.dl) + ' meters');

    def observations_curve(self, index):
        vels = self.ensembles.see_member(index, self.beam, clean=True);
        if self.detrend:
            vels = signal.detrend(vels, axis=1); 
            # This must be turned into an xr.DataArray if not already
        raw_struct = np.zeros( [vels.shape[0], 
            self.fit_range[1]-self.fit_range[0] + 1] )
        dr = self.fit_range[0]; # cycle through separations between beams
        while dr < (self.fit_range[1]+1):
            struct_for_dr = np.zeros( [vels.shape[0],] ); 
            # Sum all velocity difference for given dr
            for zz in range(self.nbins-dr):
                struct_for_dr += ( vels[:,zz] - vels[:,zz+dr] )**2
            # Now save mean structure functions for this ensemble
            raw_struct[:,dr-self.fit_range[0]] = struct_for_dr / (zz+1)
            dr += 1;
        # Erase profiles with extreme variability
        return raw_struct

    def theory_curve(self, epsilon, noise):
        SF_theory = noise +  2.1*(epsilon**(2/3))* self.l_fit**(2/3); 
        return SF_theory

    def cost_function(self, parameters, obs_curve, degfred):
        # Take in an empirical structure function and calculate error
        # against theoretical curve for given parameters=[eps, noise].
        Dth = self.theory_curve(parameters[0], parameters[1]); 
        misfit = np.sum( ( obs_curve - Dth )**2 )**(1/2); # simple L2 norm
        return misfit
       

class Veron(TurbMethod):
    ''' Take in an instance of the Ensembles class and apply the methods described
    by Veron to it.
    '''
    def __init__(self, ensemble_set, detrend=False, beam='v1'):
        # set properties defined by parent class
        super(Veron,self).__init__(ensemble_set, detrend, beam);
        # now set properties specific to this method
        self.wvnums = np.fft.rfftfreq( n=self.nbins, d=dl_aqdp )[1:];

    def observations_curve(self, index):
        # Calculate power spectrum of along-beam velocities
        vels = self.ensembles.see_member(index, self.beam, clean=True); # get vels
        powerspec = timetools.spectrum_1D( arr=vels, dt=dl_aqdp,
                             axis=1, nseg=1, hann=False); # normalized spectra
        powerspec = powerspec[:,1:] # remove frequency 0
        return powerspec

    def theory_curve(self, epsilon, spec_floor):
        psi_theory = spec_floor + (18/55)*(8*epsilon/9/0.4)**(2/3) \
                   * ( self.wvnums**(-5/3) );
        return psi_theory
    
    def cost_function(self, parameters, obs_curve, degfred):
        cost = 0; # obs vs theory at all wavenumbers
        psi_theory = self.theory_curve( \
                parameters[0], parameters[1]);
        for kk in range(len(obs_curve)-2):
            DIST = chi2.pdf( degfred * obs_curve[kk] \
                    / psi_theory[kk], df=degfred)
            cost -= np.log( (degfred/psi_theory[kk]) * DIST ); 
        return cost
    
class vehicle_motion:
    def __init__(self, ensemble):
        self.data = ensemble.raw;
        self.ens = ensemble; 
    
    def vertical_velocity(self, index):
        p0 = self.data.pressure[ self.ens.ind_start[index] ]; 
        p1 = self.data.pressure[ self.ens.ind_stop[index] ]; 
        # need to compute dp/dt
        dp = p1 - p0; 
        dt = pd.to_datetime( p1.time.values ) - pd.to_datetime( p0.time.values );
        dpdt = dp/dt.total_seconds();
        return -dpdt
    

    def find_member(self, xr_obj, index):
        find_ensemble = slice( self.ens.ind_start[index], self.ens.ind_stop[index] ); 
        if len(xr_obj.dims) == 2:
            xr_obj = xr_obj[find_ensemble,:]; 
        else:
            xr_obj = xr_obj[find_ensemble]; 
        return xr_obj
''' 
BELOW IS A SET OF FUNCTIONS THAT EXECUTE PLOTTING SUBROUTINES
FIGURES ARE DEFINED AS INSTANCES OF MODVIEW.VIZTOOLS.PANEL_PLOT class.
----------- data are invoked using grid_dat, raw_dat, and the ensemble class.
'''

import viztools

def make_pcolor(ax,variable,limits,settings):
    # Return pcolormesh of grid_dat[comp] within the time and depth
    # limits specified (dict).
    
    dat2plot = cut_grid_var(grid_dat[variable],limits); # get grid and cut it
    p = dat2plot['pressure'].values; time = dat2plot['time']; # plotting axes
    img_vals = dat2plot.values[:-1,:-1];
    img = ax.pcolormesh( time, p, img_vals, cmap=settings['cmap'],
            vmin=settings['vmin'], vmax=settings['vmax'], shading='flat')
    return img

def make_contour(ax,variable,limits,settings):
    dat2plot = cut_grid_var(grid_dat[variable],limits);
    p = dat2plot['pressure'].values; time = dat2plot['time'].values;
    img = ax.contour( time, p, dat2plot.values, levels=settings['levels'],
            colors=settings['colors'], linewidth=settings['linewidth'])
    return img

def make_uv_aqdp():
    fig_dict = {'figsize':[6,10],'widths':[1,0.05], 'heights':[1,1,1],
            'panels':([0,0],[1,0],[2,0]) }
    fig = viztools.panel_plot(fig_dict);
    
    limits = {'t0':None, 't1':None, 'z0':None, 'z1':None};
    u_settings = {'vmin':-0.3, 'vmax':0.3, 'cmap':'bwr'}

    u_plot = make_pcolor( fig.axes[0], 'u', limits, u_settings);


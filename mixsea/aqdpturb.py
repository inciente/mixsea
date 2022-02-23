import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.stats import chi2, linregress
from scipy.optimize import minimize_scalar, curve_fit
from scipy.interpolate import UnivariateSpline;
import time

def buildraw(path2file):
    matfile = sio.loadmat(path2file)
    matfile = matfile['aqdp']; 
    
    global aqdp
    aqdp=dict(); 
    
    # Coordinates
    aqdp['yday'] = np.squeeze( matfile[0]['yday'][0] );
    aqdp['pressure'] = np.squeeze( matfile[0]['p'][0]);
    aqdp['dl'] = 0.0221; aqdp['dt'] = np.mean( np.diff(aqdp['time'], n=1)*24*3600 );

	# Measured variables get their own class instances
    aqdp['v1'] = VelMeas( pd.DataFrame( data = matfile[0]['v1'][0], index = aqdp['time']), aqdp['dl']);

class grid:
	
    def __init__(self, timevec='none', path2file='none'): 
        if path2file == 'none':
            self.path2file = '/media/mydrive/AQDP/TTIDE_M5_sn107_gridded.mat';
        # Load matfile featuring aqdp data gridded onto depth and time coordinates.
        matfile = sio.loadmat(self.path2file);
        matfile = matfile['MPall'];
        datgrid = dict();
        self.yday = np.squeeze(matfile[0]['yday'][0]);
        valcheck = ~np.isnan(self.yday); 
        self.yday = self.yday[valcheck]
        
        # Store properties with no time dimension
        self.p = np.squeeze(matfile[0]['p'][0]); 
        self.z = np.squeeze(matfile[0]['z'][0]);
        self.info = np.squeeze(matfile[0]['info'][0]);
        vars2get = ["t","s","u","v","w"];
        for item in vars2get:
            values = np.squeeze( matfile[0][item][0] ); 
            datgrid[item] = values[:,valcheck];
       
        self.gridded = datgrid;
	
	def cutvar(self, var, limits):
		timecheck = self.yday > limits['t0'] and self.yday < limits['t1'];
		return self.gridded[var][:,timecheck];


# Make a class that stores aqdp data, allowing to detrend and take spectra as needed
class VelMeas:
    
    std = 0.01; # std necessary
    
    def __init__(self, vels, dl):
        
        self.v = vels;
        self.nbins = min(vels.shape)
        self.time = vels.index
        self.dl = dl;
        self.wvnum = 2*np.pi*np.fft.rfftfreq( self.nbins, d = dl)[1:];
        self.l = np.cumsum( self.dl*np.arange(0,self.nbins,1))
    
    def kspectrum(self, ind1, ind2, detrend = False ):
        # Compute wavenumber spectrum for individual velocity profiles
        if detrend:
            u2spec = signal.detrend( self.v.loc[ind1:ind2,:], axis = 1);
        else: 
            u2spec = self.v.loc[ind1:ind2,:];
        
        # Calculate variance to normalize spectrum later    
        normvar = np.expand_dims( np.var( u2spec , axis = 1 ), axis = 1);
        dk = 1/(self.nbins * self.dl )
        
        # Get power for positive wvnumbers
        spectrum = np.fft.fft( u2spec, axis = 1 );
        spectrum = spectrum*np.conj(spectrum); 
        spectrum = spectrum[:,1:int( np.ceil(self.nbins/2+0.5))];
        
        # Normalization
        spectrum2 = ( normvar * spectrum );
        normvar = np.expand_dims( np.sum( spectrum*dk, axis = 1 ) , axis = 1);
        spectrum = spectrum2 / normvar;

        return spectrum.real;
    
    def structfunc(self, limits):
        # Calculate structure function for all profiles in data
        rawStruct = np.empty( (len(self.time), limits[1]-limits[0]+1));
        print('running struct method')
        
        # Cycle through separations r to calculate struct func
        for rr in range(limits[0], limits[1]+1):
        
            psamples = self.nbins-rr; # number of samples per profile
            one_take = np.zeros( (self.v.shape[0], ) ); # calc mean Dr
        
            for zz in range(psamples): # starting points for difference
                one_take += (self.v.loc[:,zz] - self.v.loc[:,zz+rr])**2 / psamples;
    
            # Save D(r,t) for all times 
            rawStruct[:,rr-limits[0]] = one_take; 
        return rawStruct
        
class Ensemble:

    # Object that creates a route towards sets of raw profiles and methods applied to them 
    # Inputs: np.array that identifies the indices to a set of ensembles. 
    # all calculations aboard

    def __init__(self, slow_time, slow_dt, vel_dict, beam = 'v1'):
        
        self.time = slow_time; 
        self.dt = slow_dt; 
        self.vels=vel_dict;
        self.ind1, self.ind2 = self.lim_indices(); # indices of ensemble limits
        self.beam = beam; 
        self.p, self.all2ind = self.find_mean_p();

        self.veroneps = dict(); self.wileseps = dict(); 
        self.wileseps['psamples'] = [self.vels[self.beam].nbins - dr \
                          for dr in range(1, self.vels[self.beam].nbins ) ];
    
    def find_mean_p(self):
        ''' Find mean pressure within each ensemble (plist), and 
        assign correspondence indices to each profile in the raw data (all2ind)'''
        plist = np.zeros( (len(self.time),1));
        all2ind = np.ones( (len(self.vels['yday']),1))*(-999)
        for kk in range(len(self.time)):
            if self.ind1[kk] < self.ind2[kk]:
                plist[kk] = np.nanmean( \
                        self.vels['pressure'][self.ind1[kk]:self.ind2[kk]] )
                all2ind[self.ind1[kk]:self.ind2[kk]] = kk;
            else:
                plist[kk] = np.nan;
        return plist, all2ind

    def lim_indices(self):
        # Find indices where each ensemble starts or ends
        offset = self.dt/2
        ind1 = np.array( [np.searchsorted(self.vels['yday'], 
                  self.time[kk] - offset ) for kk in range(len(self.time)) ] )
        ind2 = np.array( [np.searchsorted(self.vels['yday'],
                self.time[kk] + offset ) for kk in range(len(self.time)) ] );
        return ind1, ind2

    def ensv(self, ens_index):
        # measured velocities within a single ensemble
        v = self.vels[self.beam].v;
        return v.iloc[ self.ind1[ens_index]:self.ind2[ens_index] , : ];
    
    def allv(self):
        # Make array including all velocity profiles within ensemble set
        vels = aqdp[self.beam].v.loc[self.all2ind>-1,:]; 
        checker = self.all2ind[self.all2ind > -1]; 
        return vels, checker;
        
    def dpdt(self):
        aqdp = self.vels;
        # Get all aqdp time indices
        wvel = np.zeros( ( len(self.time) , ) );
        for kk in range(len(self.time)):
            if self.ind1[kk]<self.ind2[kk]:
                pstart = aqdp['pressure'][self.ind1[kk]]; 
                tstart = aqdp[self.beam].time[self.ind1[kk]];
                pend = aqdp['pressure'][self.ind2[kk]]; 
                tend = aqdp[self.beam].time[self.ind2[kk]];
                wvel[kk] = (pend - pstart)/(tend-tstart)/24/3600; # convert to dbar/second
            else: 
                wvel[kk] = np.nan; # empty ensembles
        return wvel
    
    def allspectra(self, detrend=False):
        aqdp = self.vels
        u2spec, checkers = self.allv(); 
        print(u2spec.shape)
        u2spec = u2spec - np.expand_dims( np.nanmean(u2spec, axis=1), axis=1);
        
        if detrend:
            u2spec = signal.detrend(u2spec,axis=1);    
        normvar = np.expand_dims( np.var(u2spec, axis=1), axis=1 );
        dk = 2*np.pi/(aqdp[self.beam].nbins*aqdp[self.beam].dl); # 2 pi / record_length
        
        # Power for positive wavenumebrs
        spectrum = np.fft.rfft(u2spec, axis=1);
        spectrum = spectrum*np.conj(spectrum); 
        spectrum = spectrum[:,1:]; # get rid of wavenumber 0
        
        spectrum2 = normvar*spectrum; # variance of flow
        origvar = np.expand_dims(np.nansum( spectrum*dk, axis=1), axis=1);
        spectrum= spectrum2/origvar; # normalize by integral
        return np.real(spectrum), checkers
    
    def Veron(self, detrend = False ):
        aqdp = self.vels;
        # Compute the noise spectrum
        PhiNoise = (0.55*aqdp[self.beam].std)**2 / (max(aqdp[self.beam].wvnum)-min(aqdp[self.beam].wvnum));
        DegFred = 2*np.mean( self.ind2 - self.ind1);
        self.detrend = detrend; self.PhiNoise = PhiNoise; 
        
        # Define function for theoretical spectrum
        def PhiTheory(epsilon): 
            return self.PhiNoise + (18/55)*(8*epsilon/9/0.4)**(2/3) * (aqdp[self.beam].wvnum**(-5/3));
        
        def PhiTrials(epstrials): # all spectra for a set of epsilon values
            allphis = np.zeros( (len(epstrials), len(aqdp[self.beam].wvnum) ) );
            ii=0;
            for epsi in epstrials:
                allphis[ii,:] = PhiTheory(epsi);
                ii+=1;
            return allphis
			
        def CostFunc(allphis, jj, phihere):
            #Sth = PhiTheory(epsilon); 
            Sth = allphis[jj,:]; 
            cost = 0; 
            for kk in range(len(phihere)-5):
                DIST = chi2.pdf( DegFred * phihere[kk] / Sth[kk], df = DegFred)
                cost += np.log( (DegFred/Sth[kk]) * DIST );
            return cost
        start_time = time.time()
        spectra, checkens = self.allspectra(detrend);
        print("--- %s seconds for spectra ---" % (time.time() - start_time))
        
        def MLEfit(epstrials):
            Ntrials = len(epstrials);
            self.veroneps['epsilon'] = np.zeros( (len(self.time), 1) );
            self.veroneps['cost'] = np.zeros( (len( self.time), Ntrials) );
            self.veroneps['curvvar'] = np.zeros( (len(self.time),1) );
            self.veroneps['varspec'] = np.empty( (len(self.time), 1) );    
            
            testphis = PhiTrials(epstrials);         
            
            # Cycle through ensembles    
            for kk in range(len(self.time)):
                if self.ind1[kk] == self.ind2[kk]: 
                    continue # empty ensemble
#                spec_here = self.kspectrum( kk , detrend ); # mean obs spectrum
                spec_here = np.nanmean(spectra[checkens==kk,:],axis=0);
    
                # ----- Following block may be substituted by optimizing function or univspline in the future 
                costtrials = [CostFunc(testphis, jj, spec_here) for jj in range(Ntrials)];
                self.veroneps['cost'][kk,:] = costtrials;
                
                # Find max in costtrials
                maxcf = max(costtrials); maxcf = list(costtrials).index(maxcf);
                if maxcf in [0, Ntrials-1]:
                    continue # make sure it's local max
                self.veroneps['epsilon'][kk] = epstrials[maxcf];
                # ------
                
                # All these can be saved for a special function
                # Calculate variance
                f1p = (costtrials[maxcf]-costtrials[maxcf-1])/(epstrials[maxcf]-epstrials[maxcf-1]);
                f2p = (costtrials[maxcf+1]-costtrials[maxcf])/(epstrials[maxcf+1]-epstrials[maxcf]);
                costpeak = (f2p-f1p)/(0.5*(epstrials[maxcf+1]-epstrials[maxcf-1])); #2nd derivative
                
                self.veroneps['curvvar'][kk] = - 1/costpeak; # theoretical lower bound for variance
                self.veroneps['varspec'][kk] = np.var(spec_here / testphis[maxcf] ); # should have chi2 dist
                
#        eps['MAD'] = np.empty( (len(self.time), 1)); # for flags
        
        MLEfit(10**np.linspace(-9,-4,25));
        self.veroneps['epsilon'][self.veroneps['epsilon'] == 0] = np.nan
            # --------- I think this is how the calculation works overall. Need to solve format issues later. 
            
            # Offer to plt.imshow an class instance that represents the cost function plot
            # for a wide range of trial \epsilons.           
    
    def structfunc(self, detrend = False ):   
        # Calculate mean structure function for given ensemble
        #u2struc = aqdp[self.beam].v.iloc[self.ind1[ens_index]:self.ind2[ens_index], :];
        
        u2struc, checkens = self.allv(); u2struc = u2struc.values;
        if detrend:
            u2struc = signal.detrend( u2struc, axis = 1);    
        rawStruct = np.zeros( ( u2struc.shape[0], u2struc.shape[1]-1 ));    
            
        dr = 1; # Cycle through separations dr to calculate struct func
        while dr < u2struc.shape[1]: 
            rstruc = np.zeros( (u2struc.shape[0], ) ); # D storage for a given separation value r
            for zz in range( self.wileseps['psamples'][dr-1] ): # cycle through starting points for difference
                rstruc += (u2struc[:,zz] - u2struc[:,zz + dr])**2 / self.wileseps['psamples'][dr-1];
                # Save D(r,t) for all times 
            rawStruct[:,dr-1] = rstruc; 
            dr += 1;
                # Make sure to calculate uncertainty from psamples at some point 
        return rawStruct, checkens # no averaging

    #def PhiTheory(self, epsilon): 
    #    return self.PhiNoise + (18/55)*(8*epsilon/9/0.4)**(2/3) * (aqdp[self.beam].wvnum**(-5/3));
    
    def Wiles(self, detrend = False): 
        # Ensemble.Wiles is an object that stores all methods and variables relating to the structure function method      
        l = [aqdp[self.beam].dl * dr for dr in range(1, aqdp[self.beam].nbins) ]; # value of dr in meters
        dr0 = 4; drf = 20; # which separations to use in fit
        self.wileseps['fitlims'] = [dr0,drf]; # indices in l
        self.detrend = detrend;
        
        def CurveTheory(lvals, epsilon, noise):
            curve = 2.1 * epsilon ** (2/3) * lvals ** (2/3) + noise; 
            return curve
        
        def normerror(self, struc_here, epsilon, noise): # NOT IN USE YET
            # Use an empirical structure function and guesses on the model parameters to get misfit
            Dth = CurveTheory(epsilon, noise);
            misfit = ( struc_here - Dth)**2 / Dth; # this should be chi2-distributed
        
        # To save output
        self.wileseps['epsilon'] = np.zeros( (len(self.time), ) );
        self.wileseps['params'] = np.zeros( (len(self.time), 2));
        
        allstructs, checkens = self.structfunc(detrend);
        
        for kk in range(len(self.time)):
            if self.ind1[kk] == self.ind2[kk]:
                continue # empty ensemble
            struc_here = np.nanmean( allstructs[checkens==kk,:], axis=0); 
            # How is curvetheory being used here? it has no input... 
            popt, pcov = curve_fit( CurveTheory, l[dr0:drf], struc_here[dr0-1:drf-1] , p0 = [1e-8, 1e-6]); # may introduce uncertainty here (param sigma)
            self.wileseps['params'][kk,:] = popt; 
            self.wileseps['epsilon'][kk] = (popt[0] / 2.1)**(3/2)
        self.wileseps['epsilon'][self.wileseps['epsilon'] == 0 ] = np.nan
            
    def plotwiles(self, ens_index, axe ):
        # Plot the empirical and fitted structure functions for a given ensemble
        #fig = plt.figure; 
        l = [aqdp[self.beam].dl * dr for dr in range(1, aqdp[self.beam].nbins)];
        legtext = list();
        for kk in ens_index:
            (axe).plot( l, self.structfunc(kk, detrend = self.detrend) );
            epsi = self.wileseps['params'][kk,0]; noise = self.wileseps['params'][kk,1];
            (axe).plot( l, self.CurveTheory( np.array(l), epsi, noise), linestyle = 'dashed');  
            legtext.append( 'Ensemble ' + str(kk)); 
            legtext.append( 'Fit log eps = %0.2f' %np.log10( epsi ) );

        (axe).plot( [l[4],l[4]], [0, 1e-3], color = 'pink', linestyle = 'dotted')
        (axe).plot( [l[20], l[20]], [0, 1e-3], color = 'pink', linestyle = 'dotted')
        (axe).legend(legtext)
    
    def plotveron(self, ens_index, axe ):
        # Plot the empirical and fitted structure functions for a given ensemble
        #fig = plt.figure; 
        legtext = list(); 
        for kk in ens_index:
            # Observations
            (axe).loglog( aqdp[self.beam].wvnum, self.kspectrum(kk, self.detrend) );
            epsi = self.veroneps['values'][kk][0]; 
            #Fit
            (axe).loglog( aqdp[self.beam].wvnum, self.PhiTheory(epsi), linestyle = 'dashed');
            (axe).grid(); 
            legtext.append( 'Ensemble ' + str(kk)); 
            legtext.append( 'Fit log eps = %0.2f' %np.log10( epsi ) );
        (axe).legend( legtext )
        (axe).grid( True, which = 'both');
        
       

class Veron:
    def __init__(self, ensemble):
        self.ens = ensemble; # imports time, pressure, velocities, and refs
        self.eps = [];
        self.spectra = [];

    def all_spectra(self, beam='v1'): 
        vels = self.ens.vels[beam]; 
        


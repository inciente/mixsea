

# Define the functions to calculate epsilon
def MakeEnsemble(TimeIndex, Duration):
    SlowTime = np.arange( start = np.floor(TimeIndex[0]*12)/12, stop = np.ceil(TimeIndex[-1]*12)/12, step = Duration); 
    Correspondence = np.digitize(TimeIndex, SlowTime);
    SlowP = [np.mean(Pressure[Correspondence == kk]) for kk in range(len(SlowTime))]
    
    Ensemble = dict(); 
    Ensemble['time'] = SlowTime; Ensemble['pressure'] = SlowP; 
    return Ensemble

def WilesEpsilon(allvels, TimeIndex, SlowTime, dl, limits):
    rawStruct = np.empty( (len(TimeIndex), limits[1]-limits[0]+1));
    Epsilon = np.empty( (len(SlowTime)-1,1));
    Noise = np.empty( (len(SlowTime)-1,1));
    Residuals = Noise; 
    
    # Remove linear trend from each profile
    allvels = signal.detrend( allvels, axis = 1); 
    
    # Cycle through separations r to calculate struct func
    for rr in range(limits[0], limits[1]+1):
        
        psamples = allvels.shape[1]-rr; # number of samples per profile
        one_take = np.zeros((allvels.shape[0],)); # calc mean Dr
        
        for zz in range(psamples): # starting points for difference
            one_take += (allvels[:,zz] - allvels[:,zz+rr])**2 / psamples;
    
        # Save D(r,t) 
        rawStruct[:,rr-limits[0]] = one_take; 
    
    # Make a matrix to store avg struct funcs for each ensemble
    Dzr = np.empty( (len(SlowTime)-1, limits[1]-limits[0]+1)); 
    
    for ens in range(len(SlowTime)-1):
        # Find indices in timeindex
        ind1 = np.searchsorted(TimeIndex, SlowTime[ens]);
        ind2 = np.searchsorted(TimeIndex, SlowTime[ens+1]);
        
        Dzr[ens,:] = np.nanmean( rawStruct[ind1:(ind2-1),:], axis = 0);
    
    # Now make least squares fit
    l = np.arange( start = limits[0]*dl, stop = (limits[1]+1)*dl, step = dl );
    
    fit_matrix = np.ndarray( (len(l), 2));
    fit_matrix[:,0] = l**(2/3); fit_matrix[:,1] = 1; 
    
    #print(fit_matrix)
    #print( Dzr[10,:].transpose())
    
    for ens in range(len(SlowTime)-1):
        Epsilon[ens], Noise[ens] = np.linalg.lstsq(fit_matrix, 
                np.squeeze(Dzr[ens,:]), rcond = None)[0];   
        
        
    Epsilon = (Epsilon/2.1)**(3/2)/1025;
    Epsilon = np.hstack((Epsilon, Noise)); 
    return Epsilon, Dzr
    
def kSpectrum(allvels, dl, downax = 0):
    # Compute wavenumber spectra for velocity profiles
    normvar = np.var( allvels, axis = downax); # for normalization
    
    N = allvels.shape[downax]; T = dl*N; 
    dk = 1/T; # wavenumber resolution
    
    #allvels = signal.detrend(allvels, axis = downax); 
    #allvels = allvels * np.hanning(N); # no window because no segments
    
    spectrum = np.fft.fft( allvels, axis = downax);
    #spectrum = spectrum[:,1: int( np.ceil(N/2+0.5)) ] **2 / N**2 / dk; # power, normalize
    spectrum = spectrum*np.conj(spectrum); spectrum = spectrum[:,1:int( np.ceil(N/2+0.5))];

    spectrum2 =  (normvar.values*spectrum.transpose());

    spectrum = spectrum2 / np.sum(spectrum, axis = 1)/dk;
    spectrum = spectrum.real.transpose()
    
    freqs = np.fft.rfftfreq( N, d = dl);
    return spectrum, freqs[1:]
    
# COOL! NOW I NEED TO FIGURE OUT THE MAXIMUM LIKELIHOOD FITTING

def MLEfit(PhiObs, wvnum, DegFred):
    # Compute the noise spectrum
    aqdp_std = 0.01; 
    PhiNoise = aqdp_std**2 / (max(wvnum));
    
    # Theoretical spectrum
    tryEps = 10**np.linspace(-10, -5.5,50)
    FreeParam = 18/55*(8*tryEps/9/0.4)**(2/3); 
    
    # Loglikelihood function
    CostFunc = np.zeros( FreeParam.shape )
    for kk in range(len(wvnum)):
        
        kappa = wvnum[kk];
        phi_here = PhiObs[kk];
        # Theoretical values at this wavenumber
        PhiTheory = FreeParam*(kappa**(-5/3)) + PhiNoise; 
        
        CostFunc += np.log(DegFred/PhiTheory * \
                    chi2.pdf( DegFred*phi_here / PhiTheory, df = DegFred));
        # Find maximum 
        maxcf = max(CostFun); maxcf = list(CostFunc).index(maxcf); # get index
        if maxcf in [0,length(tryEps)-1]:
            besteps = np.nan; # make sure it's actually a peak
        else:
            besteps = tryEps[maxcf];
    
    return besteps


    
class Ensemble:
    
    # Object that creates a route towards sets of raw profiles and methods applied to them 
    # Inputs: np.array that identifies the indices to a set of ensembles. 
    # all calculations aboard
    
    def __init()__(self, ind1, ind2, beam = ['v1']):
        
        if len(ind1) ~== len(ind2):
            sys.exit("ind1 and ind2 must be the same length")
        
        self.ind1 = ind1; 
        self.ind2 = ind2; 
        self.beam = beam;	
    
    self.Nens = len(self.ind1); # number of ensembles in a given instance of this class
    # self.middle = int( self.ind1/2 + self.ind2/2 ); # index to reference middle point
    
    # ----   Mean in the middle of each ensemble
    self.time = aqdp[0]['time'][0][int( self.ind1/2 + self.ind2/2 )]; 
    self.p = aqdp[0]['pressure'][0][int( self.ind1/2 + self.ind2/2 )];
    
    self.procsetup = procsetup; # dictionary including all parameters to use in processing
    
    # List of all 
    
    # Make a dictionary vels = {'v1':pd.DataFrame, 'v2':pd.DataFrame} and use to reference vels within class
    
    def kSpectrum(self, procsetup):
        
        Spectra = dict();
            
        for velkey in self.beam:
            
            vels = signal.detrend( allvels[velkey] , axis = 1 ); # detrend all data?
            normvar = np.var( vels , axis = 1 ); # along-beam variance
            nbins = 

        
        # Compute wavenumber spectra for velocity profiles
        normvar = np.var( allvels, axis = downax); # for normalization
    
        N = allvels.shape[downax]; T = dl*N; 
        dk = 1/T; # wavenumber resolution
    
        #allvels = signal.detrend(allvels, axis = downax); 
        #allvels = allvels * np.hanning(N); # no window because no segments
    
        spectrum = np.fft.fft( allvels, axis = downax);
        #spectrum = spectrum[:,1: int( np.ceil(N/2+0.5)) ] **2 / N**2 / dk; # power, normalize
        spectrum = spectrum*np.conj(spectrum); spectrum = spectrum[:,1:int( np.ceil(N/2+0.5))];

        spectrum2 =  (normvar.values*spectrum.transpose());

        spectrum = spectrum2 / np.sum(spectrum, axis = 1)/dk;
        spectrum = spectrum.real.transpose()
    
        freqs = np.fft.rfftfreq( N, d = dl);
        return spectrum, freqs[1:]
    
    
    
    
                     # Cycle through separations r to calculate struct func
    for rr in range(limits[0], limits[1]+1):
        
        psamples = allvels.shape[1]-rr; # number of samples per profile
        one_take = np.zeros((allvels.shape[0],)); # calc mean Dr
        
        for zz in range(psamples): # starting points for difference
            one_take += (allvels[:,zz] - allvels[:,zz+rr])**2 / psamples;
    
        # Save D(r,t) 
        rawStruct[:,rr-limits[0]] = one_take; 
    
    def wiles():
        
        # Cycle through separations r to calculate struct func
        for rr in range(limits[0], limits[1]+1):
        
        psamples = allvels.shape[1]-rr; # number of samples per profile
        one_take = np.zeros((allvels.shape[0],)); # calc mean Dr
        
        for zz in range(psamples): # starting points for difference
            one_take += (allvels[:,zz] - allvels[:,zz+rr])**2 / psamples;
    
        # Save D(r,t) 
        rawStruct[:,rr-limits[0]] = one_take; 
        
        properties = dict(); 
        properties['Dr2'] = 
        
        
        return properties
    
    time = aqdp['time'][ind;
    pressure = 
    
        
    def 
        
    Ensemble.index =[()] SlowTime
    
    

    
    # ------
    # Information relevant to an ensemble (variables to load and or calculate)
    #   ENS['time']  = Time[ind1] + dt/2;
    #   ENS['pressure'] = np.nanmean( Pressure[ind1:ind2] );
    #   ENS.ind = [ind1, ind2] # this is what defines the ensemble
    
    # ------
    # Variables that come from the aqdp data:
    # AQDP = {v1, v2, v3, dl, time, p, temp, sal}. # dl has wavenumber vector as property
    # ------
    # thru = class();
    # ------ thru invokes sets of variables depending on what methods and datasources they relate to.
    # ------ needs to distinguish between v1, v2, v3 (doesn't directly know about ensemble indices)
    # ------ Some examples of use:
    #            thru.gsw = {N2, theta, DTDZ, DSDZ} (only relates to ctd data)
    #            thru.wiles = empirical( spectrum ) + fit( parameters, uncertainties, flags)
    #            thru.veron = empirical( struct_func ) + fit( parameters, uncertainties, flags)
    #            
    
    
    
    
    # 
    #
    

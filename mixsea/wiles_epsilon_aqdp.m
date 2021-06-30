function [epsilon, Dzr, l] = wiles_epsilon_aqdp( flow, dl, enslen, limits, rm_linear )

% Function to compute turbulent dissipation from an M-by-K matrix of
% flow using the method described in Wiles et 
% al (GRL, 2006). 

% M is the number of profiles captured and K is the number of bins in an
% ADCP or AQDP data matrix. Velocities must be along-beam.

% dl is the along-beam separation between bins and filter length is the 
% number of measurements that constitute a single ensemble for which 
% epsilon is averaged.
%
% limits is the range of separation (r) to be tested in number of bins. To
% measure same range as spectral method, do 2 bins to total/2 bins. e.g.
% limits = [2,17].

if size(flow,1) < size(flow,2)
    flow = flow'; % bincell must be dim 2
end


% Compute structure function for all recorded profiles
Numof = floor(size(flow,1)/enslen); % number of ensembles
Dzr_pre = NaN(size(flow,1), limits(2) - limits(1) + 1); % mean structure functions for each ensemble

% Profiles that will be averaged to produce an ensemble. (indices in flow matrix) 
periods = floor(linspace(1, size(flow,1), Numof));

% REMOVE LINEAR TREND DUE TO FLOW STAGNATION WHEN AQDP MOVES FAST
if rm_linear == 1
    
    flow_raw = flow; % save raw values before changing
    flow = ( detrend( flow' ) )'; 
    
end 
    
for r = limits(1):limits(2) % cycle through separations
   
    %How many samples will we get in one profile for each value of r?
    psamples = size(flow,2) - r;
    one_take = zeros(size(flow,1),1);
    
    for ens = 1:psamples  % cycle through starting points    
        one_take = one_take + ( flow(:,ens) - flow(:,ens + r) ).^2 /psamples;        
    end
    
    Dzr_pre(:,r - limits(1) + 1) = one_take;
    
end

% Make a matrix with the right size to store the averaged structure
% functions. dimensions # of ensembles-by-range of struct func
Dzr = NaN(Numof, limits(2) - limits(1) + 1);

for ens = 1:Numof-1 % now cycle through ensembles

    start = periods(ens);
    finish = periods(ens+1) - 1;
    
    Dzr(ens,:) = nanmean( Dzr_pre(start:finish,:),1); % average structfun within these indices
    
end

% Time to put this in r-space and make a r^2/3 + b fit
l = limits(1)*dl:dl:limits(2)*dl;

fit_matrix = [l.^(2/3)', ones(length(l),1)];
fit_matrix = inv( fit_matrix'*fit_matrix ) * fit_matrix';

epsilon = nan(Numof, 2);

% Now use least squares to find the optimal coefficients for the fit
for ens = 1:Numof-1
    
    epsilon(ens,:) = fit_matrix*(Dzr(ens,:))';
    
end

epsilon(:,1) = (epsilon(:,1)/2.1).^(3/2);
%epsilon(length(epsilon) + 1,:) = [nan, nan];

end




































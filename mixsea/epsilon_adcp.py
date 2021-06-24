import numpy as np
import math

def structure_function_zdep(uvals, enslen, dr, r_range, surfwave = 'false'):
	# Function to calculate epsilon from the structure function method.
	# the zdep suffiz indicats this function produces estimates with a z-dependence, so this is 
	# meant to be used with a 300-to-700 kHz ADCP. Use the aqdp version if you don't need depth-dependence.
	
	# uvals are along-beam velocities
	# dr is the separation between measurement bins
	# enslen is the number of measurements in each ensemble
	# r_range is a 2-list with the indices (in rvals) for the maximum 
	
	nmeas = r_range[1] - r_range[0] + 1;
	# Make the matrix for linear fit:
	mat4fit = np.ones( ( nmeas , 2));
	mat4fit[:,2] = (dr*np.linspace(r_range[0], r_range[1], nmeas))**(2/3);
	
	if surfwave:
		mat4fit = np.hstack(mat4fit, (dr*np.linspace(r_range[0], r_range[1], nmeas))**2);
		# BD Scannell's correction for surface waves
	
	# Now time to extract ensembles from uvals
	numens = math.floor( uvals.shape[1]/enslen ); # how many ensembles in total?
	epsilon = np.empty( (uvals.shape[0], numens ) );
	
	for kk in range( numens ):
		
		# Isolate uvals that we care about
		botens = kk*enslen; topens = (kk + 1)*enslen - 1;
		ens_beam = uvals[:,botens:topens];
		ens_beam = ens_beam - np.mean( ens_beam , axis = 0 ); # subtract depth-dependent mean flow
		
		# Structure function for al z and r
		D_zr = np.empty( ( uvals.shape[0], nmeas ) ); # dim0 depth, dim1 r
		D_zr[:] = np.nan;  
		
		# Now calculate D_zr by center differences
		for zz in range(math.floor(r_range[1]), uvals.shape[0] - math.floor(r_range[1]) ):
			
			# Calculate all values for this depth
			Dxr = np.empty( (1, nmeas) ); Dxr[:] = np.nan;
            for ll in range( r_range[0], r_range[1] + 1 ):
				# Separation exists at this depth?
				if ((zz - math.ceil(ll/2) >= 1) and (math.ceil(zz+ll/2) <= uvals.shape[0])):
					
					if np.mod( ll, 2) == 0:
						dif_vel = ( ens_beam[zz + ll/2,:] - ens_beam[zz-ll/2, :]).**2;
						D_zr[zz, ll-r_range[0]] = np.mean( dif_vel, axis = 1);
						
					else:
						dif_vel = 0.5*(ens_beam[zz + math.floor(ll/2), :] - ens_beam[zz- math.ceil(ll/2),:]).**2 + 
							0.5*( ens_beam[ zz + math.ceil(ll/2), :] - ens_beam[ zz - floor(ll/2),:] ).**2; 
						D_zr[zz, ll-r_range[0]] = np.mean( dif_vel, 2);
				# done with ll
			# done with zz
			
			# Now that we have D for all values of r at this depth, we fit the polynomial
			#nancheck = ~np.isnan( D_zr[zz,:] );
			x_solution = np.linalg.lstsq( mat4fit, D_zr[zz,:]);
			
			            
            clomat = inv( A_mat(nancheck,:)' * A_mat(nancheck,:) ) * A_mat(nancheck,:)';
            x =  clomat*D_zr(zz,nancheck)'; % fit parameters. Is this working okay? some levels may be thrown off because 
            % there's no data for the full range of r?
            
            epsilon(zz,mm,beamn) = (x(2)/2.1).^(3/2); % C_v2 = 2.1
            
            constant(zz,mm, beamn) = x(1);
            waves(zz,mm, beamn) = x(3); 
            fit_Error(zz,mm, beamn) = sum ( ( A_mat*x - D_zr(zz,:)' ).^2 );
            
            
        end
        
    end % close ensemble
    toc
    
end % close beam


	
	

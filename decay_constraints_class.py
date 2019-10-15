#!/usr/bin/python

import numpy as np 
import decay_constraints_functions_master as functions
############################################################################################################
## RADIAL CONSTANTS FOR DECAY RATE CALCULATION
n=[0,0,0,0,0]
l = [1,2,3,4,5]
m = [1,2,3,4,5]
# DM: what is n2?
n2=[2,3,4,5,6]
B = functions.radial_constants(m,l,n)

accuracy = 200 # number of points in BH mass and spin directions
range_length = 5 # Accuracy of axion parameter space

## DEFINE BLACK HOLE PARAMETER SPACE FOR ACCURACY OF EXCLUSION CONTOURS
a,mm,X,Y = functions.black_hole_parameter_space(accuracy)

## DEFINE PARAMETER SPACE TO SCAN OVER
# DM: exclusion_limit2 sets the limit for a fixed timescale given the axion mass. Seems weird?
f_ax, m_ax,exclusion_limit2, aa = functions.axion_parameter_space(range_length,13.,20.,-14.,-9.)


## READ IN BLACK HOLE DATA
sl_masses,sl_mass_up,sl_mass_low,sl_spins,sl_spin_up,sl_spin_low,sm_masses,sm_mass_up,sm_mass_low,sm_spins,sm_spin_up,sm_spin_low, sl_mass_error,sl_spin_error,sm_mass_error,sm_spin_error = functions.read_in_bh_data()
b_masses = np.append(sl_masses,sm_masses)
b_spins = np.append(sl_spins,sm_spins)
b_mass_error = np.append(sl_mass_error,sm_mass_error)
b_spin_error = np.append(sl_spin_error,sm_spin_error)
############################################################################################################

## BLACK HOLE SECTION INDEX
bhind = 6
# 0 - GRO J1655-40
# 1 - A 0620-00
# 2 - LMC X-3
# 3 - GW151226 (Secondary) 
# 4 - XTE J1550-564
# 5 - 4U 1543-475
# 6 - LMC X-1
# 7 - GW151226 (Primary) 
# 8 - GRS 1915+105
# 9 - Cygnus X-1
#10 - GW170104 (Secondary)
#11 - M33 X-7
#12 - GW150914  (Secondary)
#13 - GW170104 (Primary)
#14 - GW150914 (Primary)
#15 - Fairall 
#16 - Mrk79
#17 - NGC3783 
#18 - Mrk335
#19 - MCGD6
#20 - Mrk 
#21 - NGC 
#22 - Ark 
#23 - NGC

## BLACK HOLE DATA POINT 
xtem,ytem,dytem,dxtem = functions.black_hole_selection(bhind,b_masses,b_spins,b_mass_error,b_spin_error)

	
#################################################################################

## EXCLUSION FUNCTION CALUCLATION LOOP
#for ii in range (len(f_ax)):
#	index = ii
	
#	f_a=[f_ax[ii]]

f_a=[1.e20]
	# DM: something in here takes a long time: up to ~20 s per evaluation of an (m,f) point.
total_exclusion = functions.exclusion_function(m_ax, f_a, X, Y, B,n2,l,m,exclusion_limit2,xtem,ytem,dytem,dxtem,plots=True,verbose=True)
	#np.save('results_data/LMCX-1_exclusion'+str(index),total_exclusion)		


####################################################

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import sympy as sy
import functions
import diag
import eoms
import output
import spectrum_plot
import cosmology_plot
import superradiance_plot
np.set_printoptions(threshold=np.nan)

####################################################


class axiverse_parameters(object):
	
	def __init__(self):
		self.n_ax, self.beta_ax = functions.random_hodge()

		self.phi_in_range,self.phidot_in,self.mo,self.basis,self.n,self.betaK,self.betaM,self.a0,self.b0,self.fav,self.kmin,self.kmax,self.mmin,self.mmax,self.FL3,self.sbar,self.svar,self.Nbar,self.Nvar = functions.read_in_matrix()
		
	def diagonalisation(self):
		self.ma_array,self.fef,self.phiin_array,self.phidotin_array = diag.diag(self.phi_in_range,self.phidot_in,self.basis,self.mo,self.n,self.betaK,self.betaM,self.a0,self.b0,self.fav,self.kmin,self.kmax,self.mmin,self.mmax,self.FL3,self.sbar,self.svar,self.Nbar,self.Nvar)
		global ma_array, fef, phiin_array, phidotin_array, n, axms 
		phidotin_array = self.phidotin_array
		phiin_array = self.phiin_array
		axms = np.array([1.0e-11,10**-12])
	
	def spectrum_out(self):
		fefvisual = True
		mavisual = True
		spectrumshow = True
		if spectrumshow == True:
			spectrum_plot.spectrum_out(self.ma_array,self.fef,fefvisual,mavisual)
		plt.show()		


class axion_dynamic(object):
	
	def __init__(self):
		
		self.ain,self.tin,self.tfi,self.N,self.n_cross,self.rho_bar,self.rho_mat,self.rho_rad,self.rho_lam = functions.read_in_cosmology()
		self.rhoin_array = eoms.rhoinitial(phidotin_array, phiin_array, ma_array, n)
		self.y0 = eoms.yinitial(n,phiin_array,phidotin_array,self.rhoin_array,self.ain)
	
	def eq(self,y,t):
		self.crossing_index=[0]*n
		return eoms.deriv_wfromphi(y, t, n, self.n_cross,self.crossing_index,ma_array, self.rho_mat, self.rho_rad, self.rho_lam)
	
	def solver(self):
		self.t = np.logspace(np.log10(self.tin),np.log10(self.tfi),self.N)
		self.y = sp.integrate.odeint(self.eq, self.y0, self.t, mxstep=100000000)
		
	def output(self):
		N = len(self.t)
		
		self.rhoa,self.rhom, self.rhor, self.rholl,self.rhob,self.rhosum,self.omegar, self.omegam,self.omega,self.omegal,self.omegab,self.H,self.rhoo,self.rhon,self.ODE,self.ODM = output.dense(self.rho_lam,self.rho_mat,self.rho_rad,self.rho_bar,N,self.y,n,self.t,ma_array)
		self.P, self.Psum,self.w,self.a,self.add,self.z,self.zind = output.pressure(self.y,ma_array,N,n,self.rhom,self.rho_lam,self.rhor,self.rhoa,self.rhosum)
		self.phi,self.phid = output.axionphi(self.y,N)
		self.camb_param = functions.camb_params()
						
	def printout(self):
		#print(self.z[self.zind], self.H[self.zind], self.w[self.zind], self.rhoo[self.zind], self.rhon[self.zind], self.rhor[self.zind], self.rhom[self.zind], self.rholl[self.zind], self.add[self.zind])
		cosmology_plot.cosmology(self.rhoa,self.rhosum,self.rhor,self.rhom,self.y,self.rholl)
		cosmology_plot.camb_output_plot(self.camb_param)
		cosmology_plot.cmb_plot()
			
####################################################


class superradiance_calculator(object):
	global axms
	axms = np.array([ 10.70166407e-11])
	def __init__(self):
		self.axm,self.astar,self.g,self.l,self.m,self.n,self.bhml,self.bhmu,self.supermassive,self.constraint,self.accuracy= functions.read_in_blackhole()
		self.sr_spins,self.sr_masses,self.sr_spin_up,self.sr_spin_low,self.sr_mass_up,self.sr_mass_low,self.sm_spins,self.sm_masses,self.sm_spin_up,self.sm_spin_low,self.sm_mass_up,self.sm_mass_low,self.example_mass,self.example_spin,self.example_spin_error,self.example_mass_error = functions.black_hole_data()
		self.bhms,self.bhm,self.alpha,self.rg,self.rp,self.wp,self.X,self.Y = functions.parameters(self.bhml,self.bhmu,self.g,axms,self.astar,self.supermassive,self.accuracy)
		#self.time = functions.exclusion_time(self.bhms,self.constraint,self.axm)
		
	 
	def output(self):
		self.exclusion_limit = functions.exclusion_time(self.constraint,axms)
		self.rates,self.Z = functions.superradiance_rates_detweiler(self.l,self.m,self.n,self.alpha,self.astar,axms,self.rp,self.X,self.Y,self.accuracy)
		#self.leaver_rate = (sy.mpmath.findroot(functions.superradiance_rates_leaver_fraction,0.199 + 1j*10**-8))
		self.x1,self.y1=functions.regge_contour_limits(self.X,self.Y,self.Z,self.l,self.exclusion_limit)
		self.fx,self.fy=functions.regge_contour_outline(self.x1,self.y1,self.l,self.bhml,self.bhmu)
		self.ind=functions.black_hole_function_map(self.sr_masses,self.fx,self.fy)
		
	def stats(self):
		self.total_exclusion,self.probability,self.xtem,self.ytem,self.dytem,self.dxtem = functions.effective_zvalue(self.example_spin_error, self.example_mass_error, self.example_spin, self.example_mass, self.fx, self.fy,self.bhmu)	
		
	def print_out(self):
		
		print('Axion Mass = {0}'.format(self.axm))
		print('Black Hole Mass Range (Solar Mass) = {0} - {1}'.format(self.bhml,self.bhmu))
		print('Regge Plane For Modes - 1 - {0}'.format(len(self.x1)))	
		print('Exclusion Probability for Black Holes = ', self.probability)
		print('Total Exclusion Probability for Black Holes = ', self.total_exclusion)
		
		colours=['#045a8d','#2b8cbe','#74a9cf','#bdc9e1','#f1eef6']
		ratesplot=False
		if ratesplot == True:
			superradiance_plot.superradiance_rates_plot(self.alpha,self.rates)
			plt.show()

		regge_zone = True
		if regge_zone == True:
			blackholes=True
			error_ellipse=True
			reggetrajectories=True
			superradiance_plot.regge_region_plot(self.fx,self.fy,blackholes,reggetrajectories,self.xtem,self.ytem,self.dytem,self.dxtem,self.example_mass,self.example_spin,self.example_spin_error,self.example_mass_error,error_ellipse,self.bhmu)
			plt.show()

		regge_final = False
		if regge_final == True:
			superradiance_plot.regge_plane_plot(self.x1,self.y1,colours,self.sr_spins,self.sr_masses,self.sr_spin_up,self.sr_spin_low,self.sr_mass_up,self.sr_mass_low)
			superradiance_plot.quantum_levels_legend(colours,self.l)
			superradiance_plot.conf_legend()
			plt.show()			

def main():
	
	Spectra = False
	Cosmology = False
	Superradiance = True

	#############SPECTRA#############
	if Spectra == True:
		nax_spectra = axiverse_parameters()
		nax_spectra.diagonalisation()
		nax_spectra.spectrum_out()
	#################################
	
	############COSMOLOGY#############
	if Cosmology == True:
		nax_cosmology = axion_dynamic()
		nax_cosmology.solver()
		nax_cosmology.output()
		nax_cosmology.printout()
	#################################
	
	'''
	#######BAYESIAN#NETWORK##########
	if Network == True:
		nax_bayesian = network_cosmology()
		nax_bayesian.solver()
		nax_bayesian.output()
		nax_bayesian.printout()
	#################################
	'''
	
	############# SUPERRADIANCE ##############
	if Superradiance == True:
		nax_superradiance = superradiance_calculator()
		nax_superradiance.output()
		nax_superradiance.stats()
		nax_superradiance.print_out()
	##########################################
	
if __name__ == "__main__":
	main()

#########################################################################################################
#########################################################################################################



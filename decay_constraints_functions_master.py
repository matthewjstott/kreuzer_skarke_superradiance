#!/usr/bin/python

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib.ticker import LogFormatterExponent
from matplotlib import rc
from matplotlib.ticker import LogFormatterExponent
import math
import scipy as sp
import scipy.interpolate
import warnings
import pandas as pd
import itertools
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.special import erf, erfc
from scipy import interpolate

def _rectangle_intersection_(x1,y1,x2,y2):
	S1,S2,S3,S4=_rect_inter_inner(x1,x2)
	S5,S6,S7,S8=_rect_inter_inner(y1,y2)

	C1=np.less_equal(S1,S2)
	C2=np.greater_equal(S3,S4)
	C3=np.less_equal(S5,S6)
	C4=np.greater_equal(S7,S8)

	ii,jj=np.nonzero(C1 & C2 & C3 & C4)
	return ii,jj
def non_decreasing(L):
	return all(x<=y for x, y in zip(L, L[1:]))
def non_increasing(L):
	return all(x>=y for x, y in zip(L, L[1:]))		
def _rect_inter_inner(x1,x2):
	n1=x1.shape[0]-1
	n2=x2.shape[0]-1
	X1=np.c_[x1[:-1],x1[1:]]
	X2=np.c_[x2[:-1],x2[1:]]
	S1=np.tile(X1.min(axis=1),(n2,1)).T
	S2=np.tile(X2.max(axis=1),(n1,1))
	S3=np.tile(X1.max(axis=1),(n2,1)).T
	S4=np.tile(X2.min(axis=1),(n1,1))
	return S1,S2,S3,S4
def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx
def grad(x,fx,fy):
	f = interp1d( fx, fy,bounds_error=False )
	a=0.01
	xtem = np.linspace(x-a,x+a,99)
	ytem = f(xtem)
	m = (ytem[-1]-ytem[0])/(xtem[-1]-xtem[0])
	return(m)
	
def grady(y,fx,fy):
	f = interp1d( fy, fx ,bounds_error=False)
	a=0.01
	ytem = np.linspace(y-a,y+a,99)
	xtem = f(ytem)
	m = (xtem[-1]-xtem[0])/(ytem[-1]-ytem[0])
	return(m)		
def intersection(x1,y1,x2,y2):
	"""
x,y=intersection(x1,y1,x2,y2)
	Example:
	a, b = 1, 2
	phi = np.linspace(3, 10, 100)
	x1 = a*phi - b*np.sin(phi)
	y1 = a - b*np.cos(phi)
	x2=phi
	y2=np.sin(phi)+2
	x,y=intersection(x1,y1,x2,y2)
	plt.plot(x1,y1,c='r')
	plt.plot(x2,y2,c='g')
	plt.plot(x,y,'*k')
	plt.show()
	"""
	ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
	n=len(ii)

	dxy1=np.diff(np.c_[x1,y1],axis=0)
	dxy2=np.diff(np.c_[x2,y2],axis=0)

	T=np.zeros((4,n))
	AA=np.zeros((4,4,n))
	AA[0:2,2,:]=-1
	AA[2:4,3,:]=-1
	AA[0::2,0,:]=dxy1[ii,:].T
	AA[1::2,1,:]=dxy2[jj,:].T

	BB=np.zeros((4,n))
	BB[0,:]=-x1[ii].ravel()
	BB[1,:]=-x2[jj].ravel()
	BB[2,:]=-y1[ii].ravel()
	BB[3,:]=-y2[jj].ravel()

	for i in range(n):
		try:
			T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
		except:
			T[:,i]=np.NaN


	in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

	xy0=T[2:,in_range]
	xy0=xy0.T
	return xy0[:,0],xy0[:,1]
def regge_function(fx,fy,x):
	f = interp1d( fx, fy,bounds_error=False )
	y=f(x)
	return(y)		
def fxn():
	warnings.warn("deprecated", DeprecationWarning)
def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx




def black_hole_parameter_space(accuracy):
	newton = 6.7071186*10**-57
	solar_mass = 2.*10**30*5.6095886*10**35
	a = np.logspace(-2,0,accuracy)
	mm = np.logspace(-2.0,3,accuracy)*solar_mass*newton
	X,Y = np.meshgrid(mm,a)
	return(a,mm,X,Y)


def axion_parameter_space(acc,fa_min,fa_max,ma_min,ma_max):
	
	time_limit = 5*10**7 # limit in years, Salpeter time
	years = 31536000 # seconds in a year
	sinvev = 1.51926757933*10**15 # second in inverse eV
	#########
	range_length = acc # Accuracy of parameter space
	############################################################################################################
	### Axion Parameter Space
	# DM: fa in GeV, ma in eV
	############################################################################################################
	total_exclusion=[]
	f_ax = np.logspace(fa_min,fa_max,range_length)
	m_ax = np.logspace(ma_max,ma_min,range_length)
	isocontour_values=[]
	for i in range(len(m_ax)):
		isocontour_values.append(1./(time_limit*years*sinvev*m_ax[i]))
	return(f_ax,m_ax,isocontour_values)

############################################################################################################
def read_in_bh_data():
	#df=pd.read_csv('Black_hole_spin_data_Decay.csv', sep=',',header=None,encoding='latin-1')
	df=pd.read_csv('Black_hole_spin_data_Ti_plot.csv', sep=',',header=None,encoding='latin-1')

	solar_len = (df[1].str.contains(r'Solar').sum())+1	
	sm_len = (df[1].str.contains(r'Supermassive').sum())	

	sl_masses = pd.to_numeric(df[2][1:solar_len].values)
	sl_mass_up = pd.to_numeric(df[3][1:solar_len].values) 
	sl_mass_low = pd.to_numeric(df[4][1:solar_len].values)
	sl_spins =  pd.to_numeric(df[5][1:solar_len].values)
	sl_spin_up = pd.to_numeric(df[6][1:solar_len].values)
	sl_spin_low = pd.to_numeric(df[7][1:solar_len].values)
	
	sm_masses = pd.to_numeric(df[2][solar_len:sm_len+solar_len].values)
	sm_mass_up = pd.to_numeric(df[3][solar_len:sm_len+solar_len].values) 
	sm_mass_low = pd.to_numeric(df[4][solar_len:sm_len+solar_len].values)
	sm_spins =  pd.to_numeric(df[5][solar_len:sm_len+solar_len].values)
	sm_spin_up = pd.to_numeric(df[6][solar_len:sm_len+solar_len].values)
	sm_spin_low = pd.to_numeric(df[7][solar_len:sm_len+solar_len].values)

	sl_mass_error = list(map(max, sl_mass_up, sl_mass_low))
	sl_spin_error = list(map(max, sl_spin_up, sl_spin_low))

	sm_mass_error = list(map(max, sm_mass_up, sm_mass_low))
	sm_spin_error = list(map(max, sm_spin_up, sm_spin_low))
	
	return(sl_masses,sl_mass_up,sl_mass_low,sl_spins,sl_spin_up,sl_spin_low,sm_masses,sm_mass_up,sm_mass_low,sm_spins,sm_spin_up,sm_spin_low,sl_mass_error,sl_spin_error,sm_mass_error,sm_spin_error)

def radial_constants(m,l,n):
	rad1={}
	rad2={}
	for i in range(len(m)):
		# DM: added multiply by 1. to fix typing of rad2 and be non-zero
		rad1[i]=[]
		rad1[i] = (2**(4*l[i]+2)*math.factorial(2*l[i]+n[i]+1))/((l[i]+n[i]+1)**(2*l[i]+4)*math.factorial(n[i]))
		rad2[i]=[]
		rad2[i] = (1.*math.factorial(l[i])/(1.*math.factorial(2*l[i])*math.factorial(2*l[i]+1)))**2.
	B={}
	for i in range (len(m)):
		B[i]=[]
		B[i] = rad1[i]*rad2[i]
	return(B)
	
def black_hole_selection(bhind,x,y,dx,dy):
	bhmu = 200
	# DM: what is bhmu?
	# Returns mass and spin and errors for solar mass BHs
	# Weird way to do it?
	xtem=[]
	ytem=[]
	dytem=[]
	dxtem=[]
	for i in range(bhind,bhind+1):
			#if x[i]<bhmu:
			xtem.append(x[i])
			ytem.append(y[i])
			dytem.append(dy[i])
			dxtem.append(dx[i])	
	return(xtem,ytem,dytem,dxtem)				
	
def exclusion_function(m_ax,f_ax,X,Y,B,n2,l,m,isocontour_values,xtem,ytem,dytem,dxtem,plots=False,verbose=False):




	planck = 2.435*10**18 # planck mass in GeV
	newton = 6.7071186*10**-57 # Newton conversion, geometric to physical/natural units (DM?)
	solar_mass = 2.*10**30*5.6095886*10**35 # solar mass conversion to geometric units (DM?)
	delta_a = 0.3
	#m=2
	c_0 = 5 
	n = 2

	if verbose:
		print('f_a [GeV] = ', f_a)
	total_exclusion=[]



	for jjj in range(len(f_ax)):

		f_a = f_ax[jjj]
		for ii in range(len(m_ax)):
			print(jjj,ii)
			if verbose:
				print('m_ax [eV] = ', m_ax[ii])
			axm2=[m_ax[ii]]

			#############################################
			# Start computing ZZf contours for exclusion
			ZZf1=[]
			ZZf2 = []
			ZZf3 = []
			if verbose:
				print('Filling contour array ...')
			for j in range (len(axm2)):

				Zf1={}
				Zf2 = {}
				Zf3 = {}
				for i in range(len(l)):
					Zf1[i] = []
					Zf2[i] = []
					Zf3[i] = []
					prdc = 1
					for lj in range(l[i]):
						prdc = prdc*((lj+1)**2*(1-Y**2)+((m[i]*Y-2*axm2[j]*(X + X*(1-Y**2)**0.5))**2))

					#print('BH mass in Msol', X/newton/solar_mass)
					#################################


					# DM: rewritten treatment of Bosenova in  terms of self-coupling.
					# Checked this notation conversion gives exclusion probability equivalemt to 1.e-3
					# Looks wrong because expression diverges in limit f to inifinity, when it should just reproduce the free-field?


					################################
					# Arvanitaki et al N, Eq. 9, 1411.2263
					#Nbose = (10**78*c_0*(n2[i]**4/(X*axm2[j])**3)*((X/newton)/(10*solar_mass))**2*(f_a/planck)**2)
					# Canonical coupling for V = Lambda*phi^4/4! (note factorial 4)


					Lambda = axm2[j]**2./(1.e9*f_a)**2.




					Nbose = c_0*30.*n2[i]**4/Lambda/(axm2[j]*X)


					# This can be compared to the naive estimate N = (4!/2!)/Lambda = 12/Lambda
					# Q: where does the n^4/alpha dependence come from? And extra ~ 2 in pre-factor.
					# A: see detailed treatment of self-energy in Sec. 3.2.1 of 1004.3558

					# Not understood: we have n2^4, Arvanitaki has ell^4, while 1411.2263 has n^4.
					# MS: In 1411.2263 their n is the principle quantum number n2 (n+l+1) and in Arvanitaki they state
					# "Here we made use of (11) for the size of the cloud, and set n (principle) \sim l." No idea why they do this but I think all three are consistently the same.

					##################################

					# DM: Q: what is being computed here? Eq. in paper?
					# MS: This is Eq. 16 from 1704.05081 where N_m (occupation number of maximally filled cloud) is now replaced with an expression using N_bosenova in 1411.2263
					#print(((isocontour_values[j]))**2/((X**2)/m[i]*delta_a)*Nbose)





					#((isocontour_values[j])) ** 2 / ((X ** 2) / m[i] * delta_a) * Nbose

					Zf1[i]=(axm2[j]*X)**(4*l[i]+5)      *      (m[i]*Y-2*axm2[j]*((X + X*(1-Y**2)**0.5)))                     *B[i]*prdc*    (f_a/planck)**2

					Zf2[i] = (axm2[j]*X) ** (4 * l[i]+5) * (m[i]*Y-2*axm2[j]*((X+X*(1-Y**2)**0.5)))*B[i]*prdc
					#########################################

					###############################
					# Matt's original expression
					#Zf3[i]=(axm2[j]*X)**(4*l[i]+5)*(m[i]*Y-2*axm2[j]*((X + X*(1-Y**2)**0.5)))*B[i]*prdc*                      ((isocontour_values[j]))**2*(10**78*c_0*(n2[i]**4/(X*axm2[j])**3)*((X/newton)/(10*solar_mass))**2*(f_a/planck)**2)/((X**2)/m[i]*delta_a)
					#################################

				# This array is contour plotted and the contours are used to compute the probability by overlap with Gaussian data point
				ZZf1.append(Zf1.copy())
				ZZf2.append(Zf2.copy())
				ZZf3.append(Zf3.copy())

			#################################################################################

			cs={}
			p1={}
			v1={}
			x1={}
			y1={}
			if verbose:
				print('Computing contours ...')
			for i in range (len(l)):
				if verbose:
					print('ell = ',l[i])
				cs[i]=[]
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					# DM: why is the contour drawn at 500?
					# DM: ZZf goes to infinity as Nbose and f_a go to infinity, and this affects the contour position
					cs[i] = plt.contour(X/(solar_mass*newton),Y,ZZf1[0][i],[isocontour_values[j]],cmap=plt.cm.bone,alpha=0.5,linewidths = 2.5,zorder=2)
					#cs[i] = plt.contour(X / (solar_mass * newton), Y, ZZf2[0][i], [isocontour_values[j]], colors='red',linestyle='--', alpha=0.5,linewidths=2.5, zorder=2)
					#cs[i] = plt.contour(X / (solar_mass * newton), Y, ZZf3[0][i], [isocontour_values[j]],colors='blue', linestyle=':', alpha=0.5,linewidths=2.5, zorder=2)


				#print(i)
				p1[i]=[]
				v1[i]=[]
				x1[i]=[]
				y1[i]=[]
				p1[i] = cs[i].collections[0].get_paths()

				if len(p1[i])>1:
					for j in range(len(p1[i])):
						try:
							temp = p1[i][j].vertices
							x1[i].append((temp[:,0]))
							y1[i].append((temp[:,1]))
						except IndexError:
							del v1[i]
							del x1[i]
							del y1[i]
							continue
				else:
					try:
						temp = p1[i][0].vertices
						x1[i].append((temp[:,0]))
						y1[i] = np.append(y1[i],(temp[:,1]))
					except IndexError:
						del v1[i]
						del x1[i]
						del y1[i]
						continue
			plt.close()
			# DM: added kwarg for plots to help debugging
			if plots:
				plt.errorbar(xtem,ytem,xerr=dxtem,yerr=dytem,linewidth=2.5,markersize=2.5)
				plt.xscale('log')
				plt.show()

			if verbose:
				print('Computing probabilities ... ')
			for i in range(len(x1)):
				if (len(y1[i])==2):
					continue
				else:
					if non_increasing(y1[i]) == True or non_decreasing(y1[i])==True and x1[i][0][-1]>35:
						del x1[i]
						del y1[i]

			#################################################################################

			nx=[]
			ny=[]
			new_length = 4000
			fx = np.logspace(-2, 3.1, new_length)

			for i in range (len(x1)):
				# DM : this loop weird. Raises key error when f_a very large
				if len(x1[i])>1:
					ty = list(itertools.chain.from_iterable(y1[i]))
					tx = list(itertools.chain.from_iterable(x1[i]))
					ny.append(sp.interpolate.interp1d(tx, ty,bounds_error=False)(fx))
				else:
					ny.append(sp.interpolate.interp1d(x1[i][0], y1[i],bounds_error=False)(fx))
			try:
				nyy = np.vstack((ny[0::]))
			except ValueError:
					#print('here')
					total_exclusion.append(0)
					continue

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				fy = np.nanmin(nyy,axis=0)
				fxn()

			mask = ~np.isnan(fy)
			fx = fx[mask]
			fy = fy[mask]


		#################################################################################

			normprob=[]
			for i in range(len(xtem)):
				if fx[-1] < xtem[i]:
					check = 1
				elif fx[0] < xtem[i] < fx[-1]:
					check =2
				else:
					check =3

				if check == 1:
					try:
						yreduced = np.array(fy) - ytem[i]
						freduced = interpolate.UnivariateSpline(fx, yreduced, s=0)
						try :
							xroot = (freduced.roots()[-1])
						except IndexError or ValueError:
							normprob.append(1.0)
							continue
						zscore=((xroot-xtem[i])/dxtem[i])
						prob = 1.-(0.5+0.5*erf(zscore/np.sqrt(2)))
						normprob.append(np.abs(prob))
					except ValueError or  IndexError:
						normprob.append(1.0)


				if check == 2:
					ff = (regge_function(fx,fy,xtem[i]))
					fdx = (grad(xtem[i],fx,fy))

					if math.isnan(fdx) == True:
						normprob.append(1.0)
						continue

					effvar = (dytem[i]**2+fdx**2*dxtem[i]**2)
					effsd = (np.sqrt(effvar))
					zscore=((ff-ytem[i])/effsd)
					prob = 0.5+(0.5*erf(zscore/np.sqrt(2)))
					normprob.append(np.abs(prob))

				if check == 3:

					try:
						yreduced = np.array(fy) - ytem[i]
						freduced = interpolate.UnivariateSpline(fx, yreduced, s=0)
						try:
							xroot = (freduced.roots()[0])
						except IndexError or ValueError:
							normprob.append(1.0)
							continue
						zscore=((xroot-xtem[i])/dxtem[i])
						prob = (0.5+0.5*erf(zscore/np.sqrt(2)))
						normprob.append(np.abs(prob))
					except ValueError or IndexError:
						normprob.append(1.0)



			normprob = np.array(normprob)
			total_exclusion_temp = np.product(normprob)
			total_exclusion_temp = 1.-total_exclusion_temp
			total_exclusion.append(total_exclusion_temp)
			if verbose:
				print('exclusion probability = ', total_exclusion_temp)

	return (total_exclusion)
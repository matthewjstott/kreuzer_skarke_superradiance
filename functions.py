from __future__ import division
import numpy as np 
from scipy.stats import norm, chi2
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import math
import pandas as pd
import configparser
import json
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.special import erf, erfc
from scipy import interpolate
import sys
import copy
from shapely.geometry import LineString
import scipy.optimize as optimize
import warnings
from operator import itemgetter
from functools import reduce
import itertools
import camb
import site
import os

def random_hodge():
	h1=[]
	h12= []
	with open('poly.txt', 'r') as f:
		content = f.readlines()
		for x in content:
			row = x.split()
			h1.append(float(row[0]))
			h12.append(float(row[1]))
			
	fit_alpha_h1, fit_loc_h1, fit_beta_h1=sp.stats.gamma.fit(h1)
	fit_alpha_h12, fit_loc_h12, fit_beta_h12=sp.stats.gamma.fit(h12)
	h1_ran = int(np.random.gamma(fit_alpha_h1,fit_beta_h1, 1))
	h12_ran = int(np.random.gamma(fit_alpha_h12,fit_beta_h12, 1))
	beta_ran=(h1_ran/(h1_ran+h12_ran+1.))
	return(h1_ran,beta_ran)		

def read_in_matrix():
	
	config = configparser.ConfigParser()   
	config.read('configuration_card.ini')
	
	phi_in_range = config.getfloat('Initial Conditions','phi_in_range')
	phidot_in = config.getfloat('Initial Conditions','phidot_in')
	
	mo = config.getint('Model_Selection','Model')
	basis = config.getint('Model_Selection','Basis')
	n = config.getint('Hyperparameter','Number of Axions')
	betaK = config.getfloat('Hyperparameter','betaK')
	betaM = config.getfloat('Hyperparameter','betaM')
	
	a0 = config.getfloat('Hyperparameter','a0')
	b0 = config.getfloat('Hyperparameter','b0')
	fav = config.getfloat('Hyperparameter','fNflation')
	kmin = config.getfloat('Hyperparameter','kmin')
	kmax = config.getfloat('Hyperparameter','kmax')
	mmin = config.getfloat('Hyperparameter','mmin')
	mmax = config.getfloat('Hyperparameter','mmax')
	
	FL3 = config.getfloat('Hyperparameter','FL3')
	sbar = config.getfloat('Hyperparameter','sbar')
	svar = config.getfloat('Hyperparameter','svar')
	Nbar = config.getfloat('Hyperparameter','Nbar')
	Nvar = config.getfloat('Hyperparameter','Nvar')
	
	return(phi_in_range,phidot_in,mo,basis,n,betaK,betaM,a0,b0,fav,kmin,kmax,mmin,mmax,FL3,sbar,svar,Nbar,Nvar)

def read_in_cosmology():
	
	config = configparser.ConfigParser()   
	config.read('configuration_card.ini')

	a = config.getfloat('Initial Conditions','a_in')
	tin = config.getfloat('Initial Conditions','t_in')
	tfin = config.getfloat('Initial Conditions','t_fi')
	ts = config.getint('Evolution Settings','Number of time steps')
	ncross = config.getint('Evolution Settings','Number of Crossings')
	rhocrit=3.
	rho_bar = config.getfloat('Cosmo Params','Ombh2')*rhocrit
	rho_mat = config.getfloat('Cosmo Params','Omh2')*rhocrit
	rho_lam = config.getfloat('Cosmo Params','Orh2')*rhocrit
	rho_rad = config.getfloat('Cosmo Params','Olh2')*rhocrit
	
	return(a,tin,tfin,ts,ncross,rho_bar,rho_mat,rho_lam,rho_rad)

def read_in_blackhole():
	
	config = configparser.ConfigParser()   
	config.read('configuration_card.ini')
	
	axm = config.getfloat('Input', 'Axion Mass')
	astar = config.getfloat('Input', 'Black Hole Spin')
	g = config.get('Input', 'Gravitational Constant')
	l = json.loads(config.get('Input', 'Quantum Levels l'))
	m = json.loads(config.get('Input', 'Quantum Levels m'))
	n = json.loads(config.get('Input', 'Overtone Modes')) 
	bhml = config.getfloat('Input', 'Lower Black Hole Mass')
	bhmu = config.getfloat('Input', 'Upper Black Hold Mass')
	supermassive = config.get('Input', 'Supermassive')	
	constraint = config.getfloat('Input', 'Time Scale')
	accuracy = config.getint('Input', 'Accuracy')
	supermassive = str(supermassive)
	axm=float(axm)
	g=float(g)
	constraint = float(constraint)
	return(axm,astar,g,l,m,n,bhml,bhmu,supermassive,constraint,accuracy)

def camb_params():
	print('Currently using CAMB version: %s '%camb.__version__)
	sub_path = '/pysm-2.0-py3.6.egg/pysm/template'
	path = site.getsitepackages()[0]+sub_path
	filename = 'camb_lenspotentialCls.dat'
	filename = os.path.join(path, filename)
	camb_output = np.loadtxt(filename)
	
	pars = camb.CAMBparams()
	pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
	pars.InitPower.set_params(ns=0.965, r=0)
	pars.set_for_lmax(2199, lens_potential_accuracy=0);
	results = camb.get_results(pars)
	powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
	cl = results.get_lens_potential_cls(lmax=2199)

	totCL=powers['total']
	unlensedCL=powers['unlensed_scalar']

	ls = np.arange(totCL.shape[0])
	L = np.array(ls[2:])
	TT = np.array(unlensedCL[:-2,0])
	EE = np.array(unlensedCL[:-2,1])
	BB = np.array(unlensedCL[:-2,2])
	TE = np.array(unlensedCL[:-2,3])
	PP = np.array(([i[5] for i in camb_output]))
	TP = np.array(([i[6] for i in camb_output]))
	EP = np.array(([i[7] for i in camb_output]))
	
	cmb_param = np.column_stack((L,TT,EE,BB,TE,PP,TP,EP))
	np.savetxt(filename,cmb_param)
	
	return(cmb_param)



def effective_zvalue(dy,dx,y,x,fx,fy,bhmu):
	xtem=[]
	ytem=[]
	dytem=[]
	dxtem=[]
	normprob=[]
	for i in range(len(x)):
			if x[i]<bhmu:
				xtem.append(x[i])
				ytem.append(y[i])
				dytem.append(dy[i])
				dxtem.append(dx[i])		
	for i in range(len(xtem)): 
		if fx[-1] > xtem[i]:
			try:
				print('this one')
				f = (regge_function(fx,fy,xtem[i]))
				fdx = (grad(xtem[i],fx,fy))
				effvar = (dytem[i]**2+fdx**2*dxtem[i]**2)
				effsd = (np.sqrt(effvar))
				zscore=((f-ytem[i])/effsd)
				zupper=((1-ytem[i])/effsd)
				zlower=((0-ytem[i])/effsd)
				prob = 0.5+(0.5*erf(zscore/np.sqrt(2)))
				exprob = 1-prob
				prob_upper = (0.5*erfc(zupper/np.sqrt(2)))
				prob_lower = (0.5+ (0.5*erf(zlower/np.sqrt(2))))
				norm = (1-prob_upper-prob_lower)
				exprob = exprob- prob_upper-prob_lower
				normprob.append(np.abs(exprob/norm))	
				
			except ValueError:
				print('that one')
				yreduced = np.array(fy) - ytem[i]
				freduced = interpolate.UnivariateSpline(fx, yreduced, s=0)
				xroot = (freduced.roots()[0])
				zscore = ((xroot-xtem[i])/dx[i])
				normprob.append((0.5+(0.5*erf(zscore/np.sqrt(2)))))	
		else:
			
			yreduced = np.array(fy) - ytem[i]
			freduced = interpolate.UnivariateSpline(fx, yreduced, s=0)
			xroot = (freduced.roots()[-1])
			zscore = ((xroot-xtem[i])/dx[i])
			normprob.append(0.5+(0.5*erf(zscore/np.sqrt(2))))
	
	total_exclusion = np.product(normprob)					
	return(total_exclusion,normprob,xtem,ytem,dytem,dxtem)

def exclusion_time(time_limit,axm):
	years = 31536000
	sinvev = 1.51926757933*10**15
	exlusion_limit = 1./(time_limit*years*sinvev*axm)
	return(exlusion_limit)
	
def rates_time(rate,axm):
	years = 31536000
	sinvev = 1.51926757933*10**15
	time_scales = 1./rate*years*sinvev*axm 
	return(time_scales)	
		
def black_hole_function_map(masses,fx,fy):
	ind=[]
	for i in range(len(masses)):
		fx=np.array(fx)
	return(ind)	

def parameters(bhml,bhmu,g,ma_array,astar,supermassive,accuracy):
	if supermassive == False:
		bhms = np.linspace(bhml,bhmu,accuracy)
	else:
		bhms = np.linspace(bhml,bhmu,accuracy)
	bhm = bhms*2.*10**30*5.6095886*10**35
	alpha = np.einsum('i,ji->ij', ma_array, np.array([np.linspace(bhml,bhmu,accuracy)]*(len(ma_array))).T)*2.*10**30*5.6095886*10**35*g
	rg = g*bhm
	rp = rg + rg*(1-astar**2)**0.5
	wp = (1/(2*rg))*(astar/(1+(1-astar**2)**0.5))
	a = np.linspace(0.01,1.0,accuracy)
	mm = np.linspace(bhml,bhmu,accuracy)*2.*10**30*5.6095886*10**35*g
	X,Y = np.meshgrid(mm,a)
	return(bhms,bhm,alpha,rg,rp,wp,X,Y)


def black_hole_data():
	
	df=pd.read_csv('Black_hole_spin_data_Ti_plot.csv', sep=',',header=None,encoding='latin-1')
	solar_len = (df[1].str.contains(r'Solar').sum())+1	
	sl_masses = pd.to_numeric(df[2][1:solar_len].values)
	sl_mass_up = pd.to_numeric(df[3][1:solar_len].values) 
	sl_mass_low = pd.to_numeric(df[4][1:solar_len].values)
	sl_spins =  pd.to_numeric(df[5][1:solar_len].values)
	sl_spin_up = pd.to_numeric(df[6][1:solar_len].values)
	sl_spin_low = pd.to_numeric(df[7][1:solar_len].values)
	sm_masses = pd.to_numeric(df[2][solar_len:-1].values)
	sm_mass_up = pd.to_numeric(df[3][solar_len:-1].values) 
	sm_mass_low = pd.to_numeric(df[4][solar_len:-1].values)
	sm_spins =  pd.to_numeric(df[5][solar_len:-1].values)
	sm_spin_up = pd.to_numeric(df[6][solar_len:-1].values)
	sm_spin_low = pd.to_numeric(df[7][solar_len:-1].values)
	
	
	
	
			
	example_mass = [10,24]
	example_spin = [0.3,0.7]
	example_spin_error = [0.05,0.08]
	example_mass_error = [0.95,1.4]		
			
	return(sl_spins,sl_masses,sl_spin_up,sl_spin_low,sl_mass_up,sl_mass_low,sm_spins,sm_masses,sm_spin_up,sm_spin_low,sm_mass_up,sm_mass_low,example_mass,example_spin,example_spin_error,example_mass_error)	
def cov_ellipse(cov, q=None, nsig=None, **kwargs):
	if q is not None:
		q = np.asarray(q)
	elif nsig is not None:
		q = 2 * norm.cdf(nsig) - 1
	else:
		raise ValueError()
	r2 = chi2.ppf(q, 2)
	val, vec = np.linalg.eigh(cov)
	width, height = 2 * np.sqrt(val[:, None] * r2)
	rotation = np.degrees(np.arctan2(*vec[::-1, 0]))
	return width, height, rotation

def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx
def grad(x,fx,fy):
	f = interp1d( fx, fy )
	a=0.01
	xtem = np.linspace(x-a,x+a,99)
	ytem = f(xtem)
	m = (ytem[-1]-ytem[0])/(xtem[-1]-xtem[0])
	return(m)	
def fxn():
	warnings.warn("deprecated", DeprecationWarning)

def regge_function(fx,fy,x):
	f = interp1d( fx, fy )	
	y=f(x)
	return(y)

def regge_contour_outline(x1,y1,l,bhml,bhmu):
	nx=[]
	ny=[]
	new_length = 4000
	fx = np.linspace(bhml, bhmu, new_length)
	for i in range (len(x1)):
		if len(x1[i])>1:
			ty = list(itertools.chain.from_iterable(y1[i]))
			tx = list(itertools.chain.from_iterable(x1[i]))
			ny.append(sp.interpolate.interp1d(tx, ty,bounds_error=False)(fx))					
		else:
			ny.append(sp.interpolate.interp1d(x1[i][0], y1[i],bounds_error=False)(fx))
		
	nyy = np.vstack((ny[0::]))
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		fy = np.nanmin(nyy,axis=0)
		fxn()
	mask = ~np.isnan(fy)
	fx = fx[mask]
	fy = fy[mask]
	w=np.blackman(25) 
	fy=np.convolve(w/w.sum(),fy,mode='same')
	return(fx,fy)


def superradiance_rates_leaver_fraction(w):
	dps = 25
	sep = Symbol('sep')
	ar = 0.9 
	mu = 0.2
	l = 1
	m = 1
	br = cmath.sqrt(1 - ar**2)
	rminus = (1 - br)
	rplus = (1 + br)
	s = 0
	k1 = 1./2.*abs(m - s)
	k2 = 1./2.*abs(m + s)
	Alm = l*(l + 1) - s*(s + 1)
	wang = cmath.sqrt(w**2 - mu**2)
	def alpha_ang(n):
		alp_ang = -2*(n + 1)*(n + 2*k1 + 1)
		return(alp_ang)
	def beta_ang(n):
		bet_ang = (n*(n - 1) + 2*n*(k1 + k2 + 1 - 2*ar*wang) - (2*ar*wang*(2*k1 + s + 1) - (k1 + k2)*(k1 + k2 + 1)) - (ar**2*wang**2 + s*(s + 1) + sep))
		return(bet_ang)
	def gamma_ang(n):
		gam_ang = 2*ar*wang*(n + k1 + k2 + s)
		return(gam_ang)
	def alpha(n):
		alp = n**2 + (c0 + 1)*n + c0
		return(alp)	
	def beta(n):
		bet = -2*n**2 + (c1 + 2)*n + c3
		return(bet)	
	def gamma(n):
		gam = n**2 + (c2 - 3)*n + c4 + (-c2 + 2)*0
		return(gam)	
	def F2(x):
		args=tuple(x)
		return F1(*args)
	def leaver_angle():
		init = -1
		for i in range(40,0,-1):
			init = (gamma_ang(i))/(beta_ang(i) - alpha_ang(i)*init)
		return(init)	
	def leaver_solve():
		init = -1
		for i in range(80,0,-1):
			init = (gamma(i))/(beta(i) - alpha(i)*init)
		return(init)		

	leaver_ang = leaver_angle()
	ang_ratio = beta_ang(0)/alpha_ang(0)
	leaver_ang_root = ang_ratio - leaver_ang


	f_list=list([leaver_ang_root])
	sep=list([sep])
	x0=np.array([0])
	args=sep
	F1=sy.lambdify(args,f_list)
	def F2(x):
		args=tuple(x)
		return F1(*args)
		
	f3='def F3(%s):\n\treturn F2(tuple(%s))'%(str(sep)[1:-1],str(sep))
	def F3(sep):
		return F2(tuple([sep]))	
	#f3='def F3(%s):\n\treturn F2(tuple(%s))'%(str(sep)[1:-1],str(sep))
	#exec f3
	Alm = (sy.mpmath.findroot(F3,list(x0)))[0]

	q = -cmath.sqrt(mu**2 - w**2)
	c0 = 1 - 2*1j*w - (2*1j/br)*(w - ar*m/2)
	c1 = -4 + 4*1j*(w - 1j*q*(1 + br)) + 4*1j/br*(w - ar*m/2) - 2*(w**2 + q**2)/q 
	c2 = 3 - 2*1j*w - 2*1j/br*(w - ar*m/2) - 2*(q**2 - w**2)/q
	c3 = 2*1j*(w - 1j*q)**3/q + 2*(w - 1j*q)**2*br + q**2*ar**2 + 2*1j*q*ar*m - Alm - 1 - (w - 1j*q)**2/q + 2*q*br + 2*1j/br*((w - 1j*q)**2/q + 1)*(w - ar*m/2) 
	c4 = (w - 1j*q)**4/q**2 + 2*1j*w*(w - 1j*q)**2/q - 2*1j/br*(w - 1j*q)**2/q*(w - ar*m/2)
		
	func_ang = leaver_solve()	
	func_ratio = beta(0)/alpha(0) - func_ang

	print(func_ratio, w)	
	return(func_ratio)


	
def superradiance_rates_detweiler(l2,m2,n2,alpha,astar,ma_array,rp,X,Y,accuracy):
	if all(ma_array[i] >= ma_array[i+1] for i in range(len(ma_array)-1)):
		pass
	else:
		ma_array = -np.sort(-ma_array)
	scale = ma_array[0:]/ma_array[0]	
	A=[]
	for j in range (len(ma_array)):
		for i in range(len(m2)):
			tem = alpha[j]**(4*l2[i]+5)*(m2[i]*astar-2*ma_array[j]*rp)
			A = np.insert(tem,0,A)
	A = np.reshape(A,(len(ma_array),len(m2),accuracy))
	
	B=[]
	for i in range(len(m2)):
		tem = (2**(4*l2[i]+2)*math.factorial(2*l2[i]+n2[i]+1))/((l2[i]+n2[i]+1)**(2*l2[i]+4)*math.factorial(n2[i])) * (math.factorial(l2[i])/(math.factorial(2.*l2[i])*math.factorial(2.*l2[i]+1.)))**2
		B = np.insert(tem,0,B)

	C=[]
	for j in range (len(ma_array)):
		for i in range(len(m2)):
			prd = 1
			for lj in range(l2[i]):
				prd = prd*((lj+1)**2*(1-astar**2)+((m2[i]*astar-2*ma_array[j]*rp)**2))
			C = np.insert(prd,0,C)
	C = np.reshape(C,(len(ma_array),len(m2),accuracy))

	rates=np.array([])
	B = np.array(accuracy*[B]).T
	for j in range(len(ma_array)):
		rtem = (reduce(np.multiply,[A[j],B,C[j]]))
		rates = np.insert(rtem,0,rates)
	rates = np.reshape(rates,(len(ma_array),len(m2),accuracy))
	rates  = list(map(sum, zip(*rates)))

	ZZ=[]
	for k in range(len(ma_array)):
		axm=ma_array[k]
		for i in range(len(l2)):
			prdc = 1
			for lj in range(l2[i]):
				prdc = prdc*((lj+1)**2*(1-Y**2)+((m2[i]*Y-2*axm*(X + X*(1-Y**2)**0.5))**2))
			Z=(axm*X)**(4*l2[i]+5)*(m2[i]*Y-2*axm*((X + X*(1-Y**2)**0.5)))*B[i][0]*prdc
			ZZ = np.insert(Z,0,ZZ)	
	ZZ[ZZ < 0] = 0
	ZZ = np.reshape(ZZ,(len(ma_array),len(m2)*accuracy**2))
	#print(zz[0][])
	ZZ = ZZ*scale[:, None]
	Z2 = list(map(sum, zip(*ZZ)))
	Z2 = np.reshape(list(Z2),(len(m2),accuracy,accuracy))

	return(rates,Z2)
		
def non_decreasing(L):
	return all(x<=y for x, y in zip(L, L[1:]))
def non_increasing(L):
	return all(x>=y for x, y in zip(L, L[1:]))	
		
def regge_contour_limits(X,Y,Z,l,exclusion_limit):
	print(exclusion_limit)
	#exclusion_limit=1.66974099e-23
	g = 6.7071186*10**-57
	solarm = 1.98852*10**30
	kgev = 5.6095886*10**35
	cs={}
	p1={}
	v1={}
	x1={}
	y1={}
	aa= np.logspace(-17.0,-17.0,1)
	for i in range (len(l)):
		print(i)
		cs[i]=[]
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			cs[i] = plt.contour(X/(solarm*kgev*g),Y,Z[i],exclusion_limit,cmap=plt.cm.bone,alpha=0.0,linewidths = 2.5,zorder=2)
	
		print(i)
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
	for i in range(len(x1)):
		if (len(y1[i])==2):
			continue
		else:	
			if non_increasing(y1[i]) == True or non_decreasing(y1[i])==True :
				del x1[i]
				del y1[i]				
	return(x1,y1)	
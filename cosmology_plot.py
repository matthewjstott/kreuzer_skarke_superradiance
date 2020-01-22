#suimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rc
from pysm.nominal import models
import pysm
import healpy as hp
import numpy as np
plt.rcParams["figure.facecolor"] = 'w'
plt.rcParams["axes.facecolor"] = 'w'
plt.rcParams["savefig.facecolor"] = 'w'

def cosmology(rhoa,rhosum,rhor,rhom,y,rholl):
	ax = plt.subplot(1,1,1)
	#plt.title('N- Axion Cosmology')
	ax.plot(y[:,-1], rhoa,'blue',label=r'$\rho_{axion}$')
	ax.plot(y[:,-1], rhosum,'green',label=r'$\rho_{total}$')
	ax.plot(y[:,-1], rhor,'orange',label=r'$\rho_{radiation}$')
	ax.plot(y[:,-1], rhom,'red',label=r'$\rho_{matter}$')
	ax.plot(y[:,-1],rholl,c='k')
	ax.axvline(0.000333222, color='k', linestyle='--',label=r'$z_{eq}$')
	ax.legend(loc='upper right')
	plt.ylabel(r'$\rho$ (eV)$^4$', fontsize=16)
	plt.xlabel(r'$a$', fontsize=16)
	#plt.grid(True)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(10**-8,1)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	#plt.xlim(10**-9,1)
	plt.show()

def camb_output_plot(camb_param):
	lpoint=[]
	ttpoint=[]
	tt_up=[]
	tt_down=[]
	with open('cmb_data.txt', 'r') as f:
			content = f.readlines()
			for x in content:
				row = x.split()
				lpoint.append(float(row[1]))
				ttpoint.append(float(row[2]))
				tt_down.append(float(row[3]))
				tt_up.append(float(row[4]))

	lpoint = np.array(lpoint)
	ttpoint = np.array(ttpoint)
	tt_down = np.array(tt_down)
	tt_up = np.array(tt_up)
	
	camb_param = np.reshape(camb_param.T,(8,len(camb_param)))
	fig, ax = plt.subplots(2,2, figsize = (12,12))
	#ax[0,0].plot(ls,totCL[:,0], color='k')
	ax[0,0].plot(camb_param[0][2:-1],camb_param[1][2:-1], color='red')
	#ax[0,0].set_title('TT')
	ax[0,0].set_xlabel(r'${\rm Multiploe Moment} \ l$')
	ax[0,0].set_ylabel(r'$l(l+1)C_{l} \ / \ 2\pi (\mu K^2)$')
	ax[0,0].set_xscale('log')
	ax[0,0].errorbar(lpoint,ttpoint,yerr=[tt_down,tt_up], fmt='o',color='k',markersize=1, capsize=2)
	ax[1,0].plot(camb_param[0][2:-1],camb_param[2][2:-1], color='k')
	#ax[1,0].set_title(r'$EE$')
	ax[1,1].plot(camb_param[0][2:-1],camb_param[3][2:-1], color='k')
	ax[1,1].set_title(r'$TE$');
	for ax in ax.reshape(-1): ax.set_xlim([2,2500]);
	plt.show()


def cmb_plot():
	import pkg_resources
	pysm_ver = pkg_resources.get_distribution("pysm").version
	print('Currently using PySm version: %s '%pysm_ver)
	nside = 64
	sky_config = {
		'synchrotron' : models("s1", nside),
		'dust' : models("d1", nside),
		'freefree' : models("f1", nside),
		'cmb' : models("c1", nside),
		'ame' : models("a1", nside),
	}
	# initialise Sky 
	sky = pysm.Sky(sky_config)
	cmb = sky.cmb(nu = 23.)

	fig = plt.figure(figsize = (6, 4))
	hp.mollview(cmb[0], min = -30, max = 30, title = r'CMB', sub = (111))
	plt.show()
	plt.close()
	total = sky.signal()(100.)
	fig = plt.figure(figsize = (6, 4))
	hp.mollview(total[0], min = 10, max = 200, title = r'CMB + DUST', sub = (111))
	plt.show()
	plt.close()


def omega(omegar,omegam,a):
	ax = plt.subplot(1,1,1)
	ax.plot(a,omegar)
	ax.plot(a,omegam)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

def eos(z,w,a):
	ax = plt.subplot(1,1,1)
	ax.plot(a,w)
	plt.xscale('log')
	#plt.yscale('log')
	plt.show()

def phiplot(phi,phid,a):
	ax = plt.subplot(1,1,1)
	ax.plot(a,phi,c='r')
	ax.plot(a,phid,c='b')
	plt.show()
	


def massspec(mo,lma_array):
	ax =plt.subplot(1,1,1)
	n, bins, patches = plt.hist(lma_array, 200,facecolor='orange', normed=True)
	#plt.legend(loc='upper left')
	#plt.ylabel('Probability')
	#plt.xlabel(r'Log Eigenvalues')
	#plt.title(r'Mass Spectrum for Model {0}'. format(mo))
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.show()		


def fspec(mo,lfef):		
	ax = plt.subplot(1,1,1)
	n, bins, patches = plt.hist(lfef, 200,facecolor='red', normed=True,label='$f_{eff}$')
	#plt.legend(loc='upper left')
	#plt.ylabel('Probability')
	#plt.xlabel(r'Log Eigenvalues')
	#plt.title(r'$f$ Spectrum for Model {0}'. format(mo))
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.show()	
	
def mfspec(mo,lfef,lma_array):
	plt.subplot(1,1,1)
	n, bins, patches = plt.hist(lma_array, 100,facecolor='orange', normed=True,label='$Mass$')
	n, bins, patches = plt.hist(lfef, 50,facecolor='red', normed=True,label='$f_{eff}$')
	plt.legend(loc='upper left')
	plt.ylabel('Probability')
	plt.xlabel(r'Log Eigenvalues')
	plt.title(r'Mass and $f$ Spectrum for Model {0}'. format(mo))
	plt.show()	

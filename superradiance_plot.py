#!/usr/bin/python
import sys, platform, os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from matplotlib.patches import Ellipse
import camb
from camb import model, initialpower
from pysm.nominal import models
import healpy as hp
import site 
from matplotlib import rc
from scipy import arange, array, exp
import pandas as pd
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.rcParams["figure.facecolor"] = 'w'
plt.rcParams["axes.facecolor"] = 'w'
plt.rcParams["savefig.facecolor"] = 'w'

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
	def eigsorted(cov):
		vals, vecs = np.linalg.eigh(cov)
		order = vals.argsort()[::-1]
		return vals[order], vecs[:,order]

	if ax is None:
		ax = plt.gca()

	vals, vecs = eigsorted(cov)
	theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

	# Width and height are "full" widths, not radius
	width, height = 2 * nstd * np.sqrt(vals)
	ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

	ax.add_artist(ellip)
	return ellip



def conf_legend():
	conf=[99,95,68]
	concolor=['orangered','red','maroon']
	Ep_handle={}
	Ep_label={}
	for i in range(0,3):
		Ep_handle[i]=[]
		Ep_label[i]=[]
		Ep_handle[i] = [mpatches.Patch(color=concolor[i], alpha=0.6, linewidth=0)]
		Ep_label[i] = [u'${0}\%$ CL'.format(conf[i])]

	handles2=[]
	labels2=[]
	for i in range(0,3):	
		handles2.extend(Ep_handle[i])
		labels2.extend(Ep_label[i])
		
	legend22 = plt.legend(handles2,labels2,loc='center right',bbox_to_anchor = [0.9325, 0.61],
	           ncol=2,prop={'size':12},numpoints=1)
	pyplot.gca().add_artist(legend22)
	plt.legend(loc='center right',bbox_to_anchor = [0.99, 0.27])
	
	
def quantum_levels_legend(colours,l):
	p_handle={}
	p_label={}
	for i in range(0,5):
		p_handle[i]=[]
		p_label[i]=[]
		p_handle[i] = [mpatches.Patch(color=colours[i], alpha=1.0, linewidth=1.5)]
		p_label[i] = [u'$l=m={0}$'.format(l[i])]

	plt.text(13.11, 0.34, r'$\mu_{\rm ax}=10^{-11}eV$', fontsize=15,bbox={'facecolor':'white', 'alpha':1.0, 'pad':12})
	#handle, label = ax.get_legend_handles_labels()
	handles=[]
	labels=[]
	for i in range(0,5):
		handles.extend(p_handle[i])
		labels.extend(p_label[i])
		
	legend2 = plt.legend(handles,labels,loc='lower right',
	           ncol=2,prop={'size':12},numpoints=1)
	pyplot.gca().add_artist(legend2)
	
	
def regge_plane_plot(x1,y1,colours,sr_spins,sr_masses,sr_spin_up,sr_spin_low,sr_mass_up,sr_mass_low):
	fig, ax = plt.subplots(figsize=(10,6))
	for i in range(4,-1,-1):
		ax.fill_between(x1[i], y1[i], 1,facecolor=colours[i],linewidth=2.0,zorder=2)

	labels=(r'$\rm Continuum\ Fit \ Black$'
	'\n' 
	r'$\rm Hole \ Data$') 
	ax.errorbar(sr_masses, sr_spins, yerr=[sr_spin_up,sr_spin_low], xerr=[sr_mass_up,sr_mass_low], fmt='o',color='k',label=labels)
	plt.legend(loc='lower right',prop={'size':12})
	plt.xlabel(r'$\rm Black \ Hole \ Mass \ \left(\rm{M_{\rm BH}} \ / M_{\odot} \right)$', ha='center', va='center',size=20,labelpad=15)
	plt.ylabel(r'$\rm Black \ Hole \ Spin \ \left( a_{*}\right)$',size=21)	
	plt.ylim(0,1)
	plt.xlim(0,x1[4].max())
	
def regge_region_plot(fx,fy,blackholes,rt,xtem,ytem,dytem,dxtem,example_mass,example_spin,example_spin_error,example_mass_error,error_ellipse,bmhu):
	plt.plot(fx,fy,linestyle='--',color='black')
	print(xtem)
	plt.fill_between(fx, fy,1, color='cyan',alpha=0.5)
	plt.xlim(fx.min(),fx.max())	
	
	
	
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
	
	
	
	if blackholes == True:
		#for i in range(len(ytem)):
		#	plt.errorbar(xtem[i], ytem[i], yerr=dytem[i], xerr=dxtem[i], fmt='o',color='k')
		plt.errorbar(sl_masses,sl_spins,yerr=[sl_spin_low,sl_spin_up],xerr=[sl_mass_low,sl_mass_up], fmt='o',color='orangered',capsize=1, elinewidth=0.9,markeredgewidth=0.9,markersize=3.75)
		#if error_ellipse==True:
		#	for i in range (len(sl_masses)):
		#		plot_cov_ellipse([[(sl_mass_low[i])**2, 0],[0, (sl_spin_low[i])**2]],[sl_masses[i],sl_spins[i]], nstd=3, alpha=0.5, facecolor='none',zorder=1,edgecolor='black',linewidth=0.1)
		#		plot_cov_ellipse([[(sl_mass_low[i])**2, 0],[0, (sl_spin_low[i])**2]],[sl_masses[i],sl_spins[i]], nstd=2, alpha=0.5, facecolor='none',zorder=1,edgecolor='black',linewidth=0.1)
		#		plot_cov_ellipse([[(sl_mass_low[i])**2, 0],[0, (sl_spin_low[i])**2]],[sl_masses[i],sl_spins[i]], nstd=1, alpha=0.5, facecolor='none',zorder=1,edgecolor='black',linewidth=0.1)
			

	plt.xlabel(r'${\rm M_{BH}} \left( M_{\odot} \right)$', ha='center', va='center',size=20,labelpad=15)
	plt.ylabel(r'$  a_{*}$',size=21)	
	plt.ylim(0,1)
	plt.xlim(1,100)
	#plt.xscale('log')
	
	
def intersection_plot(nx,ny,indx,indx2):
	plt.plot(nx[4][indx2[3]], ny[4][indy2[3]], 'ro')
	plt.plot(nx[0][0:indx[0]],ny[0][0:indx[0]])
	plt.plot(nx[1][indx2[0]:indx[1]],ny[1][indx2[0]:indx[1]])
	plt.plot(nx[2][indx2[1]:indx[2]],ny[2][indx2[1]:indx[2]])
	plt.plot(nx[3][indx2[2]:indx[3]],ny[3][indx2[2]:indx[3]])
	plt.plot(nx[4][indx2[3]:-1],ny[4][indx2[3]:-1])	
	
	
def superradiance_rates_plot(alpha,rates):
	for i in range(0,5):
		plt.plot(alpha,rates[i]*2,linewidth=2)
	plt.yscale('log')
	plt.xlabel(r'$\mu_{\rm ax}  r_g$', size=24,labelpad=4.15)
	plt.ylabel(r'$ \log_{10}(M_{\rm BH} \ IM(\omega))$',size=21,labelpad=2)
	plt.xlim(0,2.55)
	plt.ylim(10**-16.5,10**-6.5)

		
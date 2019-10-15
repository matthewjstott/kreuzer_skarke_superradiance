#!/usr/bin/python

import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from matplotlib import rc
from matplotlib.ticker import LogFormatterExponent
import scipy as sp
import decay_constraints_functions_master as functions


rc('text', usetex=True)


range_p=50

x={}
lista=[]
for i in range(0,range_p):
	
	# Test test_exclusion (Matt's expression), against LMCX1_exclusion (my expression)
	x[i] = np.load('results_data/LMCX-1_exclusion{0}.npy'.format(i))
	#print(len(x[i]),i)

for i in range(0,range_p):
	lista.extend(x[i])

f_ax = np.logspace(-13.0,-20.0,range_p) #inverse!
m_ax = np.logspace(-11,-14,range_p)


sigma = 1.25



lista = gaussian_filter(lista, sigma)
	


lista = np.reshape(lista,(len(f_ax),len(m_ax)))

X,Y = np.meshgrid(m_ax,f_ax)
Z = lista



c1 = plt.contour(X,Y,Z,[0.95],colors='deepskyblue',label = r'$3\sigma$')
c2 = plt.contour(X,Y,Z,[0.90],colors='darkblue',label = r'$2\sigma$')
c3 = plt.contour(X,Y,Z,[0.68],colors='black',label = r'$1\sigma$')

h1,_ = c1.legend_elements()
h2,_ = c2.legend_elements()
h3,_ = c3.legend_elements()
plt.legend([h1[0], h2[0],h3[0]], [r'$3\sigma$', r'$2\sigma$',r'$1\sigma$'],frameon=True,edgecolor='k',prop={'size':12.5},title='${\\rm Exclusion}$',ncol=1)


plt.plot(X,Y,'ko',markersize=0.2,alpha=0.7)


plt.contourf(X,Y,Z,[0.95,1],colors='deepskyblue',alpha=0.75)
plt.contourf(X,Y,Z,[0.90,0.95],colors='darkblue',alpha=0.75)
plt.contourf(X,Y,Z,[0.68,0.9],colors='black',alpha=0.75)

plt.text(1.7*10**-14, 1.*10**-13.50, r'${\rm LMC\ X-1}$', fontsize=29,rotation=90,color='black')


plt.ylabel(r'$1/f_a\ [GeV]$',size=18)
plt.xlabel(r'$\mu_a\ [eV]$',size=18)

plt.xscale('log')
plt.xlim(10**-14,10**-9)
plt.ylim(10**-20,10**-12.0)
plt.yscale('log')
plt.savefig('LMCX-1_Me_NBose.png',bbox_inches='tight')


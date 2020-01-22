#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np 
from scipy import interpolate
from scipy import ndimage	
plt.rcParams["figure.facecolor"] = 'w'
plt.rcParams["axes.facecolor"] = 'w'
plt.rcParams["savefig.facecolor"] = 'w'
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as mpatches
from matplotlib import pyplot
mass=[]
prob= []
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

		
		
range_length = 50		
ma_max = -10.5	
ma_min = -13.6
#ax = plt.subplot(1,1,1)
mass = np.logspace(ma_max,ma_min,range_length)
prob= np.load('LMCX-1_exclusion.npy')


	
yreduced = np.array(prob) - 0.95

x_sm = np.array(mass)
y_sm = np.array(prob)

# resample to lots more points - needed for the smoothed curves
x_smooth = np.linspace(x_sm.min(), x_sm.max(), 2000)

sigma = 2.0
x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)

#print(mass[34])
#print(mass[83])

#axins = zoomed_inset_axes(ax, 2.5,bbox_to_anchor=(10**2.56, 10**2.484))  # zoom = 6

#x1, x2, y1, y2 =  10**-11., 10**-10.7,-0.01, 0.1,
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)

#axins.tick_params(
#	axis='y',          # changes apply to the x-axis
#	which='both',      # both major and minor ticks are affected
#	bottom='off',      # ticks along the bottom edge are off
#	top='off',         # ticks along the top edge are off
#	labelbottom='off')


idx = np.argwhere(np.diff(np.sign(y_g1d - 0.95)) != 0).reshape(-1) + 0
idx2 = np.argwhere(np.diff(np.sign(y_g1d - 0.68)) != 0).reshape(-1) + 0
print(idx)
print(idx2)

#axins.vlines(6.9179102616724927e-11, 0, y_g1d[idx[0]], colors='r', linestyles=':', label='')
#axins.vlines(mass[idx[1]], 0, y_g1d[idx[1]], colors='r', linestyles=':', label='')
#axins.hlines(10**-11., 10**-10.5, 0, colors='k', linestyles='-', label='')


plt.vlines(2.0333498194716876e-11, 0, 0.95, colors='r', linestyles='-', label='',alpha=0.5)
plt.vlines(5.92870010002381e-14, 0, 0.95, colors='r', linestyles='-', label='',alpha=0.5)

plt.vlines(2.591494837695301e-11, 0, 0.68, colors='orange', linestyles='-', label='',alpha=0.5)
plt.vlines(4.8241087041653735e-14, 0, 0.68, colors='orange', linestyles='-', label='',alpha=0.5)

plt.axhspan(0.95, 1.01, alpha=0.2, color='skyblue')
plt.axhspan(0.68, 0.95, alpha=0.2, color='dodgerblue')

plt.plot(x_g1d,y_g1d, 'black', linewidth=1)
plt.xscale('log')
plt.ylim(0,1.01)

#plt.axhline(0.95,linestyle='--',color='red',alpha=0.5)
plt.ylabel(r'${\mathbb{P}_{\rm ex}}(\mu_{\rm ax})$',size=17)
plt.xlabel(r'$\mu_{\rm ax}$',size=17)
#plt.yscale('log')		

aa=['2','1']
cols=['skyblue','dodgerblue']
p_handle={}
p_label={}
for i in range(0,2):
	p_handle[i]=[]
	p_label[i]=[]
	p_handle[i] = [mpatches.Patch(color=cols[i], alpha=0.2, linewidth=1.5)]
	p_label[i] = [u'{0}  \sigma'.format(aa[i])]

#plt.text(13.11, 0.34, r'$\mu_{\rm ax}=10^{-11}eV$', fontsize=15,bbox={'facecolor':'white', 'alpha':1.0, 'pad':12})
#handle, label = ax.get_legend_handles_labels()
handles=[]
labels=[]
for i in range(0,2):
	handles.extend(p_handle[i])
	labels.extend(p_label[i])
	
legend2 = plt.legend(handles,labels,bbox_to_anchor=(0.6,0.15),
           ncol=2,prop={'size':12},numpoints=1,edgecolor='k')
pyplot.gca().add_artist(legend2)

plt.xlim(10**-14,10**-10)


plt.show()		
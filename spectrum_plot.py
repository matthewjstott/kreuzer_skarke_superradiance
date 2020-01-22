#suimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
plt.rcParams["figure.facecolor"] = 'w'
plt.rcParams["axes.facecolor"] = 'w'
plt.rcParams["savefig.facecolor"] = 'w'

def marcenkopasturpdf(x, c):
	ub = (1 + np.sqrt(c))**2
	lb = (1 - np.sqrt(c))**2 
	mp = np.zeros(len(x))
	lbidx = np.where(x > lb)
	ubidx = np.where(x < ub)  
	a = lbidx[0][0]
	b = ubidx[-1][-1]
	xh = x[a:b+1]
	mp[a:b+1] = np.sqrt((xh - lb)*(ub - xh))/(2*np.pi*c*xh)              
	return (lb, ub, mp)

def ran11pdf(x):
	xp = 1+0+2*np.sqrt(0)
	xm = 1+0-2*np.sqrt(0)
	dis = 1./(2.*np.pi*x*0.0000001)*np.sqrt((x - xm)*(xp-x))
	return(dis)
			
def ran22pdf(x):
	dis = (1./2.*np.pi)*np.sqrt(x*(4.-x))
	return(dis)	
	
def ran31pdf(x):
	a = (2.0**1/3.0*np.sqrt(3.0))/48.0*np.pi
	b = 2.**1/3.*(27.+3.*np.sqrt(81.-12.*x)**2./3.-6.*x**1/3.)
	c = x**(2./3.)*(27.+3.*np.sqrt(81.-12.*x))**1/3.
	dis = a*(b/c)
	return(dis)	

def ran32pdf(x):
	dis1 = 9.*(1.+np.sqrt(1.-4.*x/27.))**4./3. - (4.*x)**2./3.
	dis2 = 3.**1.5*2*np.pi*(4.*x)**1./3.*(1+np.sqrt(1.-4.*x/27.))**2./3.
	dis = dis1/dis2
	return(dis)
	
def ran33pdf(x):
	dis = x*ran31(x)
	return(dis)	

def mp_pdf_plot():
	c = np.linspace(0.0,1,250)	
	x = np.linspace(0,4,1000)	
	colors = plt.cm.afmhot(np.linspace(0,1.0,250))
	sm = plt.cm.ScalarMappable(cmap='afmhot', norm=plt.Normalize(vmin=0, vmax=1.0))
	sm._A = []
	lb={}
	ub={}
	mp={}
	for ii in range (0,250):
		lb[ii], ub[ii], mp[ii] = marcenkopasturpdf(x, c[ii])
	
	for ii in range(0,250):
		plt.plot(x, mp[ii], linewidth = 1,color=colors[ii],linestyle='--',alpha=0.25)
	plt.plot(x, mp[50], linewidth = 2,linestyle='-',color=colors[50],label='$1\sigma$ lower limit',alpha=1)
	plt.plot(x, mp[100], linewidth = 2,linestyle='-',color=colors[100],label='$1\sigma$ upper limit',alpha=1)
	plt.plot(x, mp[150], linewidth = 2,linestyle='-',color=colors[150],label='$2\sigma$ lower limit',alpha=1)
	plt.plot(x, mp[200], linewidth = 2,linestyle='-',color=colors[200],label='$2\sigma$ upper limit',alpha=1)
	plt.plot(x, mp[249], linewidth = 2,linestyle='-',color=colors[249],label='$3\sigma$ lower limit',alpha=1)
	clb = plt.colorbar(sm)
	ttl= clb.ax.set_title('$\\beta_{\mathcal{M}}$',fontsize=15)
	ttl.set_position([.5, 1.02])
	clb.ax.invert_yaxis()
	plt.xlabel('$\\left ( \\frac{m_a^2}{M_H^2} \\right  )$',fontsize=23)
	plt.ylabel(r"$\mathbb{P}_{(1)}(m_{a}^2)$",fontsize=20)
	plt.ylim(0,2)
	plt.xlim(0,4)
	plt.show()
	

def MPSpectra(n,fav,b0,samp,c):
	msamples=np.array([])
	fsamples=np.array([])
	for i in range (0,len(c)):
		L=int(n/c[i]) 
		for j in range(0,samp):
			kk = np.empty((n,n))
			kk.fill(1.) 
			kk2=np.diag(kk[:,0]) 
			k2 = np.dot(kk2,kk2.T)  
			ev,p = np.linalg.eig(k2) 
			fef = np.sqrt(ev)*fav 
			fmat = np.zeros((n,n))
			np.fill_diagonal(fmat,fef)
			kDr = np.zeros((n, n))
			np.fill_diagonal(kDr, (1./(fef))) 
			X = b0*(np.random.randn(n, L)) 
			M = np.dot(X,(X.T))/L 
			mn = reduce(np.dot, [kDr,p,M,p.T,kDr.T]) 
			ma_array,mv = np.linalg.eig(mn) 
			#ma_array = np.sqrt(ma_array)
			msamples=np.append(msamples,ma_array)
	msamples = np.reshape(msamples,(len(c),samp*n))	
	return(msamples)

def WWSpectra(n,a0,b0,samp,c):
	msamples=np.array([])
	fsamples=np.array([])
	for i in range (0,len(c)):
		L=int(n/c[i]) 
		for j in range(0,samp):
			k  = a0*(np.random.randn(n, L))
			k2 = np.dot(k,(k.T))/L
			ev,p = np.linalg.eig(k2) 
			fef = np.sqrt(np.abs(2*ev))
			fsamples=np.append(fsamples,fef)	 
			kD = reduce(np.dot, [p.T, k2, p]) 
			kD[kD < 1*10**-13] = 0 
			kDr = np.zeros((n, n)) 
			np.fill_diagonal(kDr, 1/(fef))
			m = b0*(np.random.randn(n, L)) 
			M = np.dot(m,(m.T))/L
			mn = 2.*reduce(np.dot, [kDr,p,M,p.T,kDr.T]) 
			ma_array,mv = np.linalg.eig(mn) 
			ma_array = np.log10(np.sqrt(ma_array))
			msamples=np.append(msamples,ma_array)
	fsamples = np.reshape(fsamples,(len(c),samp*n))
	msamples = np.reshape(msamples,(len(c),samp*n))	
	return(fsamples,msamples)

def LFSpectra(n,kmin,kmax,mmin,mmax,samp,c):
	msamples=np.array([])
	fsamples=np.array([])
	for i in range (0,len(c)):
		L=int(n/c[i]) 
		for j in range(0,samp):
			k = (np.random.uniform(kmin,kmax,(n,L))) 
			k = 10.**k
			k2 = np.dot(k,k.T)/L 
			ev,p = np.linalg.eig(k2) 
			fef = np.sqrt(2.*ev)
			fmat = np.zeros((n,n))
			np.fill_diagonal(fmat,fef)
			kDr = np.zeros((n, n))
			np.fill_diagonal(kDr, (1./(fef)))
			m = (np.random.uniform(mmin,mmax,(n,L))) 
			m = 10.**m
			m2 = np.dot(m,m.T)/L 
			mn = 2.*reduce(np.dot, [kDr,p,m2,p.T,kDr.T]) 
			ma_array,mv = np.linalg.eig(mn) 
			ma_array = np.log10(np.sqrt(ma_array))
			msamples=np.append(msamples,ma_array)
	fsamples = np.reshape(fsamples,(len(c),samp*n))
	msamples = np.reshape(msamples,(len(c),samp*n))	
	return(fsamples,msamples)	


def MPplot(n,fav,b0,samp,c):
	msamples = MPSpectra(n,fav,b0,samp,c)
	for i in range(len(c)-1,-1,-1):
		bin = int(round(c[i]*50))
		plt.hist(msamples[i],bin,normed=True,color=np.random.rand(3,))
	plt.show()
	
def WWplot(n,a0,b0,samp,c):
	fsamples,msamples = WWSpectra(n,a0,b0,samp,c)
	for i in range(len(c)-1,-1,-1):
		bin = int(round(c[i]*50))
		plt.hist(msamples[i],bin,normed=True,color=np.random.rand(3,))
	plt.show()
	
def LFplot(n,kmin,kmax,mmin,mmax,samp,c):
	fsamples,msamples = LFSpectra(n,kmin,kmax,mmin,mmax,samp,c)
	for i in range(len(c)-1,-1,-1):
		bin = int(round(c[i]*50))
		plt.hist(msamples[i],bin,normed=True,color=np.random.rand(3,))
	plt.show()







def spectrum_out(ma_array,fef,fefvisual,mavisual):
	
	if fefvisual == True:
		fig, ax = plt.subplots()
		ax.hist(fef,100,normed=True,edgecolor = 'black',alpha=0.5,label='$\\beta_{\mathcal{K},\mathcal{M}} = 0.05$')
		ax.set_xlabel('$log_{10}\\left( \\frac{m_a^2}{M_{H}^2} \\right)$',fontsize=23)
		ax.set_ylabel('PDF',fontsize=23)
		ax.legend(loc='upper right',prop={'size':18})
			
	if mavisual == True:
		fig, ax = plt.subplots()
		ax.hist(ma_array,100,normed=True,edgecolor = 'black',alpha=0.5,label='$\\beta_{\mathcal{K},\mathcal{M}} = 0.05$')
		ax.set_xlabel('$log_{10}\\left( \\frac{m_a^2}{M_{H}^2} \\right)$',fontsize=23)
		ax.set_ylabel('PDF',fontsize=23)
		ax.legend(loc='upper right',prop={'size':18})

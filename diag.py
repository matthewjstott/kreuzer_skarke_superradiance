import scipy.integrate as integrate
import numpy as np
import numpy.random as rd
from fractions import *
import scipy as sp
import matplotlib.pyplot as plt
from functools import reduce

def poscheck(ev):

	if any(x <= 0 for x in ev):
		raise Exception('You have negative eigenvalues')
	else:
		return 0

def checkvolume(n,smin,smax,Imax,Nvolmax,Idist):

	s = np.random.uniform(smin,smax,n)
	##### 1 = uniform distribution of bij from [0,Imax - 1]
	##### 2 = Poisson distribution of bij with lambda = Imax
	##### 3 = 'diagonal' uniform distribution of bij from [0,Imax - 1]
	if Idist == 1:
		Nvol = 2*np.pi*np.random.poisson(Nvolmax,size=(n,n))
	elif Idist == 2:
		Nvol = 2*np.pi*np.random.poisson(Nvolmax,size=(n,n))
	else:
		Nvol = np.zeros((n, n))
		#np.fill_diagonal(b, 2*np.pi*np.random.randint(Imax,size=n))
		np.fill_diagonal(Nvol, 2*np.pi*np.random.poisson(Nvolmax,size=n))
	return np.dot(s,Nvol)



def diag(phi_range,phidotin,basis,mo,n,betaK,betaM,a0,b0,fav,kmin,kmax,mmin,mmax,FL3,sbar,svar,Nbar,Nvar):
	
	
	n = 1000
	
	if mo == 1:
	
		kk = np.empty((n,n))
		kk.fill(1.) 
		kk2=np.diag(kk[:,0])
		kkT = kk2.transpose() 
		k2 = np.dot(kk2,kkT)  
		ev,p = np.linalg.eig(k2) 
		fef = np.sqrt(ev)*fav 
		fmat = np.zeros((n,n))
		np.fill_diagonal(fmat,fef)
		kDr = np.zeros((n, n))
		np.fill_diagonal(kDr, (1./(fef))) 
		
		L=int(n/betaM)
		X = b0*(np.random.randn(n, L)) 
		M = np.dot(X,(X.T))/L 
		mn = reduce(np.dot, [kDr,p,M,p.T,kDr.T]) 
		ma_array,mv = np.linalg.eig(mn) 
		ma_array = np.sqrt(ma_array)
		
	####################################################################
	####################################################################

	if mo == 2:

		LK=int(n/betaK)
		LM=int(n/betaM)
		k = (np.random.uniform(kmin,kmax,(n,LK))) 
		k = 10.**k
		k2 = np.dot(k,k.T)/LK 
		ev,p = np.linalg.eig(k2) 
		fef = np.sqrt(2.*ev)
		fmat = np.zeros((n,n))
		np.fill_diagonal(fmat,fef)
		kDr = np.zeros((n, n))
		np.fill_diagonal(kDr, (1./(fef)))


		m = (np.random.uniform(mmin,mmax,(n,LM))) 
		m = 10.**m
		m2 = np.dot(m,m.T) /LM 
		mn = 2.*reduce(np.dot, [kDr,p,m2,p.T,kDr.T]) 
		ma_array,mv = np.linalg.eig(mn) 
		ma_array = np.sqrt(ma_array) 

	####################################################################
	####################################################################

	if mo == 3:
			
		LK=int(n/betaK)
		LM=int(n/betaM)
		k  = a0*(np.random.randn(n, LK))
		k2 = np.dot(k,(k.T))/LK # Factor of L
		ev,p = np.linalg.eig(k2) 
		fef = np.sqrt(np.abs(2.*ev))
		fmat = np.zeros((n,n))
		np.fill_diagonal(fmat,fef)	 
		kD = reduce(np.dot, [pT, k2, p]) 
		kD[kD < 1*10**-13] = 0 
		kDr = np.zeros((n, n)) 
		np.fill_diagonal(kDr, 1./(fef))
	
		m = b0*(np.random.randn(n, LM)) 
		M = np.dot(m,(m.T))/LM # Factor of L
		mn = 2.*reduce(np.dot, [kDr,p,M,p.T,kDr.T]) 
		eigs,mv = np.linalg.eig(mn) 
		ma_array=np.sqrt(eigs)
	

	####################################################################
	####################################################################

	if mo == 4:
	
		a0=1.
		
		L = int(n/betaM)
		s = np.abs(np.random.normal(sbar,svar,n))
		Ntilde = np.abs(np.random.normal(Nbar,Nvar,size=(n,L)))	
			
		remove_tachyons=True
		
		k = np.zeros((n,n))
		np.fill_diagonal(k,a0*a0/s/s)
		ev,p = np.linalg.eig(k) 
		fef = np.sqrt(np.abs(2.*ev))
		fmat = np.zeros((n,n))
		np.fill_diagonal(fmat,fef)
		kDr = np.zeros((n, n))
		np.fill_diagonal(kDr, (1./(fef)))
		
		Sint = np.dot(s,Ntilde)
		Esint = np.exp(-Sint/2.)
		Idar = n*[1.]
		Cb = np.sqrt(np.dot(Idar,Ntilde))
				
		A = 2.*np.sqrt(FL3)*reduce(np.multiply,[Cb,Esint,Ntilde]) 

		m = np.dot(A,A.T)/L 
		mn = 2.*reduce(np.dot, [kDr,p,m,p.T,kDr.T]) 
		ma_array,mv = np.linalg.eigh(mn) 
		
		if remove_tachyons:
			tachyons=ma_array[ma_array<0]
			ma_array[ma_array<0]=0.
		
		ma_array = np.sqrt(np.abs(ma_array))

	####################################################################
	####################################################################


	if mo == 5:
		
		k = (np.random.uniform(kmin,kmax,(n))) 
		k = 10.**k 
		fef=np.sqrt(2.*k)
		fmat = np.zeros((n,n))
		np.fill_diagonal(fmat,fef)
		p=1.
		
		m = (np.random.uniform(mmin,mmax,(n))) 
		m = 10.**m
		ma_array=np.sqrt(2.*m)
		mv=1.


	####################################################################
	####################################################################


	phiin_array = rd.uniform(0.,phi_range,n)
	
	if basis ==1:
		phiin_array=reduce(np.dot,[mv,fmat,phiin_array])
	else:
		phiin_array=reduce(np.dot,[mv,fmat,p,phiin_array])
	
	phidotin_array = [phidotin]*n


	
	return ma_array,fef,phiin_array,phidotin_array
	


	
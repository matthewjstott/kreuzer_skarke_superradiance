import scipy.integrate as integrate
import numpy as np
import numpy.random as rd
from fractions import *
import scipy as sp
import sys

def axionphi(y,N):
	phi = np.sum(y[:,0:-1:3][:],axis=-1)
	phid = np.sum(y[:,1::3][:],axis=1)
	return phi,phid 

def dense(rhol,rho_m0,rho_r0,rho_b0,N,y,n,t,ma_array):
	rhom=[]
	rhor=[]
	rhob=[]
	rhoa = np.sum(y[:,2::3][:],axis=-1)
	for i in range(N):
		rhom.append(rho_m0/y[:,-1][i]**3.)
		rhor.append(rho_r0/y[:,-1][i]**4.)
		rhob.append(rho_b0/y[:,-1][i]**4.)
	rholl = [rhol]*N	
	rhom = np.array(rhom)
	rhol = np.array(rhol)
	rhor = np.array(rhor)
	rhoa = np.array(rhoa)
	rhob = np.array(rhob)
	rhosum = rhom+rhol+rhor+rhoa+rhob	
	omegar = rhor/rhosum
	omegam = rhom/rhosum
	omegaa = rhoa/rhosum
	omegal = rhol/rhosum
	omegab = rhob/rhosum	
	H = (1.0/np.sqrt(3.0))*np.sqrt(rhosum[0:len(t)])
	
	rhoo=[]
	rhon=[]
	for ii in range (N):
		rhoa_de = 0
		rhoa_dm = 0
		rhoa_de2 = 0
		rhoa_dm2 = 0
		for j in range (n):
			if 3/np.sqrt(3)*np.sqrt(rhom[ii]+rhor[ii]+rhol+rhoa[ii])>ma_array[j]:
				rhoa_de = rhoa_de + y[ii,2::3][j]
			else:
				rhoa_dm = rhoa_dm + y[ii,2::3][j]
		rhoo.append(rhoa_dm)
		rhon.append(rhoa_de)
	for j in range (n):
			if 3/np.sqrt(3)*np.sqrt(rhom[ii]+rhor[ii]+rhol+rhoa[ii])>ma_array[j]:
				rhoa_de2 = rhoa_de + y[-1,2::3][j]
			else:
				rhoa_dm2 = rhoa_dm + y[-1,2::3][j]				
	ODE = (rhoa_de2/rhosum[-1])
	ODM = (rhoa_dm2/rhosum[-1])			
							
	return rhoa,rhom,rhor,rholl,rhob,rhosum,omegar,omegam,omegaa,omegal,omegab,H,rhoo,rhon,ODE,ODM,

def pressure(y,ma_array,N,n,rhom,rhol,rhor,rhoa,rhosum):
	Parray = np.zeros((N,n))	
	for i in range(n):
		field=y[:,3*i]
		zero_crossings = np.where(np.diff(np.sign(field)))[0]
		if np.size(zero_crossings)==0:
			last_zero=N
		else:
			last_zero=zero_crossings[-1]
		for j in range(last_zero):
			Parray[j,i]=0.5*y[j,3*i+1]**2.-0.5*ma_array[i]**2.*field[j]**2.
	P=np.sum(Parray,axis=1) 
	phom = np.array(rhom)*0.
	phor = np.array(rhor)*1./3.
	phol = np.array(rhol)*-1.
	Psum = phol+phor+phom+P
	w = P/rhoa
	a=y[:,-1][0:N]
	add=[]
	for ii in range(N):
		add.append(-a[ii]/3*(rhosum[ii]+3*Psum[ii]))
	z = 1.0/y[:,-1][0:N] - 1
	if z[-1]<0:
		zind = (next(idx for idx, value in enumerate(z) if value < 0.0))	
	else:
		zind =-1	
	return P,Psum,w,a,add,z,zind


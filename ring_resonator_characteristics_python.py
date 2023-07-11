import numpy as np
import matplotlib.pyplot as plt
import imp
import pandas as pd
from scipy.signal import find_peaks



r=100e-6
L=2*np.pi*r             #in m

lam=np.linspace(1.45e-6, 1.625e-6, 1000001)
lam0=1.55e-6

#------------------------------Waveguide and directional coupler model------------------

#Waveguide Effective index
n0=1.45
n1=-1.65e5
n2=2.8e11
neff=n0 + n1*(lam-lam0) + n2*(lam-lam0)**2


#Waveguide Loss
l0=5.6e2     #in dB/m
l1=4.5e9     
l2=-2e16 
alpha=l0 + l1*(lam-lam0) + l2*(lam-lam0)**2

#Self Coupling power(t)
x=(lam-lam0)*8.71e7
coupling_power=0.77/(1+0.382*np.exp(-x)) +0.08

#----------------------------Ring resonator model------------------------------

a=np.exp(-alpha*L/2)           # a=1 for ideal cavites with 0 attenuation
theta=2*np.pi*neff*L/lam       #Phase difference occured after total round trip

t1=np.sqrt(coupling_power)     #self coupling coefficient (amplitude) of directional coupler 1
t2=np.sqrt(coupling_power)     #For critical coupling of a lossy media, t1=t2*a

#Single bus resonator
bus=1
if (bus==1):
    T_thr=(a*a-2*t1*a*np.cos(theta)+t1*t1)/(1-2*t1*a*np.cos(theta)+(t1*a)**2)
    
    plt.plot(lam*1e9, 10*np.log10(T_thr), linewidth=2, color='darkblue', label='Through port')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Characteristics(dB)')
    plt.legend(frameon=False)
    #plt.xlim([1540, 1560])
    #plt.ylim([-45, 0])

#Double bus resonator
if (bus==2):
    T_thr=(t2*t2*a*a-2*t1*t2*a*np.cos(theta)+t1*t1)/(1-2*t1*t2*a*np.cos(theta)+(t1*t2*a)**2)
    T_d=(1-t1*t1)*(1-t2*t2)*a/(1-2*t1*t2*a*np.cos(theta)+(t1*t2*a)**2)
    
    plt.plot(lam*1e9, 10*np.log10(T_thr), linewidth=2, color='darkblue', label='Through port')
    plt.plot(lam*1e9, 10*np.log10(T_d), linewidth=2, color='darkorange', label='Drop port')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Characteristics(dB)')
    plt.legend(frameon=False)
    #plt.xlim([1540, 1560])
    #plt.ylim([-15, 0])
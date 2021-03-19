import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate



#-------input parameters--------
j=1j
fe= 0.6793      # Area fraction of Al2O3

fi=1-fe
dfe = pd.read_table('Al2O3_pvlighthouse.txt', skiprows=0) #skipping first 1 row
wle=dfe['wl']
dfi = pd.read_table('Al_pvlighthouse.txt', skiprows=0) #skipping first 1 row

def wavelength_fitting(x,y, xnew):
    # construct a polynomial
    z = np.polyfit(x, y, 5) # 3 is an order
    f = np.poly1d(z)        #function
    ynew = f(xnew)    
    return ynew

def wavelength_interpolation(x,y, xnew): 
    f = interpolate.splrep(x, y, s=0)     #s=0 -> No smoothing/regression required
    ynew = interpolate.splev(xnew, f, der=0)
    return ynew

def MG_approx(dfe, dfi, fi):
    wle=dfe['wl']
    ne=dfe['n']
    ke=dfe['k']
    
    wli=dfi['wl']
    ni=dfi['n']
    ki=dfi['k']
    ni_new=wavelength_interpolation(wli,ni, wle)
    ki_new=wavelength_interpolation(wli,ki, wle)
    
    
    ee=(ne+j*ke)**2
    ei=(ni_new+j*ki_new)**2
    
    #A=3*ee/(ei+2*ee)  #spherical inclusion
    #A=ee/ei           #nanodisk inclusion
    A=2*ee/(ei+ee)   #Nanowire inclusion
    
    num=fi*ei*A+ee*(1-fi)
    den=fi*A+(1-fi)
    
    e_eff=num/den
    return e_eff

eps_eff=MG_approx(dfe,dfi,fi)

n_eff=np.real(np.sqrt(eps_eff))
k_eff=np.imag(np.sqrt(eps_eff))
N_eff=n_eff+j*k_eff

plt.plot(wle, n_eff, label='n')
plt.plot(wle, k_eff, label='k')
plt.legend()
plt.xlabel( 'Wavelength (nm)', fontsize=12)
plt.ylabel( 'Index', fontsize=12)
plt.title( r'$Al_2O_3$ area fraction = ' + str(fe*100) + r'%')

#-------Writing----data in a file
c = [wle, n_eff, k_eff]
with open(r'effective_refractive_index_' + str(fe*100) + r'%_Al2O3.txt', "w") as file:
    for x in zip(*c):
        file.write("{0}\t{1}\t{2}\n".format(*x))
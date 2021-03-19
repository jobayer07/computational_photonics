import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
j=1j


lamda=1000e-9;                      #Wavelength (m)
n0=3.57075                          #Index of the first layer
n1=1.602786+j*0.016101              #Index of the second layer
n2=1.35+j*9.58                      #Index of the third layer
thickness=np.arange(1, 100, 1)*1e-9 #Thickness of the intermediate layer (m)

def Fresnel_reflection_calculation(angle_in_degree):
    theta=angle_in_degree/180*np.pi
    theta1=np.arcsin(n0/n1*np.sin(theta)) 
    theta2=np.arcsin(n0/n2*np.sin(theta))
    r01m=(n1*np.cos(theta)-n0*np.cos(theta1))/(n1*np.cos(theta)+n0*np.cos(theta1))
    r12m=(n2*np.cos(theta1)-n1*np.cos(theta2))/(n2*np.cos(theta1)+n1*np.cos(theta2))
    r01e=(n0*np.cos(theta)-n1*np.cos(theta1))/(n0*np.cos(theta)+n1*np.cos(theta1))
    r12e=(n1*np.cos(theta1)-n2*np.cos(theta2))/(n1*np.cos(theta1)+n2*np.cos(theta2))
    k0=2*np.pi/lamda
    kz1=np.sqrt(n1*n1*k0*k0-n0*n0*k0*k0*np.sin(theta)**2)
    
    Re=abs((r01e+r12e*np.exp(2j*kz1*thickness))/(1+r01e*r12e*np.exp(2j*kz1*thickness)))**2      #Reflectance of TE light
    Rm=abs((r01m+r12m*np.exp(2j*kz1*thickness))/(1+r01m*r12m*np.exp(2j*kz1*thickness)))**2      #Reflectance of TM light
    #Te=1-Re                   #Transmittance of TE light
    #Tm=1-Rm                   #Transmittance of TM light
    return Re, Rm

RTE0, RTM0=Fresnel_reflection_calculation(0)
RTE41, RTM41=Fresnel_reflection_calculation(41.4)

fig, ax=plt.subplots(1, figsize=(8, 6))
plt.plot(thickness*1e9,RTM41*100,'blue', label='TM, oblique incidence')
plt.plot(thickness*1e9,RTE41*100,'red', label='TE, oblique incidence')
plt.plot(thickness*1e9,RTM0*100,'k', label='TE or TM, normal incidence')
plt.xlabel('Thickness (nm)')
plt.ylabel('Reflectance (%)')
plt.legend(fontsize=12).get_frame().set_linewidth(0.0)

from matplotlib.ticker import AutoMinorLocator
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(bottom=True, top=True, left=True, right=True, which='major', direction='in', length=8, color='k')
plt.tick_params(bottom=True, top=True, left=True, right=True, which='minor', direction='in', length=4, color='k')

plt.savefig('D:/STUDY_CREOL/Research with Davis/Publications/Nanostructure Paper/Figures/Fresnel_reflection.jpg', dpi=200)
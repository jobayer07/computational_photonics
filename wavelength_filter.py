"""
@author: Mohammad Jobayer Hossain
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath

# Physical Parameters
#w0=129e-6            #Beam waist
w0=5e-6            #Beam waist
lam=535e-9          #Wavelength641

# Numerical Parameters
dx=0.05e-6             #Step size in X axis
dz=1e-5             #Step size in Z axis
L=0.24e-3            #Smaller window gives me repeated oscillation

dist=33e-6
Z=165e-3             #distance to travel in Z a7is

#dist=30e-6
#Z=103e-3             #distance to travel in Z a7is

j=complex(0, 1)     #complex number j

# Variables
Nz=int(Z/dz)        #Number of points in Z axis
k0=2*cmath.pi/lam
Nx=int(L/dx)
x_points = np.linspace(-100e-6, 100e-6, Nx)
z_points = np.linspace(0, Nz*dz, Nz)
V= k0*0.75*1e-3*np.exp(-((x_points-0)/(3e-6))**6)+k0*0.75*1e-3*np.exp(-((x_points+dist)/(3e-6))**6)

#Functions
def gaussian_function(x,w):  #this function is for initial field
    mu=0
    gaus=np.exp(-((x-mu)*(x-mu))/(2*w*w))
    #Normalization step
    P=gaus**2               #instantaneous power
    Pt=np.sum(P)*dx         #total power of the beam
    Norm=np.sqrt(Pt)        #Normalization factor. Taken in square root, for field
    y=gaus/Norm
    return y


#Main Code
kx_points=2*np.pi/L*np.linspace(-Nx/2, Nx/2, Nx)
phase=np.exp(j*kx_points*kx_points*dz/(2*k0))
phase_term=np.fft.fftshift(phase)

init_field=gaussian_function(x_points, w0)  #initial field

final_mat=np.zeros([Nx,Nz], dtype=complex)   
final_mat[:, 0]=init_field

n0=1.33
W=0.08*np.max(x_points)
V_loss=(abs(x_points/W))**25
Absorber=1#np.exp(-V_loss)

for i in range (1,Nz):                    #Nz=1000
    d1=np.fft.ifft(phase_term*(np.fft.fft(final_mat[:, i-1])))
    v1=Absorber*np.exp(-j*V*dz)*d1
    final=np.fft.ifft(phase_term*(np.fft.fft(v1)))
    final_mat[:, i]=final
    
#-------------------------------------------- plotting starts --------------------------------------------
x_min=0
x_max=149
y_min=-60
y_max=60
plt.figure(1)
plt.imshow(abs(final_mat[:, :]*final_mat[:, :]), cmap = 'inferno', extent = [x_min , x_max, y_min , y_max])
ax=plt.gca()
plt.xlabel('z axis (mm)')
plt.ylabel('x axis (x2 um)')

plt.figure(2)
ax=plt.plot(x_points*1e6, V*10, 'k', label='Potential')
plt.plot(x_points*1e6, abs(final*final), 'g', label='Final beam')
plt.xlabel('x axis (um)')
plt.ylabel('a.u.')
plt.legend()

#
plt.figure(3)
I1=abs(np.log(final_mat[int(Nx/2)+1, :])**2)          ##Intensity at the center of the first waveguide
plt.plot(z_points*1e3, I1, label='Waveguide1')
I2=abs(np.log(final_mat[int(Nx/2-dist/dx), :])**2)  #Intensity at the center of the second waveguide
plt.plot(z_points*1e3, I2, label='Waveguide2')
plt.legend()
ax=plt.gca()
ax.set_xlim([0, 162])
ax.set_ylim([0, 45])
plt.xlabel('z axis (mm)')
plt.ylabel('Intensity (a.u)')

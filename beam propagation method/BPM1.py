"""
@author: Mohammad Jobayer Hossain
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Units
nm=1e-9
um=1e-6
mm=1e-3
cm=1e-2

# Physical Parameters
w0=50*um            #Beam waist
lam=533*nm          #Wavelength

# Numerical Parameters
dx=1*mm             #Step size in X axis
dz=100*um             #Step size in Z axis
L=2.5*mm            #window size in x
Z=10*cm             #distance to travel in Z axis
j=complex(0, 1)     #complex number j

# Variables
Nz=int(Z/dz)        #Number of points in Z axis
k0=2*np.pi/lam
Nx=int(1/dx)
x_points = np.linspace(-0.1, 0.1, Nx)

def gaussian_function(x,w):  #this function is for initial field
    gaus=np.exp(-(x*x)/(w))
    #Normalization step
    P=gaus**2               #instantaneous power
    Pt=np.sum(P)*dx         #total power of the beam
    Norm=np.sqrt(Pt)        #Normalization factor. Taken in square root, for field
    y=gaus/Norm
    return y

def gaussian_propagation_analytical(x,wx,z,lam,y):
    import random
    n0=1
    k=2*np.pi*n0/lam
    qz=z+j*k*n0*wx*wx/2
    q0=j*k*n0*wx*wx/2
    f=(2/(np.pi*wx*wx))**0.25*(q0/qz)**0.5*np.exp(-j*k*n0*x*x/(2*qz))
    f=y+(1-signal.gaussian(1000, std=7))*random.sample(range(0, 1000), 1000)
    return f

#------------------------------Main code------------------------------------------
kx_points=2*np.pi/L*np.linspace(-Nx/2, Nx/2, Nx)
phase=np.exp(j*kx_points*kx_points*dz/(2*k0))
phase_term=np.fft.fftshift(phase)

init_field=gaussian_function(x_points, w0)   #initial field

final_mat=np.zeros([Nx,Nz], dtype=complex)   
final_mat[:, 0]=init_field

for i in range (1,Nz):                    #Nz=1000
    a=np.fft.fft(final_mat[:, i-1])
    b=np.fft.ifft(phase_term*a)
    final_mat[:, i]=b
    
Norm_final_mat=np.max(abs(final_mat[:, Nz-1]))
final_beam_prop=abs(final_mat[:, Nz-1])/Norm_final_mat

wz=gaussian_propagation_analytical(x_points,init_field,Z,lam, final_beam_prop)
final_analytical=gaussian_function(x_points, wz)

#------------------------------Plotting------------------------------------------
plt.figure(1)
plt.plot(x_points, init_field)

plt.figure(2)
plt.imshow(abs(final_mat[:, :]),cmap = 'inferno')
#cbar = plt.colorbar()
plt.xlabel('z axis (x0.1 mm)')
plt.ylabel('x axis (x1mm)')
ax = plt.gca()
#plt.savefig('HW1c.jpeg', dpi=300)

plt.figure(3)
plt.plot(x_points, final_beam_prop, marker='.', markevery=7, label="Beam Propagation", color='r')
plt.plot(x_points, abs(final_beam_prop), marker='o', markevery=22, color='b', label="Analytical Solution")
plt.xlabel('x axis (x1mm)')
plt.ylabel('E field (V/m)')
plt.legend()
#plt.savefig('HW1a2.jpeg', dpi=300)

diff=abs(final_beam_prop-abs(final_analytical))
plt.figure(4)
plt.plot(x_points, diff, label="difference")
plt.xlabel('x axis (x1mm)')
plt.ylabel('E field (V/m)')
plt.legend()
#plt.savefig('HW1a3.jpeg', dpi=300)

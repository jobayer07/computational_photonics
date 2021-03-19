import numpy as np
import matplotlib.pyplot as plt
import cmath
import random

# Physical Parameters
w0=10e-6            #Beam waist
lam=533e-9          #Wavelength

# Numerical Parameters
dx=1e-4             #Step size in X axis
dz=1e-6             #Step size in Z axis
L=0.001             #window size in x
Z=1e-2             #distance to travel in Z axis

# Variables
Nz=int(Z/dz)        #Number of points in Z axis
k0=2*cmath.pi/lam

n0=1.33

Nx=int(1/dx)

def gaussian_function(x,w):
    gaus=np.exp(-(x*x)/(w))
    P=gaus**2               #instantaneous power
    Pt=np.sum(P)*dx         #total power of the beam
    Norm=np.sqrt(Pt)        #Normalization factor. Taken in square root, for field 
    y=gaus/Norm
    return y
#------------------------------Main code---------------
j=complex(0, 1)

x_points = np.linspace(-0.1, 0.1, Nx)

kx_points=2*cmath.pi/L*np.linspace(-Nx/2, Nx/2, Nx)
phase=np.exp(j*kx_points*kx_points*dz/(2*k0))
phase_term=np.fft.fftshift(phase)

init_field=gaussian_function(x_points, w0)   #initial field

def beam_prop(phase_term, init_field, n):
    final_mat=np.zeros([Nx,Nz], dtype=complex)   #Nx=200, Nz=1000
    final_mat[:, 0]=init_field
    for i in range (1,Nz):                    #Nz=1000
        if n==0:   #random_media
            V=1e-3*random.random()/n0
        elif n==1: #Media with refractive index=1
            V=((n*n)-(n0*n0))/(2*n0*n0)
        d1=np.fft.ifft(phase_term*(np.fft.fft(final_mat[:, i-1])))
        v1=np.exp(-j*V*dz)*d1
        d2=np.fft.ifft(phase_term*(np.fft.fft(v1)))
        final_mat[:, i]=d2
    final=final_mat[:, i]
    return final
    
final_field_air=beam_prop(phase_term, init_field, 1)

beam_rand_media=np.zeros([Nx,1001], dtype=complex)   #Nx=200, Nz=1000
for m in range (1, 1000):
    beam_rand_media[:, m]=beam_prop(phase_term, init_field, 0)
    prop_rand_media=beam_rand_media[:, m]+beam_rand_media[:, m-1]
avg_prop_rand_media=prop_rand_media/1000

#All the Plotting
plt.figure(1)
plt.plot(x_points, init_field)

plt.figure(2)
plt.plot(x_points, abs(final_field_air), label="prop_in_air")
plt.plot(x_points, abs(avg_prop_rand_media)/0.22, linestyle='--', color='k', label="avg_rand_media")
plt.xlabel('x axis (x0.1mm)')
plt.ylabel('E field (V/m)')
plt.legend()
plt.savefig('HW1b.jpeg', dpi=100)
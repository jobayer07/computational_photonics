"""
@author: Mohammad Jobayer Hossain
"""
import numpy as np
import matplotlib.pyplot as plt
import cmath

# Physical Parameters
#w0=129e-6            #Beam waist
w0=5e-6            #Beam waist
lam=633e-9          #Wavelength

# Numerical Parameters
dx=0.1e-6             #Step size in X axis
dz=1e-5             #Step size in Z axis
L=0.25e-3            #Smaller window gives me repeated oscillation
Z=5e-2             #distance to travel in Z axis
j=complex(0, 1)     #complex number j

# Variables
Nz=int(Z/dz)        #Number of points in Z axis
k0=2*cmath.pi/lam
Nx=int(L/dx)
x_points = np.linspace(-100e-6, 100e-6, Nx)
z_points = np.linspace(0, Nz*dz, Nz)
V= k0*0.75*1e-3*np.exp(-((x_points-0)/(5e-6))**8)+k0*0.75*1e-3*np.exp(-((x_points-15e-6)/(5e-6))**8)

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

plt.figure(1)
plt.plot(x_points*1e6, V/V[1225])

#plt.figure(2)
plt.plot(x_points*1e6, init_field/init_field[1225])

plt.figure(3)
plt.imshow(abs(final_mat[:, :]*final_mat[:, :]))
#vmin=31.595,  vmax=31.69, 
#, cmap = 'inferno'
#cbar = plt.colorbar()
plt.xlabel('z axis (x10 um)')
plt.ylabel('x axis (x10 um)')

#---------------------------------------More Analysis------------------------------
#Intensity of the oscillation
I=abs(final_mat[1250, :]*final_mat[1250, :])  #Intensity at the center of the first waveguide
I_norm=I/np.max(I)
plt.figure(4)
plt.plot(z_points*1e6, I_norm)
plt.xlabel('z axis (x10 um)')
plt.ylabel('Normalized intensity (a.u.)')

#Frequency of the oscillation
I_modified=I_norm-np.mean(I_norm)
I_fouri=np.fft.fft(I_modified)
plt.figure(5)
plt.plot(np.log(I_fouri[1:4998]))
plt.xlabel('data points')
plt.ylabel('log(frequency)')


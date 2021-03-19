import numpy as np
import matplotlib.pyplot as plt
import cmath
import math

dz=300e-6             #Step size in Z axis

# Physical Parameters
w0=50e-6            #Beam waist
lam=533e-9          #Wavelength

# Numerical Parameters
dx=1e-4             #Step size in X axis

L=0.005
#L=0.005             #window size in x
#Z=6e-2             #distance to travel in Z axis
Z=6e-2

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


def Hermite(x, n):
    Har_mat=np.zeros(Nx)
    a0=np.ones([Nx, n+1])
    for k in range (n+1):
        if k==0:
            a0[:, k]=np.ones(Nx)
        if k==1:
            a0[:, k]=2*x
        elif k>1:
            a0[:, k]=2*x*a0[:, k-1] - 2*(k-1)*a0[:, k-2]
        Har_mat=a0[:, k]
    return Har_mat


def harmite_gaussian_analytical(k0,w0,a,n,z,x):
    r=1 #refractive index of the medium
    zr=1/2*k0*(w0**2)*r
    wz=w0*np.sqrt(1+(z/zr)**2)
    if z!=0:
        Rz=z*(1+(zr/z)**2)
        phas=k0*z-(1+n)*np.arctan(z/zr)+k0*x*x/(2*Rz)
        HG=(w0/wz)*Hermite(x*np.sqrt(2)/wz, n)*np.exp(-x*x/wz*wz)*np.exp(-j*phas)
    return HG
#------------------------------Main code---------------
j=complex(0, 1)
a=np.pi/(1.8*1e-2)

x_points = np.linspace(-5, 5, Nx)
init_field=gaussian_function(x_points, w0)   #initial field
kx_points=2*cmath.pi/L*np.linspace(-Nx/2, Nx/2, Nx)

final_mat=np.zeros([Nx,Nz], dtype=complex)   #Nx=200, Nz=1000
PT=np.zeros(Nz, dtype=complex)  #array of total power
H=np.zeros(Nz, dtype=complex)   #Hamiltonian
final_mat[:, 0]=init_field

'''
#---------3 step BPM--------------------
phase=np.exp(j*kx_points*kx_points*dz/(4*k0))
phase_term=np.fft.fftshift(phase)
V=-(a**2)*(x_points**2)/2   #potential for GRIN medium

for i in range (1,Nz):                    #Nz=1000
    d1=np.fft.ifft(phase_term*(np.fft.fft(final_mat[:, i-1])))
    v1=np.exp(-j*V*dz)*d1
    d2=np.fft.ifft(phase_term*(np.fft.fft(v1)))
    final_mat[:, i]=d2
    psi_sq=final_mat[:, i]**2               #instantaneous power
    PT[i]=np.sum(abs(psi_sq))*dx    #total power of the beam
    va=np.gradient(np.gradient(final_mat[:, i]))
    h=(va/(2*k0)-k0*V*psi_sq)
    H[i]=np.sum(abs(h))*dx          #Hamiltonian of the beam
'''    

#---------2 step BPM----------------------------------
phase=np.exp(j*kx_points*kx_points*dz/(2*k0))
phase_term=np.fft.fftshift(phase)
V=-(a**2)*(x_points**2)/2   #potential for GRIN medium

for i in range (1,Nz):                    #Nz=1000
    v1=np.exp(-j*V*dz)*final_mat[:, i-1]
    d2=np.fft.ifft(phase_term*(np.fft.fft(v1)))
    final_mat[:, i]=d2
    psi_sq=final_mat[:, i]**2               #instantaneous power
    PT[i]=np.sum(abs(psi_sq))*dx    #total power of the beam
    va=np.gradient(np.gradient(final_mat[:, i]))
    h=(va/(2*k0)-k0*V*psi_sq)
    H[i]=np.sum(abs(h))*dx          #Hamiltonian of the beam
 

#All the Plotting
plt.figure(1)
plt.plot(x_points, init_field)

plt.figure(2)
plt.imshow(abs(final_mat), cmap = 'inferno')
cbar = plt.colorbar()
plt.axis('off')

#---------------------------------------------------

'''
n=4
x=x_points
f=np.zeros([Nx, Nz], dtype=complex)
for p in range(1,Nz):
    f[:, p]=harmite_gaussian_analytical(k0,w0,a,n,dz*p,x)
    psi_sq=f[:, p]**2               #instantaneous power
    PT[p]=np.sum(abs(psi_sq))*dx    #total power of the beam
    va=np.gradient(np.gradient(f[:, p]))
    h=(va/(2*k0)-k0*V*psi_sq)
    H[p]=np.sum(abs(h))*dx          #Hamiltonian of the beam

plt.figure(1)
plt.imshow(abs(f), cmap = 'inferno')
plt.axis('off')
'''
plt.figure(3)
plt.plot(1-PT/PT[1])
plt.title('Relative error in Power')
plt.xlabel('Z axis (iterations)')
plt.ylabel('Pr')
plt.legend()

plt.figure(4)
plt.plot(1-H/H[1])
plt.title('Relative error in Hamiltonian')
plt.xlabel('Z axis (iterations)')
plt.ylabel('Hr')
plt.legend()

Pr300=np.sum(abs(1-PT/PT[1]))*dz


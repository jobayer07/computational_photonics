"""
@author: Mohammad Jobayer Hossain
"""
import numpy as np
import matplotlib.pyplot as plt
import cmath


# Physical Parameters
#w0=129e-6            #Beam waist
w0=12e-6            #Beam waist
lam=633e-9          #Wavelength

# Numerical Parameters
dx=1e-6             #Step size in X axis
dz=1e-5             #Step size in Z axis
L=3e-3            #window size in x
Z=5e-2             #distance to travel in Z axis
j=complex(0, 1)     #complex number j

# Variables
Nz=int(Z/dz)        #Number of points in Z axis
k0=2*cmath.pi/lam
Nx=int(L/dx)
x_points = np.linspace(-10e-5, 10e-5, Nx)

def gaussian_function(x,w):  #this function is for initial field
    #d=10e-6
    mu=0#np.sqrt(d)
    gaus=np.exp(-0.5*((x-mu)*(x-mu))/(w*w))
    #Normalization step
    P=gaus**2               #instantaneous power
    Pt=np.sum(P)*dx         #total power of the beam
    Norm=np.sqrt(Pt)        #Normalization factor. Taken in square root, for field
    y=gaus/Norm
    return y

init_field=gaussian_function(x_points, w0)


def gaussian_propagation_analytical(x,w0,z,lam):
    n=1     # in air
    k=2*cmath.pi*n/lam
    zr=cmath.pi*w0*w0*n/lam
    wz=w0*np.sqrt(1+(z/zr)**2)
    R=z*(1+(zr/z)**2)
    psi=np.arctan(z/zr)                 #Gouy phase
    f=np.exp(-x*x/wz)*np.exp(-j*(k*z+k*(x*x/(2*R)))-psi)
    #Normalization step
    Norm=np.max(abs(f))
    f=f/Norm
    return f

#------------------------------Main code BPM------------------------------------------
kx_points=2*cmath.pi/L*np.linspace(-Nx/2, Nx/2, Nx)
phase=np.exp(j*kx_points*kx_points*dz/(2*k0))
phase_term=np.fft.fftshift(phase)

init_field=gaussian_function(x_points, w0)   #initial field
plt.figure(1)
plt.plot(x_points, init_field)



final_mat=np.zeros([Nx,Nz], dtype=complex)   
final_mat[:, 0]=init_field

n0=1.33
a=np.pi/(1.8*1e-2)
W=0.08*np.max(x_points)
V= 0.75*1e-3*np.exp(-x_points/(5e-6)) #-a*a*x_points*x_points
V_loss=(abs(x_points/W))**25
Absorber=1#np.exp(-V_loss)

for i in range (1,Nz):                    #Nz=1000
    d1=np.fft.ifft(phase_term*(np.fft.fft(final_mat[:, i-1])))
    v1=Absorber*np.exp(-j*V*dz)*d1
    final=np.fft.ifft(phase_term*(np.fft.fft(v1)))
    final_mat[:, i]=final

#BPM second time
final_mat2=np.zeros([Nx,Nz], dtype=complex)   
final_mat2[:, 0]=final
for i in range (1,Nz):                    #Nz=1000
    d1=np.fft.ifft(phase_term*(np.fft.fft(final_mat2[:, i-1])))
    v1=Absorber*np.exp(-j*V*dz)*d1
    final2=np.fft.ifft(phase_term*(np.fft.fft(v1)))
    final_mat2[:, i]=final2    

#------------------------------Error Calculation --------------------------------------
#Power
psi_sq=abs(final_mat2*final_mat2)
z_points=np.linspace(0, Z, Nz)
Power=np.zeros(Nz, dtype=complex)
Perr=np.zeros(Nz, dtype=float)
for i in range(Nz):
    Power[i]=np.sum(final_mat2[:, i])*dx
    Perr[i]=1-abs(Power[i]/Power[0])
    
#Hamiltonian
#d_sq_psi=np.zeros(Nz, dtype=float)
H=np.zeros(Nz, dtype=complex)
H_err=np.zeros(Nz, dtype=float)
for i in range(Nz):
    d_sq_psi=abs(np.gradient(np.gradient(final_mat2[:, i])))
    first_term=d_sq_psi/(2*k0)
    second_term=k0*V*final_mat2[:, i]*final_mat2[:, i]
    H[i]=np.sum(first_term-second_term)*dx
    H_err[i]=1-abs(H[i]/H[0])
#------------------------------Figure Plotting------------------------------------------
plt.figure(1)
plt.plot(x_points, init_field)
plt.plot(x_points, final2)
plt.xlabel('x axis (x 10 um)')
#plt.plot(x_points-10e-6, init_field)

plt.figure(3)
plt.imshow(abs(final_mat2[:, :]),cmap = 'inferno')
#cbar = plt.colorbar()
plt.xlabel('z axis (x10 um)')
plt.ylabel('x axis (um)')
ax = plt.gca()

plt.figure(4)
plt.plot(z_points*1e6, Perr)
plt.xlabel('z axis (um)')
plt.ylabel('1-P[z]/P[0]')

plt.figure(5)
plt.plot(z_points*1e6, H_err)
plt.xlabel('z axis (um)')
plt.ylabel('1-H[z]/H[0]')

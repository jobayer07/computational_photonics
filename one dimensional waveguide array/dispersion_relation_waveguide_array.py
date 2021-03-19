import numpy as np
import matplotlib.pyplot as plt
import cmath


# Physical Parameters
#w0=129e-6            #Beam waist
w0=12e-6            #Beam waist
lam=633e-9          #Wavelength

# Numerical Parameters
dx=1e-6             #Step size in X axis
dz=0.1582e-3             #Step size in Z axis
L=18e-3            #window size in x
Z=100e-2             #distance to travel in Z axis
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

def matrix_rotation(image_in, theta):
    #clockwise rotation: theta=-1,  anti-clockwise rotation: theta=+1
    n,m=image_in.shape  #row, column
    for i in range(m):
        b=image_in[:,i]
        b = b[::theta]
        image_in[:,i]=b
    image_out=image_in.transpose()
    return image_out

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


#------------------------------Figure Plotting------------------------------------------
plt.figure(1)
plt.plot(x_points, init_field)
plt.plot(x_points, final)
plt.xlabel('x axis (x 10 um)')
#plt.plot(x_points-10e-6, init_field)

plt.figure(3)
plt.imshow(abs(final_mat[:, :]),cmap = 'inferno')
#cbar = plt.colorbar()
plt.xlabel('z axis (x10 um)')
plt.ylabel('x axis (um)')
ax = plt.gca()
'''
plt.figure(4)
plt.plot(z_points*1e6, Perr)
plt.xlabel('z axis (um)')
plt.ylabel('1-P[z]/P[0]')

plt.figure(5)
plt.plot(z_points*1e6, H_err)
plt.xlabel('z axis (um)')
plt.ylabel('1-H[z]/H[0]')
'''
#m=np.fft.fftshift(final_mat[:, :], axes=(0,))
#m=np.fft.fftshift(m, axes=(1,))
fft_prop=np.fft.fft2(final_mat[:, :],)
x_sft=np.roll(fft_prop, -9000, axis=0)
z_sft=np.roll(x_sft, -3160, axis=1)
Kz_Kx_imag=matrix_rotation(abs(z_sft), -1)

kz_points=2*np.pi/Z*np.linspace(-Nz/2, Nz/2, Nz)
plt.figure(4)
im=np.log(Kz_Kx_imag)
a=plt.imshow(im,cmap = 'inferno', extent = [1/kx_points[0]*1e6 , -1/kx_points[0]*1e6, 1/kz_points[0]*1e4 , -1/kz_points[0]*1e4])    #extent = [x_min , x_max, y_min , y_max]
plt.xlabel('Kx (1/um)')
plt.ylabel('Kz (1/cm)')

V=k0*0.75*1e-3*np.exp(-((x_points-0)/(5e-6))**8)
d=15e-6
for i in range (1, 10):
    V=V+k0*0.75*1e-3*np.exp(-((x_points-i*d)/(5e-6))**8)+k0*0.75*1e-3*np.exp(-((x_points+i*d)/(5e-6))**8)
    temp=V
    
plt.figure(5)
plt.plot(x_points*1e6, V)
plt.xlabel('x (um)')
plt.ylabel('Potential')
    
#Periodic lattice





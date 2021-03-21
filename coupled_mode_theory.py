"""
@author: Mohammad Jobayer Hossain
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
import scipy.linalg


# Physical Parameters
#w0=129e-6            #Beam waist
w0=12e-6            #Beam waist
lam=633e-9          #Wavelength

# Numerical Parameters
dx=0.5e-5             #Step size in X axis
dz=0.5e-5             #Step size in Z axis
L=0.7e-3            #window size in x
Z=5e-2             #distance to travel in Z axis
j=complex(0, 1)     #complex number j

# Variables
Nz=int(Z/dz)        #Number of points in Z axis
k0=2*cmath.pi/lam
n0=1.33
Nx=int(L/dx)

x_points = np.linspace(-0.5e-3, 0.5e-3, Nx)
#x_points = np.linspace(-0.5e-3, 0.5e-3, Nx)
def gaussian_function(x,w):
 gaus=np.exp(-(x*x)/(w))
 y=gaus
 return y

def solve_eigenmodes(V):
    #Preparation
    h=2*k0*n0*dx*dx
    q=-k0*n0*V
    I=np.zeros([Nx, Nx], dtype=complex)
    np.fill_diagonal(I, 1)

    W=np.zeros([Nx, Nx], dtype=complex)
    np.fill_diagonal(W, 2)
    np.fill_diagonal(W[1::], -1)    #filling down_diagonal with -1
    np.fill_diagonal(W[:,1:], -1)   #filling up_diagonal with -1
    
    q=-k0*n0*V
    Q=np.zeros([Nx, Nx], dtype=complex)
    np.fill_diagonal(Q, q)
    
    A=W/(h*h)+Q-np.matmul(W,Q)/12
    
    #Eigen solution Now
    [D, Modes]=np.linalg.eig(A)
    temp=D
    ind=np.argsort(np.real(temp))
    beta=temp[ind]
    Modes=Modes[:, ind]
    #Normalization
    Norm=np.sum(Modes[0, :]*Modes[0, :])*dx
    Modes=Modes/Norm
    return Modes, beta

def calculate_C_pq(Ep, Eq):
    C_pq=np.sum(np.conj(E1)*E2)*dx
    return C_pq


def calculate_X(Ep, n_p):
    const=k0*(n*n-n_p*n_p)/(2*n0*n0)
    Ep_conj=np.conj(Ep)
    X=np.sum(const*Ep_conj*Ep)*dx
    return X
'''
def calculate_K_pq(Ep, Eq, n_p):
    const=k0*(n*n-n_p*n_p)/(2*n0*n0)
    Ep_conj=np.conj(Ep)
    K_pq=np.sum(const*Ep_conj*Eq)*dx
    return K_pq
'''
'''
def calculate_X(Ep):
    Ep_conj=np.conj(Ep)
    lap_Ep=np.gradient(np.gradient(Ep))
    num=np.sum(Ep_conj*lap_Ep)*dx
    den=np.sum(Ep*Ep)*dx
    X=num/den
    return X
'''
def calculate_K_pq(Ep, Eq):
    Ep_conj=np.conj(Ep)
    lap_Eq=np.gradient(np.gradient(Eq))
    num=np.sum(Ep_conj*lap_Eq)*dx
    den=np.sum(Ep*Ep)*dx
    K_pq=num/den
    return K_pq

def calculate_H(C12, C21, X1, X2, K12, K21, d):
    H=np.zeros([2, 2], dtype=complex)
    H[0,0]=(K21*C21-X1)/(1-C21*C12)
    H[0,1]=(C12*X2-K12)/(1-C12*C21)*np.exp(j*d*dz)
    H[1,0]=(X1*C21-K21)/(1-C12*C21)*np.exp(-j*d*dz)
    H[1,1]=(K12*C21-X2)/(1-C12*C21)
    return H

#------------------------------Main code---------------
j=complex(0, 1)
a=cmath.pi/(1.8*1e-2)
x_points = np.linspace(-0.001, 0.001, Nx)
kx=2*n0*cmath.pi/L*np.linspace(-Nx/2, Nx/2, Nx)
init_field=gaussian_function(x_points, w0) #initial field
final_mat=np.zeros([Nx,Nz], dtype=complex) #Nx=200, Nz=1000
final_mat[:, 0]=init_field

V1=0.75*1e-3*np.exp(-((x_points)/(3e-6))**6)
V2=51/80*np.exp(-((x_points-(10e-6))/(3.5e-6))**6)

Modes1, beta1=solve_eigenmodes(V1)
Modes2, beta2=solve_eigenmodes(V2)

#--------------------------------------------Part C---------------------
n1=n0*np.sqrt(1+2*V1)
n2=n0*np.sqrt(1+2*V2)
n=n1+n2-n0

#Finding Coefficients
#delta
delta=abs(beta1[0]-beta2[0])

#C_pq's
E1=Modes1[0,:]
E2=Modes2[0,:]
C12=calculate_C_pq(E1, E2)
C21=calculate_C_pq(E2, E1)

#X's

X1=calculate_X(E1, n1)
X2=calculate_X(E2, n2)
'''
#k's
K12=calculate_K_pq(E1, E2, n1)
K21=calculate_K_pq(E2, E1, n2)
'''
#X's
'''
X1=calculate_X(E1)
X2=calculate_X(E2)
'''
#k's
K12=calculate_K_pq(E1, E2)
K21=calculate_K_pq(E2, E1)

#Wave Propagation Now
#H=np.zeros([2, 2], dtype=complex)

H=calculate_H(C12, C21, X1, X2, K12, K21, delta)

ex=scipy.linalg.expm(j*H*dz)

#Wave Propagation now
#Nz=500
A=np.zeros(Nz, dtype=complex)
B=np.zeros(Nz, dtype=complex)
A[0]=1
B[0]=0
AB=np.zeros([2, Nz], dtype=complex)
AB[0,0]=A[0]
AB[1,0]=B[0]
for i in range(1, Nz):
    AB[:, i]=np.matmul(ex, AB[:, i-1])
    A[i]=AB[0, i]
    B[i]=AB[1, i]
    


#------------------------------Part D now---------------------------------
#n=n[70]
V=(n*n-n0*n0)/(2*n0*n0)

#------------------------------Main code BPM----------
kx_points=2*np.pi/L*np.linspace(-Nx/2, Nx/2, Nx)
phase=np.exp(j*kx_points*kx_points*dz/(2*k0))
phase_term=np.fft.fftshift(phase)

'''
plt.figure(1)
plt.plot(x_points*1e6, init_field)
plt.plot(x_points*1e6, init_field2)
'''
Nz=500

final_mat=np.zeros([Nx,Nz], dtype=complex)   
final_mat[:, 0]=E1

for i in range (1,Nz):                    #Nz=1000
    d1=np.fft.ifft(phase_term*(np.fft.fft(final_mat[:, i-1])))
    v1=np.exp(-j*V*dz)*d1
    final=np.fft.ifft(phase_term*(np.fft.fft(v1)))
    final_mat[:, i]=final




#----------------------------Plotting--------------------------------


#plt.figure(1)
#plt.imshow(abs(AB), cmap = 'inferno')
plt.figure(1)
plt.plot(abs(A*A)+abs(B*B))

plt.figure(2)
intensity = abs(A*A)

fig, ax1 = plt.subplots()
ax1.plot(abs(A*A), color='tab:red')
ax1.set_xlabel('Z space (x50 um)')
ax1.set_ylabel('A_power', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.get_yaxis().set_ticks([])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(abs(B*B), color='tab:blue')
ax2.set_ylabel('B_power', color='tab:blue')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.get_yaxis().set_ticks([])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.figure(1)
plt.plot(x_points*1e6, abs(final_mat[:, 0]))
plt.plot(x_points*1e6, abs(final))
plt.xlabel('x axis (x 10 um)')
#plt.plot(x_points-10e-6, init_field)

plt.figure(3)
plt.imshow(abs(final_mat[:, :]*final_mat[:, :]),cmap = 'inferno')

plt.figure(4)
plt.plot(V)
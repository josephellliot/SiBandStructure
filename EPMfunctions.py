### imports, definitions and functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim

#Constants and form factors:
#Standard for Si, taken from lit

ryd_to_eV = 13.6057 #factor to convert units
form_factors_Si = {3.0: -0.0224*ryd_to_eV, 8.0: 0.085*ryd_to_eV, 11.0: 0.056*ryd_to_eV, 12.0: -0.012*ryd_to_eV}

hbar = const.hbar
m_e = const.m_e
e = const.e
#lattice constant in angstroms
a= 5.43e-10
#k values corresponding to high symmetry points on the Si lattice
Gpoint = np.array([0,0,0])
Xpoint = np.array([0,0.5,0.5])
Lpoint = np.array([1/2,1/2,1/2])
Wpoint = np.array([1/4,3/4,1/2])
Upoint = np.array([1/4,5/8,5/8])
Kpoint = np.array([3/8,3/4,3/8])

#example high symmetry path. Any path constructed from the above variables can be used
sym_path = (Gpoint,Xpoint,Wpoint,Kpoint,Gpoint,Lpoint,Upoint,Wpoint,Lpoint,Kpoint) 
#generation of reciprocal lattice vectors
def RLVgen(RealSpaceVector):                               #input: normalised real space lattice vector [[xyz],[xyz],[xyz]]
    b1 = 2*np.pi*np.cross(RealSpaceVector[1],RealSpaceVector[2])/np.dot(RealSpaceVector[0],np.cross(RealSpaceVector[1],RealSpaceVector[2]))
    b2 = 2*np.pi*np.cross(RealSpaceVector[2],RealSpaceVector[0])/np.dot(RealSpaceVector[0],np.cross(RealSpaceVector[1],RealSpaceVector[2]))
    b3 = 2*np.pi*np.cross(RealSpaceVector[0],RealSpaceVector[1])/np.dot(RealSpaceVector[0],np.cross(RealSpaceVector[1],RealSpaceVector[2]))
    RLV = np.array([b1,b2,b3])
    return RLV  

def Vg(LatVec,form_factor_array): #This is the potential term in the matrix of Eq.n in docs
#The potential term depends on the RLVs, as this is what characterises the crystal
#Depends on a structure factor and a form factor.
#Structure factor depends on the direction of the RLVs, the form factor depends on their magnitude.
    StructureFactor = np.cos(np.pi/4*(LatVec[0]+LatVec[1]+LatVec[2]))
    phase = (np.pi/2) * (LatVec[0] + LatVec[1] + LatVec[2])
    S = 1.0 + np.exp(-1j * phase)
    S = np.real(S)
#4 Possibilities for the form factor, this pics the correct one from the possibilities
    ModSquared = LatVec[0]**2+LatVec[1]**2+LatVec[2]**2
    if ModSquared in form_factor_array:    
        V = form_factor_array[ModSquared]
    else:
        V = 0.0
    return StructureFactor * V

def generate_g_indices(Nmax = 15, allowed_shells = (0,3,8,11,12)): #These are the indices in the Hamiltonian which tell us which structure factor to use
    G_list = []
    for h in range(-Nmax, Nmax+1):
        for k in range(-Nmax, Nmax+1):
            for l in range(-Nmax, Nmax+1):
             
                if ((h % 2 == k % 2 == l % 2)) and ((h**2+k**2+l**2) in allowed_shells):  
                    G_list.append((h, k, l))
    return G_list

def kinetic_energy(k,g,basis): #calculation of the kinetic energy terms which appear on the diagonal of the Hamiltonian
    nx, ny, nz = g
    gvec = nx*basis[0]+ny*basis[1]+nz*basis[2]
    kgsum = k+gvec   
    #print('gvec:', gvec)
    return ((hbar**2)/(2*m_e))*np.dot(kgsum,kgsum)/e    #should return energy in eV
def build_k_path(sym_points, recip_basis, points_per_segment=200, return_labels=True): #construction of the k space path between the specified high sym points
    k_points = []               #the hamiltonian will be evaluated at every k point, k dist is only used for plotting as we plot the cummulative k distance on the x axis
    k_dist = [0.0]
    tick_pos = [0.0]            #it is also useful to get the high symmetry points in the k points array so that they can be easily plotted
    tick_labels = ["$\Gamma$"] 

    
    name_map = {
        tuple([0,0,0]): "$\Gamma$",
        tuple([0,0.5,0.5]): "X",
        tuple([0.5,0.5,0.5]): "L",
        tuple([0.25,0.75,0.5]): "W",
        tuple([0.25,0.625,0.625]): "U",
        tuple([0.375,0.75,0.375]): "K"
    }

    start_frac = sym_points[0]
    start_phys = start_frac @ recip_basis
    k_points.append(start_phys)

    for i in range(len(sym_points) - 1):
        start_frac = sym_points[i]
        end_frac = sym_points[i+1]

        start_phys = start_frac @ recip_basis
        end_phys = end_frac @ recip_basis

        segment = [start_phys + t*(end_phys - start_phys)
                   for t in np.linspace(0, 1, points_per_segment, endpoint=False)]
        k_points.extend(segment[1:])  

        for j in range(1, len(segment)):
            dk = np.linalg.norm(segment[j] - segment[j-1])
            k_dist.append(k_dist[-1] + dk)

        # add symmetry tick after each segment
        tick_pos.append(k_dist[-1])
        tick_labels.append(name_map.get(tuple(np.round(end_frac,3)), f"P{i+1}"))
        #print(tick_labels)
    return (np.array(k_points), np.array(k_dist),
            np.array(tick_pos), tick_labels) if return_labels else (np.array(k_points), np.array(k_dist))

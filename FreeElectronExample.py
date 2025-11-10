#FREE ELECTRON CODE BEGINS
a_vecs = np.array([[a,0,0],[0,a,0],[0,0,a]]) #basis vectors for free electron
vector = RLVgen(a_vecs) 

indices = generate_g_indices()

k_points, k_dist, ticks, labels = build_k_path(sym_path, vector, points_per_segment=200)
hamiltonian_FE = [] #empty hamiltonian to be populated with KE terms
for k in k_points: #loop through the k path we have constructed
    KE_terms_kth = np.array([kinetic_energy(k,g,vector) for g in indices])
    KE_matrix = np.diag(KE_terms_kth) #calculating the kinetic energy for every point on the k path
    
    #Potential terms in the matrix are all zero for the free electron case
    
    N = len(indices)
    V_matrix = 0
    hamiltonian_FE.append(KE_matrix)
hamiltonian = np.array(hamiltonian_FE)

H_energies_FE = np.array([np.linalg.eigvalsh(p) for p in hamiltonian]) #getting the eigenvalues
#plot the band structyre
plt.figure(figsize=(12,8))
for n in range(min(15, H_energies_FE.shape[1])):  # first 10 bands
    plt.plot(k_dist*1e-10, H_energies_FE[:,n], color='black') #units in angstroms

plt.ylabel(r'$E$ (eV)', fontsize = 30)
plt.xlabel(r'$k$ $\left(\mathrm{\AA}^{-1}\right)$', fontsize = 30)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.text(0,-7, 'Free electron', fontsize = 20)
#FREE ELECTRON CODE ENDS

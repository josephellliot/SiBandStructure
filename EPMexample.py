#EPM CODE BEGINS
#the EPM method is similar to the free electron method, but the off diagonal terms are also populated by potential terms given by the form factors.
k_points_Si, k_dist_Si, ticks_Si, labels_Si = build_k_path(sym_path, vector, points_per_segment=200)
EPM_energies = []
for k in k_points: #loop through the k path we have constructed
    KE_terms_kth = np.array([kinetic_energy(k,g,vector) for g in indices])
    KE_matrix_Si = np.diag(KE_terms_kth) #calculating the kinetic energy for every point on the k path
    N = len(indices)
    V_matrix_Si = np.zeros((N,N)) #initialise an empty matrix of correct dimensions which we will populate with potential terms
    for i, gi in enumerate(indices):
        for j, gj in enumerate(indices): #loop through both dimensions of the matrix
            gdoubleprime = (gi[0]-gj[0], gi[1]-gj[1], gi[2]-gj[2]) #calculate the vector g'', which tells us which form factor to use for this term in the matrix
        #print(gdoubleprime)
            V_matrix_Si[i, j] = Vg(gdoubleprime, form_factors_Si)


    #sum the kinetic matrix and potential matrix to get the hamiltonian

    H_Si = KE_matrix_Si+V_matrix_Si
    # print(H_Si.shape, 'H_Si shape')
    #compute eigenvalues
    H_eigs_Si = np.array([np.linalg.eigvalsh(H_Si)])
    EPM_energies.append(H_eigs_Si)
    

EPM_energies = np.array(EPM_energies)
print("EPM_energies shape:", EPM_energies.shape)
EPM_energies = np.array(EPM_energies).squeeze()
print(EPM_energies.shape)


plt.figure(figsize = (16,12))
for n in range(min(15, EPM_energies.shape[1])):  # first 10 bands
    #print('Energy spectrum at n', EPM_energies[:,n])
    plt.plot(k_dist_Si*1e-10, EPM_energies[:,n], color='black') #units in angstroms
for n in range(0,len(ticks_Si)):
    plt.axvline(ticks_Si[n]*1e-10, linestyle = '--', alpha = 0.5, color = 'red')

    plt.text(ticks_Si[n]*1e-10, 44,labels_Si[n], fontsize = 25 )
plt.ylabel(r'$E$ (eV)', fontsize = 30)
plt.xlabel(r'$k$ $\left(\mathrm{\AA}^{-1}\right)$', fontsize = 30)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.text(0,-7, 'EPM method', fontsize = 20)
#EPM CODE ENDS

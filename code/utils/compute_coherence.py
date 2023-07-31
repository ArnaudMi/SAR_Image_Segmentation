import numpy as np
import matplotlib.pyplot as plt

from decomposition import Decomposition


img = np.load('../var/bretigny_lee_filter.npy')
decomp = Decomposition(img, pauli=False)
height, length, dim = img.shape
coh_robust = decomp.compute_coh_array((4,4))

np.save('../var/coh_array_robust_bretigny_lee.npy', coh_robust)

yamaguchi = decomp.apply_Yamaguchi_decomposition()
deoriented = decomp.apply_Yamaguchi_deoriented()
H,A,alpha = decomp.apply_halpha_decomposition()

h_A_alpha_img = [[[H[i,j],A[i,j],alpha[i,j]] for j in range(length)] for i in range(height)]

fig_yamaguchi = plt.figure(figsize = (20,8))
ax_1 = fig_yamaguchi.add_subplot(2,2,1)
ax_2 = fig_yamaguchi.add_subplot(2,2,2)
ax_3 = fig_yamaguchi.add_subplot(2,2,3)
ax_4 = fig_yamaguchi.add_subplot(2,2,4)

ax_1.imshow(abs(img))
ax_2.imshow(h_A_alpha_img)
ax_3.imshow(deoriented[:,:,:3])
ax_4.imshow(yamaguchi)

ax_1.set_title('Bretigny sur Orge en décomposition de Pauli')
ax_2.set_title('Bretigny Sur Orge en décomposition H/A/alpha')
ax_3.set_title('Bretigny Sur Orge en décomposition de Yamaguchi')
ax_4.set_title('Bretigny Sur Orge décomposition de Yamaguchi avec désorientation')
fig_yamaguchi.show()
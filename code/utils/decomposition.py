import numpy as np
import scipy.io as sio

from tqdm import tqdm


class Extraction:
    """
    Class specialized for extracting row MATLAB files
    """

    def __init__(self, path: str) -> None:
        self.data = sio.loadmat(path)
        self.cols = ["HH", "HV", "VV"]
        self.dic = {}
        for col in self.cols:
            self.dic[col] = self.data[col]
        self.img = np.zeros(
            (self.data[col].shape[0], self.data[col].shape[1], 3), dtype=complex
        )
        for idx, tag in enumerate(self.cols):
            self.img[:, :, idx] = self.dic[tag]

class Decomposition:
    def __init__(self, img: np.ndarray, pauli = True) -> None:
        """
        In case pauli is not declared it decomposes the image after the pauli decomposition. In case you don't want to apply pauli, set to False.\n
        Attributes : \n
        self.original_img contains img in the form ('HH','HV','VV'), no preprocessing applied \n
        self.decomposed_img contains the decomposed img \n
        self.height, self.length and self.dim are the dimensions of the image \n
        self.data contains all the differents decomposition of the original image in the form self.data['your_decomposition'] \n\n
        Methods : \n
        self.apply_pauli_decomposition \n
        self.compute_gaussian_cov \n
        self.compute_robust_cov \n
        self.compute_cov_img \n
        self.matrix_squared_norm \n
        """
        self.normalized_img = np.copy(img)
        self.normalized_img[:,:,1] = np.sqrt(2) * img[:,:,1] 
        self.original_img = img
        self.decomposed_img = np.zeros_like(img)
        self.height, self.length, self.dim = self.original_img.shape
        self.data = dict()
        if pauli == True :
            self.data['pauli'] = self.apply_pauli_decomposition()
        else : 
            self.data['pauli'] = self.original_img

    """
    ===========================================================PAULI_DECOMPOSITION================================================================
    """

    def apply_pauli_decomposition(self):
        """
        returns : Array in the same shape as self.original_img,\n\n
                  
        applies Pauli so as to visualize the image \n
        R channel is associated with double bounce, typically happens in urban zone \n
        G channel is associated with volume scattering, happens in crops \n
        B channel is associated with simple bounce, surface scattering \n
        
        """
        self.decomposed_img[:, :, 0] = (
            self.original_img[:, :, 0] - self.original_img[:, :, 2]
        )
        self.decomposed_img[:, :, 1] = 2 * self.original_img[:, :, 1]
        self.decomposed_img[:, :, 2] = (
            self.original_img[:, :, 0] + self.original_img[:, :, 2]
        )
        return 1 / np.sqrt(2) * self.decomposed_img

    """
    ===========================================================COVARIANCE_ESTIMATION==============================================================
    """

    def gaussian_coherence(
            self, i: int, j: int, neighbourhood_size: tuple
    ) :
        """
        Computes the coherence of the pixel i,j after the Gaussian maximum likelihood Estimator on the square neighbourhood\n

        Args:\n
            - i height index of the pixel in the image\n
            - j length index of the pixel in the image\n
            - neighbourhood_size is the size of the square window\n
        Returns:\n
            - Array shape(self.dim, self.dim) dtype = complex, coherence matrix of the pixel i,j\n
        """
        pauli = self.data['pauli']

        h, l = neighbourhood_size[0], neighbourhood_size[1]
        
        top = max(0, i - h)
        bottom = min(self.height, i + h)
        left = max(0, j - l)
        right = min(self.length, j + l)

        coh_estimator = np.zeros((3, 3), dtype=complex)
        for p in range(top, bottom, 1):
            for q in range(left, right, 1):
                center_pix = pauli[p,q].reshape(1,3)
                coh_estimator += np.conjugate(center_pix).T @ center_pix
        return 1/((bottom - top)*(right - left)) * coh_estimator
    

    def robust_coherence(
            self, i: int, j: int, neighbourhood_size: tuple
    ) :
        """
        Computes the coherence of the pixel i,j after the Tyler's M-Estimator on the square neighbourhood\n
        
        Args:\n
            - i height index of the pixel in the image\n
            - j length index of the pixel in the image\n
            - neighbourhood_size is the size of the square window\n
        Returns:\n
            - Array shape(self.dim, self.dim) dtype = complex, coherence matrix of the pixel i,j\n
        """
        pauli = self.data['pauli']

        h, l = neighbourhood_size[0], neighbourhood_size[1]
        
        top = max(0, i - h)
        bottom = min(self.height, i + h)
        left = max(0, j - l)
        right = min(self.length, j + l)

        coh_estimator_n = np.eye(3, dtype=complex)
        coh_estimator = np.zeros((3, 3), dtype=complex)
        count = 0
        N = (bottom - top) * (right - left)
        while (
            self.matrix_squared_norm(coh_estimator - coh_estimator_n) > 1e-4
        ):
            if count >= 1:
                coh_estimator_n = coh_estimator

            coh_estimator = np.zeros((3, 3), dtype=complex)
            inv = np.linalg.inv(coh_estimator_n)
            for p in range(top, bottom, 1):
                for q in range(left, right, 1):
                    center_pix = pauli[p, q].reshape(1, 3)
                    denom = np.conjugate(center_pix) @  inv @ center_pix.T
                    coh_estimator += np.conjugate(center_pix).T @ center_pix / denom

            coh_estimator = 1 / N * coh_estimator
            coh_estimator = 3 / np.real(np.trace(coh_estimator)) * coh_estimator
            count += 1


        return coh_estimator

    def compute_gaussian_coh_array(
        self, neighbourhood_size: tuple
    ) :
        
        coh = np.zeros((self.height, self.length, self.dim, self.dim), dtype=complex)

        for i in tqdm(range(self.height)):
            for j in range(self.length):
                coh[i, j] = self.gaussian_coherence(i, j, neighbourhood_size)

        self.data["gauss_coh"] = coh
        return coh

    # def compute_gaussian_cov(
    #     self, i: int, j: int, neighbourhood_size: tuple
    # ) -> np.ndarray[complex]:
    #     # méthode optimal pour le cas gaussien
    #     h, l = neighbourhood_size[0], neighbourhood_size[1]
    #     top = max(0, i - h)
    #     bottom = min(self.height, i + h)
    #     left = max(0, j - l)
    #     right = min(self.length, j + l)
    #     cov_estimator = np.zeros((3, 3), dtype=complex)
    #     for p in range(top, bottom, 1):
    #         for q in range(left, right, 1):
    #             cov_estimator += (
    #                 np.conjugate(self.original_img[p, q]) @ self.original_img[p, q].T
    #             )
    #     return 1 / (bottom - top) / (right - left) * cov_estimator

    # def compute_robust_cov(
    #     self, i: int, j: int, neighbourhood_size: tuple
    # ) -> np.ndarray[complex]:
    #     """
    #     Implementation of the Tyler's estimator :
    #         compute iteratively the covariance matrix defined by Taylor in [10] D. Tyler, “A distribution-free M-estimator of multivariate scatter,” The
    #         Annals of Statistics, vol. 15, no. 1, pp. 234–251, 1987.
    #         Showed to be unique in 2008 : Covariance structure maximum-likelihood estimates in compound gaussian noise, F. Pascal, J.-P. Ovarlez and P.larzabal
    #     """
    #     h, l = neighbourhood_size[0], neighbourhood_size[1]
    #     top = max(0, i - h)
    #     bottom = min(self.height, i + h)
    #     left = max(0, j - l)
    #     right = min(self.length, j + l)
    #     mean = np.mean(self.original_img[top:bottom, left:right], axis=(0, 1))
    #     cov_estimator_n = np.eye(3, dtype=complex)
    #     cov_estimator = np.eye(3, dtype=complex)
    #     count = 0
    #     N = (bottom - top) * (right - left)
    #     while (
    #         self.matrix_squared_norm(cov_estimator - cov_estimator_n) > 1e-4
    #         or count == 0
    #     ):
    #         if count >= 1:
    #             cov_estimator_n = cov_estimator

    #         cov_estimator = np.zeros((3, 3), dtype=complex)

    #         for p in range(top, bottom, 1):
    #             for q in range(left, right, 1):
    #                 center_pix = (self.original_img[p, q] - mean).reshape(1, 3)
    #                 cov_estimator += (
    #                     np.conjugate(center_pix).T
    #                     @ center_pix
    #                     / np.real(
    #                         np.conjugate(center_pix) @ np.linalg.inv(cov_estimator_n) @ center_pix.T
    #                     )
    #                 )

    #         cov_estimator = 1 / N * cov_estimator
    #         cov_estimator = 3 / np.real(np.trace(cov_estimator)) * cov_estimator
    #         count += 1
    #     return cov_estimator

    def compute_coh_array(self, neighbourhood_size : tuple) -> np.ndarray:
        """
        Computes the array containing all the coherence matrices of all the pixels of the image after the robust estimation method\n
        Args:\n
            - neighbourhood_size is the square dimension on which the coherence is computed\n
        Returns:\n
            - Array shape(self.height, self.length, self.dim, self.dim) is the matrix containing all the coherence matrices\n
        """
        
        coh = np.zeros((self.height, self.length, self.dim, self.dim), dtype=complex)

        for i in tqdm(range(self.height)):
            for j in range(self.length):
                coh[i, j] = self.robust_coherence(i, j, neighbourhood_size)

        self.data["coh"] = coh
        return coh
    
    def compute_cov_array(self, neighbourhood_size: tuple):
        #covariance is based on the raw vectors
        cov = np.zeros((self.height, self.length, self.dim, self.dim), dtype=complex)
        for i in range(self.height):
            for j in range(self.length):
                cov[i, j] = self.compute_robust_cov(i, j, neighbourhood_size)
        self.data["cov"] = cov
        return cov

    def matrix_squared_norm(self, matrix: np.ndarray[complex]):
        """
        Euclidean squared norm
        """
        s = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                s += np.real(matrix[i, j]) ** 2 + np.imag(matrix[i, j]) ** 2
        return s

    """
    ========================================COVARIANCE_ESTIMATION_END==========================================================================
    """
    """
    ========================================CLOUDE-POTTIER-DECOMPOSITION=======================================================================
    """

    def compute_H_Alpha(
        self, i: int, j: int, coh: np.ndarray[float]
    ) -> tuple[float, float]:
        """
        compute Cloude et Pottier decomposition of pixel i,j
        from the scatter matrix of a given point, compute the eigenvalues and the eigenvectors
        pi = lambdai / sum (lambdai)
        alpha = sum pi * (alpha_i)
        A = lambda2-lambda3/(lambda2+lambda3)
        H = -sum pi log(pi)

        as the covariance matrix is hermitian positive definite its eigen values are always real and positive
        eigenvalues that are irrelevant are set to 0 ie. less than 1e-4
        """

        coh_estimate = coh
        eig_vals, eig_vec = np.linalg.eig(coh_estimate)
        eig_vals = np.real(abs(eig_vals))
        trace = sum(eig_vals) #span of the matrix equals to the power of the pixel

        p = [eig_vals[i] / trace for i in range(3)] #probability distribution

        alpha = np.sum(
            [np.arccos(abs(eig_vec[:, i][0])) * p[i] for i in range(3)]
        ) #cloude, pottier angle

        H = - sum([p[i] * np.log(p[i]) /np.log(3) if p[i] != 0 else 0 for i in range(3) ])
        eig_list = sorted(eig_vals)
        A = (
            0.0
            if eig_list[1] + eig_list[0] == 0
            else (eig_list[1] - eig_list[0]) / (eig_list[1] + eig_list[0])
        )# anisotropy, difference of the two smallest eigenvalues divided by their sum
        return H, A, 90*2/np.pi * alpha

    def apply_halpha_decomposition(self, token = 'robust'):
        """
        See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=485127
        Args:\n
            - token : str is the mode for computing coherence matrices
        Returns:\n
            - entropy : Array shape(self.height, self.length)  containing the entropy for each pixel
            - anisotropy : Array shape(self.height, self.length) containing the anisotropy for each pixel
            - alpha : Array shape(self.height, self.length) containing the alpha of each pixel 
        """
        entropy = np.zeros((self.height, self.length))
        A = np.zeros((self.height, self.length))
        alpha = np.zeros((self.height, self.length))
        try :
            if token == 'robust' :
                cov_array = self.data['coh']
            else :
                cov_array = self.data['gauss_coh']
        except AttributeError:
            print('You have to compute the coherence matrix first, run : compute_coh_array first')
            return None
        for i in tqdm(range(self.height)):
            for j in range(self.length):
                entropy[i, j], A[i, j], alpha[i, j] = self.compute_H_Alpha(
                    i, j, cov_array[i, j]
                )
        self.data["entropy"] = entropy
        self.data["alpha"] = alpha
        self.data["anisotropy"] = A
        return entropy, A, alpha


    """
    
    """
    """
    ===============================================================YAMAGUCHI_DECOMPOSITION======================================================================
    """

    def module_squared(self, c: complex):
        """
        returns |c|**2 the module squared of a complex number
        """
        return np.real(c) ** 2 + np.imag(c) ** 2

    def apply_Yamaguchi_decomposition(self):
        # See Yamaguchi's Decomposition Algorithm
        yamaguchi = np.zeros((self.height, self.length, 4))
        Ps, Pd, Pv, Pc = 0, 0, 0, 0
        for i in range(self.height):
            for j in range(self.length):
                a = self.original_img[i, j, 0]
                b = self.original_img[i, j, 2]
                c = self.original_img[i, j, 1]
                Pc = 2 * abs(
                    np.imag(np.conjugate(c) * (a - b))
                )  # determines the helix scattering power
                test = np.log10(
                    self.module_squared(b) / self.module_squared(a)
                )  # determines which physical propagation is preponderous

                # three cases disjonction : test <-2 ; test in [-2,2] ; test > 2
                if 10 * test < -2:
                    Pv = 15 / 2 * self.module_squared(c) - 15 / 8 * Pc
                    if Pv <= 0:
                        Pc = 0  # a power cannot be negative
                        Pv = 15 / 2 * self.module_squared(c)
                    S = 1 / 2 * self.module_squared(a + b) - 1 / 2 * Pv
                    D = (
                        1 / 2 * self.module_squared(a - b)
                        - 7 / 4 * self.module_squared(c)
                        - 1 / 16 * Pc
                    )
                    C = 1 / 2 * (a + b) * np.conjugate(a - b) - 1 / 6 * Pv

                if -2 <= test <= 2:
                    Pv = 8 * self.module_squared(c) - 2 * Pc
                    if Pv <= 0:
                        Pc = 0  # a power cannot be negative
                        Pv = 8 * self.module_squared(c)
                    S = (
                        1 / 2 * self.module_squared(a + b)
                        - 4 * self.module_squared(c)
                        + Pc
                    )
                    D = 1 / 2 * self.module_squared(a - b) - 2 * self.module_squared(c)
                    C = 1 / 2 * (a + b) * np.conjugate(a - b)

                if test > 2:
                    Pv = 15 / 2 * self.module_squared(c) - 15 / 8 * Pc
                    if Pv <= 0:
                        Pc = 0  # a power cannot be negative
                        Pv = 15 / 2 * self.module_squared(c)
                    S = 1 / 2 * self.module_squared(a + b) - 1 / 2 * Pv
                    D = (
                        1 / 2 * self.module_squared(a - b)
                        - 7 / 4 * self.module_squared(c)
                        - 1 / 16 * Pc
                    )
                    C = 1 / 2 * (a + b) * np.conjugate(a - b) + 1 / 6 * Pv

                # Final Decomposition, function of the parameters computed above
                TP = (
                    self.module_squared(a)
                    + self.module_squared(b)
                    + 2 * self.module_squared(c)
                )  # total power  
                if Pv + Pc < TP:
                    C0 = a * np.conjugate(b) - self.module_squared(c) + 1 / 2 * Pc
                    if np.real(C0) < 0:
                        Ps = S - self.module_squared(C) / D
                        Pd = D + self.module_squared(C) / D
                        
                    else :
                        Ps = S + self.module_squared(C) / S
                        Pd = D - self.module_squared(C) / S
                    if Ps < 0 :
                        Ps = 0
                        Pd = TP - Pv - Pc
                    if Pd < 0 :
                        Pd = 0
                        Ps = TP - Pv - Pc
                else : 
                    Ps = 0
                    Pd = 0
                    Pv = TP - Pc

                yamaguchi[i, j, 0], yamaguchi[i, j, 1], yamaguchi[i, j, 2], yamaguchi[i, j, 3],= (Ps, Pd, Pv, Pc)

        self.data["yamaguchi"] = yamaguchi
        return yamaguchi

    """
    ===========================================================YAMAGUCHI_END======================================================================
    """
    """
    ============================================================YAMAGUCHI_DEORIENTATION=============================
    """
    def deorientation(self, cov) :
        deoriented_cov = np.zeros_like(cov, dtype = complex)
        B = cov[1, 1] - cov[2, 2]
        E = cov[2, 1] + cov[1, 2]
        theta = 0
        if B**2 + E**2 <1e-4 :
            return cov
        if B > 0 :
            theta = 1/4 * np.arctan(E/B)
        else :
            if E >= 0 :
                theta = 1/4 * (np.pi + np.arctan(E/B))
            else :
                theta = 1/4 * (np.arctan(E/B) - np.pi)
        rotation_matrix = np.array([np.array([1,0,0]),np.array([0,np.cos(2*theta),-np.sin(2*theta)]),np.array([0,np.sin(2*theta),np.cos(2*theta)])])
        deoriented_cov = rotation_matrix @ cov @ rotation_matrix.T
        return deoriented_cov
    
    def apply_Yamaguchi_deoriented(self) :
        """
        Applies the Yamaguchi decomposition to the self.img. See the article : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5437202
        Args: None\n
        Returns:\n
            - Array shape(self.height, self.length, 5) in the order(i,j,(Ps,Pd,Pv,alpha,beta))
        """
        deoriented = np.zeros((self.height, self.length, 5))
        for i in range(self.height):
            for j in range(self.length) :
                coh = self.data['coh'][i,j]
                T = self.deorientation(coh)  
                if np.real(T[0,0]) <= np.real(T[2,2]) :
                    alpha, beta = 0, 0
                    Pv = 3*T[0,0]
                    Ps = 0
                    Pd = T[1,1] + T[2,2] - 2*T[0,0]
                else :
                    Pv = 3*T[2,2]
                    x11 = T[0,0] - T[2,2]
                    x22 = T[1,1] - T[2,2]
                    c = self.module_squared(T[0,1])**2
                    if c > x11 * x22 :
                        if x11 > x22 :
                            alpha = 0
                            beta = T[0,1]/np.sqrt(c)*np.sqrt(x22/x11)
                            Ps = x11 + x22
                            Pd = 0
                        else :
                            beta = 0
                            alpha = T[0,1]/np.sqrt(c)*np.sqrt(x22/x11)
                            Ps = 0
                            Pd = x11 + x22
                    else :
                        if x11 > x22 :
                            alpha = 0
                            beta = T[0,1]/x11
                            Ps = x11 + c/x11
                            Pd = x22 - c/x11
                        else :
                            beta = 0
                            alpha = T[0,1]/x22
                            Ps = x11 - c/x22
                            Pd = x22 + c/x22
                deoriented[i,j] = Ps, Pd, Pv, alpha, beta
        self.data["deoriented"] = deoriented
        return deoriented 
    """
    ===============================================================HUYNEN_DECOMPOSITION=========================================================================
    """

    def apply_Huynen_decomposition(self, neighbourhood_size, cov_array):
        """
        Huynen decomposition of the sinclair matrix :
        computed in a given neighbourhood of each pixel
        """

        for i in range(self.height):
            for j in range(self.length):
                cov = cov_array[i, j] #coherence_matrix, computed after pauli decomposition
                A0 = np.real(cov[0, 0]) / 2
                B0 = np.real(cov[1, 1] + cov[2, 2]) / 2
                B = np.real(cov[1, 1] - cov[2, 2]) / 2
                C = np.real(cov[1, 0] + cov[0, 1]) / 2
                D = np.imag(cov[1, 0] - cov[0, 1]) / 2
                E = np.real(cov[1, 2])
                F = np.imag(cov[1, 2])
                H = np.real(cov[0, 2])
                G = np.imag(cov[0, 2])
        return A0, B0, B, C, D, E, F, G, H

    """
    ===============================================================END_HUYNEN_DECOMPOSITION====================================================================
    """
    """
    rvi parameter is a parameter for humid zones, see
    """

    def compute_rvi(self):
        self.data["rvi"] = (
            self.original_img[:, :, 1]
            * 8
            / (
                self.original_img[:, :, 0]
                + self.original_img[:, :, 2]
                + 2 * self.original_img[:, :, 1]
            )
        )
        return self.data["rvi"]

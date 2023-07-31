
import numpy as np
from tqdm import tqdm

from statistics import median



class Filter :

    """
    To add :    
        - Filtre de Frost, Filtre de Gamma_MAP, Kuan 
        - Autoencoder filtering ?
    """
    #class specialized for filtering SAR images formated as (height, len, (HH,HV,VV))
    def __init__(self, img : np.ndarray , kernel_size : tuple[int,int]) -> None:
        #kernel_size is the window on which we will apply our filter, example :
        #  if kernel_size == (3,3) then the mean will be computed on its direct neighbours in a 3x3 square.

        self.original_img = img
        self.kernel_size = kernel_size
        self.height, self.length, self.dim = img.shape
        self.k_height, self.k_length = kernel_size[0], kernel_size[1]
        self.filtered_img = np.zeros_like(self.original_img)

    def apply_average_filter(self):
        img = self.original_img
        filtered_img = np.zeros(img.shape, dtype = np.complex128)
        height, length, dim = img.shape

        k_height, k_length = self.kernel_size[0], self.kernel_size[1]

        filtered_img = np.zeros_like(img)
 
        for i in range(height) :
            for j in range(length) :
                top = max(0, i - k_height//2)
                bottom = min(height, i + k_height//2 + 1)
                left = max(0, j-k_length//2)
                right = min(length, j + k_length//2 + 1)
                filtered_img[i,j] = np.mean(img[top:bottom, left:right, :], axis = (0,1), dtype = complex)

        self.filtered_img = filtered_img
    
    def apply_median_filter(self) :
        #this methods applies the median on each real part, imaginary part of each component HH, HV, VV.
        for i in range(self.height) :
            for k in range(self.length) :
                top = max(0, i - self.k_height // 2 )
                bottom = min(self.height, i + self.k_height // 2 + 1)
                left = max(0, k - self.k_length // 2)
                right = min(self.length, k + self.k_length // 2 + 1)
                for d in range(self.dim) :
                    self.filtered_img[i, k, d] = median(np.real(self.original_img[top : bottom, left : right, d].reshape(-1))) + median(np.imag(self.original_img[top : bottom, left : right, d].reshape(-1))) * complex(real = 0, imag = 1)

    def apply_lee_filter(self,sigma_v = 1.15):
        """
        Applique le filtre de Lee à l'image SAR polarimetrique.

        Le résultat apparaît dans la variable self.filtered_img

        var_y est calculé localement pour chaque pixel selon l'article de Lee : Polarimetric SAR Speckle Filtering And Its Implication For Classification

        Args: 
            sigma_v est un nombre arbitrairement choisi qui représente l'écart type du speckle, bruit que l'on cherche à filtrer
        """
        img = self.original_img
        size = self.k_height
        img_mean = np.mean(img, axis = (0,1))
        var_y = np.zeros_like(img)
        var_x = np.zeros_like(img)
        b = np.zeros_like(img)
        for d in range(self.dim) :
            for i in tqdm(range(self.height)) :
                for j in range(self.length) :
                    top = max(0, i - self.k_height//2 )
                    bottom = min(self.height, i + self.k_height//2 + 1)
                    left = max(0, j - self.k_length//2)
                    right = min(self.length, j + self.k_length//2 + 1)
                    var_y[i,j,d] = np.mean(self.squared_norm(img[top:bottom, left: right,d]), axis = (0,1))-self.squared_norm(np.mean(img[top:bottom, left: right,d], axis = (0,1)))
                    var_x[i,j,d] = (var_y[i,j,d] - img_mean[d]*img_mean[d]*sigma_v*sigma_v)/(1+sigma_v*sigma_v)
                    if var_x[i,j,d] < 0 :
                        var_x[i,j,d] = 0
                    b[i,j,d] = var_x[i,j,d]/var_y[i,j,d]
                    self.filtered_img[i,j,d] = img_mean[d] + b[i,j,d] * (img[i,j,d] - img_mean[d])

        return self.filtered_img
    

    def squared_norm(self, c : complex) :
        a = np.real(c)
        b = np.imag(c)
        return a*a + b*b
   
    """
    Kuan and Frost filter are to be implemented
    """
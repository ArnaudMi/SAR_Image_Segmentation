import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from cv2 import cvtColor, COLOR_RGB2BGR

class SAM_Cluster:
    def __init__(self, polar:np.ndarray, cluster_nb:int, sam_path, distance_token ='wishart') -> None:
        """
        Implements an optimal clustering algorithm based on the distance between segments output of SAM.\n
        Args:\n
            - polar is the polarimetric image one would like to cluster\n
            - cluster_nb is the number of cluster wanted\n
            - distance_token sets the distance chosen for the clustering : 'wishart' sets to the wishart distance and anything else sets the 'Bhattacharyya' one.\n
        """
        self.polar = polar
        self.cluster_nb = cluster_nb
        self.height, self.length, self.dim = self.polar.shape
        self.coherence_array = np.zeros((self.height, self.length, self.dim))
        self.sam = sam_model_registry["default"](checkpoint=sam_path)
        self.distance = distance_token
        img = self.from_complex_to_uint8() # converts our data into a compatible format for SAM
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        img_cv2 = cvtColor(img, COLOR_RGB2BGR)
        self.masks = mask_generator.generate(img)
        

        
        
    
    def robust(self, pixel_list) :
        
        estimate = np.zeros((self.dim, self.dim), dtype= complex)
        estimate_n = np.eye(self.dim, dtype= complex)
        count = 0
        while np.sum((estimate-estimate_n)**2)>1e-4 :
            if count != 0:
                estimate_n = estimate
            estimate = np.zeros((self.dim, self.dim), dtype= complex)
            for pixel in pixel_list :
                pixel = pixel.reshape(3,1)
                estimate += pixel @ pixel.T / (np.conjugate(pixel).T @ estimate_n @ pixel)
            estimate = 3 * estimate / np.trace(estimate)
            count += 1
        return estimate
    
    def compute_loss(self, Delta, D) :
        
        _,S = Delta.shape
        s = 0
        for k in range(S) :
            intra = Delta[:,k].T@D@Delta[:,k]
            inter = 0
            for j in range(k+1,S) :
                inter += Delta[:,k].T@D@Delta[:,j] 

            s += 1/2 * intra - inter 
        return s
        
    def cluster(self, is_gaussian: bool, tol) :

        inv = np.linalg.inv

        l = len(self.masks)
        D = np.zeros((l,l))
        list_coherence = []
        for mask in self.masks :
            pixels_list = self.polar[mask['segmentation']].reshape(-1,1,3)
            N = len(pixels_list)
            if is_gaussian:
                estimate = 1/N * np.sum([pixels_list[i]], axis=0 ) #gaussian estimator
            else :
                estimate = self.robust(pixels_list)
            list_coherence.append(estimate)
        print('Coherence matrices estimated\n')
        for i in range(l) :
            for j in range(i+1, l) :
                if self.distance == 'wishart':
                    D[i,j] = self.distance_W(list_coherence[i],list_coherence[j])
                else :
                    D[i,j] = self.distance_Bhattacharyya(list_coherence[i],list_coherence[j])
                D[j,i] = D[i,j]
        S = self.cluster_nb
        n = l

        Hess = np.zeros((S * n, S * n))
        for i in range(S) :
            Hess[i*n:(i+1)*n, i*n: (i+1)*n] = np.copy(D)
            for j in range(i+1,S) :
                Hess[i*n:(i+1)*n, j*n:(j+1)*n] = -np.copy(D)

        Hess_inv = inv(Hess)
        Delta = np.zeros((n,S))
        Delta[:,0] = np.array([1]*n)

        loss = self.compute_loss(Delta, D)
        mem = 0
        epoch = 0
        def absolute(a) :
            return np.sqrt(a*a)
        
        def compute_grad(Delta) :
            n,S = Delta.shape
            grad = np.zeros((S*n,))
            for i in range(S) :
                s = np.zeros((n,))
                s = Delta[:,i] - np.sum(Delta[:,i+1:S], axis = 1)
                grad[n*i : (i+1) * n] = D @ (s)
            return grad
        
        while absolute(mem - loss)>tol :
            mem = loss
            grad = compute_grad(Delta)
            direction = - Hess_inv @ grad
            for i in range(S) :
                Delta[:,i] = Delta[:,i] + direction[n*i:n*(i+1)]
            for i in range(n) :
                a = np.argmax(Delta[i,:])
                Delta[i,a] = 1
                Delta[i,:a] = 0
                Delta[i,a+1:] = 0
            loss = self.compute_loss(Delta,D)
            epoch += 1
            print(f'epoch = {epoch}, loss = {loss}')
        print(self.compute_loss(Delta))
        img_clustering = np.zeros(self.polar.shape[:2])
        for i in range(n) :
            x, y = np.where(self.masks[i]['segmentation']== True)
            for abs, val in enumerate(x) :
                img_clustering[val,y[abs]] = np.argmax(Delta[i]) +1

        return img_clustering  
      
    def min(a : float,b : float) :
        return a*(a < b) + b * (a >= b)

    def from_complex_to_uint8(self) :
        #scale data used by matplotlib.pyplot.imshow
        img_polar = self.polar
        height, length, dim = img_polar.shape
        convert = np.zeros(img_polar.shape, dtype= np.uint8)
        abs_img = abs(img_polar)
        def min(a,b) :
            return a*(a<b) + b*(b<=a)
        convert[:,:,0] = 255 * min(abs_img[:,:,0], 1)
        convert[:,:,1] = 255 * min(abs_img[:,:,1], 1)
        convert[:,:,2] = 255 * min(abs_img[:,:,2], 1)
        return convert


    def plot_clusters(self) :
        try :
            plt.imshow(self.img_clustering)
            plt.show(False)
            plt.imshow(self.img_cv2)
        except NameError :
            print('Variable self.img_cluster does not exist, first you need to cluster, try cluster_gaussian or cluster_robust')
            

    def distance_Bhattacharyya(self, a, b):
        return np.linalg.det(1/2 * (a+b))**2/(np.linalg.det(a)*np.linalg.det(b)) - 1
    
    def distance_W(self, a, b) :
        inv = np.linalg.inv
        dim = a.shape[0]
        return np.sum(inv(a)*b + inv(b) * a) - 2*dim


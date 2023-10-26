import numpy as np
import scipy.io as sio
import sys
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from pathlib import Path

CPU_NUMBER = multiprocessing.cpu_count()



def from_mask_to_img(masks) :
    masks_img = np.zeros_like(masks[0]['segmentation'], dtype=int)
    for i in range(143) :
        masks_img += masks[i]
    plt.imshow(masks_img)
    return masks_img

def from_complex_to_uint8(img) :
    #scale data used by matplotlib.pyplot.imshow
    img_polar = np.copy(img)
    convert = np.zeros(img_polar.shape, dtype= np.uint8)
    abs_img = abs(img_polar)
    def min(a,b) :
        return a*(a<b) + b*(b<=a)
    convert[:,:,0] = 255 * min(abs_img[:,:,0], 1)
    convert[:,:,1] = 255 * min(abs_img[:,:,1], 1)
    convert[:,:,2] = 255 * min(abs_img[:,:,2], 1)
    return convert

def open_s_dataset(path: str):
    path = Path(path)
    # http://www.spectralpython.net/fileio.html#envi-headers
    s_11_meta = envi.open(path / 's11.bin.hdr', path / 's11.bin')
    s_12_meta = envi.open(path / 's12.bin.hdr', path / 's12.bin')
    s_21_meta = envi.open(path / 's21.bin.hdr', path / 's21.bin')
    s_22_meta = envi.open(path / 's22.bin.hdr', path / 's22.bin')

    s_11 = s_11_meta.read_band(0)
    s_12 = s_12_meta.read_band(0)
    s_21 = s_21_meta.read_band(0)
    s_22 = s_22_meta.read_band(0)

    assert np.all(s_21 == s_12)

    return np.stack((s_11, s_12, s_22), axis=-1).astype(np.complex64)



def open_t_dataset_t3(path: str):
    path = Path(path)
    first_read = envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0)
    T = np.zeros(first_read.shape + (6,), dtype=np.complex64)

    # Diagonal
    T[:, :, 0] = first_read
    T[:, :, 1] = envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0)
    T[:, :, 2] = envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0)

    # Upper part
    T[:, :, 3] = envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0)
    T[:, :, 4] = envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0)
    T[:, :, 5] = envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0)
    return T.astype(np.complex64)


def open_dataset_t6(path: str):
    path = Path(path)
    first_read = envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0)
    T = np.zeros(first_read.shape + (21,), dtype=np.complex64)

    # Diagonal
    T[:, :, 0] = first_read
    T[:, :, 1] = envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0)
    T[:, :, 2] = envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0)
    T[:, :, 3] = envi.open(path / 'T44.bin.hdr', path / 'T44.bin').read_band(0)
    T[:, :, 4] = envi.open(path / 'T55.bin.hdr', path / 'T55.bin').read_band(0)
    T[:, :, 5] = envi.open(path / 'T66.bin.hdr', path / 'T66.bin').read_band(0)

    T[:, :, 6] = envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0)
    T[:, :, 7] = envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0)
    T[:, :, 8] = envi.open(path / 'T14_real.bin.hdr', path / 'T14_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T14_imag.bin.hdr', path / 'T14_imag.bin').read_band(0)
    T[:, :, 9] = envi.open(path / 'T15_real.bin.hdr', path / 'T15_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T15_imag.bin.hdr', path / 'T15_imag.bin').read_band(0)
    T[:, :, 10] = envi.open(path / 'T16_real.bin.hdr', path / 'T16_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T16_imag.bin.hdr', path / 'T16_imag.bin').read_band(0)

    T[:, :, 11] = envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0)
    T[:, :, 12] = envi.open(path / 'T24_real.bin.hdr', path / 'T24_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T24_imag.bin.hdr', path / 'T24_imag.bin').read_band(0)
    T[:, :, 13] = envi.open(path / 'T25_real.bin.hdr', path / 'T25_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T25_imag.bin.hdr', path / 'T25_imag.bin').read_band(0)
    T[:, :, 14] = envi.open(path / 'T26_real.bin.hdr', path / 'T26_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T26_imag.bin.hdr', path / 'T26_imag.bin').read_band(0)

    T[:, :, 15] = envi.open(path / 'T34_real.bin.hdr', path / 'T34_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T34_imag.bin.hdr', path / 'T34_imag.bin').read_band(0)
    T[:, :, 16] = envi.open(path / 'T35_real.bin.hdr', path / 'T35_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T35_imag.bin.hdr', path / 'T35_imag.bin').read_band(0)
    T[:, :, 17] = envi.open(path / 'T36_real.bin.hdr', path / 'T36_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T36_imag.bin.hdr', path / 'T36_imag.bin').read_band(0)

    T[:, :, 18] = envi.open(path / 'T45_real.bin.hdr', path / 'T45_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T45_imag.bin.hdr', path / 'T45_imag.bin').read_band(0)
    T[:, :, 19] = envi.open(path / 'T46_real.bin.hdr', path / 'T46_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T46_imag.bin.hdr', path / 'T46_imag.bin').read_band(0)

    T[:, :, 20] = envi.open(path / 'T56_real.bin.hdr', path / 'T56_real.bin').read_band(0) + \
                  1j * envi.open(path / 'T56_imag.bin.hdr', path / 'T56_imag.bin').read_band(0)
    return T.astype(np.complex64)

def get_non_overlapping_masks(masked_img):
    """From SAM's output to non overlapping masks"""
    
    sys.setrecursionlimit(1000000)

    bool_img = np.zeros_like(masked_img, dtype=bool)
    L_mask = []
    while np.any(bool_img == False):
        x,y = np.where(bool_img == False)
        i,j = x[0], y[0]
        #ignore points not selected by SAM
        if masked_img[i,j] == 0 :
            bool_img[i,j] = True
        else :
            area = color_area(masked_img, i,j)
            bool_img[area] = True
            L_mask.append(area)
    return L_mask


def color_area(masked_img, i, j):
    """used in get_non_overlapping mask"""
    bool_area = np.zeros_like(masked_img, dtype=bool)
    n_area = masked_img[i,j]
    if n_area == 0 :
        return bool_area
    def color(i,j):
        boolean = False
        if not(i<0 or i>=bool_area.shape[0] or j<0 or j>= bool_area.shape[1]):
            if n_area == masked_img[i,j] and bool_area[i,j] == False:
                bool_area[i,j] = True
                boolean = True
                color(i+1,j)
                color(i,j+1)
                color(i-1,j)
                color(i,j-1)
        return boolean
    color(i,j)
    return bool_area
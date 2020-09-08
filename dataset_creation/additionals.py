import numpy as np
import math
import scipy
import cv2


def filter_array(array, min_val, max_val):
    masked = np.zeros((array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j]<max_val and array[i,j]>min_val:
                masked[i,j] = array[i,j]
    return masked


def cdf_gauss(array):
    _array = np.zeros((array.shape[0],array.shape[1])).astype(float)
    mean = 0
    import math
    std_avg = np.std(array)
    #print('Average std:', std_avg)
    std_array = (array-mean)/std_avg
    #print(std_array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            _array[i, j] = gaussian_CDF(array[i, j], mean, std_avg)
            #print(_array[i, j])
            std_array[i, j] = std_avg
            #print(i, j, array[i,j])
    #print(std_array)
    #print('Mean array value:', np.mean(_array))
    return (_array, std_avg )

def reconstruct_arrays(scaled, std_avg):
    import math
    import scipy.special
    _array = np.zeros((scaled.shape[0], scaled.shape[1]))
    mean = 0
    #print(std_avg)
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            _array[i, j] = inv_gaussian_CDF(scaled[i, j], mean, std_avg)
    return _array


def gaussian_CDF(x, mean, std):
    #return 0.5*(1+math.erf((x-mean)/(std*np.sqrt(2))))
    if x<=0:
        return 0.5*np.exp(x/std)
    else:
        return 1-0.5*np.exp(-x/std)


def inv_gaussian_CDF(x, mean, std):
    if x>0.99: x = 0.99
    if x<0.01: x = 0.01
    inverf = scipy.special.erfinv(2*x-1)
    #if inverf > 100:
    #    print('MASSIVE INVERF:', inverf,'from:', x)
    return inverf * std *np.sqrt(2) + mean


def open_file_image(filename):
    # open image file
    img = cv2.imread(filename, -1)

    real_m_scaled = img[:,:,0].astype(float)/255
    imag_m_scaled = img[:,:,1].astype(float)/255
    std_array     = img[:,:,2].astype(float)/255
    std_real = std_array[0,0]
    std_imag = std_array[-1,-1]
    return real_m_scaled, imag_m_scaled, std_real, std_imag

def open_file_image_not_rescaled(filename):
    # open image file
    img = cv2.imread(filename, -1)

    real_m = img[:,:,0].astype(float)/255
    imag_m = img[:,:,1].astype(float)/255
    return real_m, imag_m


def recon_stft_from_elements(real_m_scaled, real_m_std, imag_m_scaled, imag_m_std):
    # Reconstruct the real_m and imag_m
    real_m_recon = reconstruct_arrays(real_m_scaled, real_m_std)
    imag_m_recon = reconstruct_arrays(imag_m_scaled, imag_m_std)

    stftMat_recon = real_m_recon + 1j * imag_m_recon

    return stftMat_recon




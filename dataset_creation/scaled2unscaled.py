#Author Myron Sampsakis-Bakopoulos M.Sc.
#August 2020

from additionals import *
import numpy as np
import librosa.core as lc
import scipy
import cv2
import os
import math
import time
import matplotlib.pyplot as plt



source_dir = './scaled'
target_dir = './unscaled'

par = 1

files = os.listdir(source_dir)

for filename in files:
    print("Reconstruction ",filename)

    real_m_scaled, imag_m_scaled, std_real, std_imag = open_file_image(source_dir+"/"+filename)
    if std_real == 0. or std_imag == 0.:
        print(std_real, std_imag)
        continue
    print(std_real * par / 255, std_imag * par / 255)
    real_m_recon = reconstruct_arrays(real_m_scaled * par, std_real * par)
    imag_m_recon = reconstruct_arrays(imag_m_scaled * par, std_imag * par)


    '''
    stftMat_recon = real_m_recon.astype(float) + 1j * imag_m_recon.astype(float)
    iStftMat_recon = lc.istft(stftMat_recon, hop_length=_hop_length, center=True)
    print(iStftMat_recon.shape)
    '''
    transparent_img = np.stack((real_m_recon * 255, imag_m_recon * 255, np.zeros(real_m_recon.shape)), axis=-1)
    cv2.imwrite(target_dir+"/"+filename +".png",  transparent_img)


    #scipy.io.wavfile.write(recon_files+"/"+"recon"+"_"+filename+".wav", sr, iStftMat_recon)
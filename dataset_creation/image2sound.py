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


#======PARAMETERS SETTINGS=======
path_soundfiles = "./audiofiles"
path_dataset    = "./imagefiles"
recon_files     = "./reconstructed"
_n_fft = 310 #310 or 693 for 1 or 5 sec accordingly
print(str(_n_fft))
_hop_length = int(_n_fft/4)
sr = 12000
duration = 1 #1 or 5 #seconds of each audio segment
rescale = True


files = os.listdir(path_dataset+'/results')

for filename in files:
    print("Reconstruction ",filename)
    if rescale:
        real_m_scaled, imag_m_scaled, std_real, std_imag = open_file_image(path_dataset+"/results/"+filename)
        if std_real == 0. or std_imag == 0.:
            print(std_real, std_imag)
            continue
        print(std_real * 3 / 255, std_imag * 3 / 255)
        real_m_recon = reconstruct_arrays(real_m_scaled * 3, std_real * 3)
        imag_m_recon = reconstruct_arrays(imag_m_scaled * 3, std_imag * 3)

    if not rescale:
        real_m_recon, imag_m_recon = open_file_image_not_rescaled(path_dataset + "/results/" + filename)


    stftMat_recon = real_m_recon.astype(float) + 1j * imag_m_recon.astype(float)
    iStftMat_recon = lc.istft(stftMat_recon, hop_length=_hop_length, center=True)
    print(iStftMat_recon.shape)

    scipy.io.wavfile.write(recon_files+"/"+"recon"+"_"+filename+".wav", sr, iStftMat_recon)



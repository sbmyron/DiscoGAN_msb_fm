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


#======PARAMETERS SETTINGS=======
path_soundfiles = "./audiofiles"
path_dataset    = "./imagefiles"
_n_fft = 310      #310 for 1sec, 693 for 5 sec
print(str(_n_fft))
_hop_length=int(_n_fft/4)
sr = 12000
duration = 1 #1 or 5 seconds of each audio segment
std_dic = {}
std_real_list = []
std_imag_list = []
scaling = True


files_violin = os.listdir(path_soundfiles+"/violin")
files_piano = os.listdir(path_soundfiles+"/piano")
files = files_violin+files_piano

for filename in files:
    file_done = open('file_done.out', 'r+')
    done_files = []
    for line in file_done:
        done_files.append(line)

    print(done_files)
    if filename in done_files:
        print('file already done...skippin!')
        continue



    if filename.split(".")[-1] != 'wav':
        continue
    print('Loading file '+filename)
    _organ_type = filename.split("_")[0]
    if _organ_type == 'p':
        organ_type = 'piano'
    elif _organ_type == 'v':
        organ_type = 'violin'

    data, sampleRate = lc.load(path_soundfiles+'/'+organ_type+'/'+filename, sr=sr, mono=True)

    #Size of the audiofile (in waveform)
    n_segments = int(np.ceil((data.shape[0]/sampleRate)/duration)) 

    for i_segment in range(n_segments):
        _data = np.zeros((duration*sampleRate))
        _data = data[i_segment*sampleRate*duration:(i_segment+1)*sampleRate*duration]


        #Check for uniform size (practically padd the last matrix):
        if _data.shape[0] != sampleRate*duration:
            _data = np.hstack((_data, np.zeros((sampleRate*duration - _data.shape[0])))) 

        #Performing STFT of the sound file
        stftMat = lc.stft(_data, n_fft=_n_fft, hop_length=_hop_length, center=True)

        real_m = np.real(stftMat)
        imag_m = np.imag(stftMat)
 
 
        if scaling:
            real_m_scaled, real_m_std = cdf_gauss(real_m)
            imag_m_scaled, imag_m_std = cdf_gauss(imag_m) 

        #Remove any windows without sound
        if scaling:
            if real_m_std == 0. or imag_m_std == 0.:
                print(real_m_std, imag_m_std)
                continue

            std_array = np.zeros((real_m_scaled.shape)) 

            max_std = 3

            std_array[0:int(std_array.shape[0]/2), ] = min(int(real_m_std/3*255), 3)
            std_array[int(std_array.shape[0]/2):, ]  = min(int(real_m_std/3*255), 3)

        key = filename.split('_')[1].split(".wav")[0]+'_p_'+str(i_segment) 
        if scaling:
            std_dic[key] = [real_m_std, imag_m_std]
            std_real_list.append(real_m_std)
            std_imag_list.append(imag_m_std)


        powerMat = np.abs(stftMat)  

        # Define an image, and save the file to it

        if scaling:
            transparent_img = np.stack((real_m_scaled * 255, imag_m_scaled * 255, std_array * 255), axis=-1)
        else:
            print(real_m.shape)
            transparent_img = np.stack((real_m * 255, imag_m * 255, np.zeros(real_m.shape)), axis=-1)
 
        cv2.imwrite(path_dataset+"/"+organ_type+"/"+filename.split(".")[0]+'_p_'+str(i_segment)+".png", transparent_img)
        print('Segment ', i_segment+1,'/',n_segments, 'done!')

    file_done.write(filename)
    file_done.close()
 

    print('Writing std to file...')
    file = open('std.out', 'w')
    for key in std_dic.keys(): 
        file.write(str(key)+' : '+str(std_dic[key][0])+' , '+str(std_dic[key][1])+'\n')
    file.close()

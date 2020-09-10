Author: Myron Sampsakis-Bakopoulos
September 2020

dataset_maker.py:
To create the dataset, choose the path of the soundfiles, and their destination. 
For 1 sec. output spectrograms, choose _n_fft = 310, and duration=1. 
Otherwise, for 5 sec. spectrograms, choose _n_fft=693 and duration=5. 
Use scaling  = True if you want scaling, else use scaling = False
Run the code to generate the output files.
Warning!: Generating a dataset with scaling = True is much slower, due to the transformations required.


image2sound.py:
To turn the spectrogram back to a *.wav file, enter the folder of the *.png files, and their desired output directory.
Choose the same: 
_n_fft, duration and rescale
that you chose when encoding them, and run the code. Else, the files will be either elongated or shorted in time, or not be produced at all.


scaled2unscaled:
To turn a scaled spectrogram into an unscaled spectrogram, simply enter the two directories, and run the code.


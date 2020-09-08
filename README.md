# DiscoGAN_msb_fm
DiscoGAN, but with sound


### 1 sec
To run the DiscoGAN with 1 sec files, rename model_1sec.py to model.py, and within the main directory, execute the command:
        
        $ python2 ./discogan/image_translation.py --task_name='piano2violin' --image_size=156 --batch_size=8



### 5 sec
To run the DiscoGAN with 5 sec files, rename model_5sec.py to model.py, and within the main directory, execute the command:

    $ python2 ./discogan/image_translation.py --task_name='piano2violin' --image_size=347 --batch_size=8

### Hybrid Architecture 
To run the Hybrid DiscoGAN with 5x 1sec files in each batch, execute the command:


    $ python2 ./discogan/image_translation_5n1.py --task_name='piano2violin' --image_size=156 --batch_size=5


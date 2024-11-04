# convert audio to spectrogram(img file)
# place this py file into the ./Urbansound8K directory

import csv
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt 


def spectrogram_image(filename,fold,filepath):
    y, sr = librosa.load(filepath)
         
    if len(y) < 5 * sr:
        y = np.pad(y, ( int(( (5 * sr - len(y)) /2)) , int(( (5 * sr - len(y)) / 2 ) ) ) , mode = 'constant')

    mel = librosa.feature.melspectrogram(y = y, sr = sr,n_mels = 216)
    log_mel = librosa.power_to_db(mel, ref = np.max)

    fig = plt.figure(figsize = (1.5,1.5))
    
    librosa.display.specshow(log_mel, sr=sr)
    filename = filename.strip('.wav')
        
    if (fold  <= 8):
        new_filepath = './us8k_train/'+ filename +'.tif' # your path
    else:
        new_filepath = './us8k_valid/'+ filename+'.tif' # your path
        
    fig.savefig(fname = new_filepath, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)


# main loop
metadata = open('./UrbanSound8K.csv','r')
line = csv.reader(metadata)

itr = -1

for ln in line:
    itr += 1
    if itr == 0: continue
    filename = ln[0]
    fold = ln[5]
    class_num = ln[6]
    filepath = './audio/fold' + fold + '/' + filename # your path
    spectrogram_image(filename,int(fold),filepath)
# convert audio to spectrogram(img file)
# place this py file into the ./Urbansound8K directory

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt 

def cropping(y,sr,window_size):
    max_ind = np.argmax(y)
    window = int((sr * window_size)/2)
    return y[max_ind-window : max_ind + window]

def spectrogram_image(fname, filepath , newpath):
    y, sr = librosa.load(filepath)
    if (len(y) < 2 * sr):
        hop = int(sr/10)
        tmp = np.append(y,[0]*hop)
        y = np.append(tmp,y)

    y = np.pad(y, ( int(( (10 * sr - len(y)) /2)) , int(( (10 * sr - len(y)) / 2 ) ) ) , mode = 'constant')
    cropped = cropping (y,sr,2)
    if len(cropped) == 0: return 0
        
    mel = librosa.feature.melspectrogram(y = cropped, sr = sr,n_mels = 128)
    log_mel = librosa.power_to_db(mel, ref = np.max)
    H,P = librosa.decompose.hpss(mel)
    log_H = librosa.power_to_db(H, ref = np.max)
    log_P = librosa.power_to_db(P, ref = np.max)
    
    filename = fname.strip('.wav')

    if (newpath == 'train'):
        new_filepath_mel = '/data/$user/repos/dataset/unified_spect/trainset/'+ filename +'.tif' # your path
        new_filepath_H = '/data/$user/repos/dataset/unified_spect/trainset/H_'+ filename +'.tif' # your path
        new_filepath_P = '/data/$user/repos/dataset/unified_spect/trainset/P_'+ filename +'.tif' # your path
    else:
        new_filepath_mel = '/data/$user/repos/dataset/unified_spect/validset/'+ filename+'.tif' # your path
        new_filepath_H = '/data/$user/repos/dataset/unified_spect/validset/H_'+ filename +'.tif' # your path
        new_filepath_P = '/data/$user/repos/dataset/unified_spect/validset/P_'+ filename +'.tif' # your path
        

    fig = plt.figure(figsize = (1.13,1.67))
    librosa.display.specshow(log_mel, sr=sr)
     
    fig.savefig(fname = new_filepath_mel, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

    fig = plt.figure(figsize = (1.13,1.67))
    librosa.display.specshow(log_H, sr=sr)
    fig.savefig(fname = new_filepath_H, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)


    fig = plt.figure(figsize = (1.13,1.67))
    librosa.display.specshow(log_P, sr=sr)
    fig.savefig(fname = new_filepath_P, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

# main loop
train_path = '../dataset/unified_audio/unified_train'
valid_path = '../dataset/unified_audio/unified_valid'

'''
for filename in os.listdir(train_path):
    filepath = os.path.join(train_path,filename)
    spectrogram_image(filename, filepath, 'train')
'''

for filename in os.listdir(valid_path):
    filepath = os.path.join(valid_path,filename)
    spectrogram_image(filename, filepath, 'valid')

print ('execution terminated')

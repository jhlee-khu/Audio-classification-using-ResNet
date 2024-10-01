# convert audio to spectrogram(img file)
# place this py file into the ./Urbansound8K directory

import csv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def split_y_n_pad(y,patchsize):
    # split y to multiple array (patches) which length == patchsize
    # if patch size is smaller than var(patchsize) => zero pad
    num_patches = int(np.ceil(len(y) / patchsize))
    padded = np.pad (y, (0, patchsize * num_patches - len(y)),'constant')
    patches = np.reshape (padded, (-1,patchsize))

    return patches

def spectrogram_image(filename,fold,filepath,class_num):
    y, sr = librosa.load(filepath)

    patchsize = int(sr * 640/1000)
    patches = split_y_n_pad(y,patchsize)
    patch_idx = len(patches)

    hop = int(np.round(patchsize / 64)) 
    window_size = int(np.round(2.5 * hop)) # window size will be 25 ms, and hop 10 ms

    annot_file_path = './annotation.csv'
    f = open (annot_file_path,'a',newline = '')
    wr = csv.writer(f)

    for i in range(0,patch_idx):
        stft = librosa.stft(patches[i], n_fft = window_size, hop_length = hop)
        mel = librosa.filters.mel(sr = sr, n_fft = window_size, n_mels = 64)
        power = np.abs(stft) ** 2
        mel_spec = np.dot(mel, power)
        fig = plt.figure(figsize = (64/77,64/77))
        librosa.display.specshow(mel_spec, sr=sr, hop_length = hop)

        # fold 1~7 => training set
        # fold 8,9 => validation set
        # fold 10 => testset
        if (fold  <= 7):
            new_filepath = './training/' + str(i) + '-' + filename+'.png'
        elif (fold == 8 or fold == 9):
            new_filepath = './validation/' + str(i) + '-' + filename+'.png'
        else:
            new_filepath = './test/' + str(i) + '-' + filename+'.png'
        
        fig.savefig(fname = new_filepath, bbox_inches = 'tight', pad_inches = 0)

        #write annotation file
        row = [str(i) + '-' + filename+'.png',class_num]
        wr.writerow(row)
        
        plt.close()
    f.close()
    


# main loop
metadata = open('./metadata/UrbanSound8K.csv','r')
line = csv.reader(metadata)

itr = -1

for ln in line:
    itr += 1
    if itr == 0: continue
    filename = ln[0]
    fold = ln[5]
    class_num = ln[6]
    filepath = './audio/fold' + fold + '/' + filename
    spectrogram_image(filename,int(fold),filepath,class_num)

''' 
#test 
    if itr == 5:
        break
'''
# convert audio to spectrogram(img file)
# place this py file into the ./Urbansound8K directory
# urbansound8k.csv file must also be in the same directory (./Urbansound8K )

import csv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def spectrogram_image(filename,fold,filepath):
    y, sr = librosa.load(filepath)

    # right zero padded if audio length is less than 5 s.
    # As urbansound 8k dataset audios are 0~4 s
    # all spectrograms has been zero padded.
    y = np.pad(y,(0,sr * 5 - len(y)),'constant',constant_values = 0) 

    hop = int(np.round(sr/32)) 
    window_size = 2 * hop # window size will be 1/16 s, and hop 1/32 s

    stft = librosa.stft(y, n_fft = window_size, hop_length = hop)
    mel = librosa.filters.mel(sr = sr, n_fft = window_size, n_mels = 128)
    power = np.abs(stft) ** 2
    mel_spec = np.dot(mel, power)
    
    fig = plt.figure(figsize = (4,4))
    librosa.display.specshow(mel_spec, sr=sr, hop_length = hop)
    new_filepath = './convfold'+fold+'/'+filename+'.png'
    fig.savefig(fname = new_filepath, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    
# main loop
metadata = open('UrbanSound8K.csv','r')
line = csv.reader(metadata)

itr = -1

for ln in line:
    itr += 1
    if itr == 0: continue
    filename = ln[0]
    fold = ln[5]
    filepath = './audio/fold' + fold + '/' + filename
    spectrogram_image(filename,fold,filepath)
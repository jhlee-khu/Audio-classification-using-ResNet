# cell 1
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import argparse
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image


import sys
sys.path.append('/data/$USER/repos/khu_cse_capstone13_24_2') # your path
from resnet import *

# cell 2
parser = argparse.ArgumentParser(description='PyTorch Spectrogram training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args(args = [])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

best_acc = 0
start_epoch = 0

# cell 3
# user define dataset class

class audio_spectrogram(Dataset):
    def __init__(self,annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file, names =['file_name' , 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        spect_path = self.img_dir + '/' +self.img_labels.iloc[idx, 0]
        spect = img = Image.open(spect_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]

        if self.transform: spect = self.transform(spect)
        if self.target_transform: label = self.target_transform(label)

        return spect,label

# cell 4

# label map and allocate train and test set pathes
# initialize

labels = ("air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music")

trainannot_path = '/data/$USER/repos/khu_cse_capstone13_24_2/us8k_train.csv' # your path
validationannot_path = '/data/$USER/repos/khu_cse_capstone13_24_2/us8k_valid.csv' # your path

train_spect_path = '/local_datasets/us8k_train' # your path
validation_spect_path = '/local_datasets/us8k_valid' # your path

trainset = audio_spectrogram(trainannot_path, train_spect_path,transform = transforms.ToTensor())
validationset = audio_spectrogram(validationannot_path, validation_spect_path,transform = transforms.ToTensor())
# cell 5
#data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(validationset, batch_size=32, shuffle=False, num_workers=2)

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
#if True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/data/$USER/repos/khu_cse_capstone13_24_2/checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# cell 6
#model setting

train_acc = np.array([]).astype(np.float32)
test_acc = np.array([]).astype(np.float32)

from torch.optim.lr_scheduler import StepLR

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay = 0.0002)
scheduler = StepLR(optimizer, step_size=10, gamma=0.6)

# cell 7
# Training
def train(epoch):
    global train_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx == len(trainloader) - 1):
            train_acc = np.append(train_acc,100.*correct/total)
            print('train %d epoch, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# Test & validation
def test(epoch):
    global best_acc, test_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx == len(testloader) - 1):
                test_acc = np.append(test_acc,100.*correct/total)
                print('test %d epoch, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '/data/$USER/repos/khu_cse_capstone13_24_2/checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch + 80):
    train(epoch)
    test(epoch)
    scheduler.step()

np.save('/data/$USER/repos/khu_cse_capstone13_24_2/trainacc.npy', train_acc)
np.save('/data/$USER/repos/khu_cse_capstone13_24_2/testacc.npy', test_acc)
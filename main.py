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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

import sys
sys.path.append('/data/karna/repos/khu_cse_capstone13_24_2') # your path
from resnet import *
from TF_ResNet import *

# cell 2
parser = argparse.ArgumentParser(description='PyTorch Spectrogram training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
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
        spect = Image.open(spect_path).convert('RGB')
        
        pth = self.img_labels.iloc[idx, 0].split('/')

        H_spect_path = self.img_dir + '/' + pth[0] + '/H_' + pth[1] 
        H_spect = Image.open(H_spect_path).convert('RGB')

        P_spect_path = self.img_dir + '/' + pth[0] + '/P_' + pth[1] 
        P_spect = Image.open(P_spect_path).convert('RGB')

        if self.transform: 
            spect = self.transform(spect)
            H_spect = self.transform(H_spect)
            P_spect = self.transform(P_spect)
        label = int(self.img_labels.iloc[idx, 1])
        if self.target_transform: label = self.target_transform(label)

        return torch.stack([spect, H_spect, P_spect],dim = 0) ,label

# cell 4

# label map and allocate train and test set pathes
# initialize

#fold_num
foldnum = '3_'

trainannot_path = '/local_datasets/unified_spect_folds/' + foldnum + 'train_annot.csv'
validationannot_path = '/local_datasets/unified_spect_folds/' + foldnum + 'valid_annot.csv' 

train_spect_path = '/local_datasets/unified_spect_folds/' 
validation_spect_path = '/local_datasets/unified_spect_folds/'

trainset = audio_spectrogram(trainannot_path, train_spect_path,transform = transforms.ToTensor())
validationset = audio_spectrogram(validationannot_path, validation_spect_path,transform = transforms.ToTensor())

# cell 5
#data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(validationset, batch_size=64, shuffle=False, num_workers=2)


model_name = 'ResNet34_fold3'
# net = TF_ResNet34()
net = ResNet34(2)
net = net.to(device)

train_acc = np.array([]).astype(np.float32)
test_acc = np.array([]).astype(np.float32)
train_loss_arr = np.array([])
classacc = np.zeros(11)
test_loss_arr = np.array([])

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
# if True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/data/karna/repos/khu_cse_capstone13_24_2/checkpoint/' + model_name +'_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    train_acc = np.load('/data/karna/repos/khu_cse_capstone13_24_2/accdata/trainacc_'+ model_name + '.npy')
    test_acc = np.load('/data/karna/repos/khu_cse_capstone13_24_2/accdata/testacc_' + model_name + '.npy')
    classacc = np.load('/data/karna/repos/khu_cse_capstone13_24_2/accdata/classacc_'+ model_name  + '.npy')
    train_loss_arr = np.load('/data/karna/repos/khu_cse_capstone13_24_2/accdata/train_loss_'+ model_name  + '.npy')
    test_loss_arr = np.load('/data/karna/repos/khu_cse_capstone13_24_2/accdata/test_loss_' + model_name + '.npy')


# cell 6
#model setting

from torch.optim.lr_scheduler import StepLR

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)



# cell 7
# Training
def train(epoch):
    global train_acc,train_loss_arr
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
            train_loss_arr = np.append(train_loss_arr, train_loss/(batch_idx+1))
            print('train %d epoch, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

best_class_predicted = np.zeros((11,11))

# Test & validation
def test(epoch):
    class_predicted = np.zeros((11,11))
    global best_acc, test_acc,string_sec, classacc, test_loss_arr, model_name, best_class_predicted
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_classes = 11

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
                class_predicted[label][predicted[i]] += 1

            if (batch_idx == len(testloader) - 1):
                test_acc = np.append(test_acc, 100.*correct/total)
                test_loss_arr = np.append(test_loss_arr, test_loss/(batch_idx+1))
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

        for i in range(num_classes):
            classacc[i] = 100. * class_correct[i] / class_total[i]
        best_class_predicted = class_predicted.copy()

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '/data/karna/repos/khu_cse_capstone13_24_2/checkpoint/' + model_name +'_ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 80):
    train(epoch)
    test(epoch)

np.save('/data/karna/repos/khu_cse_capstone13_24_2/accdata/trainacc_'+ model_name  + '.npy', train_acc)
np.save('/data/karna/repos/khu_cse_capstone13_24_2/accdata/testacc_' + model_name + '.npy', test_acc)

np.save('/data/karna/repos/khu_cse_capstone13_24_2/accdata/train_loss_' + model_name + '.npy', train_loss_arr)
np.save('/data/karna/repos/khu_cse_capstone13_24_2/accdata/test_loss_' + model_name + '.npy', test_loss_arr)
np.save('/data/karna/repos/khu_cse_capstone13_24_2/accdata/class_predicted_' + model_name + '.npy', best_class_predicted)

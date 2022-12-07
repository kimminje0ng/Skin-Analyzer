# python EfficientNet_color_loss.py

import sys
import time
import datetime
import os
import gc
import copy
import cv2
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image, ImageFile    # 수정
from efficientnet_pytorch import EfficientNet
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 수정

# 병렬으로 수정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyper_param_batch = 16 # gpu 수만큼 할당

random_seed = 200   # 원래 100
random.seed(random_seed)
torch.manual_seed(random_seed)

num_classes = 4
model_name = 'efficientnet-b7'

train_name = 'model6_preprocessed'

PATH = './work_dir/color_loss_b16_e300_s200/'    # output 저장

data_train_path = './'+train_name+'/train'
data_validation_path = './'+train_name+'/validation'
data_test_path = './'+train_name+'/test'

image_size = EfficientNet.get_image_size(model_name)
print(image_size)

model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
if torch.cuda.device_count() > 1:   # 병렬
    model = nn.DataParallel(model)
model = model.to(device)
gc.collect()
torch.cuda.empty_cache()

def lambda_func(x):
    return x.rotate(90)

transforms_train = transforms.Compose([
                                        transforms.Resize([int(600), int(600)]),    # 수정
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.Lambda(lambda_func),    # 수정
                                        transforms.RandomRotation(10),
                                        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), #수정
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

transforms_val = transforms.Compose([
                                        transforms.Resize([int(600), int(600)], interpolation=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.Lambda(lambda_func),    # 수정
                                        transforms.RandomRotation(10),
                                        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), #수정
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])


train_data_set = datasets.ImageFolder(data_train_path, transform=transforms_train)
val_data_set = datasets.ImageFolder(data_validation_path, transform=transforms_val)

dataloaders, batch_num = {}, {}

dataloaders['train'] = DataLoader(train_data_set,
                                    batch_size=hyper_param_batch,
                                    shuffle=True,
                                    num_workers=8)  # 수정
dataloaders['val'] = DataLoader(val_data_set,
                                    batch_size=hyper_param_batch,
                                    shuffle=False,
                                    num_workers=8)  # 수정


batch_num['train'], batch_num['val'] = len(train_data_set), len(val_data_set)

print('batch_size : %d,  train/val : %d / %d' % (hyper_param_batch, batch_num['train'], batch_num['val']))

class_names = train_data_set.classes
print(class_names)

# loss function for balanced dataset
train_dict = dict(Counter(train_data_set.targets))

total_num_train = train_dict[0] + train_dict[1] + train_dict[2] + train_dict[3]
class_weight = [1 - train_dict[0]/total_num_train, 1 - train_dict[1]/total_num_train, 1 - train_dict[2]/total_num_train, 1 - train_dict[3]/total_num_train]

class_weight = torch.tensor(class_weight)
#print(torch.tensor(class_weight))

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()
    
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        epoch_start = time.time()
        print("start\n")
        for phase in ['train', 'val']:
            if phase == 'train':
                print("train working..\n") 
                model.train()
            else:
                print("eval working..\n")
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_cnt = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))
                
            epoch_end = time.time() - epoch_start
            
            print('Training epochs {} in {:.0f}m {:.0f}s'.format(epoch, epoch_end // 60, epoch_end % 60))
            print()
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))

    model.load_state_dict(best_model_wts)
        
    torch.save(model, PATH + 'aram_' + train_name+'.pt')
    torch.save(model.state_dict(), PATH + 'president_aram_'+train_name+'.pt')
    print('model saved')

    end_sec = time.time() - start_time
    end_times = str(datetime.timedelta(seconds=end_sec)).split('.')
    end_time = end_times[0]
    print("end time :", end_time)
    
    return model, best_idx, best_acc, train_loss, train_acc, val_loss, val_acc

if __name__ == "__main__":
    sys.stdout = open('/home/irteam/Skin-Analyzer/aihub_Efficient/work_dir/color_loss_b16_e300_s200/log2.txt', 'w')

    criterion = nn.CrossEntropyLoss(weight = class_weight).cuda()

    optimizer_ft = optim.Adam(model.parameters(),lr = 1e-4, weight_decay=1e-4)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    num_epochs = 300
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    sys.stdout.close()


# train
from func.load_dataset import airway_dataset
# from func.unet_3d_basic import UNet3D_basic
from func.model_arch import SegAirwayModel
from func.loss_func import dice_loss_weights, dice_accuracy, dice_loss_power_weights, dice_loss, dice_loss_power
from func.ulti import load_obj

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import time
import gc

# settings
# ----------
save_path = 'checkpoint/checkpoint_sample.pkl'
need_resume = True
load_path = "" #'checkpoint/checkpoint.pkl'
path_dataset_info_more_focus_on_airways_of_low_generation = \
    "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_more_low_generation_10"
path_dataset_info_more_focus_on_airways_of_high_generation = \
    "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_more_high_generation_1"
learning_rate = 1e-5
max_epoch = 50
freq_switch_of_train_mode_high_low_generation = 1
num_samples_of_each_epoch = 2000
batch_size = 4
train_file_format = '.npy'
crop_size = (32, 128, 128)
windowMin_CT_img_HU = -1000 # min of CT image HU value
windowMax_CT_img_HU = 600 # max of CT image HU value
model_save_freq = 1
num_workers = 4
# ----------

# init model
model=SegAirwayModel(in_channels=1, out_channels=2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

if need_resume and os.path.exists(load_path):
    print("resume model from "+str(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# optimizer
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataset
dataset_info_more_focus_on_airways_of_low_generation = load_obj(path_dataset_info_more_focus_on_airways_of_low_generation)
train_dataset_more_focus_on_airways_of_low_generation = airway_dataset(dataset_info_more_focus_on_airways_of_low_generation)
train_dataset_more_focus_on_airways_of_low_generation.set_para(file_format=train_file_format, crop_size=crop_size,\
    windowMin=windowMin_CT_img_HU, windowMax=windowMax_CT_img_HU, need_tensor_output=True, need_transform=True)

dataset_info_more_focus_on_airways_of_high_generation = load_obj(path_dataset_info_more_focus_on_airways_of_high_generation)
train_dataset_more_focus_on_airways_of_high_generation = airway_dataset(dataset_info_more_focus_on_airways_of_high_generation)
train_dataset_more_focus_on_airways_of_high_generation.set_para(file_format=train_file_format, crop_size=crop_size,\
    windowMin=windowMin_CT_img_HU, windowMax=windowMax_CT_img_HU, need_tensor_output=True, need_transform=True)

print('total epoch: '+str(max_epoch))

start_time = time.time()

for ith_epoch in range(0, max_epoch):
    
    if np.floor(ith_epoch/freq_switch_of_train_mode_high_low_generation)%2==0 or \
        ith_epoch>=(max_epoch-freq_switch_of_train_mode_high_low_generation):
        print("train_more_focus_on_airways_of_low_generation")
        sampler_of_airways_of_low_generation = RandomSampler(train_dataset_more_focus_on_airways_of_low_generation,\
            num_samples = min(num_samples_of_each_epoch, len(dataset_info_more_focus_on_airways_of_low_generation)), replacement = True)
        dataset_loader = DataLoader(train_dataset_more_focus_on_airways_of_low_generation,\
            batch_size=batch_size, sampler = sampler_of_airways_of_low_generation, num_workers=num_workers,\
                pin_memory=True, persistent_workers=(num_workers > 1))
    
    else:
        print("train_more_focus_on_airways_of_high_generation")
        sampler_of_airways_of_high_generation = RandomSampler(train_dataset_more_focus_on_airways_of_high_generation,\
            num_samples = min(num_samples_of_each_epoch, len(dataset_info_more_focus_on_airways_of_high_generation)), replacement = True)
        dataset_loader = DataLoader(train_dataset_more_focus_on_airways_of_high_generation,\
            batch_size=batch_size, sampler = sampler_of_airways_of_high_generation, num_workers=num_workers,\
                pin_memory=True, persistent_workers=(num_workers > 1))
    
    len_dataset_loader = len(dataset_loader)

    for ith_batch, batch in enumerate(dataset_loader):
        img_input=batch['image'].float().to(device)
        
        groundtruth_foreground=batch['label'].float().to(device)
        groundtruth_background=1-groundtruth_foreground
    
        fore_pix_num = torch.sum(groundtruth_foreground)
        back_pix_num = torch.sum(groundtruth_background)
        fore_pix_per = fore_pix_num/(fore_pix_num+back_pix_num)
        back_pix_per = back_pix_num/(fore_pix_num+back_pix_num)
        weights = (torch.exp(back_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))*torch.eq(groundtruth_foreground,1).float()+\
            torch.exp(fore_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))*torch.eq(groundtruth_foreground,0).float()).to(device)
        
        img_output=model(img_input)
        
        loss=dice_loss_weights(img_output[:,0,:,:,:], groundtruth_background, weights)+\
            dice_loss_power_weights(img_output[:,1,:,:,:], groundtruth_foreground, weights, alpha=2)
        accuracy=dice_accuracy(img_output[:,1,:,:,:], groundtruth_foreground)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_consumption = time.time() - start_time

        print(
            "epoch [{0}/{1}]\t"
            "batch [{2}/{3}]\t"
            "time(sec) {time:.2f}\t"
            "loss {loss:.4f}\t"
            "acc {acc:.2f}%\t"
            "fore pix {fore_pix_percentage:.2f}%\t"
            "back pix {back_pix_percentage:.2f}%\t".format(
                ith_epoch + 1,
                max_epoch,
                ith_batch,
                len_dataset_loader,
                time = time_consumption,
                loss = loss.item(),
                acc = accuracy.item()*100,
                fore_pix_percentage = fore_pix_per*100,
                back_pix_percentage = back_pix_per*100))
    
    del dataset_loader
    gc.collect()

    if (ith_epoch+1)%model_save_freq==0:
        print('epoch: '+str(ith_epoch+1)+' save model')
        model.to(torch.device('cpu'))
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        model.to(device)

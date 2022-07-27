import numpy as np
import torch
from torch import from_numpy as from_numpy
import os
import skimage.io as io

# how to run model
"""
raw_img, label = get_image_and_label(image_path, label_path)
raw_img, label = get_crop_of_image_and_label_within_the_range_of_airway_foreground(raw_img,label)
seg_result = semantic_segment_crop_and_cat(raw_img, model, device, crop_cube_size=[32, 128, 128], stride=[16, 64, 64])
seg_onehot = np.array(seg_result>threshold, dtype=np.int)
"""

def get_image_and_label(image_path, label_path):
    raw_img = io.imread(image_path, plugin='simpleitk')
    raw_img = np.array(raw_img, dtype=np.float)
    label = io.imread(label_path, plugin='simpleitk')
    label = np.array(label, dtype=np.float)
    
    return raw_img,label

def get_crop_of_image_and_label_within_the_range_of_airway_foreground(raw_img,label):
    locs = np.where(label>0)
    x_min = np.min(locs[0])
    x_max = np.max(locs[0])
    y_min = np.min(locs[1])
    y_max = np.max(locs[1])
    z_min = np.min(locs[2])
    z_max = np.max(locs[2])
    
    return raw_img[x_min:x_max, y_min:y_max, z_min:z_max], label[x_min:x_max, y_min:y_max, z_min:z_max]

class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin
    
    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm-=np.min(img_3d_norm)
        max_99_val=np.percentile(img_3d_norm, 99)
        if max_99_val>0:
            img_3d_norm = img_3d_norm/max_99_val*255
        
        return img_3d_norm

def semantic_segment_crop_and_cat(raw_img, model, device, crop_cube_size=256, stride=256, windowMin=-1000, windowMax=600):
    
    normalization=Normalization_np(windowMin, windowMax)
    raw_img = normalization(raw_img)
    
    # raw_img: 3d matrix, numpy.array
    assert isinstance(crop_cube_size, (int, list))
    if isinstance(crop_cube_size, int):
        crop_cube_size=np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size)==3
    
    assert isinstance(stride, (int, list))
    if isinstance(stride, int):
        stride=np.array([stride, stride, stride])
    else:
        assert len(stride)==3
    
    for i in [0,1,2]:
        while crop_cube_size[i]>raw_img.shape[i]:
            crop_cube_size[i]=int(crop_cube_size[i]/2)
            stride[i]=crop_cube_size[i]
    
    img_shape=raw_img.shape
    
    seg=np.zeros(img_shape)
    seg_log=np.zeros(img_shape) # 0 means this pixel has not been segmented, 1 means this pixel has been
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))\
    *len(np.arange(0, img_shape[2], stride[2]))
    count=0
    
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('Progress of segment_3d_img: '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start_input=i
                    x_end_input=i+crop_cube_size[0]
                else:
                    x_start_input=img_shape[0]-crop_cube_size[0]
                    x_end_input=img_shape[0]
                
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start_input=j
                    y_end_input=j+crop_cube_size[1]
                else:
                    y_start_input=img_shape[1]-crop_cube_size[1]
                    y_end_input=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start_input=k
                    z_end_input=k+crop_cube_size[2]
                else:
                    z_start_input=img_shape[2]-crop_cube_size[2]
                    z_end_input=img_shape[2]
                
                raw_img_crop=raw_img[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                raw_img_crop=normalization(raw_img_crop)
                
                seg_log_crop=seg_log[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                seg_crop=seg[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                
                try:
                    raw_img_crop=raw_img_crop.reshape(1, 1, crop_cube_size[0], crop_cube_size[1], crop_cube_size[2])
                except:
                    print("raw_img_crop shape: "+str(raw_img_crop.shape))
                    print("raw_img shape: "+str(raw_img.shape))
                    print("i, j, k: "+str((i,j,k)))
                    print("crop from: "+str((x_start_input, x_end_input, y_start_input, y_end_input, z_start_input, z_end_input)))
                raw_img_crop=from_numpy(raw_img_crop).float().to(device)
                
                with torch.no_grad():
                    seg_crop_output=model(raw_img_crop)
                seg_crop_output_np=seg_crop_output[:,1,:,:,:].cpu().detach().numpy()
                seg_crop_output_np = seg_crop_output_np[0,:,:,:]
                
                seg_temp=np.zeros(seg_crop.shape)
                seg_temp[seg_log_crop==1]=(seg_crop_output_np[seg_log_crop==1]+seg_crop[seg_log_crop==1])/2#np.multiply(seg_crop_output_np[seg_log_crop==1],seg_crop[seg_log_crop==1])
                seg_temp[seg_log_crop==0]=seg_crop_output_np[seg_log_crop==0]
                
                seg[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]=\
                seg_temp
                seg_log[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]=1
                
                count=count+1
                
    return seg

def dice_accuracy(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.flatten()
    tflat = target.flatten()
    
    intersection = 2. * np.sum(np.multiply(iflat, tflat))

    A_sum = np.sum(np.multiply(iflat, iflat))
    B_sum = np.sum(np.multiply(tflat, tflat))
    
    return (intersection) / (A_sum + B_sum + 0.0001)
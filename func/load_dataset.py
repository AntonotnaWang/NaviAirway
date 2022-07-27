# dataset load and transform
import numpy as np
import skimage.io as io
import random
import h5py

from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio

class Random3DCrop_np(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
        if isinstance(output_size, int):
            self.output_size=(output_size, output_size, output_size)
        else:
            assert len(output_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
            self.output_size=output_size
        
    def random_crop_start_point(self, input_size):
        assert len(input_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
        d, h, w=input_size
        d_new, h_new, w_new=self.output_size
        
        d_new = min(d, d_new)
        h_new = min(h, h_new)
        w_new = min(w, w_new)
        
        assert (d>=d_new and h>=h_new and w>=w_new), "Attention: input size should >= crop size; now, input_size is "+str((d,h,w))+", while output_size is "+str((d_new, h_new, w_new))
        
        d_start=np.random.randint(0, d-d_new+1)
        h_start=np.random.randint(0, h-h_new+1)
        w_start=np.random.randint(0, w-w_new+1)
        
        return d_start, h_start, w_start
    
    def __call__(self, img_3d, start_points=None):
        img_3d=np.array(img_3d)
        
        d, h, w=img_3d.shape
        d_new, h_new, w_new=self.output_size
        
        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)
        
        d_start, h_start, w_start = start_points
        d_end = min(d_start+d_new, d)
        h_end = min(h_start+h_new, h)
        w_end = min(w_start+w_new, w)
        
        crop=img_3d[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return crop

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


class airway_dataset(Dataset):
    def __init__(self, data_dict, num_of_samples = None):
        # each item of data_dict is {name:{"image": img path, "label": label path}}
        self.data_dict = data_dict
        if num_of_samples is not None:
            num_of_samples = min(len(data_dict), num_of_samples)
            chosen_names = np.random.choice(np.array(list(data_dict)), num_of_samples, replace=False)
        else:
            chosen_names = np.array(list(data_dict))
        self.name_list = chosen_names
        self.para = {}
        self.set_para()
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        return self.get(idx, file_format=self.para["file_format"],\
            crop_size=self.para["crop_size"],\
                windowMin=self.para["windowMin"],\
                    windowMax=self.para["windowMax"],\
                        need_tensor_output=self.para["need_tensor_output"],\
                            need_transform=self.para["need_transform"])

    def set_para(self, file_format='.npy', crop_size=32, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True):
        self.para["file_format"] = file_format
        self.para["crop_size"] = crop_size
        self.para["windowMin"] = windowMin
        self.para["windowMax"] = windowMax
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform

    def get(self, idx, file_format='.npy', crop_size=32, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True):

        random3dcrop=Random3DCrop_np(crop_size)
        normalization=Normalization_np(windowMin, windowMax)

        name = self.name_list[idx]

        if file_format == ".npy":
            raw_img = np.load(self.data_dict[name]["image"])
            label_img = np.load(self.data_dict[name]["label"])
        elif file_format == '.nii.gz':
            raw_img = io.imread(self.data_dict[name]["image"], plugin='simpleitk')
            label_img = io.imread(self.data_dict[name]["label"], plugin='simpleitk')
        elif file_format == ".h5":
            hf = h5py.File(self.data_dict[name]["path"], 'r+')
            raw_img = np.array(hf["image"])
            label_img = np.array(hf["label"])
            hf.close()
        
        assert raw_img.shape == label_img.shape
            
        start_points=random3dcrop.random_crop_start_point(raw_img.shape)
        raw_img_crop=random3dcrop(np.array(raw_img, float), start_points=start_points)
        label_img_crop=random3dcrop(np.array(label_img, float), start_points=start_points)
        raw_img_crop=normalization(raw_img_crop)

        raw_img_crop = np.expand_dims(raw_img_crop, axis=0)
        label_img_crop = np.expand_dims(label_img_crop, axis=0)

        output = {"image": raw_img_crop, "label": label_img_crop}

        if need_tensor_output:
            output = self.to_tensor(output)
            if need_transform:
                output = self.transform_the_tensor(output, prob=0.5)

        return output

    def to_tensor(self, images):
        for item in images.keys():
            images[item]=from_numpy(images[item]).float()
        return images
    
    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio={}
        
        for item in image_tensors.keys():
            dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose([
            tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=prob),tio.RandomAffine(p=prob)])
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        transform_val = tio.Compose([
            tio.RandomBlur(p=prob),\
                tio.RandomNoise(p=prob),\
                    tio.RandomMotion(p=prob),\
                        tio.RandomBiasField(p=prob),\
                            tio.RandomSpike(p=prob),\
                                tio.RandomGhosting(p=prob)])
        transformed_subject_all_imgs['image'] = transform_val(transformed_subject_all_imgs['image'])
        
        for item in subject_all_imgs.keys():
            image_tensors[item] = transformed_subject_all_imgs[item].data
        
        return image_tensors

if __name__=="__main__":
    import pickle
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    data_dict = load_obj("../dataset_info/train_test_set_dict_EXACT09_LIDC_IDRI_128_set_1_extended_more_big_10")
    dataset = airway_dataset(data_dict)
    output = dataset.get(0)
    for name in output.keys():
        print(name, output[name].shape)
    num_workers = 1
    dataset.set_para(file_format='.npy', crop_size=64, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True)
    Dataset_loader = DataLoader(dataset, batch_size=3, shuffle=True, \
        num_workers=num_workers, pin_memory=False, persistent_workers=False)
    batch = next(iter(Dataset_loader))
    for name in batch.keys():
        print(name, batch[name].shape)
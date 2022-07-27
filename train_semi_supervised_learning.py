# train semi supervised
from func.load_dataset import airway_dataset
from func.loss_func import dice_loss_weights, dice_accuracy, dice_loss_power_weights, dice_loss, dice_loss_power
from func.ulti import load_obj, crop_one_3d_img, load_one_CT_img
from func.semi_supervise_learning import get_the_whole_img_set_dict, init_the_teacher_model,\
    use_teacher_model_to_process_one_img_generating_study_materials,\
        save_the_study_materials_for_one_img,\
            use_teacher_model_to_process_one_list_of_img_and_save_study_materials,\
                get_balanced_data_dict, get_data_dict_of_current_unlabled_image_and_pseudolabels
from func.model_arch import SegAirwayModel

import numpy as np
import torch
import os
import h5py
import shutil
import time
import torch
from torch.utils.data import DataLoader, RandomSampler

# settings
# ----------
load_path_teacher = "checkpoint/checkpoint.pkl"
load_path_student = "checkpoint/checkpoint.pkl"
save_path_student_model = "checkpoint/checkpoint_semi_supervise_learning_sample.pkl"
unlabel_data_path = "/data/Airway/LIDC-IDRI_3D/raw_data" #"/data/Airway/QMH_airway_data/Unlabeled_3d_data"
unlabel_data_indicator = "LIDC_IDRI_"
path_dataset_info_more_focus_on_airways_of_low_generation = \
    "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_more_low_generation_10"
path_dataset_info_more_focus_on_airways_of_high_generation = \
    "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_more_high_generation_1"
file_path_of_study_materials = "/data/Airway/QMH_airway_data/temp"

crop_cube_size_for_study_material_generation = [32, 128, 128]
stride_for_study_material_generation = [16, 64, 64]
crop_cube_size_for_study_material_saving = (128, 128, 128)
stride_for_study_material_saving = (64,64,64)

max_iteration_teacher_student_training = 5 # iteration of teacher-student training
num_of_imgs_used_for_each_tea_stu_iteration = 22
max_epoch = 2
num_samples_of_dataloader = 1000

learning_rate = 1e-5
batch_size=2
crop_size_for_train=(32, 128, 128)
windowMin_CT_img_HU = -1000 # min of CT image HU value
windowMax_CT_img_HU = 600 # max of CT image HU value
labeled_train_file_format = '.npy'
unlabeled_train_file_format = ".h5"
model_save_freq = 1
num_workers = 4

device_for_teacher = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_for_student = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# ----------

# init the teacher model
teacher_model = init_the_teacher_model(SegAirwayModel,\
    in_channel=1, out_channel=2,\
        load_path = load_path_teacher, strict=False)
teacher_model.to(device_for_teacher)

# init the student model
student_model=SegAirwayModel(in_channels=1, out_channels=2)
checkpoint = torch.load(load_path_student)
student_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
student_model.to(device_for_student)

# get the dict of unlabeled images
unlabeled_img_set_dict = get_the_whole_img_set_dict(unlabel_data_path, indicator = unlabel_data_indicator)

# optimizer setting 
optimizer=torch.optim.Adam(student_model.parameters(), lr=learning_rate)

# datasets of labeled imgs 
labeled_dataset_info_more_focus_on_airways_of_low_generation = load_obj(path_dataset_info_more_focus_on_airways_of_low_generation)
labeled_train_dataset_low_gen = airway_dataset(labeled_dataset_info_more_focus_on_airways_of_low_generation)
labeled_train_dataset_low_gen.set_para(file_format=labeled_train_file_format, crop_size=crop_size_for_train,\
    windowMin=windowMin_CT_img_HU, windowMax=windowMax_CT_img_HU, need_tensor_output=True, need_transform=True)

dataset_info_more_focus_on_airways_of_high_generation = load_obj(path_dataset_info_more_focus_on_airways_of_high_generation)
labeled_train_dataset_high_gen = airway_dataset(dataset_info_more_focus_on_airways_of_high_generation)
labeled_train_dataset_high_gen.set_para(file_format=labeled_train_file_format, crop_size=crop_size_for_train,\
    windowMin=windowMin_CT_img_HU, windowMax=windowMax_CT_img_HU, need_tensor_output=True, need_transform=True)

#deleting the file_path_of_study_materials
try:
    shutil.rmtree(file_path_of_study_materials)
except:
    print("no such filefolder")
    pass

start_time = time.time()
for ith_tea_stu_iteration in range(max_iteration_teacher_student_training):
    
    # under ith_iteration, devide the unlabeled images to smaller batches, and train on each batch
    pic_name_list = np.array(list(unlabeled_img_set_dict.keys()))
    np.random.shuffle(pic_name_list)
    pic_name_batch_list = []
    start_loc = 0
    while start_loc < len(pic_name_list):
        if start_loc+num_of_imgs_used_for_each_tea_stu_iteration<=len(pic_name_list):
            pic_name_batch_list.append(pic_name_list[start_loc:start_loc+num_of_imgs_used_for_each_tea_stu_iteration])
        else:
            pic_name_batch_list.append(pic_name_list[start_loc:len(pic_name_list)])
        start_loc+=num_of_imgs_used_for_each_tea_stu_iteration
    
    # so, we get pic_name_batch_list, let's do it one by one
    for ith_pic_name_batch, pic_name_batch in enumerate(pic_name_batch_list):
        # use teacher model generating pseudolabels for this batch, and getting the data dict
        data_dict_of_current_unlabled_image_and_pseudolabels = \
        use_teacher_model_to_process_one_list_of_img_and_save_study_materials(pic_name_batch, unlabeled_img_set_dict,
                                                                              teacher_model,
                                                                              device = device_for_teacher,
                                                                              file_path_of_study_materials = file_path_of_study_materials,
                                                                              crop_cube_size_for_study_material_generation = crop_cube_size_for_study_material_generation,
                                                                              stride_for_study_material_generation = stride_for_study_material_generation,
                                                                              crop_cube_size_for_study_material_saving = crop_cube_size_for_study_material_saving,
                                                                              stride_for_study_material_saving = stride_for_study_material_saving,
                                                                              min_crop_cube_size=crop_cube_size_for_study_material_generation)

        # extend the data dict
        # datasets of unlabeled imgs 
        unlabeled_datset_info_more_focus_on_airways_of_low_generation = \
        get_balanced_data_dict(data_dict_of_current_unlabled_image_and_pseudolabels, is_more_big = True, copy_times_I = 10)
        unlabeled_train_dataset_low_gen = airway_dataset(unlabeled_datset_info_more_focus_on_airways_of_low_generation)
        unlabeled_train_dataset_low_gen.set_para(file_format=unlabeled_train_file_format, crop_size=crop_size_for_train,\
            windowMin=windowMin_CT_img_HU, windowMax=windowMax_CT_img_HU, need_tensor_output=True, need_transform=True)

        unlabeled_datset_info_more_focus_on_airways_of_high_generation = \
        get_balanced_data_dict(data_dict_of_current_unlabled_image_and_pseudolabels, is_more_big = False, copy_times_I = 1)
        unlabeled_train_dataset_high_gen = airway_dataset(unlabeled_datset_info_more_focus_on_airways_of_high_generation)
        unlabeled_train_dataset_high_gen.set_para(file_format=unlabeled_train_file_format, crop_size=crop_size_for_train,\
            windowMin=windowMin_CT_img_HU, windowMax=windowMax_CT_img_HU, need_tensor_output=True, need_transform=True)
        
        # labeled and unlabeled datasets
        dataset_list = [labeled_train_dataset_low_gen, unlabeled_train_dataset_low_gen,\
            labeled_train_dataset_high_gen, unlabeled_train_dataset_high_gen, labeled_train_dataset_low_gen]
        dataset_list_str = ["labeled, low gen",\
            "unlabeled, low gen",\
                "labeled images, high gen",\
                    "unlabeled, high gen",\
                        "labeled, low gen"]
        # dataset_list_str = ["on labeled images, focus more on airways of low generation",\
        #     "on unlabeled images, focus more on airways of low generation",\
        #         "on labeled images, focus more on airways of high generation",\
        #             "on unlabeled images, focus more on airways of high generation",\
        #                 "on labeled images, focus more on airways of low generation"]
        
        for ith_epoch in range(0, max_epoch):
            
            for ith_dataset, dataset in enumerate(dataset_list):
                sampler = RandomSampler(dataset, num_samples = num_samples_of_dataloader, replacement = True)
                dataset_loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler,\
                    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 1))
                
                len_dataset_loader = len(dataset_loader)
                for ith_batch, batch in enumerate(dataset_loader):

                    # get the number of unconnected components in label
                    if torch.sum(batch['label'])>0:
                        unique_vals_of_label = torch.unique(batch['label'])
                        unique_vals_of_label = unique_vals_of_label[unique_vals_of_label>0]
                        num_of_unconnected_components_in_label= len(unique_vals_of_label)
                    else:
                        num_of_unconnected_components_in_label = 0
                    num_of_unconnected_components_in_label = max(num_of_unconnected_components_in_label, 1)

                    img_input=batch['image'].float().to(device_for_student)
                    groundtruth_foreground=batch['label'].float().to(device_for_student)
                    groundtruth_background=1-groundtruth_foreground
            
                    fore_pix_num = torch.sum(groundtruth_foreground)
                    back_pix_num = torch.sum(groundtruth_background)
                    fore_pix_per = fore_pix_num/(fore_pix_num+back_pix_num)
                    back_pix_per = back_pix_num/(fore_pix_num+back_pix_num)
                    weights = (torch.exp(back_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))*torch.eq(groundtruth_foreground,1).float()+\
                        torch.exp(fore_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))*torch.eq(groundtruth_foreground,0).float()).to(device_for_student)
            
                    #print("num_of_unconnected_components_in_label: "+str(num_of_unconnected_components_in_label))

                    img_output=student_model(img_input)
                    
                    loss=1/num_of_unconnected_components_in_label * (dice_loss_weights(img_output[:,0,:,:,:], groundtruth_background, weights)+\
                        dice_loss_power_weights(img_output[:,1,:,:,:], groundtruth_foreground, weights, alpha=2))
                    accuracy=dice_accuracy(img_output[:,1,:,:,:], groundtruth_foreground)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    time_consumption = time.time() - start_time
                    print(
                        "tea_stu_iteration [{0}/{1}]\t"
                        "unlabeled_img_batch [{2}/{3}]\t"
                        "epoch [{4}/{5}]\t"
                        "dataset [{6}]\t"
                        "train_batch [{7}/{8}]\t"
                        "time(sec) {time:.2f}\t"
                        "loss {loss:.4f}\t"
                        "acc {acc:.2f}%\t"
                        "fore pix {fore_pix_percentage:.2f}%\t"
                        "back pix {back_pix_percentage:.2f}%\t".format(
                            ith_tea_stu_iteration + 1,
                            max_iteration_teacher_student_training,
                            ith_pic_name_batch + 1,
                            len(pic_name_batch_list),
                            ith_epoch + 1,
                            max_epoch,
                            dataset_list_str[ith_dataset],
                            ith_batch,
                            len_dataset_loader,
                            time = time_consumption,
                            loss = loss.item(),
                            acc = accuracy.item()*100,
                            fore_pix_percentage = fore_pix_per*100,
                            back_pix_percentage = back_pix_per*100))

            if (ith_epoch+1)%model_save_freq==0:
                print('save stu model')
                student_model.to(torch.device('cpu'))
                torch.save({'model_state_dict': student_model.state_dict()}, save_path_student_model)
                student_model.to(device_for_student)
        
        #deleting the file_path_of_study_materials
        try:
            shutil.rmtree(file_path_of_study_materials)
        except:
            print("no such filefolder")
            pass
    
    print("update teacher model")
    teacher_model.to(torch.device('cpu'))
    checkpoint_of_stu_model = torch.load(save_path_student_model)
    teacher_model.load_state_dict(checkpoint_of_stu_model['model_state_dict'])
    teacher_model.to(device_for_teacher)
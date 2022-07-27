from func.model_run import semantic_segment_crop_and_cat
from func.ulti import load_obj, crop_one_3d_img, load_one_CT_img
from func.post_process import post_process, get_super_vox, Cluster_super_vox

import numpy as np
import torch
import os
import h5py
import edt
import copy

def get_the_whole_img_set_dict(unlabelled_img_file_path = "/data/Airway/LIDC-IDRI_3D/raw_data", indicator = "LIDC_IDRI_"):
    whole_img_set_dict = dict()

    unlabelled_img_names = os.listdir(unlabelled_img_file_path)
    unlabelled_img_names.sort()

    for case in unlabelled_img_names:
        temp = case.split(".")[0]
        whole_img_set_dict[indicator+temp]=unlabelled_img_file_path+"/"+case
    
    return whole_img_set_dict

def init_the_teacher_model(model_arch, in_channel, out_channel, load_path, strict=False):
    model=model_arch(in_channels=in_channel, out_channels=out_channel)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    return model

def use_teacher_model_to_process_one_img_generating_study_materials(raw_img_path, teacher_model, threshold = 0.5,
                                                                    crop_cube_size = [32, 128, 128],
                                                                    stride = [16, 64, 64],
                                                                    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
    raw_img = load_one_CT_img(raw_img_path)
    seg_result = semantic_segment_crop_and_cat(raw_img, teacher_model,
                                               crop_cube_size=crop_cube_size,stride=stride,
                                               device=device)
    seg_result_onehot = np.array(seg_result>=threshold, dtype=np.int)
    seg_onehot_connected, airway_prob_map, seg_onehot_cluster = \
    post_process(seg_result, threshold=threshold, return_seg_onehot_cluster = True)
    
    return raw_img, seg_onehot_cluster, seg_onehot_connected

def save_the_study_materials_for_one_img(raw_img, seg_onehot, chosen_pic_name,
                                         data_dict_of_current_unlabled_image_and_pseudolabels = None,
                                         output_file_path = "/data/Airway/LIDC-IDRI_3D/temp",
                                         crop_cube_size=(128, 128, 128),
                                         stride=(64,64,64), min_crop_cube_size=[32, 128, 128]):

    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)

    raw_img_crop_list = crop_one_3d_img(raw_img, crop_cube_size=crop_cube_size, stride=stride)
    label_img_crop_list = crop_one_3d_img(seg_onehot, crop_cube_size=crop_cube_size, stride=stride)
    
    if data_dict_of_current_unlabled_image_and_pseudolabels is None:
        data_dict_of_current_unlabled_image_and_pseudolabels = {}

    for idx in range(len(raw_img_crop_list)):
        print("progress: "+str(idx)+"th crop | "+str(chosen_pic_name), end="\r")
        
        assert raw_img_crop_list[idx].shape == label_img_crop_list[idx].shape
        
        if raw_img_crop_list[idx].shape[0] >= min_crop_cube_size[0] and raw_img_crop_list[idx].shape[1] >= min_crop_cube_size[1] and \
        raw_img_crop_list[idx].shape[2] >= min_crop_cube_size[2]:
        
            h = h5py.File(output_file_path+"/"+chosen_pic_name+"_"+str(idx)+".h5", 'w')
            h.create_dataset("image",data=raw_img_crop_list[idx])
            h.create_dataset("label",data=label_img_crop_list[idx])

            data_dict_of_current_unlabled_image_and_pseudolabels[chosen_pic_name+"_"+str(idx)] = {}
            data_dict_of_current_unlabled_image_and_pseudolabels[chosen_pic_name+"_"+str(idx)]["path"] = output_file_path+"/"+chosen_pic_name+"_"+str(idx)+".h5"

            airway_pixel_num_cal = np.array(label_img_crop_list[idx]>0, dtype = np.int)

            airway_pixel_num = np.sum(airway_pixel_num_cal)
            data_dict_of_current_unlabled_image_and_pseudolabels[chosen_pic_name+"_"+str(idx)]["airway_pixel_num"] = airway_pixel_num

            if airway_pixel_num>0:
                label_temp=np.array(airway_pixel_num_cal, dtype=np.uint32, order='F')
                label_temp_edt=edt.edt(
                    label_temp,
                    black_border=True, order='F',
                    parallel=1)

                data_dict_of_current_unlabled_image_and_pseudolabels[chosen_pic_name+"_"+str(idx)]["airway_pixel_num_boundary"] = \
                len(np.where(label_temp_edt==1)[0])
                data_dict_of_current_unlabled_image_and_pseudolabels[chosen_pic_name+"_"+str(idx)]["airway_pixel_num_inner"] = \
                len(np.where(label_temp_edt>1)[0])

            else:
                data_dict_of_current_unlabled_image_and_pseudolabels[chosen_pic_name+"_"+str(idx)]["airway_pixel_num_boundary"] = 0
                data_dict_of_current_unlabled_image_and_pseudolabels[chosen_pic_name+"_"+str(idx)]["airway_pixel_num_inner"] = 0
        else:
            pass
    
    return data_dict_of_current_unlabled_image_and_pseudolabels

def use_teacher_model_to_process_one_list_of_img_and_save_study_materials(pic_name_list, whole_img_set_dict,
                                                                          teacher_model,
                                                                          device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                                                          file_path_of_study_materials = "/data/Airway/LIDC-IDRI_3D/temp",
                                                                          crop_cube_size_for_study_material_generation = [32, 128, 128],
                                                                          stride_for_study_material_generation = [16, 64, 64],
                                                                          crop_cube_size_for_study_material_saving = (128, 128, 128),
                                                                          stride_for_study_material_saving = (64,64,64),
                                                                          min_crop_cube_size=[32, 128, 128]):
    for idx, chosen_pic_name in enumerate(pic_name_list):
        print("use teacher model to generate study_materials")
        print("process img: "+str(chosen_pic_name)+"; the path is "+str(whole_img_set_dict[chosen_pic_name]))
        
        if np.random.rand()>=0.1:
            threshold = 0.5
        else:
            threshold = 0.7
        
        print("choose threshold: "+str(threshold))
        
        raw_img, seg_onehot_cluster, seg_onehot_connected = \
        use_teacher_model_to_process_one_img_generating_study_materials(whole_img_set_dict[chosen_pic_name],
                                                                        teacher_model,
                                                                        threshold = threshold,
                                                                        crop_cube_size = crop_cube_size_for_study_material_generation,
                                                                        stride = stride_for_study_material_generation,
                                                                        device = device)
        
        if np.random.rand()>=0.2:
            seg_onehot = seg_onehot_cluster
            print("use original model output as pseudolabel")
            
        else:
            seg_onehot = seg_onehot_connected
            print("use processed model output (the largest connected body) as pseudolabel")
        
        if idx == 0:
            data_dict_of_current_unlabled_image_and_pseudolabels = \
            save_the_study_materials_for_one_img(raw_img, seg_onehot, chosen_pic_name,
                                                 data_dict_of_current_unlabled_image_and_pseudolabels = None,
                                                 output_file_path = file_path_of_study_materials,
                                                 crop_cube_size=crop_cube_size_for_study_material_saving,
                                                 stride=stride_for_study_material_saving, min_crop_cube_size=min_crop_cube_size)
        else:
            data_dict_of_current_unlabled_image_and_pseudolabels = \
            save_the_study_materials_for_one_img(raw_img, seg_onehot, chosen_pic_name,
                                                 data_dict_of_current_unlabled_image_and_pseudolabels = data_dict_of_current_unlabled_image_and_pseudolabels,
                                                 output_file_path = file_path_of_study_materials,
                                                 crop_cube_size=crop_cube_size_for_study_material_saving,
                                                 stride=stride_for_study_material_saving, min_crop_cube_size=min_crop_cube_size)
        
    return data_dict_of_current_unlabled_image_and_pseudolabels

def get_balanced_data_dict(data_dict_org, is_more_big = True, copy_times_I = 10):
    data_dict_extended = copy.deepcopy(data_dict_org)
    for idx, case in enumerate(data_dict_org.keys()):
        if data_dict_org[case]["airway_pixel_num"]>0:
            if is_more_big:
                copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_inner"]/data_dict_org[case]["airway_pixel_num_boundary"])
            else:
                if data_dict_org[case]["airway_pixel_num_inner"]==0:
                    copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"])
                else:
                    copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"]/data_dict_org[case]["airway_pixel_num_inner"])

            for i in range(int(copy_times_I*copy_times_II)):
                data_dict_extended[case+"_copy_"+str(i+1)]=data_dict_org[case]
    return data_dict_extended

def get_data_dict_of_current_unlabled_image_and_pseudolabels(data_dict_of_current_unlabled_image_and_pseudolabels = None,
                                                             output_file_path = "/data/Airway/LIDC-IDRI_3D/temp"):
    
    if data_dict_of_current_unlabled_image_and_pseudolabels is None:
        data_dict_of_current_unlabled_image_and_pseudolabels = {}

    name_list = os.listdir(output_file_path)
    
    for idx, name in enumerate(name_list):
        print("get_data_dict_of_current_unlabled_image_and_pseudolabels; processing "+name+"; progress: "+str(int(idx/len(name_list)*100))+"%", end="\r")
        hf = h5py.File(output_file_path+"/"+name, 'r+')
        label_img = np.array(hf["label"])
        hf.close()
        
        airway_pixel_num_cal = np.array(label_img>0, dtype = np.int)
        
        name_split = name.split(".")[0]
        
        data_dict_of_current_unlabled_image_and_pseudolabels[name_split] = {}
        data_dict_of_current_unlabled_image_and_pseudolabels[name_split]["path"] = output_file_path+"/"+name

        airway_pixel_num = np.sum(airway_pixel_num_cal)
        data_dict_of_current_unlabled_image_and_pseudolabels[name_split]["airway_pixel_num"] = airway_pixel_num

        if airway_pixel_num>0:
            label_temp=np.array(airway_pixel_num_cal, dtype=np.uint32, order='F')
            label_temp_edt=edt.edt(
                label_temp,
                black_border=True, order='F',
                parallel=1)

            data_dict_of_current_unlabled_image_and_pseudolabels[name_split]["airway_pixel_num_boundary"] = \
            len(np.where(label_temp_edt==1)[0])
            data_dict_of_current_unlabled_image_and_pseudolabels[name_split]["airway_pixel_num_inner"] = \
            len(np.where(label_temp_edt>1)[0])

        else:
            data_dict_of_current_unlabled_image_and_pseudolabels[name_split]["airway_pixel_num_boundary"] = 0
            data_dict_of_current_unlabled_image_and_pseudolabels[name_split]["airway_pixel_num_inner"] = 0
    
    return data_dict_of_current_unlabled_image_and_pseudolabels
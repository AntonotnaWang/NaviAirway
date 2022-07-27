import pickle
import numpy as np
import pandas as pd
import skimage.io as io
import SimpleITK as sitk
import os

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def crop_one_3d_img(input_img, crop_cube_size, stride):
    # input_img: 3d matrix, numpy.array
    assert isinstance(crop_cube_size, (int, tuple))
    if isinstance(crop_cube_size, int):
        crop_cube_size=np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size)==3
    
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
        stride=np.array([stride, stride, stride])
    else:
        assert len(stride)==3
    
    img_shape=input_img.shape
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    
    count=0
    
    crop_list = []
    
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('crop one 3d img progress : '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start=i
                    x_end=i+crop_cube_size[0]
#                     x_start_output=i
#                     x_end_output=i+stride[0]
                else:
                    x_start=img_shape[0]-crop_cube_size[0]
                    x_end=img_shape[0]
#                     x_start_output=i
#                     x_end_output=img_shape[0]
                
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start=j
                    y_end=j+crop_cube_size[1]
#                     y_start_output=j
#                     y_end_output=j+stride[1]
                else:
                    y_start=img_shape[1]-crop_cube_size[1]
                    y_end=img_shape[1]
#                     y_start_output=j
#                     y_end_output=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start=k
                    z_end=k+crop_cube_size[2]
#                     z_start_output=k
#                     z_end_output=k+stride[2]
                else:
                    z_start=img_shape[2]-crop_cube_size[2]
                    z_end=img_shape[2]
#                     z_start_output=k
#                     z_end_output=img_shape[2]
                
                crop_temp=input_img[x_start:x_end, y_start:y_end, z_start:z_end]
                crop_list.append(np.array(crop_temp, dtype=np.float))
                
                count=count+1
                
    return crop_list

def load_one_CT_img(img_path):
    return io.imread(img_path, plugin='simpleitk')

def loadFile(filename):
    ds = sitk.ReadImage(filename)
    #pydicom.dcmread(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    #print("frame_num, width, height: "+str((frame_num, width, height)))
    return img_array, frame_num, width, height

def get_3d_img_for_one_case(img_path_list, img_format="dcm"):
    img_3d=[]
    for idx, img_path in enumerate(img_path_list):
        print("progress: "+str(idx/len(img_path_list))+"; "+str(img_path), end="\r")
        img_slice, frame_num, _, _ = loadFile(img_path)
        assert frame_num==1
        img_3d.append(img_slice)
    img_3d=np.array(img_3d)
    return img_3d.reshape(img_3d.shape[0], img_3d.shape[2], img_3d.shape[3])

def get_and_save_3d_img_for_one_case(img_path, output_file_path, img_format="dcm"):
    case_names=os.listdir(img_path)
    case_names.sort()
    img_path_list = []
    for case_name in case_names:
        img_path_list.append(img_path+"/"+case_name)
    img_3d = get_3d_img_for_one_case(img_path_list)
    sitk.WriteImage(sitk.GetImageFromArray(img_3d),output_file_path)
    
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
    
def get_CT_image(image_path, windowMin=-1000, windowMax=600, need_norm=True):
    raw_img = io.imread(image_path, plugin='simpleitk')
    raw_img = np.array(raw_img, dtype=np.float)
    
    if need_norm:
        normalization=Normalization_np(windowMin=windowMin, windowMax=windowMax)
        return normalization(raw_img)
    else:
        return raw_img

# show the airway centerline
def get_df_of_centerline(connection_dict):
    d = {}
    d["x"] = []
    d["y"] = []
    d["z"] = []
    d["val"] = []
    d["text"] = []
    for item in connection_dict.keys():
        print(item, end="\r")
        d["x"].append(connection_dict[item]['loc'][0])
        d["y"].append(connection_dict[item]['loc'][1])
        d["z"].append(connection_dict[item]['loc'][2])
        d["val"].append(connection_dict[item]['generation'])
        d["text"].append(str(item)+": "+str({"before":connection_dict[item]["before"], "next":connection_dict[item]["next"]}))
    df = pd.DataFrame(data=d)
    return df

# show the airway centerline
def get_df_of_line_of_centerline(connection_dict):
    d = {}
    for label in connection_dict.keys():
        if connection_dict[label]["before"][0]==0:
            start_label = label
            break
    def get_next_point(connection_dict, current_label, d, idx):
        while (idx in d.keys()):
            idx+=1
        
        d[idx]={}
        if "x" not in d[idx].keys():
            d[idx]["x"]=[]
        if "y" not in d[idx].keys():
            d[idx]["y"]=[]
        if "z" not in d[idx].keys():
            d[idx]["z"]=[]
        if "val" not in d[idx].keys():
            d[idx]["val"]=[]
        
        before_label = connection_dict[current_label]["before"][0]
        if before_label not in connection_dict.keys():
            before_label = current_label
        d[idx]["x"].append(connection_dict[before_label]["loc"][0])
        d[idx]["y"].append(connection_dict[before_label]["loc"][1])
        d[idx]["z"].append(connection_dict[before_label]["loc"][2])
        d[idx]["val"].append(connection_dict[before_label]["generation"])
        
        d[idx]["x"].append(connection_dict[current_label]["loc"][0])
        d[idx]["y"].append(connection_dict[current_label]["loc"][1])
        d[idx]["z"].append(connection_dict[current_label]["loc"][2])
        d[idx]["val"].append(connection_dict[current_label]["generation"])
        
        if connection_dict[current_label]["number_of_next"]==0:
            return
        else:
            for next_label in connection_dict[current_label]["next"]:
                get_next_point(connection_dict, next_label, d, idx+1)
    
    get_next_point(connection_dict, start_label, d,0)
    return d
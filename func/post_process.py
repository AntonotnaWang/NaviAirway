import numpy as np
from skimage.segmentation import watershed
import edt
import copy
import torch
from skimage.measure import label as label_regions

def post_process(model_output, threshold=0.5, return_seg_onehot_cluster = False, need_erosion_or_expansion = False, kernel_size=3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    seg_onehot = np.zeros(model_output.shape)
    seg_onehot[model_output>=threshold]=1
    
    seg_onehot_super_vox = get_super_vox(seg_onehot, connectivity=2, offset=[1,1,1])
    
    cluster_super_vox=Cluster_super_vox(min_touching_area=2, min_touching_percentage=0.5)
    cluster_super_vox.fit(seg_onehot_super_vox)
    seg_onehot_cluster = cluster_super_vox.output_3d_img
    del cluster_super_vox
    
    unique_vals, unique_val_counts = np.unique(seg_onehot_cluster, return_counts=True)
    unique_val_counts = unique_val_counts[unique_vals>0]
    unique_vals = unique_vals[unique_vals>0]
    sort_locs = np.argsort(unique_val_counts)[::-1]
    seg_onehot_final = np.zeros(seg_onehot_cluster.shape)
    seg_onehot_final[seg_onehot_cluster==unique_vals[sort_locs][0]]=1
    
    seg_onehot_final = fill_inner_hole(seg_onehot_final,
                                       need_erosion_or_expansion=need_erosion_or_expansion,
                                       kernel_size=kernel_size, device = device)
    
    airway_prob_map = copy.deepcopy(model_output)
    airway_prob_map[seg_onehot_final==0]=0
    
    if return_seg_onehot_cluster:
        return seg_onehot_final, airway_prob_map, seg_onehot_cluster
    else:
        return seg_onehot_final, airway_prob_map

def img_3d_erosion_or_expansion(img_3d, kernel_size=3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    org_shape = img_3d.shape
    
    padding = int((kernel_size - 1)/2)
    
    img_3d = img_3d.reshape(1,1,img_3d.shape[0],img_3d.shape[1],img_3d.shape[2])
    img_3d = torch.from_numpy(img_3d).float().to(device)
    
    pool_operation = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=1, padding=padding, dilation=1)
    img_3d = pool_operation(img_3d)
    
    img_3d = torch.nn.functional.interpolate(img_3d, size=org_shape, mode='nearest')
    
    img_3d=img_3d.detach().cpu().numpy()
    img_3d=img_3d.reshape(img_3d.shape[2],img_3d.shape[3],img_3d.shape[4])
    
    return img_3d
    
def fill_inner_hole(seg_onehot_final, need_erosion_or_expansion = False, kernel_size=3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    seg_onehot_background = np.array(seg_onehot_final==0, dtype=np.int)
    labels_bg = label_regions(seg_onehot_background)
    
    unique_vals, unique_val_counts = np.unique(labels_bg, return_counts = True)
    unique_val_counts = unique_val_counts[unique_vals>0]
    unique_vals = unique_vals[unique_vals>0]
    sort_locs = np.argsort(unique_val_counts)[::-1]
    
    bg = np.array(labels_bg==unique_vals[sort_locs][0], dtype=np.int)
    
    if need_erosion_or_expansion:
        bg = -img_3d_erosion_or_expansion(-bg, kernel_size=kernel_size, device = device)
        bg = img_3d_erosion_or_expansion(bg, kernel_size=kernel_size, device = device)
    
    seg_onehot_final = 1 - bg
    
    return seg_onehot_final

def delete_fragments(seg_onehot_cluster):
    unique_vals, unique_val_counts = np.unique(seg_onehot_cluster, return_counts=True)
    unique_val_counts = unique_val_counts[unique_vals>0]
    unique_vals = unique_vals[unique_vals>0]
    sort_locs = np.argsort(unique_val_counts)[::-1]
    
    seg_onehot_final = np.zeros(seg_onehot_cluster.shape)
    seg_onehot_final[seg_onehot_cluster==unique_vals[sort_locs][0]]=1
    return seg_onehot_final

def get_super_vox(seg_onehot, connectivity=2, offset=[1,1,1]):
    seg_onehot_edt=edt.edt(np.array(seg_onehot, dtype=np.uint32, order='F'), black_border=True, order='F', parallel=1)
    seg_onehot_super_vox = watershed(-seg_onehot_edt, mask=np.array(seg_onehot>0), connectivity=connectivity, offset=offset)
    return seg_onehot_super_vox

### cluster on super voxxel
class Cluster_super_vox():
    def __init__(self, min_touching_area=2, min_touching_percentage=0.5, boundary_extend=2):
        super(Cluster_super_vox, self).__init__
        self.min_touching_area = min_touching_area
        self.min_touching_percentage = min_touching_percentage
        
        self.boundary_extend = boundary_extend
        
        self.UN_PROCESSED = 0
        self.LONELY_POINT = -1
        self.A_LARGE_NUM = 100000000
        
    def fit(self, input_3d_img, restrict_area_3d=None):
        self.input_3d_img = input_3d_img
        
        if restrict_area_3d is None:
            self.restrict_area_3d = np.array(input_3d_img==0, dtype=np.int8)
        else:
            self.restrict_area_3d = restrict_area_3d
        
        unique_vals, unique_val_counts = np.unique(self.input_3d_img, return_counts=True)
        unique_val_counts = unique_val_counts[unique_vals>0]
        unique_vals = unique_vals[unique_vals>0]
        sort_locs = np.argsort(unique_val_counts)[::-1]
        self.unique_vals = unique_vals[sort_locs]
        
        self.val_labels = dict()
        for unique_val in self.unique_vals:
            self.val_labels[unique_val] = self.UN_PROCESSED
        
        self.val_outlayer_area = dict()
        for idx, unique_val in enumerate(self.unique_vals):
            print("get val_outlayer area of all vals: "+str(idx/len(self.unique_vals)), end="\r")
            self.val_outlayer_area[unique_val] = self.A_LARGE_NUM
        
        for idx, current_val in enumerate(self.unique_vals):
            print('processing: '+str(idx/len(self.unique_vals))+' pixel val: '+str(current_val), end="\r")
            if self.val_labels[current_val]!=self.UN_PROCESSED:
                continue
            valid_neighbor_vals = self.regionQuery(current_val)
            if len(valid_neighbor_vals)>0:
                print('Assign label '+str(current_val)+' to current val\'s neighbors: '+str(valid_neighbor_vals), end="\r")
                self.val_labels[current_val] = current_val
                self.growCluster(valid_neighbor_vals, current_val)
            else:
                self.val_labels[current_val] = self.LONELY_POINT
        
        self.output_3d_img = self.input_3d_img
    
    def fit_V2(self, input_3d_img, restrict_area_3d=None):
        self.input_3d_img = input_3d_img
        
        if restrict_area_3d is None:
            self.restrict_area_3d = np.array(input_3d_img==0, dtype=np.int8)
        else:
            self.restrict_area_3d = restrict_area_3d
        
        unique_vals, unique_val_counts = np.unique(self.input_3d_img, return_counts=True)
        unique_val_counts = unique_val_counts[unique_vals>0]
        unique_vals = unique_vals[unique_vals>0]
        sort_locs = np.argsort(unique_val_counts)[::-1]
        self.unique_vals = unique_vals[sort_locs]
        
        self.val_labels = dict()
        for unique_val in self.unique_vals:
            self.val_labels[unique_val] = self.UN_PROCESSED
        
        self.val_outlayer_area = dict()
        for idx, unique_val in enumerate(self.unique_vals):
            print("get val_outlayer area of all vals: "+str(idx/len(self.unique_vals)), end="\r")
            self.val_outlayer_area[unique_val] = self.get_outlayer_area(unique_val)
        
        for idx, current_val in enumerate(self.unique_vals):
            print('processing: '+str(idx/len(self.unique_vals))+' pixel val: '+str(current_val), end="\r")
            if self.val_labels[current_val]!=self.UN_PROCESSED:
                continue
            valid_neighbor_vals = self.regionQuery(current_val)
            if len(valid_neighbor_vals)>0:
                print('Assign label '+str(current_val)+' to current val\'s neighbors: '+str(valid_neighbor_vals), end="\r")
                self.val_labels[current_val] = current_val
                self.growCluster(valid_neighbor_vals, current_val)
            else:
                self.val_labels[current_val] = self.LONELY_POINT
        
        self.output_3d_img = self.input_3d_img
    
    def get_outlayer_area(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        boundary_extend=self.boundary_extend,
                                                                        crop_another_3d_img_by_the_way=self.restrict_area_3d)
        current_crop_img_onehot = np.array(current_crop_img==current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)
        
        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape
        
        current_crop_img_onehot_outlayer[current_restrict_area>0]=0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)
        
        return current_crop_outlayer_area
    
    def regionQuery(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        boundary_extend=self.boundary_extend,
                                                                        crop_another_3d_img_by_the_way=self.restrict_area_3d)
        
        current_crop_img_onehot = np.array(current_crop_img==current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)
        
        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape
        
        current_crop_img_onehot_outlayer[current_restrict_area>0]=0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)
        
        neighbor_vals, neighbor_val_counts = np.unique(current_crop_img[current_crop_img_onehot_outlayer>0], return_counts=True)
        neighbor_val_counts = neighbor_val_counts[neighbor_vals>0]
        neighbor_vals = neighbor_vals[neighbor_vals>0]
        
        print("current_crop_outlayer_area: "+str(current_crop_outlayer_area), end="\r")
        
        valid_neighbor_vals = self.neighborCheck(neighbor_vals, neighbor_val_counts, current_crop_outlayer_area)
        
        print("valid_neighbor_vals: "+str(valid_neighbor_vals), end="\r")
        print("number of valid_neighbor_vals: "+str(len(valid_neighbor_vals)), end="\r")
        
        return valid_neighbor_vals
        
    def neighborCheck(self, neighbor_vals, neighbor_val_counts, current_crop_outlayer_area):
        neighbor_val_counts = neighbor_val_counts[neighbor_vals>0]
        neighbor_vals = neighbor_vals[neighbor_vals>0]
        
        valid_neighbor_vals = []
        
        for idx, neighbor_val in enumerate(neighbor_vals):
            if neighbor_val_counts[idx]>=self.min_touching_area or \
            (neighbor_val_counts[idx]/current_crop_outlayer_area)>=self.min_touching_percentage or \
            (neighbor_val_counts[idx]/self.val_outlayer_area[neighbor_val])>=self.min_touching_percentage:
                print("touching_area: "+str(neighbor_val_counts[idx]), end="\r")
                print("touching_percentage: "+str(neighbor_val_counts[idx]/current_crop_outlayer_area)+\
                      " and "+str(neighbor_val_counts[idx]/self.val_outlayer_area[neighbor_val]), end="\r")
                valid_neighbor_vals.append(neighbor_val)
        
        double_checked_valid_neighbor_vals = []
        for valid_neighbor_val in valid_neighbor_vals:
            if self.val_labels[valid_neighbor_val]==self.UN_PROCESSED or \
            self.val_labels[valid_neighbor_val]==self.LONELY_POINT:
                double_checked_valid_neighbor_vals.append(valid_neighbor_val)
                
        return np.array(double_checked_valid_neighbor_vals)
    
    def growCluster(self, valid_neighbor_vals, current_val):
        valid_neighbor_vals = valid_neighbor_vals[valid_neighbor_vals>0]
        if len(valid_neighbor_vals)>0:
            for idx, valid_neighbor_val in enumerate(valid_neighbor_vals):
                self.val_labels[valid_neighbor_val]=current_val
                self.input_3d_img[self.input_3d_img==valid_neighbor_val]=current_val
            new_valid_neighbor_vals = self.regionQuery(current_val)
            print('Assign label '+str(current_val)+' to current val\'s neighbors: '+str(new_valid_neighbor_vals), end="\r")
            self.growCluster(new_valid_neighbor_vals, current_val)
        else:
            return

def get_outlayer_of_a_3d_shape(a_3d_shape_onehot, layer_thickness=1):
    shape=a_3d_shape_onehot.shape
    
    a_3d_crop_diff_x1 = a_3d_shape_onehot[0:shape[0]-1,:,:]-a_3d_shape_onehot[1:shape[0],:,:]
    a_3d_crop_diff_x2 = -a_3d_shape_onehot[0:shape[0]-1,:,:]+a_3d_shape_onehot[1:shape[0],:,:]
    a_3d_crop_diff_y1 = a_3d_shape_onehot[:,0:shape[1]-1,:]-a_3d_shape_onehot[:,1:shape[1],:]
    a_3d_crop_diff_y2 = -a_3d_shape_onehot[:,0:shape[1]-1,:]+a_3d_shape_onehot[:,1:shape[1],:]
    a_3d_crop_diff_z1 = a_3d_shape_onehot[:,:,0:shape[2]-1]-a_3d_shape_onehot[:,:,1:shape[2]]
    a_3d_crop_diff_z2 = -a_3d_shape_onehot[:,:,0:shape[2]-1]+a_3d_shape_onehot[:,:,1:shape[2]]

    outlayer = np.zeros(shape)
    outlayer[1:shape[0],:,:] += np.array(a_3d_crop_diff_x1==1, dtype=np.int8)
    outlayer[0:shape[0]-1,:,:] += np.array(a_3d_crop_diff_x2==1, dtype=np.int8)
    outlayer[:,1:shape[1],:] += np.array(a_3d_crop_diff_y1==1, dtype=np.int8)
    outlayer[:,0:shape[1]-1,:] += np.array(a_3d_crop_diff_y2==1, dtype=np.int8)
    outlayer[:,:,1:shape[2]] += np.array(a_3d_crop_diff_z1==1, dtype=np.int8)
    outlayer[:,:,0:shape[2]-1] += np.array(a_3d_crop_diff_z2==1, dtype=np.int8)
    
    outlayer = np.array(outlayer>0, dtype=np.int8)
    
    if layer_thickness==1:
        return outlayer
    else:
        return outlayer+get_outlayer_of_a_3d_shape(outlayer+a_3d_shape_onehot, layer_thickness-1)

def get_crop_by_pixel_val(input_3d_img, val, boundary_extend=2, crop_another_3d_img_by_the_way=None):
    locs = np.where(input_3d_img==val)
    
    shape_of_input_3d_img = input_3d_img.shape
    
    min_x = np.min(locs[0])
    max_x =np.max(locs[0])
    min_y = np.min(locs[1])
    max_y =np.max(locs[1])
    min_z = np.min(locs[2])
    max_z =np.max(locs[2])
    
    x_s = np.clip(min_x-boundary_extend, 0, shape_of_input_3d_img[0])
    x_e = np.clip(max_x+boundary_extend+1, 0, shape_of_input_3d_img[0])
    y_s = np.clip(min_y-boundary_extend, 0, shape_of_input_3d_img[1])
    y_e = np.clip(max_y+boundary_extend+1, 0, shape_of_input_3d_img[1])
    z_s = np.clip(min_z-boundary_extend, 0, shape_of_input_3d_img[2])
    z_e = np.clip(max_z+boundary_extend+1, 0, shape_of_input_3d_img[2])
    
    #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
    crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    if crop_another_3d_img_by_the_way is not None:
        assert input_3d_img.shape == crop_another_3d_img_by_the_way.shape
        crop_another_3d_img = crop_another_3d_img_by_the_way[x_s:x_e,y_s:y_e,z_s:z_e]
        return crop_3d_img,crop_another_3d_img
    else:
        return crop_3d_img

# 2nd post process
def add_broken_parts_to_the_result(connection_dict, model_output_prob_map, seg_processed_onehot, threshold=0.5,
                                   search_range = 10, delta_threshold = 0.05, min_threshold = 0.4):
    
    center_map_end_point_dict = find_end_point_of_the_airway_centerline(connection_dict)
    
    seg_processed_onehot_II = copy.deepcopy(seg_processed_onehot)
    
    for idx, item in enumerate(center_map_end_point_dict.keys()):
        print(idx/len(center_map_end_point_dict.keys()), end="\r")
        search_center = center_map_end_point_dict[item]
        model_output_prob_map_crop, crop_coord = get_crop(model_output_prob_map, search_center, search_range=search_range)
        seg_processed_onehot_crop, _ = get_crop(seg_processed_onehot, search_center, search_range=search_range)

        size_of_seg_crop = np.sum(seg_processed_onehot_crop)

        model_output_crop_delete_current_seg = model_output_prob_map_crop-seg_processed_onehot_crop
        current_threshold = threshold
        model_output_crop_revised = np.array(model_output_crop_delete_current_seg>current_threshold, dtype=np.int)

        while np.sum(model_output_crop_revised)<=size_of_seg_crop and current_threshold>=min_threshold:
            current_threshold-=delta_threshold
            model_output_crop_revised = np.array(model_output_crop_delete_current_seg>current_threshold, dtype=np.int)

        seg_processed_onehot_II[crop_coord[0]:crop_coord[1],
                         crop_coord[2]:crop_coord[3],
                         crop_coord[4]:crop_coord[5]]+=model_output_crop_revised
    seg_processed_onehot_II = np.array(seg_processed_onehot_II>0, dtype = np.int)
    
    return seg_processed_onehot_II

def get_crop(input_3d_img, search_center, search_range=1):
    shape_of_input_3d_img = input_3d_img.shape
    
    x = search_center[0]
    y = search_center[1]
    z = search_center[2]
    
    x_s = np.clip(x-search_range, 0, shape_of_input_3d_img[0])
    x_e = np.clip(x+search_range+1, 0, shape_of_input_3d_img[0])
    y_s = np.clip(y-search_range, 0, shape_of_input_3d_img[1])
    y_e = np.clip(y+search_range+1, 0, shape_of_input_3d_img[1])
    z_s = np.clip(z-search_range, 0, shape_of_input_3d_img[2])
    z_e = np.clip(z+search_range+1, 0, shape_of_input_3d_img[2])
    
    #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
    crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    
    crop_coord = [x_s, x_e, y_s, y_e, z_s, z_e]
    
    return crop_3d_img, crop_coord

def find_end_point_of_the_airway_centerline(connection_dict):
    center_map_end_point_dict = {}
    for item in connection_dict.keys():
        if connection_dict[item]["number_of_next"]==0:
            center_map_end_point_dict[item]=connection_dict[item]["loc"]
    
    return center_map_end_point_dict
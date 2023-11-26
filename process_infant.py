import os 
import numpy as np
import json

root = 'data/mini-rgbd/MINI-RGBD/MINI-RGBD_web/'
l = os.listdir(root)
# pose_2d = []
d = {'train':{},'validate':{}}
# pose_3d = []
file_name = []
file_name_3d = [ ]
for i in l:
    if i not  in ['01','02','03','04','05','06','07','08','09','10','11','12']:    continue
    if i in ['01','02','03','04','05','06','07','08','09','10']:
        temp_dict = d['train']
    else:
        temp_dict = d['validate']
    path_3d  = os.path.join(root,i,'joints_3D')
    path_2d  = os.path.join(root,i,'joints_2Ddep')
    for j in os.listdir(path_2d):
        with open(os.path.join(path_2d,j),'r') as f:
            file_name.append(j)
            lines = f.readlines()
            pose_2d = []
            for line in lines:
                # print(type(lines))
                # print(line.split(' ')[0:2])
                
                    pose_2d.append(np.array([line.split(' ')[0:2]]))
            pose_2d = np.array(pose_2d).reshape(-1,2).astype('float32')
            
            if str(i+'_'+j) not in temp_dict.keys():  
                temp_dict[i+'_'+j] = {}
            temp_dict[i+'_'+j]['pose_2d'] = pose_2d
    # print(len(temp_dict.keys()))
                
    for j in os.listdir(path_3d):
        with open(os.path.join(path_3d,j),'r') as f:
            file_name_3d.append(j)
            lines = f.readlines()
            pose_3d = []
            for line in lines:
                pose_3d.append(np.array([line.split(' ')[0:3]]))
            pose_3d = np.array(pose_3d).reshape(-1,3).astype('float32')
            temp_string = (i+'_'+j).replace('joints_3D','joints_2Ddep')
            if   temp_string not in temp_dict.keys():
                temp_dict[temp_string] = {}
            temp_dict[temp_string]['pose_3d'] = pose_3d
    # import ipdb;ipdb.set_trace()
np.save('data/mini-rgbd/MINI-RGBD.npy',d)
# assert temp == file_name_3d
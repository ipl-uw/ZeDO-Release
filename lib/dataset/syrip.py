import numpy as np
import os, sys
import pickle
from prettytable import PrettyTable
import random
import math as m
import math
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from lib.utils.transforms import image_to_camera_frame, align_to_gt

# from multiprocessing import Pool

def flip_data(data):
    """
    horizontal flip
        data: [N, 17*k] or [N, 17, k], i.e. [x, y], [x, y, confidence] or [x, y, z]
    Return
        result: [2N, 17*k] or [2N, 17, k]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    flipped_data = data.copy().reshape((len(data), 17, -1))
    flipped_data[:, :, 0] *= -1  # flip x of all joints
    flipped_data[:, left_joints+right_joints] = flipped_data[:, right_joints+left_joints]
    flipped_data = flipped_data.reshape(data.shape)

    result = np.concatenate((data, flipped_data), axis=0)

    return result

def unflip_data(data):
    """
    Average original data and flipped data
        data: [2N, 17*3]
    Return
        result: [N, 17*3]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    data = data.copy().reshape((2, -1, 17, 3))
    data[1, :, :, 0] *= -1  # flip x of all joints
    data[1, :, left_joints+right_joints] = data[1, :, right_joints+left_joints]
    data = np.mean(data, axis=0)
    data = data.reshape((-1, 17*3))

    return data

def denormalize_data(data, which='scale'):
    """
    data: [B, j, 3]
    Return: [B, j, 3]
    """
    res_w, res_h = 1000, 1000
    assert data.ndim >= 3
    if which == 'scale':
        data = data.copy()
        data[..., :2] = (data[..., :2] + [1, res_h / res_w]) * res_w / 2
        data[..., 2:] = data[..., 2:] * res_w / 2
    else:
        assert 0
    return data

def normalize_data(data):
    """
    data: [B, j, 3]
    Return: [B, j, 3]
    """
    res_w, res_h = 1000, 1000
    assert data.ndim >= 3
    data = data.copy()
    data[..., :2] = data[..., :2] / res_w * 2 - [1, res_h / res_w]
    data[..., 2:] = data[..., 2:] / res_w * 2
    return data

def worker(args):
    multi_pred, box, camera_param, root_depth, gt, protocol2 = args
    multi_results = []
    for pred in multi_pred:
        pred = image_to_camera_frame(pose3d_image_frame=pred,
            box=box,
            camera=camera_param, rootIdx=0,
            root_depth=root_depth)
        if protocol2:
            pred = align_to_gt(pose=pred, pose_gt=gt)
        error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
        multi_results.append(np.mean(error_per_joint))  # scala
    return np.amin(multi_results)  # min error among multi-hypothesis


class syrip:
    def __init__(self,  subset='train', 
        gt2d=True, read_confidence=True, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0, abs_coord=False, rot=False,
        num_joint=17,norm_2d=False,truncated=False,aug=False):
        
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.root = None
        self.subset = subset
        self.gt2d = gt2d
        self.read_confidence = read_confidence
        self.sample_interval = sample_interval
        self.flip = flip
        self.camera_param = None
        self.abs_coord = abs_coord
        self.rot = rot
        self.image_name = []
        self.action = []
        self.num_joint = num_joint
        self.norm_2d = norm_2d
        self.joint_match = {
            17: [i for i in range(17)],
            15: [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
        }
        self.change_2d = [-1,-3,-5,-6,-4,-2,-7,-9,-11,-12,-10,-8]
        self.change_12 = [2,1,0,3,4,5,-3,-2,-1,-4,-5,-6]
        self.K = []
        self.left_joints = [3,4,5,9,10,11]
        self.right_joints = [0,1,2,6,7,8]
        self.cond_3d_prob = cond_3d_prob
        self.truncated = truncated
        self.aug  = aug
        self.db_2d, self.db_3d, self.frame_name= self.read_data()
    
        if self.sample_interval:
            self._sample(sample_interval)

        self.rep = rep
        if self.rep > 1:
            print(f'stack dataset {self.rep} times for multi-sample eval')

        self.real_data_len = len(self.db_2d)

        
        
        

    def __getitem__(self, idx):
        """
        Return: [17, 2], [17, 3] for data and labels
        """
        data_2d = self.db_2d[idx % self.real_data_len]
        data_3d = self.db_3d[idx % self.real_data_len]
        data_2d = data_2d[:,:2]
        K = np.zeros((3, 3), dtype=np.float32)
        return data_2d, data_3d,K

    def __len__(self,):
        return len(self.db_3d) * self.rep

    def _random_flip(self, data, p=0.5):
        """
        Flip with prob p
        data: [17, 2] or [17, 3]
        """
        if np.random.rand(1,)[0] < p:
            data = data.copy()
            data[:, 0] *= -1  # flip x of all joints
            data[self.left_joints+self.right_joints] = data[self.right_joints+self.left_joints]
        return data
    
    def _random_rotate(self, data, p=0.5):
        """
        Flip with prob p
        data: [17, 2] or [17, 3]
        """
       
        try:
            data = data.reshape(data.shape[0],-1)
        except: import ipdb;ipdb.set_trace()
        if data.shape[-1]==2:
            ones_column = np.ones((data.shape[0], 1))
            data = np.hstack((data,ones_column))
        if np.random.rand(1,)[0] < p:
            data = data.copy()
            data = R.random().as_matrix().dot(data.T).T
        return data
    
    def save_action(self,action):
       
        self.action = action
        assert len(self.db_3d)==len(self.action)
        return self.action

    def add_noise(self, pose2d, std=5, noise_type='gaussian'):
        """
        pose2d: [B, j, 2]
        """
        if noise_type == 'gaussian':
            noise = std * np.random.randn(*pose2d.shape).astype(np.float32)
            pose2d = pose2d + noise
        elif noise_type == 'uniform':
            # a range of [-0.5std, 0.5std]
            noise = std * (np.random.rand(*pose2d.shape).astype(np.float32) - 0.5)
            pose2d = pose2d + noise
        else:
            raise NotImplementedError
        return pose2d

    def _sample(self, sample_interval):
        print(f'Class H36MDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.db_2d = self.db_2d[::sample_interval]
        self.db_3d = self.db_3d[::sample_interval]
        self.image_name = self.image_name[::sample_interval]
        self.h = self.h[::sample_interval]
        self.w = self.w[::sample_interval]
        self.K = self.K[::sample_interval]


    def read_data(self):
        # read 3d labels
        img_name = []
        data_3d = []
        data_2d = []
        frame_name = []
        
        self.img_root = '/home/zhongyuj/Infant-Pose-Estimation/data/syrip/images/train_infant' if self.subset=='train' else '/home/zhongyuj/Infant-Pose-Estimation/data/syrip/images/validate_infant'
        self.root = '/home/zhongyuj/Infant-Postural-Symmetry/data/SyRIP_3d_correction'
        all_name = np.load('/home/zhongyuj/Infant-Postural-Symmetry/data/SyRIP_3d_pred/output_imgnames.npy')
        train_pose_2d = np.load('/home/zhongyuj/Infant-Postural-Symmetry/train_pose2d.npy',allow_pickle=True).item()
        test_pose_2d = np.load('/home/zhongyuj/Infant-Postural-Symmetry/test_pose2d.npy',allow_pickle=True).item()
        pose_3d = np.load(os.path.join(self.root,'correct_3D.npy'))
        if self.subset!='train': self.subset = 'test'
        
        img_name = np.load(f'/home/zhongyuj/Infant-Postural-Symmetry/{self.subset}_rysip.npy',allow_pickle=True).item()
        h = []
        w = []
        K = []
        for i, item in enumerate(all_name):
            item = item.split('/')[-1]
            if item not in img_name.keys():
                continue
            if item in img_name.keys():
                frame_name.append( os.path.join(self.img_root,img_name[item][0]))
            else:
                continue 
            # if self.flip and np.random.rand(1,)[0]<0.5:
            #     data_3d.append(self._random_rotate(self. _random_flip((pose_3d[i]), p=0.5),p=0.5))
            
            
            data_3d.append(pose_3d[i])
            if img_name[item][0] in train_pose_2d.keys():
                
                temp_h = train_pose_2d[img_name[item][0]]['h']
                temp_w = train_pose_2d[img_name[item][0]]['w']
                temp_pose = np.array(train_pose_2d[img_name[item][0]]['keypoints'])
                temp_pose = temp_pose[self.change_2d]
                data_2d.append(temp_pose)
                h.append(temp_h)
                w.append(temp_w)
                K.append(np.array([[2000,0,temp_w/2],[0,2000,temp_h/2],[0,0,1]]))
                
            else:
                
                temp_h = test_pose_2d[img_name[item][0]]['h']
                temp_w = test_pose_2d[img_name[item][0]]['w']
                temp_pose = np.array(test_pose_2d[img_name[item][0]]['keypoints'])
                temp_pose = temp_pose[self.change_2d]
                data_2d.append(temp_pose)
                h.append(temp_h)
                w.append(temp_w)
                K.append(np.array([[2000,0,temp_w/2],[0,2000,temp_h/2],[0,0,1]]))
                

        data_3d = np.array(data_3d,dtype=np.float32)
        h = np.array(h)
        w = np.array(w)
        data_2d = np.array(data_2d,dtype=np.float32)
        
        frame_name = np.array(frame_name)
        if not self.gt2d:
            new_2d = np.load('dt_syripdata.npy',allow_pickle=True).item()
            new_2d = new_2d['train'] if self.subset=='train' else new_2d['test']
            for i in range(len(frame_name)):
                data_2d[i] = new_2d[frame_name[i].split('/')[-1]][self.change_2d]
        
            data_2d = np.array(data_2d,dtype=np.float32)
        

            
        self.h = h
        self.w = w
        self.K = np.array(K)
        data_3d = data_3d[:,:-2,:]
        
       
        if self.num_joint == 12:
            data_2d = data_2d[:,self.change_12]
            data_3d = data_3d[:,self.change_12]
            data_3d_pelvis = (data_3d[:,0,:]+data_3d[:,3,:])/2
            data_3d = data_3d - data_3d_pelvis[:,None,:]
           
      
        if self.truncated:
            data_2d = data_2d
            data_3d = data_3d
            frame_name = frame_name
       
        
        if self.aug:   
            aug_data = np.load('cls_aug_data.npy')
            for aug in aug_data:
                aug/= np.random.uniform(2.5,3.5)
            data_3d = np.concatenate([data_3d,aug_data])
        
        return data_2d, data_3d, frame_name


    

    def eval_multi(self, preds, protocol2=False, print_verbose=False, sample_interval=None,valid_ind=None,sample=None,mask_tok=None):
        
        print('eval multi-hypothesis...')
        all_gt = []

        

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        multi_preds_cam = []
        max_error = 0
        PCK1 = 0
        PCK2 = 0
        
        for idx, multi_pred in enumerate(preds):
            
            multi_results = []
            pred_store = []
            index = []
            for sec_idx, pred in enumerate(multi_pred):
                if valid_ind is not None and sec_idx not in valid_ind[idx]:
                    continue
                gt = self.db_3d[idx]
                pred_store.append(pred)
                all_gt.append(gt)
                    
                if protocol2:
                    pred = align_to_gt(pose=pred, pose_gt=gt)
                error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
                multi_results.append(np.mean(error_per_joint))  # scala
               
            current_index = np.argmin(multi_results)
            index.append(current_index)
            
            results.append(np.amin(multi_results))  # min error among multi-hypothesis
            if results[-1]>max_error:
                max_error = results[-1]
            multi_preds_cam.append(pred_store)  # [M, j, 3]
        
        results = np.array(results)  # [N]
        index = np.array(index)
        all_gt = np.array(all_gt)
        if len(preds.shape) > 3:
            preds.squeeze(1)
       
        error = np.mean(results)
        print(f'mean MPJPE error: {error}')
        return error

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
        [8, 14], [14, 15], [15, 16]]

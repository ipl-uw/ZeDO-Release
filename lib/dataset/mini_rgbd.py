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


class mini_rgbd:
    def __init__(self,  subset='train', 
        gt2d=True, read_confidence=True, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0, abs_coord=False, rot=False,
        num_joint=17,norm_2d=False,aug=False,cls=False,scale=1.0,normed=False):
        
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.root = None
        self.subset = subset
        self.gt2d = gt2d
        self.K = []
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
        self.change = [0,2,5,11,1,4,10,3,9,12,15,13,18,20,14,19,21]
        self.left_joints = [1, 4, 7, 10,13,16,18,20,22]
        self.right_joints = [2, 5, 8, 11, 14, 17, 19, 21, 23]
        self.cond_3d_prob = cond_3d_prob
        self.change_to_12 = [1, 2, 3, 4, 5, 6,  11, 12, 13, 14, 15, 16]
        self.aug = aug
        self.normed= normed
        
        
        self.db_2d, self.db_3d, self.frame_name = self.read_data()
      
    
        if self.sample_interval:
            self._sample(sample_interval)

        self.rep = rep
        if self.rep > 1:
            print(f'stack dataset {self.rep} times for multi-sample eval')

        self.real_data_len = len(self.db_2d)
        self.cls = cls
        self.scale = scale
        
       
            

        
    def norm(self,pose_3d):
        pose_3d = 2*(pose_3d - pose_3d.min())/(pose_3d.max()-pose_3d.min())-1
        return pose_3d
        import ipdb;ipdb.set_trace()
        

    def __getitem__(self, idx):
        """
        Return: [17, 2], [17, 3] for data and labels
        """
        data_2d = self.db_2d[idx % self.real_data_len]
        data_3d = self.db_3d[idx % self.real_data_len]
        w =480
        h= 640
        K = self.K[idx % self.real_data_len]
        
        if self.scale > 1:
            data_3d *= self.scale 
        if self.cls:
            data_2d = np.concatenate([data_2d,np.ones((data_2d.shape[0],1))],axis=-1)
            return data_2d, data_3d,K,np.array([0,1])
        else:
            return data_2d, data_3d,K
  
        
        

    def __len__(self,):
        return len(self.db_2d) * self.rep

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
        self.K = self.K[::sample_interval]

    def read_data(self):
       
       
        data = np.load('data/mini-rgbd/MINI-RGBD.npy',allow_pickle=True).item()
        data = data[self.subset]
        
       

        pose_3d = []
        pose_2d = []
        frame_name = []
        
        for idx, item in enumerate(tqdm(data.keys())):
            
           
            if self.flip and np.random.rand(1,)[0]<0.5:
                pose_3d.append(self._random_rotate(self. _random_flip((data[item]['pose_3d']-data[item]['pose_3d'][0:1]).reshape(-1,3), p=0.5),p=0.5))
                pose_2d.append(data[item]['pose_2d'])
            
            pose_3d.append(data[item]['pose_3d'])
            pose_2d.append(data[item]['pose_2d'])
            K = np.zeros((3, 3), dtype=np.float32)
            fx = 588.67905803875317
            fy = 590.25690113005601
            cx = 322.22048191353628
            cy = 237.46785983766890
            K[0][0] = fx
            K[1][1] = fy
            K[0][2] = cx
            K[1][2] = cy
            K[2][2] = 1
            self.K.append(K)
            frame_name.append(item)
            
            
        
        pose_3d = np.array(pose_3d,dtype=np.float32)
        pose_2d = np.array(pose_2d,dtype=np.float32)
        frame_name = np.array(frame_name)
       
        if not self.abs_coord:
            self.root = pose_3d[:,0:1]
            pose_3d  = pose_3d[:,:] - pose_3d[:,0:1]
        if self.normed:
            pose_3d = self.norm(pose_3d)
        
        
        if self.num_joint == 17:
            pose_2d = pose_2d[:,self.change]
            pose_3d = pose_3d[:,self.change]
            
     
        if self.aug: 
            aug_data = np.load('aug_mini.npy')
            
            for aug in aug_data:
                aug/= np.random.uniform(0.8,1.2)
            pose_3d = np.concatenate([pose_3d,aug_data],axis=0)
 
        if len(pose_2d) != len(pose_3d):
            pose_2d = np.zeros_like(pose_3d)
            frame_name = np.zeros_like(pose_3d)
            self.K = np.zeros_like(pose_3d)
            
        if self.num_joint==12:
            pose_2d = pose_2d[:,self.change_to_12,:]
            pose_3d = pose_3d[:,self.change_to_12,:]
        np.save('mini_gt_gt.npy',pose_3d)
        return pose_2d, pose_3d, frame_name

 
    def eval_multi(self, preds, protocol2=False, print_verbose=False, sample_interval=None,valid_ind=None,sample=None,mask_tok=None):
        """
        Eval action-wise MPJPE
        preds: [N, m, j, 3], N:len of dataset, m: multi-hypothesis number
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval multi-hypothesis...')
        pck_thres = [26.1,28.6,30.1]
        all_gt = []
       
        

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        multi_preds_cam = []
        max_error = 0
        for idx, multi_pred in enumerate(preds):
            
            multi_results = []
            pred_store = []
            index = []
            for sec_idx, pred in enumerate(multi_pred):
                if valid_ind is not None and sec_idx not in valid_ind[idx]:
                    continue
                gt = self.db_3d[idx]
                gt = (gt-gt[0:1])
                pred_store.append(pred)
                all_gt.append(gt)

                if gt.shape[-2] == 12:
                    pred = np.concatenate([pred[1:7,:],pred[11:,:]])
                    gt = np.concatenate([gt[1:7,:],gt[11:,:]])
                if protocol2:
                    pred = align_to_gt(pose=pred, pose_gt=gt)

                error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
                multi_results.append(np.mean(error_per_joint))  # scala
               
              
            current_index = np.argmin(multi_results)
            index.append(current_index)
            
            results.append(np.amin(multi_results))  # min error among multi-hypothesis
            if results[-1]>max_error:
                max_error = results[-1]
                max_cluster_index = current_index
                max_sample_index = idx
            multi_preds_cam.append(pred_store)  # [M, j, 3]
        
        results = np.array(results)  # [N]
        index = np.array(index)
        all_gt = np.array(all_gt)
        if len(preds.shape) > 3:
            preds.squeeze(1)
       
        #  MPJPE
        error = np.mean(results) 
        # import ipdb;ipdb.set_trace()
        print(f'mean MPJPE error: {error}')
        
        return error

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
        [8, 14], [14, 15], [15, 16]]

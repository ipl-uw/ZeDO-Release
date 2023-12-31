import numpy as np
import os, sys
import pickle
from prettytable import PrettyTable
import random
import math as m
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


class H36MDataset3D:
    def __init__(self, root_path, subset='train', 
        gt2d=True, read_confidence=True, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0, abs_coord=False, rot=False):
        
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.root_path = root_path
        self.subset = subset
        self.gt2d = gt2d
        self.read_confidence = read_confidence
        self.sample_interval = sample_interval
        self.flip = flip
        self.camera_param = None
        self.abs_coord = abs_coord
        self.rot = rot
        self.image_name = []
        
        self.db_2d, self.db_3d, self.gt_dataset,self.camera_param = self.read_data()

        if self.sample_interval:
            self._sample(sample_interval)

        self.rep = rep
        if self.rep > 1:
            print(f'stack dataset {self.rep} times for multi-sample eval')

        self.real_data_len = len(self.db_2d)

        self.left_joints = [4, 5, 6, 11, 12, 13]
        self.right_joints = [1, 2, 3, 14, 15, 16]

        self.cond_3d_prob = cond_3d_prob

    def __getitem__(self, idx):
        """
        Return: [17, 2], [17, 3] for data and labels
        """
        data_2d = self.db_2d[idx % self.real_data_len]
        data_3d = self.db_3d[idx % self.real_data_len]
        K = self.camera_param[idx % self.real_data_len]

        # always return [17, 3] for data_2d
        n_joints = len(data_2d)
        data_2d = np.concatenate(
            (data_2d, np.zeros((n_joints, 1), dtype=np.float32)),
            axis=-1,
        )  # [17, 3]

        # return gt3d in some prob while training
        if self.cond_3d_prob and self.subset == 'train':
            if np.random.rand(1,)[0] < self.cond_3d_prob:
                # return 3d
                data_2d = data_3d

        # only random flip during training
        if self.flip and self.subset == 'train':
            data_3d = self._random_flip(data_3d)
        
        if self.rot and self.subset == 'train':
            data_3d = self._random_rotate(data_3d)

        return data_2d, data_3d

    def __len__(self,):
        # assert len(self.db_2d) == len(self.db_3d)
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
        if np.random.rand(1,)[0] < p:
            data = data.copy()
            data = R.random().as_matrix().dot(data.T).T
        return data

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
        self.gt_dataset = self.gt_dataset[::sample_interval]
        self.camera_param = self.camera_param[::sample_interval]
        self.image_name = self.image_name[::sample_interval]

    def read_data(self):
        # read 3d labels
        file_name = 'h36m_%s.pkl' % self.subset
        
        print('loading %s' % file_name)
        file_path = os.path.join(self.root_path, file_name)
        with open(file_path, 'rb') as f:
            gt_dataset = pickle.load(f)

        labels_3d = []
        labels_image_3d = []
        camera_params = []
        seq_ind = []
        
        for idx, item in enumerate(tqdm(gt_dataset)):
            labels_3d.append(item['joint_3d_camera'])
            labels_image_3d.append(item['joint_3d_image'])
            
            temp_camera_params = np.zeros((3, 3), dtype=np.float32)
            temp_camera_params[0][0] = item['camera_param']['fx'].item()
            temp_camera_params[1][1] = item['camera_param']['fy'].item()
            temp_camera_params[0][2] = item['camera_param']['cx'].item()
            temp_camera_params[1][2] = item['camera_param']['cy'].item()
            temp_camera_params[2][2] = 1
            
            camera_params.append(temp_camera_params)
            seq_ind.append(idx)
            
            self.image_name.append(item['image_path'])

        labels_3d = np.array(labels_3d,dtype=np.float32)
        labels_image_3d = np.array(labels_image_3d,dtype=np.float32)
        camera_params = np.array(camera_params,dtype=np.float32)
        if not self.abs_coord:
            labels_3d  = labels_3d[:,:] - labels_3d[:,0:1]
        labels_3d = labels_3d/1000.0
        
        # read 2d
        if self.gt2d:
            data_2d = labels_image_3d[..., :2].copy()  # [N, 17, 2]
            
            if self.read_confidence:
                data_2d = np.concatenate((data_2d, np.ones((len(data_2d), 17, 1))), axis=-1)  # [N, 17, 3]
        else:
            file_name = 'h36m_sh_dt_ft.pkl'
            file_path = os.path.join(self.root_path, file_name)
            print('loading dt_2d %s' % file_name)
            with open(file_path, 'rb') as f:
                dt_dataset = pickle.load(f)

            data_2d =  dt_dataset[self.subset]['joint3d_image'][:, :, :2].copy()  # [N, 17, 2]
            if self.read_confidence:
                dt_confidence = dt_dataset[self.subset]['confidence'].copy()  # [N, 17, 1]
                data_2d = np.concatenate((data_2d, dt_confidence), axis=-1)  # [N, 17, 3]
            data_2d = data_2d.astype(np.float32)


        return data_2d, labels_3d, gt_dataset, camera_params

    def eval(self, preds, protocol2=False, print_verbose=False, sample_interval=None):
        """
        Eval action-wise MPJPE
        preds: [N, j, 3]
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval...')

        # read testset
        if (self.subset == 'test'  and getattr(self, 'gt_dataset', False)) or self.seq5678:
            dataitem_gt = self.gt_dataset
        else:
            # read 3d labels
            file_name = 'h36m_test.pkl'
            print('loading %s' % file_name)
            file_path = os.path.join(self.root_path, file_name)
            with open(file_path, 'rb') as f:
                dataitem_gt = pickle.load(f)

        assert len(preds) == len(dataitem_gt)

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        for idx, pred in enumerate(preds):
            gt = dataitem_gt[idx]['joint_3d_camera']
            gt = (gt-gt[0:1])/1000.0
            if protocol2:
                pred = align_to_gt(pose=pred, pose_gt=gt)
            error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
            results.append(error_per_joint)
        results = np.array(results)  # [N ,17]

        # action-wise MPJPE
        final_result = []
        action_index_dict = {}
        for i in range(2, 17):
            action_index_dict[i] = []
        for idx, dataitem in enumerate(dataitem_gt):
            action_index_dict[dataitem['action']].append(idx)
        for i in range(2, 17):
            final_result.append(np.mean(results[action_index_dict[i]]))
        error = np.mean(np.array(final_result))
        final_result.append(error)

        # print error
        if print_verbose:
            table = PrettyTable()
            table.field_names = ['H36M'] + [i for i in range(2, 17)] + ['avg']
            table.add_row(['p2' if protocol2 else 'p1'] + ['%.4f' % d for d in final_result])
            print(table)

        return error

    
    def dataset_eval( self,preds,dataset, protocol2=True, print_verbose=False, sample_interval=None):
        """
        Eval action-wise MPJPE
        preds: [N, j, 3]
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval...')
       
        assert len(preds) == len(dataset)

        if sample_interval is not None:
            preds = preds[::sample_interval]

        # gt_dataset = dataset.datasets[0].gt_dataset+dataset.datasets[1].gt_dataset
        gt_dataset = dataset.gt_dataset
        assert len(preds) == len(gt_dataset)
        results = []
        for idx, pred in enumerate(preds):

            gt = gt_dataset[idx]['joint_3d_camera']
            gt = (gt-gt[0:1])/1000.0
            if protocol2:
                pred = align_to_gt(pose=pred, pose_gt=gt)
            error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
            results.append(error_per_joint)

        results = np.array(results)  # [N ,17]

        # action-wise MPJPE
        final_result = []
        action_index_dict = {}
        for i in range(2, 17):
            action_index_dict[i] = []
        for idx, dataitem in enumerate(gt_dataset):
            action_index_dict[dataitem['action']].append(idx)
        for i in range(2, 17):
            final_result.append(np.mean(results[action_index_dict[i]]))
        error = np.mean(np.array(final_result))
        final_result.append(error)

        return error
    
    def eval_multi(self, preds, protocol2=False, print_verbose=False, sample_interval=None,valid_ind=None):
        """
        Eval action-wise MPJPE
        preds: [N, m, j, 3], N:len of dataset, m: multi-hypothesis number
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval multi-hypothesis...')

        # read testset
        if (self.subset == 'test'  and getattr(self, 'gt_dataset', False)) or self.seq5678:
            dataitem_gt = self.gt_dataset
        else:
            # read 3d labels
            file_name = 'h36m_test.pkl'
            print('loading %s' % file_name)
            file_path = os.path.join(self.root_path, file_name)
            with open(file_path, 'rb') as f:
                dataitem_gt = pickle.load(f)
        assert len(preds) == len(dataitem_gt)

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        multi_preds_cam = []
        max_error = 1000
        max_cluster_index = -1
        max_sample_index =  -1
        for idx, multi_pred in enumerate(preds):
            
            multi_results = []
            pred_store = []
            index = []
            for sec_idx, pred in enumerate(multi_pred):
                if valid_ind is not None and sec_idx not in valid_ind[idx]:
                    continue
                gt = dataitem_gt[idx]['joint_3d_camera']
                gt = (gt-gt[0:1])/1000.0
                pred_store.append(pred)
                if protocol2:
                    pred = align_to_gt(pose=pred, pose_gt=gt)
                error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
                multi_results.append(np.mean(error_per_joint))  # scala
            current_index = np.argmin(multi_results)
            index.append(current_index)
            
            results.append(np.amin(multi_results))  # min error among multi-hypothesis
            if results[-1]<max_error:
                max_error = results[-1]
                max_cluster_index = current_index
                max_sample_index = idx
            multi_preds_cam.append(pred_store)  # [M, j, 3]
        
        results = np.array(results)  # [N]
        index = np.array(index)
        
        # action-wise MPJPE
        print(f'maximum MPJPE error: {max_error} and it is at index: {max_sample_index}, {max_cluster_index}')
        final_result = []
        action_index_dict = {}
        for i in range(2, 17):
            action_index_dict[i] = []
        for idx, dataitem in enumerate(dataitem_gt):
            action_index_dict[dataitem['action']].append(idx)
        for i in range(2, 17):
            final_result.append(np.mean(results[action_index_dict[i]]))
        error = np.mean(np.array(final_result))
        final_result.append(error)

        # print error
        if print_verbose:
            table = PrettyTable()
            table.field_names = ['H36M'] + [i for i in range(2, 17)] + ['avg']
            table.add_row(['p2' if protocol2 else 'p1'] + ['%.5f' % d for d in final_result])
            print(table)

        return error

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
        [8, 14], [14, 15], [15, 16]]

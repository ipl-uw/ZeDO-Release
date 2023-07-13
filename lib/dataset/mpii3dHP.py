import numpy as np
import os, sys
import pickle
from prettytable import PrettyTable
from collections import defaultdict
import heapq
import math as m
import random as random
import torch
from lib.algorithms.advanced.utils import compute_AUC, compute_PCK

from lib.utils.transforms import image_to_camera_frame, align_to_gt
from scipy.spatial.transform import Rotation as R


action_convertor = [15,17,10,18,19,20,21]
dt_len = [6030, 6074, 5619, 5826, 253, 491]
MPII_K = [{'cx': 1017.3768231769433,
  'cy': 1043.0617066309674,
  'fx': 1500.0026763683243,
  'fy': 1500.653563770609},
 {'cx': 1015.2332835036037,
  'cy': 1038.6779735645273,
  'fx': 1503.7547333381692,
  'fy': 1501.2960541197708},
 {'cx': 1017.38890576427,
  'cy': 1043.0479217185737,
  'fx': 1499.9948168861915,
  'fy': 1500.5952584161635},
 {'cx': 1017.3629901820193,
  'cy': 1042.9893946483614,
  'fx': 1499.889694845776,
  'fy': 1500.7589012253272},
 {'cx': 939.9366622036999,
  'cy': 560.196743470783,
  'fx': 1683.4033373885632,
  'fy': 1671.9980973522306},
 {'cx': 939.8504013098557,
  'cy': 560.1146111183259,
  'fx': 1683.9052204148456,
  'fy': 1672.674313185811}]

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


class MPII3DHP:
    def __init__(self, root_path, subset='train', 
        gt2d=True, read_confidence=True, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0,abs_coord=False,rot=False):
        
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.root_path = root_path
        self.subset = subset
        self.gt2d = gt2d
        self.read_confidence = read_confidence
        self.sample_interval = sample_interval
        self.flip = flip
        self.valid_id = None
        self.abs_coord = abs_coord
        self.camera_param = None
        self.confidence = None
        self.rot = rot
        self.image_path=[]

        self.db_2d, self.db_3d, self.gt_dataset,self.valid_id,self.camera_param = self.read_data()

        if self.sample_interval:
            self._sample(sample_interval)

        self.rep = rep
        if self.rep > 1:
            print(f'stack dataset {self.rep} times for multi-sample eval')

        self.real_data_len = len(self.db_2d)

        self.left_joints = [4, 5, 6, 11, 12, 13]
        self.right_joints = [1, 2, 3, 14, 15, 16]

        self.cond_3d_prob = cond_3d_prob
        self.abs_coord = abs_coord

    def __getitem__(self, idx):
        """
        Return: [17, 2], [17, 3] for data and labels
        """
        data_2d = self.db_2d[idx % self.real_data_len]
        data_3d = self.db_3d[idx % self.real_data_len]


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
        
        # print(f'Class MPIIDataset({self.subset}): sample dataset every {sample_interval} frame')
        
            
        if len(self.valid_id)!=0:
            self.db_2d = self.db_2d[self.valid_id,:]
            self.db_3d = self.db_3d[self.valid_id,:]
            self.gt_dataset = [self.gt_dataset[i] for i in self.valid_id]
            self.camera_param = self.camera_param[self.valid_id,:]
            self.image_path = self.image_path[self.valid_id]
            self.db_2d = self.db_2d[::sample_interval]
            self.db_3d = self.db_3d[::sample_interval]
            self.gt_dataset = self.gt_dataset[::sample_interval]
            self.camera_param = self.camera_param[::sample_interval]
            self.image_path = self.image_path[::sample_interval]
            
        else:
            self.db_2d = self.db_2d[::sample_interval]
            self.db_3d = self.db_3d[::sample_interval]
            self.gt_dataset = self.gt_dataset[::sample_interval]
            self.camera_param = self.camera_param[::sample_interval]
            self.image_path = self.image_path[::sample_interval]
            

    def read_data(self):
        # read 3d labels
        if not self.gt2d :
            file_name = 'mpii_gt_%s.pkl' % self.subset
        else :
            
            file_path = 'mpii3d_%s.pkl' % self.subset
            print('loading %s' % file_path)
            file_path = os.path.join(self.root_path, file_path)
            # file_path = os.path.join(self.root_path, file_name)
            with open(file_path, 'rb') as f:
                gt_dataset = pickle.load(f)

            valid_id = []
            # normalize
            
            res_w, res_h = 2048, 2048
            res_w = np.empty((len(gt_dataset)), dtype=np.float32)
            res_h = np.empty((len(gt_dataset)), dtype=np.float32)
            labels_3d = np.empty((len(gt_dataset), 17, 3), dtype=np.float32)  # [N, 17, 3]
            labels_2d = np.empty((len(gt_dataset), 17, 3), dtype=np.float32)  # [N, 17, 3]
            camera_params = np.zeros((len(gt_dataset), 3, 3), dtype=np.float32)
            # map to [-1, 1]
            for idx, item in enumerate(gt_dataset):
                # labels_3d[idx] = item['joint3D_image']
                labels_3d[idx] = item['joint_3d_camera']
                labels_2d[idx] = item['joint_2d']
                res_w[idx] = item['w']
                res_h[idx]= item['h']
                camera_params[idx][0][0] = item['camera_param']['fx']
                camera_params[idx][1][1] = item['camera_param']['fy']
                camera_params[idx][0][2] = item['camera_param']['cx']
                camera_params[idx][1][2] = item['camera_param']['cy']
                camera_params[idx][2][2] = 1
                self.image_path.append(item['imageid'])
                
                if self.subset == 'test' and int(item['valid_i']) == 1:
                    valid_id.append(idx)
                    item['action'] = action_convertor[int(item['action'])-1]
            
            if not self.abs_coord:
                labels_3d = (labels_3d[:,:]-labels_3d[:,0:1])
            labels_3d = labels_3d/1000.0
        # labels_3d = (labels_3d-labels_3d[0:1])/1000
        
        # for idx,item in enumerate(labels_2d):
        #     labels_2d[idx,:,:2] = labels_2d[idx,:, :2]/res_w[idx]*2-[1,res_h[idx] / res_w[idx]]

        # labels_2d[..., :2] = labels_2d[..., :2] / res_w * 2 - [1, res_h / res_w]
        # labels_3d[..., 2:] = labels_3d[..., 2:] / res_w * 2

        # # reshape
        # labels = labels.reshape((-1, 17*3))
       
        # read 2d
        if self.gt2d:
            data_2d = labels_2d[..., :2].copy()  # [N, 17, 2]
            # data_2d = data_2d[:,:] - data_2d[:,0:1]
            # data_2d/=1000.0
            if self.read_confidence:
                data_2d = np.concatenate((data_2d, np.ones((len(data_2d), 17, 1))), axis=-1)  # [N, 17, 3]
        else:
            file_name = 'mpii_dt_test.npz'
            file_path = os.path.join(self.root_path, file_name)
            print('loading dt_2d %s' % file_name)
            # with open(file_path, 'rb') as f:
            #     dt_dataset = np.load(f,allow_pickle=True)

            
            labels_3d,data_2d = self.fetch_3dhp(file_path)
            
            # data_2d =  dt_dataset.item()['positions_2d'][:, :, :2].copy()  # [N, 17, 2]
            # labels_3d =  dt_dataset.item()['positions_3d'][:, :, :].copy()
            camera_params = np.zeros((sum(dt_len),3,3))
            prev = 0
            for num in range(len(dt_len)):
                    cam_p= MPII_K[num]
                    camera_params[prev:prev+dt_len[num]][0][0] = cam_p['fx']
                    camera_params[prev:prev+dt_len[num]][1][1] = cam_p['fy']
                    camera_params[prev:prev+dt_len[num]][0][2] = cam_p['cx']
                    camera_params[prev:prev+dt_len[num]][1][2] = cam_p['cy']
                    camera_params[prev:prev+dt_len[num]][2][2] = 1
                    prev += dt_len[num]
            
            data_2d = np.array(data_2d).astype(np.float32)
            real_data_2d = np.array((data_2d.shape[0],17,2))
            real_data_2d[:,0:10,:] = data_2d[:,0:10,:]
            real_data_2d[:,11:,:] = data_2d[:,10:,:]
            labels_3d = np.array(labels_3d).astype(np.float32)
            real_labels_3d = np.array((labels_3d.shape[0],17,3))
            real_labels_3d[:,0:10,:] = real_labels_3d[:,0:10,:]
            real_labels_3d[:,11:,:] = real_labels_3d[:,10:,:]
            valid_id = []
            
        self.image_path = np.array(self.image_path)

        return data_2d, labels_3d, gt_dataset, np.array(valid_id),camera_params 

    def eval(self, preds, protocol2=False, print_verbose=False, sample_interval=None):
        """
        Eval action-wise MPJPE
        preds: [N, j, 3]
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval...')

        # read testset
        if self.subset == 'test' and getattr(self, 'gt_dataset', False):
            dataitem_gt = self.gt_dataset
        else:
            # read 3d labels
            file_name = 'mpii3d_dt_test.npz'
            print('loading %s' % file_name)
            file_path = os.path.join(self.root_path, file_name)
            with open(file_path, 'rb') as f:
                dataitem_gt = np.load(f)

        # read preds
        # result_path = os.path.join(ROOT_PATH, 'experiment', test_name, 'result_%s.pkl' % mode)
        # with open(result_path, 'rb') as f:
        #     preds = pickle.load(f)['result']  # [N, 17, 3]
        # preds = np.reshape(preds, (-1, 17, 3))

        assert len(preds) == len(dataitem_gt)

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        for idx, pred in enumerate(preds):
            # pred = image_to_camera_frame(pose3d_image_frame=pred, box=dataitem_gt[idx]['box'],
            #     camera=dataitem_gt[idx]['camera_param'], rootIdx=0,
            #     root_depth=dataitem_gt[idx]['root_depth'])
            gt = dataitem_gt[idx]['joint_3d_camera']
            gt = (gt-gt[0:1])/1000.0
            if protocol2:
                pred = align_to_gt(pose=pred, pose_gt=gt)
            error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
            results.append(error_per_joint)
            # if idx % 10000 == 0:
            #     print('step:%d' % idx + '-' * 20)
            #     print(np.mean(error_per_joint))
        results = np.array(results)  # [N ,17]

        # action-wise MPJPE
        final_result = []
        action_index_dict = {}
        temp_range = [15,17,18,19,20,21]
        for i in temp_range:
            action_index_dict[i] = []
        for idx, dataitem in enumerate(dataitem_gt):
            action_index_dict[dataitem['action']].append(idx)
        for i in temp_range:
            if action_index_dict[i] is not None and len(action_index_dict[i])!=0: 
                final_result.append(np.mean(results[action_index_dict[i]]))
        error = np.mean(np.array(final_result))
        final_result.append(error)

        # print error
        temp_range = [15,17,18,19,20,21,10]
        if print_verbose:
            table = PrettyTable()
            table.field_names = ['3DHP'] + [i for i in temp_range] + ['avg']
            table.add_row(['p2' if protocol2 else 'p1'] + ['%.5f' % d for d in final_result])
            print(table)

        return error

    def eval_multi(self, preds, protocol2=False, print_verbose=False, sample_interval=None, valid_ind=None):
        """
        Eval action-wise MPJPE
        preds: [N, m, j, 3], N:len of dataset, m: multi-hypothesis number
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval multi-hypothesis...')
        max_error = 1000
        max_cluster_index = -1
        max_sample_index =  -1

        # read testset
        if self.subset == 'test' and getattr(self, 'gt_dataset', False):
            dataitem_gt = self.gt_dataset
       
            

            assert len(preds) == len(dataitem_gt)

            if sample_interval is not None:
                preds = preds[::sample_interval]

            results = []
            min_pred = []
            multi_preds_cam = []
            for idx, multi_pred in enumerate(preds):
                multi_results = []
                pred_store = []
                for pred in multi_pred:
                    # pred = image_to_camera_frame(pose3d_image_frame=pred, box=dataitem_gt[idx]['box'],
                    #     camera=dataitem_gt[idx]['camera_param'], rootIdx=0,
                    #     root_depth=dataitem_gt[idx]['root_depth'])
                    gt = dataitem_gt[idx]['joint_3d_camera']
                    gt = (gt-gt[0:1])/1000.0
                    pred_store.append(pred)
                    if protocol2:
                        pred = align_to_gt(pose=pred, pose_gt=gt)
                    error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
                    multi_results.append(np.mean(error_per_joint))  # scala
                    
                current_index = np.argmin(multi_results)
                results.append(np.amin(multi_results))  # min error among multi-hypothesis
                min_pred.append(multi_pred[np.argmin(multi_results)])
                multi_preds_cam.append(pred_store)  # [M, j, 3]
            
            results = np.array(results)  # [N]
            min_pred = np.array(min_pred)
            multi_preds_cam = np.array(multi_preds_cam)  # [N, M, j, 3]
            
            if results[-1]<max_error:
                max_error = results[-1]
                max_cluster_index = current_index
                max_sample_index = idx
           
            
            pck = compute_PCK(preds=min_pred.reshape((-1,17,3)),gts=(self.db_3d[:]-self.db_3d[:,0:1,:]))
            auc = compute_AUC(preds=min_pred.reshape((-1,17,3)),gts=(self.db_3d[:]-self.db_3d[:,0:1,:]))
            print('PCK :',pck)
            print('AUC :',auc)
            
            print(f'maximum MPJPE error: {max_error} and it is at index: {max_sample_index}, {max_cluster_index}')
            # diversity in std, expcet root joints
            multi_preds_cam_eval = multi_preds_cam - multi_preds_cam[:, :, [0], :]
            multi_preds_cam_eval = multi_preds_cam_eval[:, :, 1:, :]  # [N, M, j-1, 3]
            print(f'std: x{multi_preds_cam_eval[..., 0].std(axis=1).mean()}, \
                y{multi_preds_cam_eval[..., 1].std(axis=1).mean()}, z{multi_preds_cam_eval[..., 2].std(axis=1).mean()}')

            # action-wise MPJPE
            final_result = []
            action_index_dict = {}
            action_3dhp = [15,10,17,18,19,20,21]
            for i in action_3dhp:
                action_index_dict[i] = []
            for idx, dataitem in enumerate(dataitem_gt):
                action_index_dict[dataitem['action']].append(idx)
            for i in action_3dhp:
                final_result.append(np.mean(results[action_index_dict[i]]))
            error = np.mean(np.array(final_result))
            final_result.append(error)

            # print error
            if print_verbose:
                table = PrettyTable()
                table.field_names = ['3DHP'] + [i for i in action_3dhp] + ['avg']
                table.add_row(['p2' if protocol2 else 'p1'] + ['%.5f' % d for d in final_result])
                print(table)

            return error
        else:
                # read 3d labels
            # file_name = 'mpii3d_dt_test.npz'
            # print('loading %s' % file_name)
            # file_path = os.path.join(self.root_path, file_name)
            # with open(file_path, 'rb') as f:
            #     dataitem_gt = np.load(f)
            # data_3d,data_2d = self.fetch_3dhp(file_path)
            assert len(preds) == len(self.db_3d)    
            
            if sample_interval is not None:
                preds = preds[::sample_interval]
                
            preds[:,:,10,:] = self.db_3d[:,:,10,:]

            results = []
            multi_preds_cam = []
            for idx, multi_pred in enumerate(preds):
                multi_results = []
                pred_store = []
                for pred in multi_pred:
                    # pred = image_to_camera_frame(pose3d_image_frame=pred, box=dataitem_gt[idx]['box'],
                    #     camera=dataitem_gt[idx]['camera_param'], rootIdx=0,
                    #     root_depth=dataitem_gt[idx]['root_depth'])
                    gt = self.db_3d[idx]
                   
                    pred_store.append(pred)
                    if protocol2:
                        pred = align_to_gt(pose=pred, pose_gt=gt)
                    
                    error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
                    multi_results.append(np.mean(error_per_joint))  # scala
                current_index = np.argmin(multi_results)
                results.append(np.amin(multi_results))  # min error among multi-hypothesis
                multi_preds_cam.append(pred_store)  # [M, j, 3]
            results = np.array(results)  # [N]
            multi_preds_cam = np.array(multi_preds_cam)  # [N, M, j, 3]
            print(np.mean(results))
            
            if results[-1]<max_error:
                max_error = results[-1]
                max_cluster_index = current_index
                max_sample_index = idx
            
            
            # multi_preds_cam_eval = multi_preds_cam - multi_preds_cam[:, :, [0], :]
            # multi_preds_cam_eval = np.concatenate([multi_preds_cam_eval[:, :, 1:10, :],multi_preds_cam_eval[:, :, 11:, :]])  # [N, M, j-1, 3]
            # print(f'std: x{multi_preds_cam_eval[..., 0].std(axis=1).mean()}, \
            #     y{multi_preds_cam_eval[..., 1].std(axis=1).mean()}, z{multi_preds_cam_eval[..., 2].std(axis=1).mean()}')

            # action-wise MPJPE
            # final_result = []
            # action_index_dict = {}
            # action_3dhp = [15,10,17,18,19,20,21]
            # for i in action_3dhp:
            #     action_index_dict[i] = []
            # for idx, dataitem in enumerate(dataitem_gt):
            #     action_index_dict[dataitem['action']].append(idx)
            # for i in action_3dhp:
            #     final_result.append(np.mean(results[action_index_dict[i]]))
            # error = np.mean(np.array(final_result))
            # final_result.append(error)

            # print error
            # if print_verbose:
            #     table = PrettyTable()
            #     table.field_names = ['3DHP'] + [i for i in action_3dhp] + ['avg']
            #     table.add_row(['p2' if protocol2 else 'p1'] + ['%.5f' % d for d in final_result])
                # print(table)
            print(f'maximum MPJPE error: {max_error} and it is at index: {max_sample_index}, {max_cluster_index}')
            return error
            
            
    def fetch_3dhp(self,data_path):
        
        data = np.load(data_path, allow_pickle=True)
        data3d = data['positions_3d'].item()
        data2d= data['positions_2d'].item()

        data_3d=[]
        data_2d=[] 
        subjects=['TS1','TS2','TS3','TS4','TS5','TS6']  
        for subject in subjects:
            data3d[subject]-=data3d[subject][:,:1]
            # row=np.sum(np.abs(data2d[subject])>1,axis=(-1,-2))==0
            # data_3d.append(data3d[subject][row]/1000)
            # data_2d.append(data2d[subject][row])   
            if subject in ['TS3','TS4']:
                
                data_3d.append(data3d[subject][100:]/1000)
                data_2d.append(data2d[subject][100:])
            else:
                
                data_3d.append(data3d[subject]/1000)
                data_2d.append(data2d[subject])       
        
        
        return data_3d,data_2d

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
        [8, 14], [14, 15], [15, 16]]

import numpy as np
import os
import pickle
from prettytable import PrettyTable
import math as m
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from lib.utils.transforms import align_to_gt

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
    flipped_data[:, left_joints+right_joints] = flipped_data[:,
                                                             right_joints+left_joints]
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


class PW3D:
    def __init__(self, root_path, subset='train',
                 gt2d=True, read_confidence=True, sample_interval=None, rep=1,
                 flip=False, cond_3d_prob=0, abs_coord=False, seq1=False, seq5678=False, rot=False):

        self.w = None
        self.h = None
        self.image_name = None
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
        self.seq1 = seq1
        self.seq5678 = seq5678
        self.rot = rot
        self.order = [5, 2, 6, 3, 11, 14, 12, 15, 13, 16, 1, 4, 8, 10, 0, 7, 9]

        self.db_2d, self.db_3d, self.camera_param, self.w, self.h, self.image_name = self.read_data()

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
            data[self.left_joints +
                 self.right_joints] = data[self.right_joints+self.left_joints]
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
            noise = std * \
                (np.random.rand(*pose2d.shape).astype(np.float32) - 0.5)
            pose2d = pose2d + noise
        else:
            raise NotImplementedError
        return pose2d

    def _sample(self, sample_interval):
        print(
            f'Class 3DPWDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.db_2d = self.db_2d[::sample_interval]
        self.db_3d = self.db_3d[::sample_interval]
        self.h = self.h[::sample_interval]
        self.w = self.w[::sample_interval]
        self.camera_param = self.camera_param[::sample_interval]
        self.image_name = self.image_name[::sample_interval]

    def order_change(self, data):

        b = np.empty_like(data)
        for i in range(17):
            b[self.order[i]] = data[i]
        return b

    def read_data(self):
        # read 3d labels
        file_name = 'pw3d_%s.npz' % self.subset

        print('loading %s' % file_name)
        file_path = os.path.join(self.root_path, file_name)

        data = np.load(file_path, allow_pickle=True)

        labels_3d = []
        labels_2d = []
        camera_params = []
        w = []
        h = []
        image_name = []

        print(len(data['keypoints3d17_relative']))
        data_keypoints3d = data['keypoints3d17_relative']
        data_root_cam = data['root_cam']
        data_cam_param = data['cam_param'].item()
        width = data['image_width']
        height = data['image_height']
        imgpath = data['image_path']
        for i in tqdm(range(len(data_keypoints3d))):
            keypoints3d = data_keypoints3d[i, :,
                                           :3] + data_root_cam[i, None, :]
            keypoints3d = self.order_change(keypoints3d)

            labels_3d.append(keypoints3d)
            temp_camera_params = np.array([[data_cam_param['f'][i, 0], 0, data_cam_param['c'][i, 0]],
                                           [0, data_cam_param['f'][i, 1],
                                               data_cam_param['c'][i, 1]],
                                           [0, 0, 1]])

            keypoint2d = temp_camera_params.dot(keypoints3d.T).T
            keypoint2d = (keypoint2d / keypoint2d[:, 2:])
            camera_params.append(temp_camera_params)
            labels_2d.append(keypoint2d)
            w.append(width[i])
            h.append(height[i])
            image_name.append(imgpath[i])

        labels_3d = np.array(labels_3d, dtype=np.float32)
        labels_2d = np.array(labels_2d, dtype=np.float32)
        camera_params = np.array(camera_params, dtype=np.float32)
        w = np.array(w, dtype=np.float32)
        h = np.array(h, dtype=np.float32)
        if not self.abs_coord:
            labels_3d = labels_3d[:, :] - labels_3d[:, 0:1]

        return labels_2d, labels_3d,  camera_params, w, h, image_name

    def eval(self, preds, protocol2=False, print_verbose=False, sample_interval=None):
        """
        Eval action-wise MPJPE
        preds: [N, j, 3]
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval...')

        # read testset
        if (self.subset == 'test' and getattr(self, 'gt_dataset', False)) or self.seq5678:
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
            table.add_row(['p2' if protocol2 else 'p1'] +
                          ['%.4f' % d for d in final_result])
            print(table)

        return error

    def eval_multi(self, preds, protocol2=False, print_verbose=False, sample_interval=None, valid_ind=None, joint=17):
        """
        Eval action-wise MPJPE
        preds: [N, m, j, 3], N:len of dataset, m: multi-hypothesis number
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval multi-hypothesis...')

        assert len(preds) == len(self.db_3d)

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        for idx, multi_pred in enumerate(preds):

            multi_results = []
            pred_store = []
            index = []
            for sec_idx, pred in enumerate(multi_pred):
                if valid_ind is not None and sec_idx not in valid_ind[idx]:
                    continue

                gt = self.db_3d[idx]
                #0 is pelvis, 7 is spine, 9 is neck-base or nose,10 is head
                gt = (gt-gt[0:1])
                gt_14 = np.empty((14, 3))
                gt_14[0:6, :] = gt[1:7, :]
                gt_14[6:7, :] = gt[8:9, :]
                gt_14[7:, :] = gt[10:, :]

                if protocol2:
                    pred = align_to_gt(pose=pred, pose_gt=gt)
                pred_14 = np.empty((14, 3))
                pred_14[0:6, :] = pred[1:7, :]
                pred_14[6:7, :] = pred[8:9, :]
                pred_14[7:, :] = pred[10:, :]
                pred_store.append(pred_14)
                error_per_joint = np.sqrt(
                    np.square(pred-gt).sum(axis=1))  # [17]
                # error_per_joint = np.sqrt(np.square(pred_14-gt_14).sum(axis=1))  # [14]

                multi_results.append(np.mean(error_per_joint))  # scala
            current_index = np.argmin(multi_results)
            index.append(current_index)

            # min error among multi-hypothesis
            results.append(np.amin(multi_results))

        results = np.array(results)  # [N]
        index = np.array(index)
        error = np.mean(results)

        if protocol2:
            print(f'mean PA-MPJPE : {error}')
        else:
            print(f'mean MPJPE : {error}')

        return error

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]]

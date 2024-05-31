import numpy as np
import os
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from lib.utils.transforms import align_to_gt

class CustomDataset:
    def __init__(self, 
                 root_path,
                 sample_interval=None):

        self.w = None
        self.h = None
        self.image_name = None
        self.root_path = root_path
        self.sample_interval = sample_interval
        self.camera_param = None

        self.db_2d, self.db_3d, self.camera_param, self.image_name = self.read_data()

        if self.sample_interval:
            self._sample(sample_interval)

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
        return data_2d, data_3d

    def __len__(self,):
        return len(self.db_2d)

    def _sample(self, sample_interval):
        print(
            f'Class CustomDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.db_2d = self.db_2d[::sample_interval]
        self.db_3d = self.db_3d[::sample_interval]

        self.camera_param = self.camera_param[::sample_interval]

    def read_data(self):
        # read 3d labels
        #TODO read 2d keypoints: [N, 17, 3] with confidence score 
        #          3d keypoints: [N, 17, 3] - for evaluation purpose only. can be zero array for inferencing
        #          camera parameters: [N, 3, 3]
        #          image name: [N]

        return labels_2d, labels_3d,  camera_params, image_name

    def eval_multi(self, preds, protocol2=False, print_verbose=False, sample_interval=None, valid_ind=None):
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
                gt = (gt-gt[0:1])
                pred_store.append(pred)
                if protocol2:
                    pred = align_to_gt(pose=pred, pose_gt=gt)
                error_per_joint = np.sqrt(
                    np.square(pred-gt).sum(axis=1))  # [17]
                multi_results.append(np.mean(error_per_joint))  # scala
            current_index = np.argmin(multi_results)
            index.append(current_index)

            # min error among multi-hypothesis
            results.append(np.amin(multi_results))

        results = np.array(results)  # [N]
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

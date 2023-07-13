import numpy as np
import os
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from lib.utils.transforms import align_to_gt

class skiPose:
    def __init__(self, root_path, subset='train',
                 gt2d=True, read_confidence=True, sample_interval=None, rep=1,
                 flip=False, cond_3d_prob=0, abs_coord=False, rot=False):

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
        self.rot = rot

        self.db_2d, self.db_3d, self.camera_param, self.image_name = self.read_data()

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
            f'Class H36MDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.db_2d = self.db_2d[::sample_interval]
        self.db_3d = self.db_3d[::sample_interval]

        self.camera_param = self.camera_param[::sample_interval]

    def order_change(self, data):

        b = np.empty_like(data)
        for i in range(17):
            b[self.order[i]] = data[i]
        return b

    def read_data(self):
        # read 3d labels
        file_name = os.path.join(self.root_path, 'ski_test.h5')
        h5_label_file = h5py.File(file_name, 'r')

        print('loading %s' % file_name)

        labels_3d = []
        labels_2d = []
        camera_params = []
        image_name = []

        for index in tqdm(range(len(h5_label_file['seq']))):

            cam = h5_label_file['cam_intrinsic'][index]
            cam = cam * 256
            cam[2, 2] = 1
            pose_3D = h5_label_file['3D'][index].reshape([-1, 3])
            pose_2D = np.ones_like(pose_3D)
            pose_2D[:, :2] = h5_label_file['2D'][index].reshape(
                [-1, 2])*256  # in range 0..1
            seq = int(h5_label_file['seq'][index])
            camId = int(h5_label_file['cam'][index])
            frame = int(h5_label_file['frame'][index])
            image_name.append(
                'test/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(seq, camId, frame))
            labels_3d.append(pose_3D)

            labels_2d.append(pose_2D)
            camera_params.append(cam)

        labels_3d = np.array(labels_3d, dtype=np.float32)
        labels_2d = np.array(labels_2d, dtype=np.float32)
        camera_params = np.array(camera_params, dtype=np.float32)

        if not self.abs_coord:
            labels_3d = labels_3d[:, :] - labels_3d[:, 0:1]

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

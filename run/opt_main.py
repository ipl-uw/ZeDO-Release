import os
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# torch related
import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from lib.algorithms.advanced.simple_zeroshot_opt import gradient_field_gen, RotOpt

from lib.dataset.h36m import H36MDataset3D
from lib.dataset.mpii3dHP import MPII3DHP
from lib.dataset.pw3d import PW3D
from lib.dataset.skiPose import skiPose
from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.advanced import sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage

import copy
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

N_JOINTS = 17
JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3

output_dir = './output'


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='valid score model')
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--ckpt_name', type=str)
    parser.add_argument('--gt', action='store_true', default=False, help='use gt2d as condition')
    parser.add_argument('--hypo', type=int, default=1, help='number of hypotheses')
    args = parser.parse_args(argv[1:])

    return args

def normalize(values, actual_bounds, desired_bounds):
    return [desired_bounds[0] + (x - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0]) for x in values]

def main(args):
    config = FLAGS.config

    if config.data.dataset == 'h36m':
        sample_poses = np.load(f'clusters/h36m_cluster{args.hypo}.npy')
    elif config.data.dataset == '3dhp':
        sample_poses = np.load(f'clusters/3dhp_cluster{args.hypo}.npy')
    elif config.data.dataset == '3dpw':
        sample_poses = np.load(f'clusters/h36m_cluster{args.hypo}.npy')
    elif config.data.dataset == 'ski':
        sample_poses = np.load(f'clusters/h36m_sitting_cluster{args.hypo}.npy')
    device = torch.device("cuda")

    ''' setup score networks '''
    model = ScoreModelFC_Adv(
        config,
        n_joints=N_JOINTS,
        joint_dim=JOINT_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        cond_dim=CONDITION_DIM,
    )
    model.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=None, model=model, ema=ema, step=0)
    
    if config.data.dataset == 'h36m':
        test_dataset = H36MDataset3D(
                Path('data', 'h36m'),
                'test',
                gt2d=args.gt,
                abs_coord=True,
                sample_interval=config.ZeDO.sample,
                flip=False)
    elif config.data.dataset == '3dhp':
        test_dataset = MPII3DHP(
                Path('data', '3dhp'),
                'test',
                gt2d=args.gt,
                abs_coord=True,
                sample_interval=config.ZeDO.sample,
                flip=False)
    elif config.data.dataset == '3dpw':
        test_dataset = PW3D(
                Path('data', '3dpw'),
                'test',
                gt2d=args.gt,
                abs_coord=True,
                sample_interval=config.ZeDO.sample,
                flip=False)
    elif config.data.dataset == 'ski':
        test_dataset = skiPose(
                Path('data', 'ski'),
                'test',
                gt2d=args.gt,
                abs_coord=True,
                sample_interval=config.ZeDO.sample,
                flip=False)

    gt_3d = test_dataset.db_3d  # [n, j, 3]
    K = test_dataset.camera_param #[n, 3, 3] 
    
    gt_2d = test_dataset.db_2d 
    
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
                
    print(f'loading model from {ckpt_path}')
    map_location = {'cuda:0': 'cuda:0'}
    
    old_checkpoint = torch.load(ckpt_path, map_location=map_location)
    
    checkpoint = copy.deepcopy(old_checkpoint)
    checkpoint['model_state_dict'] = {}
        
    for k, v in old_checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        checkpoint['model_state_dict'][name] = v

    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema'])
    state['step'] = checkpoint['step']
    print(f"=> loaded checkpoint '{ckpt_path}' (step {state['step']})")

    model.eval()
    inverse_scaler = lambda x: x

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
            N=config.model.num_scales, T=config.model.t)
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
            N=config.model.num_scales, T=config.model.t)
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
            N=config.model.num_scales, T=config.model.t)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Setup sampling functions
    sampling_shape = (config.ZeDO.batch, N_JOINTS, JOINT_DIM)
    config.sampling.probability_flow = True
    sampling_eps = config.ZeDO.sampling_eps
    sampling_fn = sampling.get_sampling_fn(config, sde,
        sampling_shape, inverse_scaler, sampling_eps, device=device)

    assert config.ZeDO.batch == len(gt_3d), f'batch: {config.ZeDO.batch}, dataset len: {len(gt_3d)}'
    
    batch_results = []
    
    for sid in tqdm(range(args.hypo)):
        noisy_gt_3d = torch.ones_like(torch.tensor(gt_3d))
        noisy_gt_3d = noisy_gt_3d * torch.tensor(sample_poses - sample_poses[:, 0:1, :])[sid:sid+1, :, :]

        condition = torch.tensor(gt_2d[:, :, :2], device=device).float()
        conf = torch.tensor(gt_2d[:,:,2],device=device).float()

        denoise_x = noisy_gt_3d[:].clone().to(device)
        
        K = torch.tensor(K,device=device).float()
        
        pelvis_keypoints = torch.cat((condition[:, 0, :], torch.ones((condition.shape[0], 1), device=device)), axis=-1)
        T = torch.inverse(K).bmm(pelvis_keypoints[:, :, None]).permute(0, 2, 1)
        T = T / torch.norm(T, dim=-1, keepdim=True) * config.ZeDO.IPO_T
        rot_opt = RotOpt(denoise_x.shape[0], axis=config.ZeDO.RotAxes, 
                         minT=config.ZeDO.IPO_minScaleT,
                         maxT=config.ZeDO.IPO_maxScaleT)
        rot_opt.to(device)
        rot_optimizer = optim.Adam(rot_opt.parameters(), lr=0.1)
        criterion = torch.nn.L1Loss(reduction='none')
        keypoint_list = config.ZeDO.IPO_keylist
        for i in range(config.ZeDO.IPO_iterations):
            rot_optimizer.zero_grad()
            rot2d = rot_opt(denoise_x[:, keypoint_list, :], T, K)
            loss = criterion(rot2d[:, :, :2], condition[:, keypoint_list, :2])
            loss = torch.mean(loss)
            loss.backward()
            rot_optimizer.step()
        T = T * torch.clamp(rot_opt.scale, min=0.5, max=2)
        rot_mat = rot_opt.generate_matrix()
        
        sample_num = config.ZeDO.OIL_iterations
        timestamp = torch.linspace(sde.T, sampling_eps, sample_num, device=device)

        with torch.no_grad():
            denoise_x = rot_mat.bmm(denoise_x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            for i in range(0, sample_num): 
                if i < sample_num // 5:
                    joint_gradient = gradient_field_gen(condition,denoise_x,K,t=T,conf=conf,returnT=False)
                else:
                    joint_gradient,T = gradient_field_gen(condition,denoise_x,K,conf=conf,returnT = True)
                    
                denoise_x += joint_gradient

                trajs, results = sampling_fn(
                        model,
                        condition=condition * 0,
                        gradient = joint_gradient,
                        denoise_x=denoise_x,
                        t = timestamp[i],
                        t_step = i,
                        args=args
                    ) 
                
                denoise_x = torch.tensor(results).to(device)
            
            batch_results.append(results)
            
    batch_results = np.swapaxes(np.array(batch_results),0,1)
    
    print('eval...')
    test_dataset.eval_multi(batch_results, protocol2=False, print_verbose=True)
    test_dataset.eval_multi(batch_results, protocol2=True, print_verbose=True)

if __name__ == '__main__':

    app.run(main, flags_parser=parse_args)
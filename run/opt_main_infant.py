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
from lib.dataset.syrip import syrip
from lib.dataset.skiPose import skiPose
from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.advanced.control_model import Control_ScoreModelFC_Adv
from lib.algorithms.advanced import sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage
from lib.algorithms.advanced.model_cond import ScoreModelFC_Adv_cond
# from lib.algorithms.advanced.action_cls import actionCLS
from scipy.cluster import vq

import copy
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from lib.dataset.mini_rgbd import mini_rgbd

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

N_JOINTS = 17
JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3
torch.manual_seed(0)
np.random.seed(0)
CHANGE_TABLE = [0,2,5,11,1,4,10,3,9,12,15,13,18,20,14,19,21] 


output_dir = './output'


def find_closest(data,dataset):
    dist = torch.norm(torch.tensor(dataset)-torch.tensor(data), dim=-1, p=None)
    dist = torch.sum(dist,dim=-1)
    nearest = dist.topk(1, largest=False)     
    print(nearest.indices[0].item())
    nearest = dataset[nearest.indices[0].item(),:,:]
    return nearest

def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='valid score model')
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--ckpt_name', type=str)
    parser.add_argument('--gt', action='store_true', default=False, help='use gt2d as condition')
    parser.add_argument('--hypo', type=int, default=1, help='number of hypotheses')
    parser.add_argument('--control',  default=False, action='store_true')
    parser.add_argument('--cond',  default=False, action='store_true')
    args = parser.parse_args(argv[1:])

    return args


    
def get_dataloader(subset='train', sample_interval=None, gt2d=False, flip=False,num_replicas=1, rank=0, cond_3d_prob=0,abs=False,
                   config=None, rot=False,cond=False):
    
    if config.data.dataset == 'mini':
        dataset = mini_rgbd(
        subset,gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob,
        rot=rot,
        num_joint=17)  
    elif config.data.dataset == 'syrip':
        dataset = syrip(
        subset,gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob,
        rot=rot,
        num_joint=12) 
  
        
    
    
    if subset == 'train':
        dataloader = DataLoader(dataset, 
            batch_size=FLAGS.config.training.batch_size * len(config.GPUs), 
            shuffle=True, 
            num_workers=8,
            pin_memory=True)
    else:
        dataloader = DataLoader(dataset, 
            batch_size=FLAGS.config.eval.batch_size * len(config.GPUs),
            shuffle=False, 
            num_workers=8,
            pin_memory=True)

    return dataloader, dataset
def main(args):
    config = FLAGS.config

    
    device = torch.device("cuda")

    ''' setup score networks '''
    if args.control:
        model = Control_ScoreModelFC_Adv(
            config,
            n_joints=config.DATASET.NUM_JOINT,
            joint_dim=JOINT_DIM,
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBED_DIM,
            # n_blocks=1,
        )
    elif  args.cond:
            model = ScoreModelFC_Adv_cond(
            config,
            n_joints=config.DATASET.NUM_JOINT,
            joint_dim=JOINT_DIM,
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBED_DIM,
            # n_blocks=1,
        )
    else:
        model = ScoreModelFC_Adv(
            config,
            n_joints=config.DATASET.NUM_JOINT,
            joint_dim=JOINT_DIM,
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBED_DIM,
            # n_blocks=1,
        )
    model.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=None, model=model, ema=ema, step=0)
    skeletons = [ [0, 1], [1, 2], [3, 4], [4, 5],
                       [6, 7],
                    [7,8], [9,10],[10,11]]

    train_loader, train_dataset = get_dataloader('train', 1, True,
        cond_3d_prob=config.training.cond_3d_prob, config=config,cond=args.cond)  # always sample testset to save time
    
    test_loader, test_dataset = get_dataloader('validate', 1, True, flip=False,rot=False,
        cond_3d_prob=0, config=config,cond=args.cond)  # always sample testset to save time
    
    ckpt_path_root = args.ckpt_dir
                
    print(f'loading model from {ckpt_path_root}')
    map_location = {'cuda:0': 'cuda:0'}
   
    ckpt_path = os.path.join(ckpt_path_root,args.ckpt_name)
    old_checkpoint = torch.load(ckpt_path, map_location=map_location)
    
    checkpoint = copy.deepcopy(old_checkpoint)
    checkpoint['model_state_dict'] = {}
        
    for k, v in old_checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        checkpoint['model_state_dict'][name] = v
        
    

    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    ema.load_state_dict(checkpoint['ema'])
    state['step'] = checkpoint['step']
    print(f"=> loaded checkpoint '{ckpt_path}' (step {state['step']})")

    model.eval()
    inverse_scaler = lambda x: x
    
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
            N=config.model.num_scales,T=config.model.t)
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
            N=config.model.num_scales, T=config.model.t)
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
            N=config.model.num_scales,T=config.model.t)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Setup sampling functions
    sampling_shape = (config.ZeDO.batch, N_JOINTS, JOINT_DIM)
    config.sampling.probability_flow = True
    sampling_eps = config.ZeDO.sampling_eps
    sampling_fn = sampling.get_sampling_fn(config, sde,
        sampling_shape, inverse_scaler, sampling_eps, device=device)

    
    batch_results = []
    N_CLUSTER = args.hypo
    TABLE17TO12 = [-1,-3,-5,-6,-4,-2,-7,-9,-11,-12,-10,-8]
    
    num = 2022

    projection_error = []
    
    train_gt_3d = train_dataset.db_3d
    gt_3d = test_dataset.db_3d
    gt_2d = test_dataset.db_2d
    
    K  = test_dataset.K if config.data.dataset == 'syrip' else None
    
    
    if config.data.dataset == 'mini' :
        fx = 588.67905803875317
        fy = 590.25690113005601
        cx = 322.22048191353628
        cy = 237.46785983766890
        K = np.zeros((gt_2d.shape[0],3, 3), dtype=np.float32)
        K[:,0,0] = fx
        K[:,1,1] = fy
        K[:,0,2] = cx
        K[:,1,2] = cy
        K[:,2,2] = 1
    

    
    if config.data.dataset == 'syrip':
    
        sample_poses = train_dataset.db_3d[0]
        
        
    else:
        
        
        sample_poses = np.load (f'mini_cluster_{N_CLUSTER}.npy') # one can use a random training sample for a cluster center 
        sample_poses = sample_poses[0][test_dataset.change]
        sample_poses = sample_poses.reshape(-1,17,3)
        
   
    for sid in tqdm(range(args.hypo)):

        noisy_gt_3d = torch.ones_like(torch.tensor(gt_3d))
        noisy_gt_3d = noisy_gt_3d * torch.tensor(sample_poses) 
        denoise_x = noisy_gt_3d[:].clone().to(device)       
        condition = torch.tensor(gt_2d[:, :, :2], device=device).float()
        
        K = torch.tensor(K,device=device).float()
       
        if config.data.dataset == 'mini':
            pelvis_keypoints = torch.cat((condition[:, 0, :], torch.ones((condition.shape[0], 1), device=device)), axis=-1)
        else:
            pelvis_keypoints = torch.cat(((condition[:, 0, :]+condition[:,3,:])/2, torch.ones((condition.shape[0], 1), device=device)), axis=-1)
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
        T = T * torch.clamp(rot_opt.scale, min=config.ZeDO.IPO_minScaleT, max=config.ZeDO.IPO_maxScaleT)
        rot_mat = rot_opt.generate_matrix()
        ray = torch.inverse(K).bmm(torch.cat((condition, 
                                            torch.ones((condition.shape[0], condition.shape[1], 1), device=device)), 
                                            axis=-1).permute(0, 2, 1)).permute(0, 2, 1)
        
        if config.data.dataset == 'mini':
            ray = ray / torch.norm(ray[:, 0:1, :], dim=-1, keepdim=True)
            ray = ray * torch.norm(T, dim=-1, keepdim=True) #+ torch.randn((ray.shape[0], ray.shape[1], 1)).cuda() * ray / 20
            denoise_x = (ray - ray[:, 0:1, :]).contiguous()
        else:
            ray = ray / torch.norm((ray[:, 0:1, :]+ray[:, 3:4, :])/2, dim=-1, keepdim=True)
            ray = ray * torch.norm(T, dim=-1, keepdim=True) #+ torch.randn((ray.shape[0], ray.shape[1], 1)).cuda() * ray / 20
            denoise_x = (ray - (ray[:, 0:1, :]+ray[:, 3:4, :])/2).contiguous()

        sample_num = config.ZeDO.OIL_iterations
        

        batch_results = []
        vis_output = []
        timestamp = torch.linspace(sde.T, sampling_eps, sample_num, device=device)
        with torch.no_grad():
            
            
            denoise_x = rot_mat.bmm(denoise_x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    
            for i in range(0, sample_num): 
                
                proj2d = K.bmm((denoise_x+T).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
                proj2d = proj2d[...,:2]/proj2d[...,2:]
                projection_error.append(abs(proj2d-condition).cpu().numpy())
                if i < 950:
                    joint_gradient,ray2d = gradient_field_gen(condition,denoise_x,K,t=T,conf=None,returnT=False)
                else:
                    joint_gradient,ray2d,T = gradient_field_gen(condition,denoise_x,K,conf=None,returnT = True)
                    
                
                hip_length = torch.zeros((len(skeletons),denoise_x.shape[0]))
                for ip,sk in enumerate(skeletons):
                    hip_length[ip] = torch.norm(denoise_x[:,sk[0]] - denoise_x[:,sk[1]],dim=-1,keepdim=True).squeeze(1)
            
                hip_length= torch.max(hip_length,dim=0).values
               
                
                denoise_x += joint_gradient
                
                trajs, results = sampling_fn(
                        model,
                        condition=condition * 0,
                        denoise_x=denoise_x,
                        t = timestamp[i],
                        t_step = i,
                        args=args
                    ) 
                
                denoise_x = torch.tensor(results).to(device)
                
                
        
        batch_results.append(results)
        
    batch_results = np.swapaxes(np.array(batch_results),0,1)
       
        
    
    print('eval...')
    
    test_dataset.eval_multi(batch_results, protocol2=False, print_verbose=False)
           
    

if __name__ == '__main__':

    app.run(main, flags_parser=parse_args)






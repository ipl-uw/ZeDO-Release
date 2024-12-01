import os
import numpy as np
import functools
import pprint
import sys
import traceback
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from lib.utils.transforms import image_to_camera_frame, align_to_gt

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.dataset.EvaSampler import DistributedEvalSampler

# torch related
import torch
import pickle
import copy
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

from lib.utils.generic import create_logger



from lib.algorithms.advanced.model import ScoreModelFC_Adv
# from torch.utils.data import Dataset, DataLoader
from lib.dataset.mini_rgbd import mini_rgbd
from lib.algorithms.advanced import losses, sampling_train, sde_lib
from lib.algorithms.ema import ExponentialMovingAverage
# from lib.algorithms.advanced.model  import ScoreModelInfant
from lib.dataset.h36m import H36MDataset3D, denormalize_data
from lib.dataset.syrip import syrip
from lib.dataset.mpii3dHP import MPII3DHP
# torch.manual_seed(0)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

# global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3
# BATCH_SIZE = 100000
# TEST_BATCH_SIZE = 10000
N_EPOCHES = 8000
EVAL_FREQ = 500  # 20

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def dataset_eval( preds,dataset, protocol2=False, print_verbose=False, sample_interval=None,concate=False):
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
        if concate:
            gt_dataset = dataset.datasets[0].gt_dataset+dataset.datasets[1].gt_dataset
        else:
            gt_dataset = dataset.gt_dataset
        assert len(preds) == len(gt_dataset)
        results = []
        for idx, pred in enumerate(preds):
            # pred = image_to_camera_frame(pose3d_image_frame=pred, box=dataitem_gt[idx]['box'],
            #     camera=dataitem_gt[idx]['camera_param'], rootIdx=0,
            #     root_depth=dataitem_gt[idx]['root_depth'])
            gt = gt_dataset[idx]['joint_3d_camera']
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
        for i in range(2, 22):
            action_index_dict[i] = []
        for idx, dataitem in enumerate(gt_dataset):
            action_index_dict[dataitem['action']].append(idx)
        for i in range(2, 22):
            if action_index_dict[i] is not None and len(action_index_dict[i])!=0:
                final_result.append(np.mean(results[action_index_dict[i]]))
        error = np.mean(np.array(final_result))
        final_result.append(error)

        return error

def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='train score model')

    # parser.add_argument('--prior-t0', type=float, default=0.5)
    # parser.add_argument('--test-num', type=int, default=20)
    # parser.add_argument('--sample-steps', type=int, default=2000)
    parser.add_argument('--restore-dir', type=str)
    parser.add_argument('--sample', type=int, help='sample trainset to reduce data')
    parser.add_argument('--flip', default=False, action='store_true', help='random flip pose during training')
    parser.add_argument('--restore_dir', default=False)
    parser.add_argument('--rotflip', default=False, action='store_true')
    # parser.add_argument('--gpus', type=int, help='num gpus to inference parallel')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    # optional
    parser.add_argument('--name', type=str, default='', help='name of checkpoint folder')
    parser.add_argument('--log_name', type=str, help='name of log folder')
    parser.add_argument('--aug', default=False, action='store_true',help='name of log folder')
    parser.add_argument('--scaled', default=False, action='store_true',help='name of log folder')
    args = parser.parse_args(argv[1:])

    return args


def get_dataloader(subset='train', sample_interval=None, gt2d=False, flip=False,num_replicas=1, rank=0, cond_3d_prob=0,
                   config=None, rot=False, fine_tune=False,aug=False):
    
   
    
        
    
    if   config.data.dataset == 'syrip_concat':
        dataset = mini_rgbd(
        subset,gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob,
        rot=rot,
        num_joint=config.DATASET.NUM_JOINT) 
        
        dataset2 = syrip(
        subset,gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob,
        rot=rot,
        num_joint=config.DATASET.NUM_JOINT) 
        # dataset2 = dataset2[0:200]
        dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
        # import ipdb;ipdb.set_trace()
        
    elif config.data.dataset == 'mini':
        dataset = mini_rgbd(
        subset,gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob,
        rot=rot,
        num_joint=17,aug=aug)  
    elif config.data.dataset == 'syrip':
        dataset = syrip(
        subset,gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob,
        rot=rot,
        num_joint=config.DATASET.NUM_JOINT,aug=aug) 
        
    
    if subset == 'train':
        # train_labels = torch.FloatTensor(train_labels).reshape((-1, 17, 3)) # [N, 17, 3]
        dataloader = DataLoader(dataset, 
            batch_size=FLAGS.config.training.batch_size * len(config.GPUs), 
            shuffle=True, 
            num_workers=8,
            # sampler = sampler,
            pin_memory=True)
    else:
        # test_labels = torch.FloatTensor(test_labels).reshape((-1, 17, 3)) # [N, 17, 3]
        dataloader = DataLoader(dataset, 
            batch_size=FLAGS.config.eval.batch_size * len(config.GPUs),
            shuffle=False, 
            num_workers=8,
            # sampler = sampler,
            pin_memory=True)

    return dataloader, dataset

def main(args):
    # args = parse_args()
    config = FLAGS.config
    
    writer = SummaryWriter()

    
    # setup(rank, args.gpus)
    device = torch.device("cuda")

    logger, final_output_dir, tb_log_dir = create_logger(
            config, 'train', folder_name=args.name,log_name=args.log_name)
    logger.info(pprint.pformat(config))
    logger.info(pprint.pformat(args))
    writer = SummaryWriter(tb_log_dir)

    train_loader, train_dataset = get_dataloader('train', 1, True, flip=args.rotflip,
        cond_3d_prob=config.training.cond_3d_prob, config=config,rot=args.rotflip,fine_tune=args.fine_tune,aug=args.aug)
    test_loader, test_dataset = get_dataloader('validate', 10000, True, flip=False,rot=False,
        cond_3d_prob=0, config=config,fine_tune=args.fine_tune)  # always sample testset to save time
    logger.info(f'total train samples: {len(train_dataset)}')
    logger.info(f'total test samples: {len(test_dataset)}')

    ''' setup score networks '''
    # checkpoint = copy.deepcopy(old_checkpoint)
    # print(checkpoint.keys())
    
    # if args._concate_bb or args._3dhp_bb:
    #     checkpoint['model_state_dict'] = {}
        
    #     for k, v in old_checkpoint['model_state_dict'].items():
    #         name = k[7:] # remove `module.`
    #         checkpoint['model_state_dict'][name] = v
    # old_keys = list(checkpoint['model_state_dict'].keys())
    # old_keys.remove('module.pre_dense_cond.weight')
    # old_keys.remove('module.pre_dense_cond.bias')
    # import ipdb;ipdb.set_trace()
    model = ScoreModelFC_Adv(
        config,
        config.DATASET.NUM_JOINT,
        joint_dim=JOINT_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        # n_blocks=1,
    )
    
    
    
    
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # if args.restore_dir is False:
    # model.freeze()
    # model.copy_weight()
    
    # model.freeze()
    # import ipdb;ipdb.set_trace()
    # model.to(device)
    model = torch.nn.DataParallel(model, device_ids=config.GPUs).cuda()
    
    # model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=LR, betas=(BETA1, 0.999))
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    
    # optimizer = losses.get_optimizer(config, model.parameters())
    optimizer = losses.get_optimizer(config,filter(lambda p: p.requires_grad, model.parameters()))
    map_location = {'cuda:0': 'cuda:0'}
    if args.fine_tune:
        if config.data.dataset == 'mini':
            ckpt_path  = '/home/zhongyuj/Nitre/ZeDO_plus/checkpoint/h36mbb/checkpoint_300.pth'
        if config.data.dataset == 'syrip':
            ckpt_path = '/home/zhongyuj/Nitre/ZeDO_plus/output/concate_concate/h36m_12/checkpoint_3000.pth'
        checkpoint = torch.load(ckpt_path, map_location=map_location)
    
        # checkpoint = copy.deepcopy(old_checkpoint)
        # checkpoint['model_state_dict'] = {}
            
        # for k, v in old_checkpoint['model_state_dict'].items():
        #     name = k[7:] # remove `module.`
        #     checkpoint['model_state_dict'][name] = v
            
        

        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        ema.load_state_dict(checkpoint['ema'])
        # state['step'] = checkpoint['step']
        print(f"=> loaded checkpoint '{ckpt_path}'")
    
    # if args.restore_dir is False:
    start_epoch = 0
        
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)  # based on iteration instead of epochs
    # if args.restore_dir :
    #     state['step'] = checkpoint['step']
   

    # Identity func
    scaler = lambda x: x
    inverse_scaler = lambda x: x

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=False, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    sampling_shape = (config.eval.batch_size, config.DATASET.NUM_JOINT, JOINT_DIM)
    config.sampling.probability_flow = False
    sampling_fn = sampling_train.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,device=device)

    # num_train_steps = config.training.n_iters
    best_error = 1e5
    try:
        ''' training loop '''
        # WARNING!!! This code assumes all poses are normed into [-1, 1]
        for epoch in range(start_epoch, N_EPOCHES):
            model.train()
            logger.info(
                f'EPOCH: [{epoch}/{N_EPOCHES}, {epoch/N_EPOCHES*100:.2f}%]'
            )
            avg_loss = AverageMeter()
            for idx, (data_2d, labels_3d,K) in enumerate(tqdm(train_loader)):
                # import ipdb;ipdb.set_trace()
                
                labels_3d = labels_3d.to(device, non_blocking=True).float() * config.training.data_scale
                data_2d = data_2d.to(device, non_blocking=True)* config.training.data_scale
                data_2d = data_2d*0
                # if (args.condition):
                #     data_2d = data_2d.to(device, non_blocking=True) * config.training.data_scale
                #     cur_loss = train_step_fn(state, batch=labels_3d, condition=data_2d)
                # else:
                
                cur_loss = train_step_fn(state, batch=labels_3d, condition=data_2d)
                writer.add_scalar('train_loss', cur_loss.item(), idx + epoch * len(train_loader))
                avg_loss.update(cur_loss.item())

            # if rank == 0:
            logger.info(
                f'EPOCH: [{epoch}/{N_EPOCHES}, {epoch/N_EPOCHES*100:.2f}%][{idx}/{len(train_loader)}],\t'
                f'Loss: {avg_loss.avg}'
            )
            writer.add_scalar('Loss/train', avg_loss.avg, epoch)
           
            for i,param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'opt_LR_{i+1}', param_group['lr'], epoch)
            
            ''' eval '''
            if epoch % EVAL_FREQ == 0:
                model.eval()
                with torch.no_grad():
                    for idx, (data_2d, labels_3d,K) in enumerate(test_loader):
                        # import ipdb;ipdb.set_trace()
                        labels_3d = labels_3d.to(device, non_blocking=True).float()
                        data_2d = data_2d.to(device, non_blocking=True) * config.training.data_scale
                        data_2d = data_2d*0
                        # data_2d = data_2d.to(device, non_blocking=True)
                        # Generate and save samples
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        trajs, results = sampling_fn(
                            model,
                            condition=data_2d
                        )  # [b ,j ,3]
                        ema.restore(model.parameters())

                        # # trajs: [t, b, j, 3], i.e., the pose-trajs
                        # # results: [b, j, 3], i.e., the end pose of each traj
                        results = results / config.training.data_scale
                        np.save(os.path.join(final_output_dir, f'results_{epoch}.pth'),results)
                
                logger.info(f'Save checkpoint to {final_output_dir}')
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema': state['ema'].state_dict(),
                    'step': state['step'],
                }
                
                torch.save(save_dict, os.path.join(final_output_dir, f'checkpoint_{epoch}.pth'))
            # lr_scheduler.step()
    except Exception as e:
        traceback.print_exc()
    finally:
        writer.close()
        logger.info(f'End. Final output dir: {final_output_dir}')


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)

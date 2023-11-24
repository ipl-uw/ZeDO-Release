# ZeDO: Back to Optimization: Diffusion-based Zero-Shot 3D Human Pose Estimation (Accepted by WACV 2024)

[![](http://img.shields.io/badge/cs.CV-arXiv%3A2307.03833-B31B1B.svg)](https://arxiv.org/abs/2307.03833)
<a href='https://zhyjiang.github.io/ZeDO-proj/'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
<p align="center"><img src="images/overall.png" width="50%" alt="" /></p>

This is the official implementation of this paper:

> Zhongyu Jiang, Zhuoran Zhou, Lei Li, Wenhao Chai, Cheng-Yen Yang, and Jenq-Neng Hwang. [Back to Optimization: Diffusion-based Zero-Shot 3D Human Pose Estimation](https://arxiv.org/abs/2307.03833) WACV 2024.

And its follow-up paper:

> Zhuoran Zhou, Zhongyu Jiang, Wenhao Chai, Cheng-Yen Yang,  Lei Li and Jenq-Neng Hwang. [Efficient Domain Adaptation via Generative Prior for 3D Infant Pose Estimation](https://arxiv.org/abs/2311.12043) WACVWorkshop 2024.

<p align="center">
  <img src="./images/demo1.gif" width="20%" />
  <img src="./images/demo2.gif" width="20%" />
  <img src="./images/demo3.gif" width="20%" />
  <img src="./images/demo4.gif" width="20%" />
</p>


## 3D human pose estimation
### Results on 3DPW
Under PA-MPJPE and MPJPE

<table>
    <tr>
        <td>Methods</td>
        <td>CE</td>
        <td>Opt</td>
        <td>PA-MPJPE &darr;</td>
        <td>MPJPE &darr;</td>
    </tr>
    <tr>
        <td>SPIN</td>
        <td></td>
        <td></td>
        <td>59.2</td>
        <td>96.9</td>
    </tr>
    <tr>
        <td>VIBE</td>
        <td></td>
        <td></td>
        <td>51.9</td>
        <td>82.9</td>
    </tr>
    <tr>
        <td>PARE</td>
        <td></td>
        <td></td>
        <td>46.4</td>
        <td>74.7</td>
    </tr>
    <tr>
        <td>HybrIK</td>
        <td></td>
        <td></td>
        <td>45.0</td>
        <td>74.1</td>
    </tr>
    <tr>
        <td>HybrIK</td>
        <td><span>&#10003;</span></td>
        <td></td>
        <td>50.9</td>
        <td>82.0</td>
    </tr>
    <tr>
        <td>PoseAug</td>
        <td><span>&#10003;</span></td>
        <td></td>
        <td>58.5</td>
        <td>94.1</td>
    </tr>
    <tr>
        <td>AdaptPose</td>
        <td><span>&#10003;</span></td>
        <td></td>
        <td>46.5</td>
        <td>81.2</td>
    </tr>
    <tr>
        <td>PoseDA</td>
        <td><span>&#10003;</span></td>
        <td></td>
        <td>55.3</td>
        <td>87.7</td>
    </tr>
    <tr>
        <td>ZeDO (J=17)</td>
        <td><span>&#10003;</span></td>
        <td><span>&#10003;</span></td>
        <td><b>40.3</b></td>
        <td><b>69.7</b></td>
    </tr>
    <tr>
        <td>ZeDO (J=14)</td>
        <td><span>&#10003;</span></td>
        <td><span>&#10003;</span></td>
        <td><b>43.1</b></td>
        <td><b>76.6</b></td>
    </tr>
    <tr>
        <td>ZeDO (J=17, Additional Training Data)</td>
        <td><span>&#10003;</span></td>
        <td><span>&#10003;</span></td>
        <td><b>38.4</b></td>
        <td><b>68.3</b></td>
    </tr>
    <tr>
        <td>ZeDO (J=17, S=50, Additional Training Data)</td>
        <td><span>&#10003;</span></td>
        <td><span>&#10003;</span></td>
        <td><b>30.6</b></td>
        <td><b>54.7</b></td>
    </tr>
</table>


# Evaluate model

### Envrionment Setup
- pytorch >= 1.10
```
conda create -n ZeDO python==3.9
conda activate ZeDO
pip install -r requirements.txt
```

### Data and Model Preparation
Evaluation dataset, clusters and checkpoint: [Google Drive](https://drive.google.com/drive/folders/1A15nQ4-Rbp-rKDzfae4e2xAd9V7d4TEA?usp=sharing)

```
${POSE_ROOT}
|-- configs
|-- lib
|-- run
|-- checkpoint
    |-- concatebb
        |-- checkpoint_1500.pth
|-- data
    |-- h36m
        |-- h36m_test.pkl
        |-- h36m_sh_dt_ft.pkl
    |-- 3dpw
        |-- pw3d_test.npz
    |-- 3dhp
        |-- mpii3d_test.pkl
|-- clusters
    |-- 3dhp_cluster1.pkl
    |-- h36m_cluster1.pkl
    |-- 3dhp_cluster50.pkl
    |-- h36m_cluster50.pkl
```

### Evaluation script
```
python -m run.opt_main --config configs/subvp/concat_pose_optimization_<dataset>.py --ckpt_dir ./checkpoint/concatebb --ckpt_name checkpoint_1500.pth --hypo 1 <--gt> 
```

### In the wild inference
Coming Soon.


## 3D infant pose estimation

### Data and Model Preparation
We use [Mini-RGBD](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html) and [SyRIP](https://github.com/ostadabbas/Infant-Postural-Symmetry) in experiments, please follow the offical instructons to download the datasets.

### Evaluation script

For MINI-RGBD dataset, finetuning with data augmentation 

```
python -m run.opt_main_infant  --config configs/subvp/concat_pose_optimization_mini.py --ckpt_dir checkpoint/mini-rgbd --ckpt_name ft_aug.pth  --gt --hypo 1
```

For MINI-RGBD dataset, controlling branch with data augmentation 

```
python -m run.opt_main_infant  --config configs/subvp/concat_pose_optimization_mini.py --ckpt_dir checkpoint/mini-rgbd --ckpt_name ctl_aug.pth  --control --gt --hypo 1
```


For SyRIP dataset, finetuning with data augmentation 
```
python -m run.opt_main_infant  --config configs/subvp/concat_pose_optimization_syrip.py --ckpt_dir checkpoint/syrip --ckpt_name ft_aug.pth  --gt -hypo 1
```


# Citation
If you find this code useful in your project, please consider citing:
```
@inproceedings{Jiang2024ZeDO,
  title={Back to Optimization: Diffusion-based Zero-Shot 3D Human Pose Estimation},
  author={Jiang, Zhongyu and Zhou, Zhuoran and Li, Lei and Chai, Wenhao and Yang, Cheng-Yen and Hwang, Jenq-Neng},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2024}
}
```


# Acknowledgement
This repo is built on the excellent work [score_sde](https://github.com/yang-song/score_sde_pytorch) by Yang Song and [GFPose](https://github.com/Embracing/GFPose) by Hai Ci.

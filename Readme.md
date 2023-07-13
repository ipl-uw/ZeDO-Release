# ZeDO: Back to Optimization: Diffusion-based Zero-Shot 3D Human Pose Estimation

<p align="center"><img src="images/overall.png" width="50%" alt="" /></p>

This is the official implementation of this paper:

> Zhongyu Jiang, Zhuoran Zhou, Lei Li, Wenhao Chai, Cheng-Yen Yang, and Jenq-Neng Hwang. [Back to Optimization: Diffusion-based Zero-Shot 3D Human Pose Estimation](https://arxiv.org/abs/2307.03833) arXiv preprint arXiv:2307.03833 (2023).

<p align="center">
  <img src="./images/demo1.gif" width="20%" />
  <img src="./images/demo2.gif" width="20%" />
  <img src="./images/demo3.gif" width="20%" />
  <img src="./images/demo4.gif" width="20%" />
</p>

### Results on 3DPW
Under PA-MPJPE and MPJPE

<style>
    .result {
        width: 70%;
        text-align: center;
    }
    .result th {
        background: grey;
        word-wrap: break-word;
        text-align: center;
    }
    .result tr:nth-child(6) { background: grey; }
    .result tr:nth-child(7) { 
        background: grey; }
    .result tr:nth-child(8) { background: grey; }
    .result tr:nth-child(9) { background: grey; }
    .result tr:nth-child(10) { background: grey; }
    .result tr:nth-child(11) { background: grey; }
</style>

<div class="result">

| Methods | CE | Opt | PA-MPJPE $\downarrow$ | MPJPE $\downarrow$ |
|:---:|:---:|:---:|:---:|:---:|
| SPIN |  |  | 59.2 | 96.9 |
| VIBE |  |  | 51.9 | 82.9 |
| PARE |  |  | 46.4 | 74.7 |
| HybrIK |  |  | <u>45.0</u> | <u>74.1</u> |
| VirtualMarker |  |  | **41.3** | **67.5** |
| HybrIK | $\checkmark$ |  | 50.9 | 82.0 |s
| PoseAug | $\checkmark$ |  | 58.5 | 94.1 |
| AdaptPose | $\checkmark$ |  | 46.5 | <u>81.2</u> |
| PoseDA | $\checkmark$ |  | 55.3 | 87.7 |
| ZeDO (J=17) | $\checkmark$ | $\checkmark$ | **42.6** | **80.9** |
| ZeDO (J=14) | $\checkmark$ | $\checkmark$ | <u>45.4</u> | 88.6 |

</div>


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
python -m run.opt_main --config configs/subvp/concat_pose_optimization_<dataset>.py --ckpt_dir ./checkpoint/concatebb --ckpt_name checkpoint_1500.pth  --hypo 1 <--gt> 
```

# Citation
If you find this code useful in your project, please consider citing:
```
@article{jiang2023back,
  title={Back to Optimization: Diffusion-based Zero-Shot 3D Human Pose Estimation},
  author={Jiang, Zhongyu and Zhou, Zhuoran and Li, Lei and Chai, Wenhao and Yang, Cheng-Yen and Hwang, Jenq-Neng},
  journal={arXiv preprint arXiv:2307.03833},
  year={2023}
}
```

# Acknowledgement
This repo is built on the excellent work [score_sde](https://github.com/yang-song/score_sde_pytorch) by Yang Song and [GFPose](https://github.com/Embracing/GFPose) by Hai Ci.

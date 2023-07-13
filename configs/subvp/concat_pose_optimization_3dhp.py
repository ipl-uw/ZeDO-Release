# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from configs.default_pose_gen_configs import get_default_configs
import ml_collections


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'subvpsde'
  training.continuous = True
  training.n_iters = 950001
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True
  data.dataset = '3dhp'

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.fourier_scale = 16
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = False
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'positional'
  model.init_scale = 0.0
  model.conv_size = 3
  model.t = 0.1
  
  #optimization
  config.ZeDO = ml_collections.ConfigDict()
  ZeDO = config.ZeDO
  ZeDO.IPO_iterations = 500
  ZeDO.IPO_keylist = [0, 1, 4]
  ZeDO.RotAxes='z'
  ZeDO.IPO_T = 3
  ZeDO.IPO_minScaleT = 0.5
  ZeDO.IPO_maxScaleT = 2
  ZeDO.OIL_iterations = 1000
  ZeDO.sample = 3
  ZeDO.batch = 959
  ZeDO.sampling_eps = 0.01

  return config

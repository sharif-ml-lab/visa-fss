"""
Experiment configuration file
Extended from config file from original PANet Repository
"""

import sys
import socket

import os
import re
import glob
import itertools

import torch
import sacred
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
# from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

from platform import node
from datetime import datetime

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = sacred.Experiment('mt')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    seed = 1234
    gpu_id = 0
    mode = 'train' # for now only allows 'train' 
    num_workers = 4 # 0 for debugging. 

    dataset = 'CHAOST2_Superpix' # i.e. abdominal MRI
    use_coco_init = True # initialize backbone with MS_COCO initialization. Anyway coco does not contain medical images

    ### Training
    n_steps = 100100
    batch_size = 1
    lr_milestones = [ (ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 100
    save_snapshot_every = 25000
    validate_every = 10000
    max_iters_per_load = 1000 # epoch size, interval for reloading the dataset
    scan_per_load = -1 # numbers of 3d scans per load for saving memory. If -1, load the entire dataset to the memory
    which_aug = 'sabs_aug' # standard data augmentation with intensity and geometric transforms
    input_size = (256, 256)
    min_fg_data='100' # when training with manual annotations, indicating number of foreground pixels in a single class single slice. This empirically stablizes the training process
    label_sets = 0 # which group of labels taking as training (the rest are for testing)
    exclude_cls_list = [2, 3] # testing classes to be excluded in training. Set to [] if testing under setting 1
    usealign = True # see vanilla PANet
    use_wce = True
    query_length = 1
    query_range = 1
    
    fg_loss_coeff = 0
    qloss_coeff = 1
    sloss_coeff = 0.2
    l2_loss_coeff = 0
    dice_loss_coeff = 0
    sz_loss_coeff = 0

    ### Validation
    z_margin = 0 
    eval_fold = 0 # which fold for 5 fold cross validation
    support_idx=[-1] # indicating which scan is used as support in testing. 
    val_wsize=2 # L_H, L_W in testing
    n_sup_part = 3 # number of chuncks in testing

    # Network
    modelname = 'dlfcn_res101' # resnet 101 backbone from torchvision fcn-deeplab
    clsname = None # 
    reload_model_path = None # path for reloading a trained model (overrides ms-coco initialization)
    reload_model_path_ref = None
    reload_model_path_prop = None
    proto_grid_size = 8 # L_H, L_W = (32, 32) / 8 = (4, 4)  in training
    feature_hw = [32, 32] # feature map size, should couple this with backbone in future

    # Baseline
    fg_proto = 'mask'
    bg_proto = 'mask'

    # SSP
    ssp_refine_steps = 1
    ssp_average = False
    test_ssp_refine_steps = 1
    test_ssp_average = False

    # SSL
    superpix_scale = 'MIDDLE' #MIDDLE/ LARGE

    # sslreg
    refinement = None
    proptst_refinement = None
    trg_ref_mode = None

    device = 'cuda:0'
    torch_device = torch.device(device)

    model = {
        'align': usealign,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size' : proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path
    }

    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': query_length,
        'npart': n_sup_part 
    }

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3, 
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_prefix = ''

    exp_str = '_'.join(
        [exp_prefix]
        + [dataset,]
        + [f'sets_{label_sets}_{task["n_shots"]}shot'])

    path = None
    px_dir = None
    save_models = None
    
    path = {
        'log_dir': './runs',
        'SABS':{'data_dir': "../medical-data/ABD-CT/ABD-CT-Preprocessed"
            },
        'C0':{'data_dir': "feed your dataset path here"
            },
        'CHAOST2':{'data_dir': "../medical-data/ABD-MRI"
            },
        'SABS_Superpix':{'data_dir': "../medical-data/ABD-CT/ABD-CT-Preprocessed"},
        'SABS_PANet':{'data_dir': "../medical-data/ABD-CT/ABD-CT-Preprocessed"},
        'C0_Superpix':{'data_dir': "feed your dataset path here"},
        'CHAOST2_Superpix':{'data_dir': "../medical-data/ABD-MRI"},
        
        'ABD':{'data_dir': "../medical-data/ABD-CT/ABD-CT-Preprocessed"},
        }

    save_models = {
        'pretrained': './pretrained',
        'pretrained_model': './pretrained_model'
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config

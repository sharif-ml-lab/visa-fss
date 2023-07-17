import os
import sys
import socket

sys.path.append(os.getcwd())

import tqdm
import torch
import shutil
import torch.optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from losses.dice import compute_dice_loss, dice_interslice

from util.metric import Metric
from config_ssl_upload import ex
from util.eval import evaluate_prop
import dataloaders.augutils as myaug
from models.reg_model import RegModel
from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dataset_utils import DATASET_INFO
from dataloaders.VolumeDataset import VolumeDataset
from losses.local_slice_windows import fg_size_loss
from dataloaders.dev_customized_med import med_fewshot
from util.refinement import superpixel_refine, s2v_refine
from util.utils import set_seed, t2n, to01, compose_wt_simple
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.SSLRegGeomMultiQDataset import SSLRegGeomMultiQDataset

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])
    
    # pth = f'exps/visa-fss_seed_2895/mt_train_SABS_Superpix_lbgroup0' \
    #       f'_vfold{_config["eval_fold"]}_seed2895_SABS_Superpix_sets_0_1shot/1/snapshots/final.pth'

    pth = f'exps/visa-fss-mri_seed_2895/mt_train_CHAOST2_Superpix_lbgroup0' \
          f'_vfold{_config["eval_fold"]}_seed2895_CHAOST2_Superpix_sets_0_1shot/1/snapshots/final.pth'
    
    model.load_state_dict(torch.load(pth))
    model = model.to(_config['device'])
    model.train()
    
    nb_enc_features = [32, 32, 32, 32]
    nb_dec_features = [32, 32, 32, 32, 32, 16]
    reg_model = RegModel(2, (256, 256), nb_enc_features, nb_dec_features).float()
    # reg_model = reg_model.to(_config['device'])
    reg_model.load_state_dict(torch.load(f'{_config["save_models"]["pretrained"]}/reg_model_mse&grad.pth'))
    reg_model.eval()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    ### Transforms for data augmentation
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    tr_parent = SSLRegGeomMultiQDataset( # base dataset
        which_dataset = baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split = _config['eval_fold'],
        mode='train',
        min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
        transforms=tr_transforms,
        nsup = _config['task']['n_shots'],
        device = _config['device'],
        query_length=_config['query_length'],
        query_range=_config['query_range'],
        scan_per_load = _config['scan_per_load'],
        exclude_list = _config["exclude_cls_list"],
        superpix_scale = _config["superpix_scale"],
        refine_module=_config["refinement"],
        fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') or (data_name == 'SABS_Superpix') else None,
        reg_model=reg_model
    )
    ### dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name).to(_config['device'])
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0, 'fg_loss': 0, 'dice_loss': 0}

    _log.info('###### Training ######')
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for batch_index, sample_batched in enumerate(trainloader):
            
            model.train()
            # if batch_index >= 10:
            #     exit()
            # continue
            
            # Prepare input
            i_iter += 1
            # add writers
            support_images = [[shot.to(_config['device']) for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().to(_config['device']) for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().to(_config['device']) for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.to(_config['device'])
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().to(_config['device']) for query_label in sample_batched['query_labels']], dim=0)

            optimizer.zero_grad()
            # FIXME: in the model definition, filter out the failure case where pseudolabel falls outside of image or too small to calculate a prototype
            query_pred, align_loss, debug_vis, assign_mats = model(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)

            query_loss = criterion(query_pred, query_labels)
            if _config['fg_loss_coeff'] > 0:
                fg_loss = _config['fg_loss_coeff'] * fg_size_loss(query_pred)
            else:
                fg_loss = 0

            if _config['dice_loss_coeff'] > 0 and (sample_batched['batch_type'] == 2):
                dice_loss = _config['dice_loss_coeff'] * dice_interslice(query_pred[:-1, :, :, :], query_pred[1:, :, :, :])
            else:
                dice_loss = 0

            loss = query_loss + align_loss + fg_loss + dice_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
            fg_loss = fg_loss.detach().data.cpu().numpy() if fg_loss != 0 else 0
            dice_loss = dice_loss.detach().data.cpu().numpy() if dice_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            _run.log_scalar('fg_loss', fg_loss)
            _run.log_scalar('dice_loss', dice_loss)
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss
            log_loss['fg_loss'] += fg_loss
            log_loss['dice_loss'] += dice_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']
                fg_loss = log_loss['fg_loss'] / _config['print_interval']
                dice_loss = log_loss['dice_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0
                log_loss['fg_loss'] = 0
                log_loss['dice_loss'] = 0

                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, fg_loss: {fg_loss}, dice_loss: {dice_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up

            if (i_iter + 1) % 1000 == 0:
                evaluate_prop(_run, _config, _log, model, reg_model) 


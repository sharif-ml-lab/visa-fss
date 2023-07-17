"""
Validation script
"""
import os
import sys
import socket

sys.path.append(os.getcwd())

import tqdm
import torch
import shutil
import numpy as np
import torch.optim
import einops as eo
import torch.nn as nn
import SimpleITK as sitk
from util.metric import Metric
import torch.nn.functional as F
from config_ssl_upload import ex
from util.dice import dice_score
import torch.backends.cudnn as cudnn
from models.reg_model import RegModel
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataloaders.niftiio import convert_to_sitk
from torch.optim.lr_scheduler import MultiStepLR
from models.grid_proto_fewshot import FewShotSeg
from dataloaders.VolumeDataset import VolumeDataset
from util.refinement import superpixel_refine, s2v_refine
from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from monai.metrics.utils import get_mask_edges, get_surface_distance
from monai.metrics import compute_hausdorff_distance, compute_surface_dice

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

def compute_hausdorff_monai(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 1.0
    return dist / max_dist

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    
    max_distance = np.sqrt(256 ** 2 + 256 ** 2)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = FewShotSeg(pretrained_path = _config['reload_model_path'], cfg=_config['model'])
    model = model.cuda()
    model.eval()

    nb_enc_features = [32, 32, 32, 32]
    nb_dec_features = [32, 32, 32, 32, 32, 16]
    reg_model = RegModel(2, (256, 256), nb_enc_features, nb_dec_features).float()
    reg_model = reg_model.to(_config['device'])
    reg_model.load_state_dict(torch.load(f'{_config["save_models"]["pretrained"]}/reg_model_mse&grad.pth'))
    reg_model.eval()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    # test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all']

    ### Transforms for data augmentation
    te_transforms = None

    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if baseset_name == 'SABS': # for CT we need to know statistics of 
        tr_parent = SuperpixelDataset( # base dataset
            which_dataset = baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split = _config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transforms=None,
            nsup = _config['task']['n_shots'],
            scan_per_load = _config['scan_per_load'],
            exclude_list = _config["exclude_cls_list"],
            superpix_scale = _config["superpix_scale"],
            fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None
        )
        norm_func = tr_parent.norm_func
    else:
        norm_func = get_normalize_op(modality = 'MR', fids = None)


    te_dataset = VolumeDataset(
                which_dataset = baseset_name, 
                base_dir = _config['path'][data_name]['data_dir'], 
                idx_split = _config['eval_fold'], 
                mode = 'val', 
                min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
                organs_set = DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]],
                transforms = te_transforms, 
                nsup = _config['task']['n_shots'],
                query_length = _config['task']['n_queries'],
                scan_per_load = _config['scan_per_load'],
                exclude_list = _config["exclude_cls_list"])

    testloader = DataLoader(
            te_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=_config['num_workers'],
            pin_memory=True,
            drop_last=True
        )

    _log.info('###### Set validation nodes ######')
    mar_val_metric_node = Metric(max_label=max_label, n_scans= len(te_dataset.pid_curr_load) - _config['task']['n_shots'])

    _log.info('###### Starting validation ######')
    model.eval()
    mar_val_metric_node.reset()

    with torch.no_grad():
        save_pred_buffer = {} # indexed by class

        hds = []
        sds = []
        sd_thresh = 1
        for curr_lb in test_labels:
            # te_dataset.set_curr_cls(curr_lb)

            support_batched = te_dataset.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

            # way(1 for now) x part x shot x 3 x H x W] #
            support_images = [[shot.cuda() for shot in way]
                                for way in support_batched['support_images']] # way x part x [shot x C x H x W]
            suffix = 'mask'
            support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]

            _lb_buffer = {} # indexed by scan

            last_qpart = 0 # used as indicator for adding result to buffer

            labelname = te_dataset.label_name[curr_lb]

            hds_per_label = []
            sds_per_label = []
            for curr_scan_count, sample_batched in enumerate(testloader):

                # Get a new 3D volume
                _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                scan = sample_batched['scan'][0]
                label = sample_batched['label'][0]
                label = (label == curr_lb) * 1
                _pred = np.zeros(label.shape)
                if _scan_id in te_dataset.potential_support_sid: # skip the support scan, don't include that to query
                    continue

                # Find the reference slices in query volume
                npart = _config['task']['npart']
                if npart == 1:
                    center_pivots = [0.5]
                else:
                    half_part = 1 / (npart * 2)
                    part_interval = (1.0 - 1.0 / npart) / (npart - 1)
                    center_pivots = [ half_part + part_interval * ii for ii in range(npart) ]
                _zlist = te_dataset.tp1_cls_map[labelname][_scan_id]
                center_pivots = [_zlist[int(p * len(_zlist))] for p in center_pivots]
                pivots = None
                if _config['trg_ref_mode'] == 'min_reg_error':
                    _zlist = te_dataset.tp1_cls_map[labelname][_scan_id]
                    z_min = min(te_dataset.tp1_cls_map[labelname][_scan_id])
                    z_max = max(te_dataset.tp1_cls_map[labelname][_scan_id])
                    reg_errors = {n:{} for n in range(npart)}

                    for z in range(z_min, z_max):
                        zpart = int((z - z_min) // ((z_max - z_min) / npart))
                        ref = (support_images[0][zpart])[:, 0:1, ...]
                        trg = scan[None, None, :, :, z].cuda()
                        if ((z - center_pivots[zpart]) < 5) and ((z - center_pivots[zpart]) > -5):
                        # if True:
                            # print(trg.shape, ref.shape)
                            moved, _ = reg_model(ref, trg)
                            error = F.mse_loss(trg, moved)
                            reg_errors[zpart][z] = error

                    pivots = [min(reg_errors[n], key=reg_errors[n].get) for n in range(npart)]
                else:
                    pivots = center_pivots

                # Predict the mask of reference slices in query support
                for q_part, p in enumerate(pivots):
                    query_images = [scan[:, :, p].repeat([3, 1, 1])[None, ...].cuda()]
                    query_labels = label[:, :, p:p+1].permute((2, 0, 1)).cuda()                
                    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]
                    query_pred, _, _, assign_mats = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )
                    query_pred = np.array(query_pred.argmax(dim=1)[0].cpu())
                    _pred[..., p] = query_pred.copy()
                    mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
                    
                    qp_onehot = eo.repeat(eo.pack([query_pred, 1-query_pred], '* h w')[0], 'c h w -> b c h w', b=1)
                    qp_onehot = torch.from_numpy(qp_onehot)
                    gt_onehot = eo.repeat(eo.pack([query_labels[0], 1-query_labels[0]], '* h w')[0], 'c h w -> b c h w', b=1)
                    hd = compute_hausdorff_distance(qp_onehot, gt_onehot)
                    sd = compute_surface_dice(qp_onehot, gt_onehot, [sd_thresh])
                    # sd = torch.tensor([0])
                    hds_per_label.append(hd.item())
                    sds_per_label.append(sd.item())
                    
                    print(f'Reference {p}: Dice: {dice_score(_pred[..., p], np.array(query_labels[0].cpu()))}')

                # Propagate the mask to the entire volume
                z_min = min(te_dataset.tp1_cls_map[labelname][_scan_id])
                z_max = max(te_dataset.tp1_cls_map[labelname][_scan_id])
                for q_part, p in enumerate(pivots): 
                    # Forward propagation
                    for z_id in range(p + 1, z_max + 1):
                        part_assign = int((z_id - z_min) // ((z_max - z_min) / npart))
                        if part_assign != q_part: break
                        query_images = [scan[:, :, z_id].repeat([3, 1, 1])[None, ...].cuda()]
                        query_labels = label[:, :, z_id:z_id+1].permute((2, 0, 1)).cuda()
                        
                        # sup_img_part = [[scan[:, :, z_id-1].repeat([3, 1, 1])[None, ...].cuda()]]
                        # sup_fgm_part = [[((torch.from_numpy(_pred[:, :, z_id-1]) == 1) * 1.0).unsqueeze(0).cuda()]]
                        # sup_bgm_part = [[((torch.from_numpy(_pred[:, :, z_id-1]) != 1) * 1.0).unsqueeze(0).cuda()]]
                        sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]
                        sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                        sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]
                        
                        query_pred, _, _, assign_mats = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )
                        query_pred = query_pred.argmax(dim=1)[0].cpu()

                        if _config['proptst_refinement'] == 's2v':
                            refined_query_pred = torch.zeros_like(query_pred)
                            hold = s2v_refine(scan[:, :, z_id-1].numpy(),
                                              scan[:, :, z_id].numpy(),
                                              _pred[:, :, z_id-1],
                                              query_pred.numpy(),
                                              dilation_kernel_size=2)
                            refined_query_pred = torch.from_numpy(hold)
                            query_pred = refined_query_pred

                        query_pred = np.array(query_pred)
                        _pred[..., z_id] = query_pred.copy()
                        mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
                        
                        qp_onehot = eo.repeat(eo.pack([query_pred, 1-query_pred], '* h w')[0], 'c h w -> b c h w', b=1)
                        qp_onehot = torch.from_numpy(qp_onehot)
                        gt_onehot = eo.repeat(eo.pack([query_labels[0], 1-query_labels[0]], '* h w')[0], 'c h w -> b c h w', b=1)
                        hd = compute_hausdorff_distance(qp_onehot, gt_onehot)
                        sd = compute_surface_dice(qp_onehot, gt_onehot, [sd_thresh])
                        # sd = torch.tensor([0])
                        hds_per_label.append(hd.item())
                        sds_per_label.append(sd.item())
                        
                        print(f'Forward Ref -> {z_id}: Dice: {dice_score(_pred[..., z_id], np.array(query_labels[0].cpu()))}')

                    # Backward propagation
                    for z_id in range(p - 1, z_min - 1, -1):
                        part_assign = int((z_id - z_min) // ((z_max - z_min) / npart))
                        if part_assign != q_part: break
                        query_images = [scan[:, :, z_id].repeat([3, 1, 1])[None, ...].cuda()]
                        query_labels = label[:, :, z_id:z_id+1].permute((2, 0, 1)).cuda()
                        
                        # sup_img_part = [[scan[:, :, z_id+1].repeat([3, 1, 1])[None, ...].cuda()]]
                        # sup_fgm_part = [[((torch.from_numpy(_pred[:, :, z_id+1]) == 1) * 1.0).unsqueeze(0).cuda()]]
                        # sup_bgm_part = [[((torch.from_numpy(_pred[:, :, z_id+1]) != 1) * 1.0).unsqueeze(0).cuda()]]
                        sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]
                        sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                        sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]
                        
                        query_pred, _, _, assign_mats = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )
                        query_pred = query_pred.argmax(dim=1)[0].cpu()

                        if _config['proptst_refinement'] == 's2v':
                            refined_query_pred = torch.zeros_like(query_pred)
                            hold = s2v_refine(scan[:, :, z_id+1].numpy(),
                                              scan[:, :, z_id].numpy(),
                                              _pred[:, :, z_id+1],
                                              query_pred.numpy(),
                                              dilation_kernel_size=2)
                            refined_query_pred = torch.from_numpy(hold)
                            query_pred = refined_query_pred

                        query_pred = np.array(query_pred)
                        _pred[..., z_id] = query_pred.copy()
                        mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
                        
                        qp_onehot = eo.repeat(eo.pack([query_pred, 1-query_pred], '* h w')[0], 'c h w -> b c h w', b=1)
                        qp_onehot = torch.from_numpy(qp_onehot)
                        gt_onehot = eo.repeat(eo.pack([query_labels[0], 1-query_labels[0]], '* h w')[0], 'c h w -> b c h w', b=1)
                        hd = compute_hausdorff_distance(qp_onehot, gt_onehot)
                        sd = compute_surface_dice(qp_onehot, gt_onehot, [sd_thresh])
                        # sd = torch.tensor([0])
                        hds_per_label.append(hd.item())
                        sds_per_label.append(sd.item())
                        
                        print(f'Backward Ref ->{z_id}: Dice: {dice_score(_pred[..., z_id], np.array(query_labels[0].cpu()))}')
                        
                # Save results in _lb_buffer
                if _config['dataset'] != 'C0':
                    _lb_buffer[_scan_id] = _pred.transpose(2,0,1) # H, W, Z -> to Z H W
                else:
                    _lb_buffer[_scan_id] = _pred
                 
            save_pred_buffer[str(curr_lb)] = _lb_buffer
            hds.append(sum(hds_per_label)/len(hds_per_label))
            sds.append(sum(sds_per_label)/len(sds_per_label))

        ### save results
        for curr_lb, _preds in save_pred_buffer.items():
            for _scan_id, _pred in _preds.items():
                _pred *= float(curr_lb)
                itk_pred = convert_to_sitk(_pred, te_dataset.info_by_scan[_scan_id])
                fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}.nii.gz')
                sitk.WriteImage(itk_pred, fid, True)
                _log.info(f'###### {fid} has been saved ######')

        del save_pred_buffer

    del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred
    
    print(f'hds: {hds}')
    print(f'sds: {sds}')

    # compute dice scores by scan
    m_classDice,_, m_meanDice,_, m_rawDice = mar_val_metric_node.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw = True)
    m_classPrec,_, m_meanPrec,_,  m_classRec,_, m_meanRec,_, m_rawPrec, m_rawRec = mar_val_metric_node.get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw = True)
    mar_val_metric_node.reset() # reset this calculation node

    # write validation result to log file
    _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
    _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
    _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

    _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
    _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
    _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

    _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
    _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
    _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

    _log.info(f'mar_val batches classDice: {m_classDice}')
    _log.info(f'mar_val batches meanDice: {m_meanDice}')

    _log.info(f'mar_val batches classPrec: {m_classPrec}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

    _log.info(f'mar_val batches classRec: {m_classRec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')

    print("============ ============")

    _log.info(f'End of validation')
    return m_meanDice

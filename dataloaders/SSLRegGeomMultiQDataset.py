import re
import os
import cv2
import json
import copy
import glob
import torch
import random
import platform
import numpy as np
# import neurite as ne
from pdb import set_trace
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from util.utils import CircularList
import dataloaders.augutils as myaug
from dataloaders.dataset_utils import *
from util.save_fig import overlay_img_mask
from dataloaders.common import BaseDataset, Subset
from util.refinement import superpixel_refine, s2v_refine

class SSLRegGeomMultiQDataset(BaseDataset):
    def __init__(self, which_dataset, base_dir, idx_split, mode, transforms, scan_per_load, min_fg = '',
                 fix_length = None, tile_z_dim = 3, nsup = 1, query_length=5, query_range=2, device = None,
                 exclude_list = [], extern_normalize_func = None, reg_model=None, refine_module=None, **kwargs):
        """
        Manually labeled dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation
            mode:               'train', 'val'. 
            transforms:         data transform (augmentation) function
            min_fg:             minimum number of positive pixels in a 2D slice, mainly for stablize training when trained on manually labeled dataset
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images
            nsup:               number of support scans
            query_length:       number of query scans in each episode
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            extern_normalize_function:  normalization function used for data pre-processing  
        """
        super(SSLRegGeomMultiQDataset, self).__init__(base_dir)
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.device = device
        self.fix_length = fix_length
        self.all_label_names = self.label_name
        self.nclass = len(self.label_name)
        self.tile_z_dim = tile_z_dim
        self.base_dir = base_dir
        self.nsup = nsup
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ]
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x))) # make it circular for the ease of spliting folds
        self.query_length = query_length
        self.query_range = query_range
        self.exclude_lbs = exclude_list
        self.refine_module = refine_module
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')

        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split)                   # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)
        self.scan_per_load = scan_per_load
        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids()                      # information of scans of the entire fold

        if extern_normalize_func is not None: # helps to keep consistent between training and testing dataset.
            self.norm_func = extern_normalize_func
            print(f'###### Dataset: using external normalization statistics ######')
        else:
            self.norm_func = get_normalize_op(self.img_modality, [ fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])
            print(f'###### Dataset: using normalization statistics calculated from loaded data ######')

        if self.is_train:
            if scan_per_load > 0: # buffer needed
                self.pid_curr_load = np.random.choice( self.scan_ids, replace = False, size = self.scan_per_load)
            else: # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
            self.potential_support_sid = []
        else:
            raise Exception

        self.scans, self.labels = self.read_dataset()
        self.size = len(self.scans)
        self.overall_slice_by_cls = self.read_classfiles()
        # self.update_subclass_lookup()

        # Voxelmorph model
        self.reg_model = reg_model

    def get_scanids(self, mode, idx_split):
        val_ids  = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        self.potential_support_sid = val_ids[-self.nsup:] # this is actual file scan id, not index
        if mode == 'train':
            return [ ii for ii in self.img_pids if ii not in val_ids ]
        elif mode == 'val':
            return val_ids

    def reload_buffer(self):
        """
        Reload a portion of the entire dataset, if the dataset is too large
        1. delete original buffer
        2. update self.ids_this_batch
        3. update other internel variables like __len__
        """
        if self.scan_per_load <= 0:
            print("We are not using the reload buffer, doing notiong")
            return -1

        del self.actual_dataset
        del self.info_by_scan
        self.pid_curr_load = np.random.choice( self.scan_ids, size = self.scan_per_load, replace = False )
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.update_subclass_lookup()
        print(f'Loader buffer reloaded with a new size of {self.size} slices')

    def organize_sample_fids(self):
        out_list = {}
        for curr_id in self.scan_ids:
            curr_dict = {}
            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            _lb_fid  = os.path.join(self.base_dir, f'superpix-MIDDLE_{curr_id}.nii.gz')
            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fid"] = _lb_fid
            out_list[str(curr_id)] = curr_dict
        return out_list

    def read_dataset(self):
        """
        Build index pointers to individual slices
        Also keep a look-up table from scan_id, slice to index
        """
        scans = {}
        labels = {}
        self.info_by_scan = {} # meta data of each scan

        for scan_id, itm in self.img_lb_fids.items():
            
            if scan_id not in self.pid_curr_load:
                continue

            img, _info = read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out

            img = img.transpose(1,2,0)

            self.info_by_scan[scan_id] = _info

            img = np.float32(img)
            img = self.norm_func(img)

            # self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]

            lb = read_nii_bysitk(itm["lbs_fid"])
            lb = lb.transpose(1,2,0)

            lb = np.float32(lb)

            img = img[:256, :256, :] # FIXME a bug in shape from the pre-processing code
            lb = lb[:256, :256, :]

            assert img.shape[-1] == lb.shape[-1]
            base_idx = img.shape[-1] // 2 # index of the middle slice

            scans[scan_id] = img
            labels[scan_id] = lb

        return scans, labels

    def read_classfiles(self):
        with open(   os.path.join(self.base_dir, f'classmap_{self.min_fg}.json') , 'r' ) as fopen:
            cls_map =  json.load( fopen)
            fopen.close()

        with open(   os.path.join(self.base_dir, 'classmap_1.json') , 'r' ) as fopen:
            self.tp1_cls_map =  json.load( fopen)
            fopen.close()

        return cls_map

    def __getitem__(self, index):

        # TODO: Exclude labels if in setting 2
        # TODO: Handle augmentations for multi-query setting

        batch_type = None
        
        # We will fill these until the end of this function
        support_imgs = []
        support_msks = []
        query_images = []
        query_labels = []

        # Choose a source volume from which we extract support and queries
        v_index = torch.randint(0, self.size, size=(1,)).item()
        scan_keys = list(self.scans.keys())
        volume = self.scans[scan_keys[v_index]]
        v_mask = self.labels[scan_keys[v_index]]
        num_slices = volume.shape[-1]

        # Choose support and queries
        random_center_index = torch.randint(self.query_range, num_slices-self.query_range, size=(1,)).item()
        candidate_indices = torch.arange(random_center_index-self.query_range, random_center_index+self.query_range+1)
        candidate_indices = candidate_indices[candidate_indices != random_center_index]                                     # remove support from query options
        candidate_slices = volume[..., candidate_indices]
        selected_indices, _ = torch.sort(torch.randperm(candidate_indices.shape[0])[:self.query_length])
        selected_queries = candidate_slices[..., selected_indices]
        selected_queries = torch.from_numpy(selected_queries)
        if len(selected_queries.shape) == 2:
            selected_queries = selected_queries.unsqueeze(-1)

        if torch.rand(1) < 0.5:
            
            support = volume[:, :, random_center_index]
            sup_lbl = v_mask[:, :, random_center_index]
            superpixel_options = np.sort(np.unique(sup_lbl))[1:]
            if superpixel_options.shape[0] <= 0:
                return self.__getitem__(index)
            superpixel_index = superpixel_options[torch.randint(0, superpixel_options.shape[0], (1,))]
            sup_lbl = ((sup_lbl == superpixel_index) * 1.0).astype(np.float32)
            stacked = np.concatenate((support[..., None], sup_lbl[..., None]), axis=-1)

            support, sup_lbl = self.transforms(stacked, c_img=1, c_label=1, nclass=self.nclass,  is_train=True, use_onehot=False)
            support = torch.from_numpy(support.squeeze())
            sup_lbl = torch.from_numpy(sup_lbl.squeeze())
            if self.tile_z_dim:
                support = support.repeat([self.tile_z_dim, 1, 1])
            sup_msk = self.getMaskMedImg(sup_lbl, 1)
            support_imgs.append(support)
            support_msks.append(sup_msk)

            for i in range(self.query_length):
                query, query_lbl = self.transforms(stacked, c_img=1, c_label=1, nclass=self.nclass,  is_train=True, use_onehot=False)
                query = torch.from_numpy(query.squeeze())
                query_lbl = torch.from_numpy(query_lbl.squeeze())
                if self.tile_z_dim:
                    query = query.repeat([self.tile_z_dim, 1, 1])
                qry_msk = (query_lbl == superpixel_index) * 1
                support_imgs.append(support)
                support_msks.append(sup_msk)
                query_images.append(query)
                query_labels.append(qry_msk)

            batch_type = 1

        else:
            
            support = volume[:, :, random_center_index]
            sup_lbl = v_mask[:, :, random_center_index]
            support = torch.from_numpy(support.squeeze())
            sup_lbl = torch.from_numpy(sup_lbl.squeeze())
            if self.tile_z_dim:
                support = support.repeat([self.tile_z_dim, 1, 1])
            superpixel_options = np.sort(np.unique(sup_lbl))[1:]
            superpixel_index   = superpixel_options[torch.randint(0, superpixel_options.shape[0], (1,))]
            sup_msk = self.getMaskMedImg(sup_lbl, superpixel_index)
            support_imgs.append(support)
            support_msks.append(sup_msk)

            src_imgs = torch.from_numpy((volume[None, None, :, :, random_center_index])).expand((self.query_length, -1, -1, -1))
            src_msks = torch.from_numpy((v_mask[None, None, :, :, random_center_index])).expand((self.query_length, -1, -1, -1))
            src_msks = (src_msks == superpixel_index) * 1.0
            trg_imgs = selected_queries.permute((2, 0, 1))[:, None, :, :]
            with torch.no_grad():
                _, disp = self.reg_model(src_imgs, trg_imgs)
            trg_msks = self.reg_model.spatial_transform(src_msks, disp)
            trg_msks = trg_msks.round()
            
            # (superpixel refinement)
            if self.refine_module == 'sp':
                refined_trg_masks = torch.zeros_like(trg_msks)
                for i in range(refined_trg_masks.shape[0]):
                    hold = superpixel_refine(trg_imgs[i, 0, :, :].numpy(), trg_msks[i, 0, :, :].numpy(), 0.5)
                    refined_trg_masks[i, 0, :, :] = torch.from_numpy(hold)
                trg_msks = refined_trg_masks

            # (sli2vol refinement)
            elif self.refine_module == 's2v':
                refined_trg_masks = torch.zeros_like(trg_msks)
                for i in range(refined_trg_masks.shape[0]):
                    hold = s2v_refine(src_imgs[i, 0, :, :].numpy(),
                                    trg_imgs[i, 0, :, :].numpy(),
                                    src_msks[i, 0, :, :].numpy(),
                                    trg_msks[i, 0, :, :].numpy(),
                                    dilation_kernel_size=2)
                    refined_trg_masks[i, 0, :, :] = torch.from_numpy(hold)
                trg_msks = refined_trg_masks

            queries = selected_queries
            q_labls = trg_msks.squeeze(1).permute((1, 2, 0))
            for i in range(queries.shape[-1]):
                if self.tile_z_dim:
                    q = queries[:, :, i].repeat([self.tile_z_dim, 1, 1])
                query_images.append(q)
                query_labels.append(q_labls[:, :, i])

            batch_type = 2
        
        return {
            'superpixel_index': superpixel_index,
            'support_images'  : [support_imgs],
            'support_mask'    : [support_msks],
            'query_images'    : query_images,
            'query_labels'    : query_labels,
            'batch_type'      : batch_type
        }

    def __len__(self):
        return self.fix_length

    def update_subclass_lookup(self):
        """
        Updating the class-slice indexing list
        Args:
            [internal] overall_slice_by_cls:
                {
                    class1: {pid1: [slice1, slice2, ....],
                                pid2: [slice1, slice2]},
                                ...}
                    class2:
                    ...
                }
        out[internal]:
                {
                    class1: [ idx1, idx2, ...  ],
                    class2: [ idx1, idx2, ...  ],
                    ...
                }

        """
        # delete previous ones if any
        assert self.overall_slice_by_cls is not None

        if not hasattr(self, 'idx_by_class'):
             self.idx_by_class = {}
        # filter the new one given the actual list
        for cls in self.label_name:
            if cls not in self.idx_by_class.keys():
                self.idx_by_class[cls] = []
            else:
                del self.idx_by_class[cls][:]
        for cls, dict_by_pid in self.overall_slice_by_cls.items():
            for pid, slice_list in dict_by_pid.items():
                if pid not in self.pid_curr_load:
                    continue
                self.idx_by_class[cls] += [ self.scan_z_idx[pid][_sli] for _sli in slice_list ]
        print("###### index-by-class table has been reloaded ######")


    def getMaskMedImg(self, label, class_id):
        """
        Generate FG/BG mask from the segmentation mask

        Args:
            label:          semantic mask
            class_id:       semantic class of interest
        """
        fg_mask = torch.where(label == class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id,
                              torch.ones_like(label), torch.zeros_like(label))

        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}
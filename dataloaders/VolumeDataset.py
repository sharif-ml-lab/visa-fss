import glob
import numpy as np
import dataloaders.augutils as myaug
import torch
import random
import os
import copy
import platform
import json
import re
from dataloaders.common import BaseDataset, Subset
# from common import BaseDataset, Subset
from dataloaders.dataset_utils import*
from pdb import set_trace
from util.utils import CircularList

class VolumeDataset(BaseDataset):
    def __init__(self, which_dataset, base_dir, idx_split, mode, transforms, scan_per_load, organs_set, min_fg = '', tile_z_dim = 3, nsup = 1, query_length=5, exclude_list = [], extern_normalize_func = None, **kwargs):
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
            exclude_list:       Labels to be excluded
            extern_normalize_function:  normalization function used for data pre-processing  
        """
        super(VolumeDataset, self).__init__(base_dir)
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.all_label_names = self.label_name
        self.nclass = len(self.label_name)
        self.tile_z_dim = tile_z_dim
        self.base_dir = base_dir
        self.nsup = nsup
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ]
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x))) # make it circular for the ease of spliting folds
        
        self.organs_set = organs_set
        self.query_length = query_length
        self.exclude_lbs = exclude_list
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')

        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split) # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)

        self.scan_per_load = scan_per_load

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids() # information of scans of the entire fold

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
        print('pid current load:', self.pid_curr_load)
        # self.update_subclass_lookup()

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
            _lb_fid  = os.path.join(self.base_dir, f'label_{curr_id}.nii.gz')

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
        scan_id = self.scan_ids[index]
        scan = self.scans[scan_id]
        lbl = self.labels[scan_id]
        return {
            'scan_id': scan_id,
            'scan': scan,
            'label': lbl
        }

    def __len__(self):
        return len(self.pid_curr_load)

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
    
    def get_support(self, curr_class: int, class_idx: list, scan_idx: list, npart: int, slice=None):
        """
        getting (probably multi-shot) support set for evaluation
        sample from 50% (1shot) or 20 35 50 65 80 (5shot)
        Args:
            curr_cls:       current class to segment, starts from 1
            class_idx:      a list of all foreground class in nways, starts from 1
            npart:          how may chunks used to split the support
            scan_idx:       a list, indicating the current **i_th** (note this is idx not pid) training scan
        being served as support, in self.pid_curr_load
        """
        assert npart % 2 == 1
        assert curr_class != 0; assert 0 not in class_idx
        assert not self.is_train

        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        print(f'###### Using {len(scan_idx)} shot evaluation!')

        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [ half_part + part_interval * ii for ii in range(npart) ]

        print(f'###### Parts percentage: {pcts} ######')

        out_buffer = [] # [{scanid, img, lb}]
        for _part in range(npart):
            concat_buffer = [] # for each fold do a concat in image and mask in batch dimension
            for scan_order in scan_idx:
                _scan_id = self.pid_curr_load[ scan_order ]
                print(f'Using scan {_scan_id} as support!')

                # for _pc in pcts:
                _zlist = self.tp1_cls_map[self.label_name[curr_class]][_scan_id] # list of indices
                if slice is None:
                    _zid = _zlist[int(pcts[_part] * len(_zlist))]
                else:
                    _zid = slice
                
                print(f'support pivot number: {_zid}')    
                
                img = self.scans[_scan_id][:, :, _zid, None]
                lb = self.labels[_scan_id][:, :, _zid]

                img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
                lb  = torch.from_numpy( lb )

                if self.tile_z_dim:
                    img = img.repeat( [ self.tile_z_dim, 1, 1] )
                    assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

                sample = {"image": img,
                          "label":lb,}

                concat_buffer.append(sample)
            
            out_buffer.append({
                "image": torch.stack([itm["image"] for itm in concat_buffer], dim = 0),
                "label": torch.stack([itm["label"] for itm in concat_buffer], dim = 0),
                })

            # do the concat, and add to output_buffer

        # post-processing, including keeping the foreground and suppressing background.
        support_images = []
        support_mask = []
        support_class = []
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            support_mask.append(  self.getMaskMedImg( itm["label"], curr_class ))

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
        }

    def get_support_multi(self, curr_class: int, class_idx: list, scan_idx: list, npart: int, slice=None):
        """
        getting (probably multi-shot) support set for evaluation
        sample from 50% (1shot) or 20 35 50 65 80 (5shot)
        Args:
            curr_cls:       current class to segment, starts from 1
            class_idx:      a list of all foreground class in nways, starts from 1
            npart:          how may chunks used to split the support
            scan_idx:       a list, indicating the current **i_th** (note this is idx not pid) training scan
        being served as support, in self.pid_curr_load
        """
        assert npart % 2 == 1
        assert curr_class != 0; assert 0 not in class_idx
        assert not self.is_train

        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        print(f'###### Using {len(scan_idx)} shot evaluation!')

        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [ half_part + part_interval * ii for ii in range(npart) ]

        print(f'###### Parts percentage: {pcts} ######')

        out_buffer = [] # [{scanid, img, lb}]
        for _part in range(npart):
            concat_buffer = [] # for each fold do a concat in image and mask in batch dimension
            for scan_order in scan_idx:
                _scan_id = self.pid_curr_load[ scan_order ]
                print(f'Using scan {_scan_id} as support!')

                # for _pc in pcts:
                _zlist = self.tp1_cls_map[self.label_name[curr_class]][_scan_id] # list of indices
                if slice is None:
                    _zid = _zlist[int(pcts[_part] * len(_zlist))]
                else:
                    _zid = slice
                img = self.scans[_scan_id][:, :, _zid-1:_zid+2]
                lb = self.labels[_scan_id][:, :, _zid]

                img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
                lb  = torch.from_numpy( lb )

                # if self.tile_z_dim:
                #     img = img.repeat( [ self.tile_z_dim, 1, 1] )
                #     assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

                sample = {"image": img,
                          "label":lb,}

                concat_buffer.append(sample)
            
            out_buffer.append({
                "image": torch.stack([itm["image"] for itm in concat_buffer], dim = 0),
                "label": torch.stack([itm["label"] for itm in concat_buffer], dim = 0),
                })

            # do the concat, and add to output_buffer

        # post-processing, including keeping the foreground and suppressing background.
        support_images = []
        support_mask = []
        support_class = []
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            support_mask.append(  self.getMaskMedImg( itm["label"], curr_class ))

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
        }

    def get_support_multi_spaced(self, curr_class: int, class_idx: list, scan_idx: list, npart: int, slice=None):
        """
        getting (probably multi-shot) support set for evaluation
        sample from 50% (1shot) or 20 35 50 65 80 (5shot)
        Args:
            curr_cls:       current class to segment, starts from 1
            class_idx:      a list of all foreground class in nways, starts from 1
            npart:          how may chunks used to split the support
            scan_idx:       a list, indicating the current **i_th** (note this is idx not pid) training scan
        being served as support, in self.pid_curr_load
        """
        assert npart % 2 == 1
        assert curr_class != 0; assert 0 not in class_idx
        assert not self.is_train

        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        print(f'###### Using {len(scan_idx)} shot evaluation!')

        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [ half_part + part_interval * ii for ii in range(npart) ]

        print(f'###### Parts percentage: {pcts} ######')

        out_buffer = [] # [{scanid, img, lb}]
        for _part in range(npart):
            concat_buffer = [] # for each fold do a concat in image and mask in batch dimension
            for scan_order in scan_idx:
                _scan_id = self.pid_curr_load[ scan_order ]
                print(f'Using scan {_scan_id} as support!')

                # for _pc in pcts:
                _zlist = self.tp1_cls_map[self.label_name[curr_class]][_scan_id] # list of indices
                if slice is None:
                    _zid = _zlist[int(pcts[_part] * len(_zlist))]
                else:
                    _zid = slice
                img = self.scans[_scan_id][:, :, _zid-2:_zid+3:2]
                lb = self.labels[_scan_id][:, :, _zid]

                img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
                lb  = torch.from_numpy( lb )

                # if self.tile_z_dim:
                #     img = img.repeat( [ self.tile_z_dim, 1, 1] )
                #     assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

                sample = {"image": img,
                          "label":lb,}

                concat_buffer.append(sample)
            
            out_buffer.append({
                "image": torch.stack([itm["image"] for itm in concat_buffer], dim = 0),
                "label": torch.stack([itm["label"] for itm in concat_buffer], dim = 0),
                })

            # do the concat, and add to output_buffer

        # post-processing, including keeping the foreground and suppressing background.
        support_images = []
        support_mask = []
        support_class = []
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            support_mask.append(  self.getMaskMedImg( itm["label"], curr_class ))

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
        }

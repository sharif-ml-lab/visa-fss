import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.segmentation import slic as skimage_slic


# img_bname = f'~/medical-data/ABD-CT/ABD-CT-Preprocessed/superpix-MIDDLE_*.nii.gz'
# img_bname = f'~/medical-data/ABD-CT/ABD-CT-Preprocessed/supervox_*.nii.gz'
img_bname = f'~/medical-data/ABD-CT/ABD-CT-Preprocessed/supervox_NEW_*.nii.gz'
imgs = glob.glob(img_bname)
imgs = sorted(imgs, key = lambda x: int(x.split('_')[-1].split('.nii.gz')[0]) )

# imgs = ['~/medical-data/ABD-CT/ABD-CT-Preprocessed/image_0.nii.gz']
# msks = ['~/medical-data/ABD-CT/ABD-CT-Preprocessed/fgmask_0.nii.gz']

def supervox(img, **kwargs):
    c = skimage_slic(np.moveaxis(img, 0, -1),
                        n_segments=120, 
                        compactness=0.01, #should be low
                        max_num_iter=10,
                        sigma=2, 
                        spacing=None, 
                        multichannel=False,        # Fixed
                        convert2lab=None, 
                        enforce_connectivity=True, # Fixed
                        min_size_factor=0.2, #0.5
                        max_size_factor=5,
                        slic_zero=False, 
                        start_label=1,             # Fixed
                        mask=None,                 # Fixed
                        channel_axis=- 1)
    return np.moveaxis(c, -1, 0)

num_superpixels = []
for path in imgs:
    vol = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(vol) 
    for idx in range(vol.shape[0]):
        slc = vol[idx, :, :]
        count = len(np.unique(slc)) - 1
        num_superpixels.append(count)
num_superpixels = sorted(num_superpixels)
plt.hist(num_superpixels, bins=40, edgecolor='black')
plt.savefig('hist.png')

size_superpixels = []
for path in imgs:
    vol = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(vol) 
    for idx in range(vol.shape[0]):
        slc = vol[idx, :, :]
        for sp in np.unique(slc):
            if sp == 0:
                continue
            sze = (slc == sp).sum()
            size_superpixels.append(sze)
size_superpixels = sorted(size_superpixels)
plt.hist(size_superpixels, bins=40, edgecolor='black')
plt.savefig('tmp_plots/size_hist2.png')

# size_superpixels = []
# for path, fgpath in zip(imgs, msks):
#     scan = sitk.ReadImage(path)
#     scan = sitk.GetArrayFromImage(scan)
#     scan_msk = sitk.ReadImage(fgpath)
#     scan_msk = sitk.GetArrayFromImage(scan_msk)
#     vol = supervox(scan)
#     vol = vol * scan_msk.astype(np.int32)

#     # for idx in range(20):
#     #     slc = vol[idx, :, :]
#     #     plt.imshow(slc)
#     #     plt.savefig(f'tmp_plots/{idx}.png')

#     for idx in range(vol.shape[0]):
#         slc = vol[idx, :, :]
#         for sp in np.unique(slc):
#             if sp == 0:
#                 continue
#             sze = (slc == sp).sum()
#             size_superpixels.append(sze)
# size_superpixels = sorted(size_superpixels)
# plt.hist(size_superpixels, bins=40, edgecolor='black')
# plt.savefig('tmp_plots/size_hist2.png')

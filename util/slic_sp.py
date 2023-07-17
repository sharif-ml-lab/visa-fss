import numpy as np
from skimage.segmentation import slic as skimage_slic

def extract_SuperVoxel_Slic(img):
    c = skimage_slic(img,
                     n_segments=200, 
                     compactness=0.01, #should be low
                     max_num_iter=10,
                     sigma=2, 
                     spacing=None, 
                     multichannel=False,        # Fixed
                     convert2lab=None, 
                     enforce_connectivity=True, # Fixed
                     min_size_factor=0.01, #0.5
                     max_size_factor=5,
                     slic_zero=False, 
                     start_label=1,             # Fixed
                     mask=None,                 # Fixed
                     channel_axis=- 1)
    return c
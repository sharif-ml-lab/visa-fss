import cv2
import torch
import skimage
import numpy as np
import matplotlib.pyplot as plt

def superpixel_refine(img, lbl, ratio):

    seg_func = skimage.segmentation.felzenszwalb
    segments = seg_func(img*255, min_size=200, sigma=0.1) + 1
    refined_label = np.copy(lbl)

    for i_s in range(int(segments.max())):
        sindex = i_s + 1
        s_s = np.where(segments == sindex, 1, 0)
        q_s = np.sum(s_s)
        t_s = s_s * refined_label
        r_s = np.sum(t_s)

        ratio_s = r_s / q_s
        if ratio_s < ratio:
            refined_label[s_s == 1] = 0

    return refined_label

def s2v_refine(s1, s2, m1, m2, dilation_kernel_size=15):
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=np.float32)

    P = m1
    P_dilated = cv2.dilate(P, kernel)
    N = P_dilated - P

    p = (s1 * P).sum() / P.sum()
    n = (s1 * N).sum() / N.sum()

    condition1 = (((s2 - p) ** 2) < ((s2 - n) ** 2)) * 1
    condition2 = m2
    
    m2_verified = condition1 * condition2

    # plt.imshow(P)
    # plt.savefig('tmp_plots/P.png')
    # plt.imshow(N)
    # plt.savefig('tmp_plots/N.png')
    # plt.imshow(m2)
    # plt.savefig('tmp_plots/m2.png')
    # plt.imshow(m2_verified)
    # plt.savefig('tmp_plots/m2_verified.png')
    # exit()

    return m2_verified 
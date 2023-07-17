import torch

def format_ssp(support_images, support_fg_mask, support_bg_mask, query_images, query_labels):
    
    # image_s_list
    image_s_list = []
    assert len(support_images) == 1         # ssp only supports 1 way setting
    for shot in support_images[0]:          # shot shape -> (1, 3, h, w)
        image_s_list.append(shot)

    # mask_s_list
    mask_s_list = []
    assert len(support_fg_mask) == 1 
    assert len(support_bg_mask) == 1
    for (fg, bg) in zip(support_fg_mask[0], support_bg_mask[0]):
        mask_s_list.append(fg)

    # query_images
    image_q_list = torch.cat(query_images, dim=0)
    mask_q_list = query_labels
    
    return image_s_list, mask_s_list, image_q_list, mask_q_list

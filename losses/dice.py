from torch.nn import functional as F

def compute_dice_loss(pred, gt):
    fg_pred = pred[:, 1, :, :]
    fg_pred = F.sigmoid(fg_pred)    
    intersection = (fg_pred * gt).sum()
    dice = (2. * intersection) / (fg_pred.sum() + gt.sum())  
    return 1 - dice

def dice_interslice(srcs, trgs):
    a = F.softmax(srcs, dim=1)[:, 1, :, :]
    b = F.softmax(trgs, dim=1)[:, 1, :, :]
    intersections = (a * b).sum((1, 2))
    unions = a.sum((1, 2)) + b.sum((1, 2))
    mean_dice = (2 * intersections / unions).mean()
    return 1 - mean_dice

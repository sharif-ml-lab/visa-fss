import torch
import torch.nn.functional as F

def fg_size_loss(query_preds):

    query_preds = F.softmax(query_preds, dim=1)
    
    fg_sizes = query_preds[:, 1, :, :].sum(axis=(1, 2))
    bg_sizes = query_preds[:, 0, :, :].sum(axis=(1, 2))

    loss = (fg_sizes[1:] - fg_sizes[:-1]).abs().mean()
    return loss

def sz_loss_function(query_preds, support):
    sup_size = support.sum()
    query_preds = F.softmax(query_preds, dim=1)[:, 1, :, :]
    preds_sizes = query_preds.sum((1, 2))
    loss = ((preds_sizes - sup_size) ** 2).mean()
    return loss

def l2_loss_function(query_fts):
    means = 0
    for i in range(query_fts.shape[0] - 1):
        mn = ((query_fts[i] - query_fts[i+1]) ** 2).sum(1).mean()
        means += mn
    return means / (query_fts.shape[0] - 1)

if __name__ == '__main__':
    print(fg_size_loss(None))

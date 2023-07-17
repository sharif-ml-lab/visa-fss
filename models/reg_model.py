import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, ndims, vol_size, enc_nf, dec_nf, full_size=True, src_feats=1, tgt_feats=1):
        super(UNet, self).__init__()
        self.ndims = ndims
        self.vol_size = vol_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        self.full_size = full_size
        self.src_feats = src_feats
        self.tgt_feats = tgt_feats
        
        self.down_convs = nn.ModuleList()
        features = self.src_feats + self.tgt_feats
        for i in range(len(self.enc_nf)):
            self.down_convs.append(self.conv_block(self.ndims, features, self.enc_nf[i], 2))
            features = self.enc_nf[i]
        
        self.up_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(5):
            self.up_convs.append(self.conv_block(self.ndims, features, self.dec_nf[i]))
            features = self.dec_nf[i]
            if i < 3:
                w = int(vol_size[0] / 2 ** (3 - i))
                h = int(vol_size[1] / 2 ** (3 - i))
                self.upsamples.append(self.upsample_layer(size=(w, h)))
                features += self.enc_nf[-i - 2]

        if self.full_size:
            self.fs_upsample = self.upsample_layer(size=vol_size)
            features += self.src_feats + self.tgt_feats
            self.fs_conv = self.conv_block(self.ndims, features, self.dec_nf[5])
            features = self.dec_nf[5]
        
        if len(self.dec_nf) == 7:
            self.extra_conv = self.conv_block(self.ndims, features, self.dec_nf[6])
        
        self.out_features = features
    
    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)

        x_enc = [x]
        for i in range(len(self.enc_nf)):
            x = self.down_convs[i](x)
            x_enc.append(x)
        
        for i in range(5):
            x = self.up_convs[i](x)
            if i < 3:
                x = self.upsamples[i](x)
                x = torch.cat([x, x_enc[-i - 2]], dim=1)
        
        if self.full_size:
            x = self.fs_upsample(x)
            x = torch.cat([x, x_enc[0]], dim=1)
            x = self.fs_conv(x)
        
        if len(self.dec_nf) == 7:
            x = self.extra_conv(x)
        
        return x
        
    def upsample_layer(self, size):
        return nn.Upsample(size=size)

    def conv_block(self, ndims, in_features, out_features, strides=1):
        Conv = getattr(nn, 'Conv%dd' % ndims)
        conv = Conv(in_features, out_features, kernel_size=3, padding=1, stride=strides)
        nn.init.xavier_normal_(conv.weight)
        return nn.Sequential(conv, nn.LeakyReLU(.2, True))

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow 

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

class RegModel(nn.Module):
    def __init__(self, ndims, vol_size, enc_nf, dec_nf, full_size=True, src_feats=1, tgt_feats=1, mode='bilinear'):
        super(RegModel, self).__init__()
        
        self.unet = UNet(ndims, vol_size, enc_nf, dec_nf, full_size=True, src_feats=1, tgt_feats=1)
        self.disp_layer = nn.Conv2d(dec_nf[-1], ndims, kernel_size=3, padding=1)
        self.spatial_transformer = SpatialTransformer(size=vol_size, mode=mode)
        
    def forward(self, src, tgt):
        unet_out = self.unet(src, tgt)
        disp = self.disp_layer(unet_out)
        moved = self.spatial_transform(src, disp)
        return moved, disp
    
    def spatial_transform(self, src, disp):
        return self.spatial_transformer(src, disp)



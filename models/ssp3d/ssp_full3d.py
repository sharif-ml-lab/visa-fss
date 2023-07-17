import models.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
import pdb

class SSP_FULL3D_MatchingNet(nn.Module):
    def __init__(self, backbone, refine_steps=1):
        super(SSP_FULL3D_MatchingNet, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.refine_steps = refine_steps
        print(f'SSP3D: refine steps: {self.refine_steps}')

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        h, w = img_q.shape[-2:]

        # feature maps of support images
        feature_s_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k])
                s_0 = self.layer1(s_0)
            s_0 = self.layer2(s_0)
            s_0 = self.layer3(s_0)
            feature_s_list.append(s_0)
            del s_0
        # feature map of query image
        with torch.no_grad():
            q_0 = self.layer0(img_q)
            q_0 = self.layer1(q_0)
        q_0 = self.layer2(q_0)
        feature_q = self.layer3(q_0)

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        supp_out_ls = []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0

                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        out_0 = self.similarity_func(feature_q, FP, BP)
        current_out = out_0
        for rf in range(self.refine_steps):
            ##################### Self-Support Prototype (SSP) #####################
            SSFP, SSBP, ASFP, ASBP = self.SSP3D_func(feature_q, current_out)

            a = 0.9
            b = 0.8

            FP = FP * (1 - a) + SSFP * a
            BP = SSBP * (1 - b) + ASBP * b

            current_out = self.similarity_func(feature_q, FP, BP)

        current_out = F.interpolate(current_out, size=(h, w), mode="bilinear", align_corners=True)
        out_ls = [current_out]

        if self.training:
            
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)

            out_ls.append(self_out)
            out_ls.append(supp_out)

        return out_ls

    def SSP3D_func(self, feature_q, out):
       
        bs, _, f_h, f_w = feature_q.shape
        pred_1 = out.softmax(1)
        pred_1 = pred_1.permute((1, 0, 2, 3)).reshape((2, -1))
        pred_fg = pred_1[1, :]
        pred_bg = pred_1[0, :]

        fg_thres, bg_thres = 0.7, 0.6
        cur_feat = feature_q.permute((1, 0, 2, 3)).reshape((1024, -1))
        
        if (pred_fg > fg_thres).sum() > 0:
            fg_feat = cur_feat[:, (pred_fg>fg_thres)] #.mean(-1)
        else:
            fg_feat = cur_feat[:, torch.topk(pred_fg, 12).indices] #.mean(-1)
        if (pred_bg > bg_thres).sum() > 0:
            bg_feat = cur_feat[:, (pred_bg>bg_thres)] #.mean(-1)
        else:
            bg_feat = cur_feat[:, torch.topk(pred_bg, 12).indices] #.mean(-1)

        fg_proto = fg_feat.mean(-1)                                                     # 1024
        bg_proto = bg_feat.mean(-1)                                                     # 1024

        fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True)                        # 1024, N1
        bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)                        # 1024, N2
        cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)                     # 1024, HWD 
        cur_feat_norm_t = cur_feat_norm.t()                                             # DHW, 1024
        fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0                      # DHW, N1
        bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0                      # DHW, N2
        fg_sim = fg_sim.softmax(-1)
        bg_sim = bg_sim.softmax(-1)
        fg_proto_local = torch.matmul(fg_sim, fg_feat.t())                              # DHW, 1024
        bg_proto_local = torch.matmul(bg_sim, bg_feat.t())                              # DHW, 1024
        fg_proto_local = fg_proto_local.t().reshape((1024, bs, f_h, f_w))                    # 1024, D, H, W
        bg_proto_local = bg_proto_local.t().reshape((1024, bs, f_h, f_w))                    # 1024, D, H, W

        # global proto
        new_fg = fg_proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)               # 1, 1024, 1, 1
        new_bg = bg_proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)               # 1, 1024, 1, 1
        # local proto
        new_fg_local = fg_proto_local.permute((1, 0, 2, 3))                      # D, 1024, H, W
        new_bg_local = bg_proto_local.permute((1, 0, 2, 3))                      # D, 1024, H, W

        return new_fg, new_bg, new_fg_local, new_bg_local

    def SSP_func(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7 #0.9 #0.6
            bg_thres = 0.6 #0.6
            cur_feat = feature_q[epi].view(1024, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)               # Average fg proto across all queries (3D case)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # print('new fg bg:', new_fg.shape, new_bg.shape)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

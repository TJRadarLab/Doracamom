"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.runner import force_fp32
import torch
from torch import nn
from torchvision.models.resnet import resnet18
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from torchvision.utils import save_image
from mmdet3d.models.fusion_layers import apply_3d_transformation
import torch.nn.functional as F
from projects.mmdet3d_plugin.ops.bev_pool import bev_pool
from projects.mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmcv.cnn import build_conv_layer, build_norm_layer

# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.shape[2:],  mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

#----------预测深度概率并将特征与概率相乘-----
class CamEncode(nn.Module):
    def __init__(self, D, C, inputC):
        super(CamEncode, self).__init__()
        self.D = D #--41
        self.C = C #--64
        self.depthnet = DepthNet(in_channels=inputC, mid_channels=inputC,context_channels=C,depth_channels=D, norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01))

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # Depth
        x = self.depthnet(x) #--[6,123,135,240]

        depth = self.get_depth_dist(x[:, :self.D]) #--[6,59,135,240]
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)#--对应元素相乘
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x) #--torch.Size([6, 64, 59, 135, 240])

        return x, depth


class LiftSplatShoot_Depth(nn.Module):
    def __init__(self, lss=False, final_dim=(900, 1600), camera_depth_range=[4.0, 45.0, 1.0], pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, grid=3, inputC=256, camC=64):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            final_dim: actual RGB image size for actual BEV coordinates, default (900, 1600)
            downsample (int): the downsampling rate of the input camera feature spatial dimension (default (224, 400)) to final_dim (900, 1600), default 4. 
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            pc_range: point cloud range.
            inputC: input camera feature channel dimension (default 256).
            grid: stride for splat, see https://github.com/nv-tlabs/lift-splat-shoot.

        """
        super(LiftSplatShoot_Depth, self).__init__()
        self.pc_range = pc_range
        self.grid_conf = {
            'xbound': [pc_range[0], pc_range[3], grid],
            'ybound': [pc_range[1], pc_range[4], grid],
            'zbound': [pc_range[2], pc_range[5], grid],
            'dbound': camera_depth_range,
        }
        self.camera_depth_range = camera_depth_range
        self.final_dim = final_dim 
        self.grid = grid

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'], )
        #------------与BaseModule中的init冲突，longtensor无法取mean----------------
        #------------参数会在cuda下，需要修改-----
        # self.dx = nn.Parameter(dx, requires_grad=False)  #---tensor([0.5000, 0.5000, 0.5000], device='cuda:0')--
        # self.bx = nn.Parameter(bx, requires_grad=False) #----tensor([-49.7500, -49.7500,  -4.7500], device='cuda:0')
        # self.nx = nn.Parameter(nx, requires_grad=False) #---tensor([200, 200,  16], device='cuda:0')
        
        self.dx = dx #---tensor([0.5000, 0.5000, 0.5000], device='cuda:0')--
        self.bx = bx#tensor([-59.7500, -39.7500,  -2.7500])
        self.nx = nx #---tensor([240, 160,  16], device='cuda:0')
 
        self.downsample = downsample
        self.fH, self.fW = self.final_dim[0] // self.downsample, self.final_dim[1] // self.downsample
        self.camC = camC
        self.inputC = inputC
        self.frustum = self.create_frustum() #--#--torch.Size([59, 135, 240, 3]),代表原始图像大小和深度
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.inputC)
        
        self.constant_std = 0.5
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        z = self.grid_conf['zbound']
        cz = int(self.camC * ((z[1] - z[0]) // z[2]))
        self.lss = lss
        self.bevencode = nn.Sequential(
            nn.Conv2d(cz, cz, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cz),
            nn.ReLU(inplace=True),
            nn.Conv2d(cz, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, inputC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inputC),
            nn.ReLU(inplace=True)
        )
        if self.lss:
          self.bevencode = nn.Sequential(
            nn.Conv2d(cz, camC, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(camC),
            BevEncode(inC=camC, outC=inputC)

        )
        
    #--[41,112,200,3]在图像平面创建网格，每个代表的尺寸为
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim  #---[1080,1920]
        fH, fW = self.fH, self.fW  #--[135,240]
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) 
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3按xyz順序
        frustum = torch.stack((xs, ys, ds), -1) #--[69,135,240,3]
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, post_rots=None, post_trans=None,extra_rots=None,extra_trans=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  #--torch.Size([1, 6, 3])
        # ADD
        # undo post-transformation
        # B x N x D x H x W x 3
        if post_rots is not None or post_trans is not None:
            if post_trans is not None:
                points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots is not None:
                points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        else:
            points = self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)  # B x N x D x H x W x 3 x 1 torch.Size([1, 6, 69,135,240,3, 1])

        # cam_to_ego rots和trans已经是img2lidar了
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) #torch.Size([1, 6, 59, 135, 240, 3, 1])
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3) #--torch.Size([1, 6, 59, 135, 240, 3])
        #---转到lidar坐标系下
        if extra_rots is not None or extra_trans is not None:
            if extra_rots is not None:
                points = extra_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            if extra_trans is not None:
                points += extra_trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        x, depth = self.camencode(x) #--torch.Size([6, 64, 59, 135, 240])
        x = x.view(B, N, self.camC, self.D, H, W) #--torch.Size([1, 6, 64, 59, 135, 240])
        x = x.permute(0, 1, 3, 4, 5, 2) #--torch.Size([1, 6, 59, 135, 240, 64])
        depth = depth.view(B, N, self.D, H, W) #torch.Size([1, 6, 59, 135, 240])
        return x, depth

    @force_fp32()
    def bev_pool(self, geom_feats, x): #--torch.Size([1, 6, 59, 135, 240, 3])torch.Size([1, 6, 59, 135, 240, 64])
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx.to(x.device) - self.dx.to(x.device) / 2.)) / self.dx.to(x.device)).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        batch_ix = batch_ix.to(geom_feats.device)
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box这里过滤掉超出范围的点，视锥体和点云范围交集
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]  #---特征
        geom_feats = geom_feats[kept]  #--坐标

        x = bev_pool(x, geom_feats, B, self.nx.to(x.device)[2], self.nx.to(x.device)[0], self.nx.to(x.device)[1])

        # collapse Z
        # final = torch.cat(x.unbind(dim=2), 1)

        return x  #--torch.Size([1, 64, 16, 240, 160])



    def get_voxels(self, x, rots=None, trans=None, post_rots=None, post_trans=None,extra_rots=None,extra_trans=None):
        geom = self.get_geometry(rots, trans, post_rots, post_trans,extra_rots,extra_trans)#--torch.Size([1, 6, 59, 135, 240, 3])
#         #---------------绘制一下自车下面的点-------------
#         import matplotlib.pyplot as plt
#         # 生成示例数据
#         fig, ax = plt.subplots(figsize=(64, 48)) 
#         geom_points = geom.detach().cpu().numpy()
#         geom_points = geom_points.reshape(-1, 3)
#         #_---过滤掉超出范围的点--
#         geom_points = geom_points[(geom_points[:, 0] >= -60) & (geom_points[:, 0] <= 60) & 
#                               (geom_points[:, 1] >= -40) & (geom_points[:, 1] <= 40)]
#         ax.scatter(geom_points[:, 0], geom_points[:, 1], color='blue', label='geom Points')  # 设置颜色为蓝色，点的大小为30
        
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_title('Scatter Plot')
#         ax.legend()  # 显示图例
#         # 保存图像
#         plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/geom_points.png',bbox_inches='tight', pad_inches=0.1, dpi=150)
# # #-----------------------------------------------------


        #---------------------------------------------
        x, depth = self.get_cam_feats(x) ##--torch.Size([6, 64, 59, 135, 240])
        # x = self.voxel_pooling(geom, x) #改成bevpool
        x = self.bev_pool(geom, x)
        return x, depth

    def s2c(self, x):
        B, C, H, W, L = x.shape #---W是x，L是y# griddify (B x C x Z x X x Y)
        bev = torch.reshape(x, (B, C*H, W, L)) #--直接reshape掉高度维torch.Size([1, 1024, 240, 160])
        bev = bev.permute(0,1,3,2).contiguous()  #torch.Size([1, 1024, 160, 240])BCYX
        return bev

    def forward(self, x, rots, trans, lidar2img_rt=None, img_metas=None, post_rots=None, post_trans=None, extra_rots=None,extra_trans=None):
        x, depth = self.get_voxels(x, rots, trans, post_rots, post_trans,extra_rots,extra_trans) # [B, C, Z, X, Y]torch.Size([1, 64, 16, 288, 224])
        bev = self.s2c(x)
        x = self.bevencode(bev) #--torch.Size([1, 256, 160, 240])
        return x, depth


#----------------计算depth_loss--------------------------

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
    
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss

    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        #--torch.Size([1, 6, 1080, 1920]) torch.Size([1, 6, 59, 135, 240])
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.camera_depth_range, constant_std=self.constant_std)
        #--6,135,240----
        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.camera_depth_range[0]) & (depth_values <= (self.camera_depth_range[1] - self.camera_depth_range[2]))        
        
        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)[fg_mask]
        
        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)
        
        return depth_loss, depth_values[fg_mask]
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds, loss_depth_type):
        if loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        
        elif loss_depth_type == 'kld':
            depth_loss,min_depth = self.get_klv_depth_loss(depth_labels, depth_preds)
        
        else:
            pdb.set_trace()
        
        return depth_loss,min_depth


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, norm_cfg=dict(type='BN2d')):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels,norm_cfg=None):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # nn.BatchNorm2d(mid_channels),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                context_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            ASPP(mid_channels, mid_channels, norm_cfg=norm_cfg),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x):
        
        x = self.reduce_conv(x)
        context = self.context_conv(x)  
        depth = self.depth_conv(x)

        return torch.cat([depth, context], dim=1)
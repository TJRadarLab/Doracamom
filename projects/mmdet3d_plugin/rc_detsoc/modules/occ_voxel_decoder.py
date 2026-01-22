from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import build_conv_layer, build_norm_layer
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VoxelDecoderV1(BaseModule):

    def __init__(self,
                 occ_size=[240,160,16],
                 bev_h=80,
                 bev_w=120,
                 bev_z=8,
                 conv_up_layer = 2,
                 embed_dim = 256,
                 out_dim = 64,

                 norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01,requires_grad=True)):
        super(VoxelDecoderV1, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.out_dim = out_dim

        self.conv_up_layer = conv_up_layer
        assert occ_size[0]//bev_w == occ_size[1]//bev_h  #--加一个判断
        upsample_scale = int(np.math.log2(occ_size[0] // bev_w))
        upsample = []
        if upsample_scale == 0:
            upsample.append(nn.ConvTranspose3d(embed_dim, self.out_dim, (1, 3, 3), stride=(1, 1, 1),padding=(0,1,1)))
            upsample.append(build_norm_layer(norm_cfg, self.out_dim)[1])
            upsample.append(nn.ReLU(inplace=True))
        else:
            for _ in range(upsample_scale-1):
                upsample.append(nn.ConvTranspose3d(embed_dim, embed_dim, (1, 4, 4), stride=(1, 2, 2), padding=(0,1,1)))
                upsample.append(build_norm_layer(norm_cfg, embed_dim)[1])
                upsample.append(nn.ReLU(inplace=True))
            #----这里Z也上采样
            upsample.append(nn.ConvTranspose3d(embed_dim, self.out_dim, (2, 4, 4), stride=(2, 2, 2),padding=(0,1,1)))
            upsample.append(build_norm_layer(norm_cfg, self.out_dim)[1])
            upsample.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*upsample)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, inputs):
        #-----torch.Size([1, 256, 8, 80, 120])
        voxel_input = inputs.view(1,self.bev_h,self.bev_w,self.bev_z, -1).permute(0,4,3,1,2)

        voxel_feat = self.upsample(voxel_input) #---torch.Size([1, 32, 16, 160, 240])


        return voxel_feat
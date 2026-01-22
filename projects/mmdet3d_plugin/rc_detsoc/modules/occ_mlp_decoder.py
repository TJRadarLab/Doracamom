from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.utils import build_from_cfg
#---------------------这里加入radar--------------
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MLP_DecoderV1(BaseModule):

    def __init__(self,
                 num_classes,
                 input_dim_radar=256,
                 out_dim = 64,
                 inter_up_rate = [2,2,2],
                 add_radar=False,
                 occ_det_aux = True,
                 upsampling_method='trilinear',
                 align_corners=False,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 norm_cfg_3dconv=dict(type='BN3d', eps=1e-3, momentum=0.01, requires_grad=True),
                 occ_det_decoder = dict(type='FRPN_OCC',in_channels=32,loss_weight=1.0)):
        super(MLP_DecoderV1, self).__init__()
        self.num_classes = num_classes
        self.upsampling_method = upsampling_method
        self.out_dim = out_dim
        self.align_corners = align_corners
        self.inter_up_rate = inter_up_rate
        #------------卷积加融合------------
        self.conv_radar =  ConvModule(
                input_dim_radar,
                32, #--384
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                inplace=False)
        # self.conv_radar = MLP(dim_x=16,filter_size=16, act_fn='relu',layer_size=1)
        self.fusion_module = FusionBlock_3D(32,32,norm_cfg_3dconv,add_radar) #----融合模块
        self.occ_det_aux = occ_det_aux
        if self.occ_det_aux:
            self.occ_det_decoder = build_from_cfg(occ_det_decoder, FUSION_LAYERS)
        self.mlp_decoder = MLP(dim_x=self.out_dim,act_fn='softplus',layer_size=2)
        self.classifier = nn.Linear(self.out_dim, self.num_classes)
                
    def forward(self, img_voxel,radar_bev):
        

        b, _, z, h, w = img_voxel.shape
        radar_feat = self.conv_radar(radar_bev) #torch.Size([1,32,160,240])
        radar_feat = radar_feat.unsqueeze(-1).repeat(1,1,1,1,z).permute(0,1,4,2,3).contiguous() #torch.Size([1, 32, 160, 240, 16])
        #--------------------------------------------------------
        #----torch.Size([1, 32, 16, 160, 240])
        voxel_point = self.fusion_module(img_voxel,radar_feat)
        if self.occ_det_aux:
            voxel_det = self.occ_det_decoder(voxel_point)
        else:
            voxel_det = None
        voxel_point = voxel_point.permute(0,2,3,4,1).contiguous().view(1,-1,self.out_dim) #--这里是否contiguous都可以用view
        voxel_point_feat = self.mlp_decoder(voxel_point) #--torch.Size(torch.Size([1, 614400, 32]))
        point_cls = self.classifier(voxel_point_feat) #--torch.Size(torch.Size([1, 614400, 12]))
        #---torch.Size([1, 12, 16, 160, 240])
        voxel_point_cls = point_cls.view(1,z,h,w,-1).permute(0,4,1,2,3).contiguous()
        
        return voxel_point_cls,voxel_det

class MLP(torch.nn.Module):
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            elif act_fn == 'softplus':
                self.nn_layers.append(torch.nn.Softplus())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
                elif act_fn == 'softplus':
                    self.nn_layers.append(torch.nn.Softplus())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        for layer in self.nn_layers:
            x = layer(x)
                
        return x

class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
        norm_cfg=dict(type='BN3d', requires_grad=True),
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              bias=False)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)






class FusionBlock_3D(nn.Module):
    def __init__(self, img_channels, pts_channels,norm_cfg,add_radar):
        super(FusionBlock_3D, self).__init__()
        self.fuse_conv = ConvBnReLU3D(img_channels+pts_channels,img_channels,norm_cfg=norm_cfg)
        self.add_radar = add_radar
        self.attention = nn.Sequential(
            nn.Conv3d(img_channels, img_channels,
                      kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, img_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv3d(img_channels, img_channels,
                      kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, img_channels)[1],
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_voxel, pts_voxel):
        cat_feature = torch.cat((img_voxel, pts_voxel), dim=1)
        
        fuse_out = self.fuse_conv(cat_feature)
        attention_map = self.attention(fuse_out)
        if self.add_radar:
            out = fuse_out*attention_map + img_voxel + pts_voxel
        else:
            out = fuse_out*attention_map + img_voxel
        
        return out
    
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccbevFusion2D(BaseModule):
    def __init__(self, img_channels, radar_channels,img_bev_conv_channel=512,add_radar=False,norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01, requires_grad=True)):
        super(OccbevFusion2D, self).__init__()
        self.add_radar = add_radar

        self.img_bev_conv = nn.Sequential(
            nn.Conv2d(img_bev_conv_channel, 256,
                      kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(img_channels+radar_channels, img_channels,
                      kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, img_channels)[1],
            nn.ReLU(inplace=True),
            
        )

        self.attention = nn.Sequential(
            nn.Conv2d(img_channels, img_channels,
                      kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, img_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(img_channels, img_channels,
                      kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, img_channels)[1],
            nn.Sigmoid()
        )
        if self.add_radar and radar_channels==384:
            self.reshape_radar = nn.Sequential(
            nn.Conv2d(radar_channels, 256,
                      kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                       # mode='fan_out',
                                        #nonlinearity='relu'
                                        )
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, voxel_feat,bev_radar_feat,voxel_det=None):
        # if voxel_det is not None:
        #     det_occ_score = voxel_det.sigmoid() #--torch.Size([1, 240, 160, 16, 1])
        #     det_occ_score = det_occ_score.permute(0,4,3,2,1).repeat(1,voxel_feat.shape[1],1,1,1)
        #     voxel_feat = det_occ_score*voxel_feat
        b,c,z,h,w = voxel_feat.shape
        voxel_feat = voxel_feat.reshape(b,c*z,h,w)
        img_feat = self.img_bev_conv(voxel_feat)

        cat_feature = torch.cat((img_feat,bev_radar_feat), dim=1)
        fuse_out = self.fuse_conv(cat_feature)
        attention_map = self.attention(fuse_out)
        if self.add_radar:
            if bev_radar_feat.shape[1]==384:
                bev_radar_feat = self.reshape_radar(bev_radar_feat)
            out = fuse_out*attention_map + img_feat + bev_radar_feat 
        else:
            out = fuse_out*attention_map + img_feat 
        return out
    

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MLP_DecoderV1_ablation(BaseModule):

    def __init__(self,
                 num_classes,
                 input_dim_radar = 384,
                 out_dim = 64,
                 inter_up_rate = [2,2,2],
                 
                 occ_det_aux = True,
                 upsampling_method='trilinear',
                 align_corners=False,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 norm_cfg_3dconv=dict(type='BN3d', eps=1e-3, momentum=0.01, requires_grad=True),
                 occ_det_decoder = dict(type='FRPN_OCC',in_channels=32,loss_weight=1.0)):
        super(MLP_DecoderV1_ablation, self).__init__()
        self.num_classes = num_classes
        self.upsampling_method = upsampling_method
        self.out_dim = out_dim
        self.align_corners = align_corners
        self.inter_up_rate = inter_up_rate
        #------------卷积加融合------------
        self.conv_radar =  ConvModule(
                input_dim_radar,
                32, #--384
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                inplace=False)
        # self.conv_radar = MLP(dim_x=16,filter_size=16, act_fn='relu',layer_size=1)
        # self.fusion_module = FusionBlock_3D(32,32,norm_cfg_3dconv) #----融合模块
        # self.fusion_module = ConvBnReLU3D(64,32,norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01, requires_grad=True))
        self.occ_det_aux = occ_det_aux
        if self.occ_det_aux:
            self.occ_det_decoder = build_from_cfg(occ_det_decoder, FUSION_LAYERS)
        self.mlp_decoder = MLP(dim_x=self.out_dim,act_fn='softplus',layer_size=2)
        self.classifier = nn.Linear(self.out_dim, self.num_classes)
                
#----#torch.Size([1, 32, 16, 160, 240]),torch.Size([1, 384, 160, 240])
    def forward(self, img_voxel,radar_bev):
        
        # z h w的顺序
        #------torch.Size([1, 307200, 32])----
        b, _, z, h, w = img_voxel.shape
        
        radar_feat = self.conv_radar(radar_bev) #torch.Size([1,32,160,240])
        radar_feat = radar_feat.unsqueeze(-1).repeat(1,1,1,1,z).permute(0,1,4,2,3).contiguous() #torch.Size([1, 32, 160, 240, 16])
        #--------------------------------------------------------
        #----torch.Size([1, 32, 16, 160, 240])
        cat_feature = torch.add(img_voxel, radar_feat)
        voxel_point = cat_feature

        if self.occ_det_aux:
            voxel_det = self.occ_det_decoder(voxel_point)
        else:
            voxel_det = None
        voxel_point = voxel_point.permute(0,2,3,4,1).contiguous().view(1,-1,self.out_dim) #--这里是否contiguous都可以用view
        voxel_point_feat = self.mlp_decoder(voxel_point) #--torch.Size(torch.Size([1, 614400, 32]))
        point_cls = self.classifier(voxel_point_feat) #--torch.Size(torch.Size([1, 614400, 12]))
        #---torch.Size([1, 12, 16, 160, 240])
        voxel_point_cls = point_cls.view(1,z,h,w,-1).permute(0,4,1,2,3).contiguous()
        return voxel_point_cls,voxel_det
    


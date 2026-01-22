
from mmcv.cnn.bricks.registry import ATTENTION
import torch
from torch import nn
import torch.utils.checkpoint as cp


from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn import kaiming_init, constant_init
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.cnn import ConvModule
from .residual_block_3d import Bottleneck
from .x3d_block import ResBlock, X3DTransform
from .temporal_self_attention import TemporalSelfAttention
from .deformable_self_attention_3D_custom import DeformSelfAttention3DCustom
from mmcv.cnn.bricks.transformer import build_positional_encoding

#------------------双分支radar/cam时序编码器---------------------
@ATTENTION.register_module()
class OccTemporalEncoderV1(BaseModule):
    def __init__(self, bev_h: int, bev_w: int, bev_z: int, num_bev_queue: int, embed_dims: int, num_block: int,
                 block_type: str, conv_cfg,  conv_cfg_radar, norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01,requires_grad=True),norm_cfg_radar=dict(type='BN2d', eps=1e-3, momentum=0.01,requires_grad=True), use_pos=True, init_cfg=None) ->None:
        super().__init__(init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.num_bev_queue = num_bev_queue
        self.embed_dims = embed_dims
        self.block_type = block_type
        # in_channels = num_bev_queue * embed_dims if num_bev_queue > 1 else (num_bev_queue+1) *embed_dims
        # in_channels_radar = num_bev_queue * 384 if num_bev_queue > 1 else (num_bev_queue+1)*384
        out_channels = embed_dims
        self.use_pos = use_pos
        if block_type == "x3d":
            temporal_block = [ResBlock(in_channels, in_channels, 3, 1,  trans_func=X3DTransform, dim_inner=in_channels//4) for _ in range(num_block-1)]
            temporal_block.append(ResBlock(in_channels, out_channels, 3, 1,  trans_func=X3DTransform, dim_inner=in_channels//4))
            self.temporal_block = nn.Sequential(*temporal_block)

        elif block_type == "c3d":
            in_channels = num_bev_queue * embed_dims if num_bev_queue > 1 else (num_bev_queue+1) *embed_dims
            in_channels_radar = num_bev_queue * 384 if num_bev_queue > 1 else (num_bev_queue+1)*384
            out_channels = embed_dims


            temporal_block = [Bottleneck(in_channels, in_channels//4, conv_cfg=conv_cfg, norm_cfg=norm_cfg) for _ in range(num_block-1)]
            temporal_block.append(Bottleneck(in_channels, out_channels, downsample=nn.Sequential(
                build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ), conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            self.temporal_block = nn.Sequential(*temporal_block)

            #-----------加入Radar BEV 时序编码-------------

            self.radar_temporal_block = nn.Sequential(
                Bottleneck(in_channels_radar, out_channels, downsample=nn.Sequential(
                    build_conv_layer(conv_cfg_radar, in_channels_radar, out_channels, kernel_size=1, stride=1, bias=False),
                    build_norm_layer(norm_cfg_radar, out_channels)[1]
                ), conv_cfg=conv_cfg_radar, norm_cfg=norm_cfg_radar)
            )


        elif block_type == "self_attention":
            if self.num_bev_queue != 1:
                in_channels = (num_bev_queue-1) * embed_dims if num_bev_queue > 1 else num_bev_queue *embed_dims
                in_channels_radar = (num_bev_queue-1) * 384 if num_bev_queue > 1 else num_bev_queue*384
                out_channels = embed_dims
                temporal_block = [Bottleneck(in_channels, in_channels//4, conv_cfg=conv_cfg, norm_cfg=norm_cfg) for _ in range(num_block-1)]
                temporal_block.append(Bottleneck(in_channels, out_channels, downsample=nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1]
                ), conv_cfg=conv_cfg, norm_cfg=norm_cfg))
                self.img_temporal_block = nn.Sequential(*temporal_block)

                #-----------加入Radar BEV 历史帧编码-------------

                self.radar_temporal_block = nn.Sequential(
                    Bottleneck(in_channels_radar, out_channels, downsample=nn.Sequential(
                        build_conv_layer(conv_cfg_radar, in_channels_radar, out_channels, kernel_size=1, stride=1, bias=False),
                        build_norm_layer(norm_cfg_radar, out_channels)[1]
                    ), conv_cfg=conv_cfg_radar, norm_cfg=norm_cfg_radar)
                )

            self.radar_current_reshape_channel = nn.Sequential(
                nn.Conv2d(384, 256,kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),)

            self.temporal_block_radar_attention = TemporalSelfAttention(embed_dims=embed_dims, num_levels=1, num_bev_queue=2)
            self.temporal_block_cam_attention = DeformSelfAttention3DCustom(embed_dims=embed_dims, num_levels=1, num_bev_queue=2)
            if self.use_pos:
                self.positional_encoding_3d = build_positional_encoding(dict(
                                                            type='Learned3DPositionalEncoding',
                                                            num_feats=128,
                                                            row_num_embed=bev_h,
                                                            col_num_embed=bev_w,
                                                            z_num_embed = bev_z,
                                                            ),)
                self.positional_encoding_2d = build_positional_encoding(dict(
                                                            type='LearnedPositionalEncoding',
                                                            num_feats=128,
                                                            row_num_embed=2*bev_h,
                                                            col_num_embed=2*bev_w,
                                                            ),)
        
        self.init_weights()
    def init_weights(self):
        if self.block_type == "self_attention":
            self.temporal_block_radar_attention.init_weights()
            self.temporal_block_cam_attention.init_weights()
            for module in self.modules():
                if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                    kaiming_init(module)
                elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
                    constant_init(module, 1)
        else:
            for module in self.modules():
                if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                    kaiming_init(module)
                elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
                    constant_init(module, 1)

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        if dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
        elif dim == '3d':
            ref_y, ref_x, ref_z = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device),
                torch.linspace(
                    0.5, Z - 0.5, Z, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_z = ref_z.reshape(-1)[None] / Z
            ref_2d = torch.stack((ref_x, ref_y, ref_z), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def forward(self, bev_feat, pts_feats, prev_bev, ref_2d: torch.Tensor = None, bev_pos: torch.Tensor = None) -> torch.Tensor:
        #---prev_bev={'prev_bev_img':torch.Size([1, 2, 256, 80, 120, 8]),
#                   'prev_bev_radar':torch.Size([1, 2, 384, 160, 240])}
        # Change bev_feat shape to [bs, embed_dims, H, W, Z]
        bev_feat = bev_feat.permute(0, 2, 1) #--torch.Size([1, 256, 76800])
        bev_feat = bev_feat.reshape(bev_feat.shape[0], -1, self.bev_h, self.bev_w, self.bev_z) #--torch.Size([1, 256, 80, 120, 8])
        assert bev_feat.shape[1] == self.embed_dims, "bev features dims do not match!"

        if len(prev_bev['prev_bev_img']) == 0:
            # first frame has no prev_bev
            #---这里判断一下如果queue设置为1，那么复制一下--------
            if self.num_bev_queue == 1:
                prev_bev_img = torch.cat(
                    [bev_feat for _ in range(1)], dim=1
                )  # [bs, (num_queue-1)*embed_dims, H, W]
                prev_bev_radar = torch.cat(
                    [pts_feats[0] for _ in range(1)], dim=1
                )
            else:
                prev_bev_img = torch.cat(
                    [bev_feat for _ in range(self.num_bev_queue - 1)], dim=1
                )  # [bs, (num_queue-1)*embed_dims, H, W]
                prev_bev_radar = torch.cat(
                    [pts_feats[0] for _ in range(self.num_bev_queue - 1)], dim=1
            )
        else:
            #---------prev_img_bev-------------
            padding_bev_img = [
                prev_bev['prev_bev_img'][:, 0:1] for _ in range(prev_bev['prev_bev_img'].shape[1], self.num_bev_queue - 1)
            ] 
            prev_bev_img = torch.cat([*padding_bev_img, prev_bev['prev_bev_img']], dim=1)  # [bs, num_queue-1, dim, H, W,Z]torch.Size([1, 3, 256, 80, 120, 8])
            prev_bev_img = prev_bev_img.reshape(
                prev_bev_img.shape[0], -1, self.bev_h, self.bev_w, self.bev_z
            )  # [bs, (num_queue-1)*embed_dims, H, W, Z]torch.Size([1, 512, 80, 120, 8])
            
            #------------------prev_radar_bev----------------
            #---------prev_img_bev-------------
            padding_bev_radar = [
                prev_bev['prev_bev_radar'][:, 0:1] for _ in range(prev_bev['prev_bev_radar'].shape[1], self.num_bev_queue - 1)
            ] 
            prev_bev_radar = torch.cat([*padding_bev_radar, prev_bev['prev_bev_radar']], dim=1)  # [bs, num_queue-1, dim, H, W]
            prev_bev_radar = prev_bev_radar.reshape(
                prev_bev_radar.shape[0], -1, prev_bev_radar.shape[-2],prev_bev_radar.shape[-1]
            )  # torch.Size([1, 384*2, 160, 240])



        if self.block_type == "self_attention" and self.num_bev_queue !=1:
            bs = bev_feat.shape[0]
            ref_3d = self.get_reference_points(self.bev_h, self.bev_w, self.bev_z, dim='3d', bs=bs, device=bev_feat.device, dtype=bev_feat.dtype)
            bs, len_bev, num_bev_level, _ = ref_3d.shape
            hybird_ref_3d = torch.stack([ref_3d, ref_3d], 1).reshape(
                bs*2, len_bev, num_bev_level, 3)
            #--------------cam voxel---------

            prev_bev_img = prev_bev_img.permute(0, 1, 4, 2, 3)# [bs, dim, Z, H, W]
            temporal_fused_prev_bev_img = self.img_temporal_block(prev_bev_img)  # (bs, embed_dims, Z, H, W)torch.Size([1, 256, 8, 80, 120])
            temporal_fused_prev_bev_img = temporal_fused_prev_bev_img.permute(0, 1, 3, 4, 2)  # (bs, embed_dims, H, W, Z)
            temporal_fused_prev_bev_img = temporal_fused_prev_bev_img.reshape(bs,self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W*Z, embed_dims) torch.Size([1, 76800, 
            bev_feat = bev_feat.reshape(bs,self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W*Z, embed_dims) torch.Size([1, 76800, 256])
        
            img_value = torch.stack([temporal_fused_prev_bev_img, bev_feat], dim=1).reshape(bs*2, -1, self.embed_dims)

            #-------------3d pos_encoding--------------
            if self.use_pos:
                pos_3d_mask = torch.zeros((bs, self.bev_h, self.bev_w, self.bev_z),device=bev_feat.device).to(bev_feat.dtype)
                pos_3d = self.positional_encoding_3d(pos_3d_mask).to(bev_feat.dtype)
                pos_3d = pos_3d.flatten(2).permute(0, 2, 1)
            else:
                pos_3d = None
            #---------------------------------------------

            temporal_fused_img_feat = self.temporal_block_cam_attention(query=bev_feat, value=img_value,
                                                          query_pos=pos_3d,
                                                          reference_points=hybird_ref_3d,
                                                          spatial_shapes=torch.tensor(
                                                              [[self.bev_h, self.bev_w, self.bev_z]], device=bev_feat.device),
                                                          level_start_index=torch.tensor([0], device=bev_feat.device),
                                                          )
            
            #---------------radar bev----------------
            bs,_,h,w = pts_feats[0].shape
            ref_2d = self.get_reference_points(h, w, dim='2d', bs=bs, device=pts_feats[0].device, dtype=pts_feats[0].dtype)
            bs, len_bev, num_bev_level, _ = ref_2d.shape
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
            
            temporal_fused_prev_radar = self.radar_temporal_block(prev_bev_radar) #--b,c,h,w torch.Size([1, 256, 160, 240])
            temporal_fused_prev_radar = temporal_fused_prev_radar.reshape(bs, self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W, embed_dims)
            current_radar_feat = self.radar_current_reshape_channel(pts_feats[0]).reshape(bs, self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W, embed_dims)
            radar_value = torch.stack([temporal_fused_prev_radar, current_radar_feat], dim=1).reshape(bs*2, -1, self.embed_dims)
            
             #-------------2d pos_encoding--------------
            if self.use_pos:
                pos_2d_mask = torch.zeros((bs, h,w ),device=current_radar_feat.device).to(current_radar_feat.dtype)
                pos_2d = self.positional_encoding_2d(pos_2d_mask).to(current_radar_feat.dtype)
                pos_2d = pos_2d.flatten(2).permute(0, 2, 1)
            else:
                pos_2d = None
            
            temporal_fused_radar_feat = self.temporal_block_radar_attention(query=current_radar_feat, value=radar_value,
                                                          query_pos=pos_2d,
                                                          reference_points=hybird_ref_2d,
                                                          spatial_shapes=torch.tensor(
                                                              [[h, w]], device=current_radar_feat.device),
                                                          level_start_index=torch.tensor([0], device=current_radar_feat.device),
                                                          )
            temporal_fused_radar_feat = temporal_fused_radar_feat.reshape(bs,h,w,self.embed_dims).permute(0,3,1,2)
            return {'temporal_img': temporal_fused_img_feat, 'temporal_radar': temporal_fused_radar_feat}
        
        elif self.block_type == "self_attention" and self.num_bev_queue ==1:
            bs = bev_feat.shape[0]
            ref_3d = self.get_reference_points(self.bev_h, self.bev_w, self.bev_z, dim='3d', bs=bs, device=bev_feat.device, dtype=bev_feat.dtype)
            bs, len_bev, num_bev_level, _ = ref_3d.shape
            hybird_ref_3d = torch.stack([ref_3d, ref_3d], 1).reshape(
                bs*2, len_bev, num_bev_level, 3)
            #--------------cam voxel，核对这俩一致---------
            
            temporal_fused_prev_bev_img = prev_bev_img.reshape(bs,self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W*Z, embed_dims) torch.Size([1, 76800, 
            bev_feat = bev_feat.reshape(bs,self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W*Z, embed_dims) torch.Size([1, 76800, 256])

            img_value = torch.stack([temporal_fused_prev_bev_img, bev_feat], dim=1).reshape(bs*2, -1, self.embed_dims)
                        #-------------3d pos_encoding--------------
            if self.use_pos:
                pos_3d_mask = torch.zeros((bs, self.bev_h, self.bev_w, self.bev_z),device=bev_feat.device).to(bev_feat.dtype)
                pos_3d = self.positional_encoding_3d(pos_3d_mask).to(bev_feat.dtype)
                pos_3d = pos_3d.flatten(2).permute(0,2,1)
            else:
                pos_3d = None
            #---------------------------------------------
            
            
            temporal_fused_img_feat = self.temporal_block_cam_attention(query=bev_feat, value=img_value,
                                                          query_pos=pos_3d,
                                                          reference_points=hybird_ref_3d,
                                                          spatial_shapes=torch.tensor(
                                                              [[self.bev_h, self.bev_w, self.bev_z]], device=bev_feat.device),
                                                          level_start_index=torch.tensor([0], device=bev_feat.device),
                                                          )
            
            #---------------radar bev----------------
            bs,_,h,w = pts_feats[0].shape
            ref_2d = self.get_reference_points(h, w, dim='2d', bs=bs, device=pts_feats[0].device, dtype=pts_feats[0].dtype)
            bs, len_bev, num_bev_level, _ = ref_2d.shape
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
            
            #-------------------核对一致------------------
            temporal_fused_prev_radar = self.radar_current_reshape_channel(prev_bev_radar).reshape(bs, self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W, embed_dims)
            current_radar_feat = self.radar_current_reshape_channel(pts_feats[0]).reshape(bs, self.embed_dims, -1).permute(0, 2, 1)  # (bs, H*W, embed_dims)
            radar_value = torch.stack([temporal_fused_prev_radar, current_radar_feat], dim=1).reshape(bs*2, -1, self.embed_dims)
            
             #-------------2d pos_encoding--------------
            if self.use_pos:
                pos_2d_mask = torch.zeros((bs, h,w ),device=current_radar_feat.device).to(current_radar_feat.dtype)
                pos_2d = self.positional_encoding_2d(pos_2d_mask).to(current_radar_feat.dtype)
                pos_2d = pos_2d.flatten(2).permute(0, 2, 1)
            else:
                pos_2d = None
            
            temporal_fused_radar_feat = self.temporal_block_radar_attention(query=current_radar_feat, value=radar_value,
                                                          query_pos=pos_2d,
                                                          reference_points=hybird_ref_2d,
                                                          spatial_shapes=torch.tensor(
                                                              [[h, w]], device=current_radar_feat.device),
                                                          level_start_index=torch.tensor([0], device=current_radar_feat.device),
                                                          )
            temporal_fused_radar_feat = temporal_fused_radar_feat.reshape(bs,h,w,self.embed_dims).permute(0,3,1,2)
            return {'temporal_img': temporal_fused_img_feat, 'temporal_radar': temporal_fused_radar_feat} 
        
        
        elif self.block_type == "c3d":
            # bev_queue with shape (bs, num_queue*embed_dims, H, W, Z)torch.Size([1, 1024, 80, 120, 8])
            #------------------img_queue--------------------torch.Size([1, 768, 80, 120, 8])
            bev_queue_img = torch.cat([prev_bev_img, bev_feat], dim=1)

            bev_queue_img = bev_queue_img.permute(0, 1, 4, 2, 3)  # [bs, dim, Z, H, W]
            temporal_fused_bev_feat_img = self.temporal_block(bev_queue_img)  # (bs, embed_dims, Z, H, W)torch.Size([1, 256, 8, 80, 120])
            temporal_fused_bev_feat_img = temporal_fused_bev_feat_img.permute(0, 1, 3, 4, 2)  # (bs, embed_dims, H, W, Z)
            temporal_fused_bev_feat_img = temporal_fused_bev_feat_img.reshape(
                temporal_fused_bev_feat_img.shape[0], self.embed_dims, -1)
            temporal_fused_bev_feat_img = temporal_fused_bev_feat_img.permute(0, 2, 1)  # (bs, H*W*Z, embed_dims) torch.Size([1, 76800, 256])

            #------------------radar_queue--------------------torch.Size([1, 1152, 160, 240])
            bev_queue_radar = torch.cat([prev_bev_radar, pts_feats[0]], dim=1)
            temporal_bev_radar = self.radar_temporal_block(bev_queue_radar) #--b,c,h,w torch.Size([1, 256, 160, 240])

        
        return {'temporal_img': temporal_fused_bev_feat_img, 'temporal_radar': temporal_bev_radar}

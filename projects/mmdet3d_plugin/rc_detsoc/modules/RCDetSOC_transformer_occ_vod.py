import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.utils import build_from_cfg
from typing import Optional

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3DV1
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.builder import FUSION_LAYERS

@TRANSFORMER.register_module()
class RCDetSOCTransformer_vod(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 cam_encoder=None,
                 temporal_encoder=None,
                 decoder=None,
                 voxel_decoder = None,
                 seg_decoder = None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 with_det = False,
                 with_occ = False,
                 occbevfusion2d = None,
                 occ_det_aux = True,
                 bev_seg_aux = True,
                 bev_seg_decoder = dict(type='FRPN',in_channels=256,loss_weight=1.0),
                 vis_tag = 'vod',
                 **kwargs):
        super(RCDetSOCTransformer_vod, self).__init__(**kwargs)
        #------------加入voxel_bev融合的分支，并且with_occ也加入--------------
        self.cam_encoder = build_transformer_layer_sequence(cam_encoder)
        self.temporal_encoder = build_from_cfg(temporal_encoder, ATTENTION)
        self.voxel_decoder = build_transformer_layer_sequence(voxel_decoder)
        self.with_det = with_det
        self.with_occ = with_occ
        #----occ解码器------------
        if self.with_occ:
            self.seg_decoder = build_transformer_layer_sequence(seg_decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        #--------加入辅助分支----------
        self.occ_det_aux = occ_det_aux
        self.bev_seg_aux = bev_seg_aux
        if self.bev_seg_aux:
            self.bev_seg_decoder = build_from_cfg(bev_seg_decoder, FUSION_LAYERS)

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        if self.with_det:

            #------加入bev融合分支------
            self.decoder = build_transformer_layer_sequence(decoder)
            self.occbevfusion2d = build_transformer_layer_sequence(occbevfusion2d)
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

        #-----   可视化------
        self.vis_tag = vis_tag     
        self.draw_interval = 2000
        self.vis_time_bev = -1
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # self.can_bus_mlp = nn.Sequential(
        #     nn.Linear(18, self.embed_dims // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(inplace=True),
        # )
        # if self.can_bus_norm:
        #     self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        if self.with_det:
            self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3DV1) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        # xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            bev_z,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) #--torch.Size([76800, 1, 256])

        # can_bus = bev_queries.new_tensor(
        #     [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        # can_bus = self.can_bus_mlp(can_bus)[None, :, :] #--1,1,256
        # bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = [] #--[(68, 120), (34, 60), (17, 30), (9, 15)]
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) #--torch.Size([6, 1, 10845, 256])
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) #--tensor([    0,  8160, 10200, 10710], device='cuda:0')

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)torch.Size([6, 10845, 1, 256])

        bev_embed = self.cam_encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_z=bev_z,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )

        return bev_embed
#----------prev_bev={'prev_bev_img':torch.Size([1, 2, 256, 80, 120, 8]),
#                   'prev_bev_radar':torch.Size([1, 2, 384, 160, 240])}
    def align_img_prev_bev(self, prev_bev, bev_h, bev_w, bev_z, **kwargs):
        

        pc_range = self.cam_encoder.pc_range
        ref_y, ref_x, ref_z = torch.meshgrid(
                torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=prev_bev.dtype, device=prev_bev.device),
                torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=prev_bev.dtype, device=prev_bev.device),
                torch.linspace(0.5, bev_z - 0.5, bev_z, dtype=prev_bev.dtype, device=prev_bev.device),
            )
        ref_y = ref_y / bev_h
        ref_x = ref_x / bev_w
        ref_z = ref_z / bev_z

        GROUND_HEIGHT = -2
        grid = torch.stack(
                (ref_x,
                ref_y,
                # ref_x.new_full(ref_x.shape, GROUND_HEIGHT),
                ref_z,
                ref_x.new_ones(ref_x.shape)), dim=-1) #--torch.Size([80, 120, 8, 4])

        min_x, min_y, min_z, max_x, max_y, max_z = pc_range
        grid[..., 0] = grid[..., 0] * (max_x - min_x) + min_x
        grid[..., 1] = grid[..., 1] * (max_y - min_y) + min_y
        grid[..., 2] = grid[..., 2] * (max_z - min_z) + min_z
        grid = grid.reshape(-1, 4)  #----torch.Size([76800, 4])和ref_3d好像是一样的

        bs = prev_bev.shape[0]
        len_queue = prev_bev.shape[1]  #--这里在测试的时候确保之前多少个bev就有多少个+1个ego2global_transform
        assert bs == 1
        for i in range(bs):
            lidar_to_ego = kwargs['img_metas'][i]['lidar2ego_transformation']
            curr_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][-1] #--当前帧的pose

            curr_grid_in_prev_frame_lst = []  #----ref3D
            
            for j in range(len_queue): #---下面的lidar都是ego
                prev_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][j] #--第一帧开始的pose
                prev_ego_to_curr_ego =  np.linalg.inv(curr_ego_to_global) @ prev_ego_to_global #--前一帧ego到当前ego
                curr_ego_to_prev_ego = np.linalg.inv(prev_ego_to_curr_ego) #--当前ego到前一帧ego
                curr_ego_to_prev_ego = grid.new_tensor(curr_ego_to_prev_ego)
                #------当前的参考点转到前一帧 torch.Size([80, 120, 8, 3])
                curr_grid_in_prev_frame = torch.matmul(curr_ego_to_prev_ego, grid.T).T.reshape(bev_h, bev_w, bev_z, -1)[..., :3]
                curr_grid_in_prev_frame[..., 0] = (curr_grid_in_prev_frame[..., 0] - min_x) / (max_x - min_x)
                curr_grid_in_prev_frame[..., 1] = (curr_grid_in_prev_frame[..., 1] - min_y) / (max_y - min_y)
                curr_grid_in_prev_frame[..., 2] = (curr_grid_in_prev_frame[..., 2] - min_z) / (max_z - min_z)
                curr_grid_in_prev_frame = curr_grid_in_prev_frame * 2.0 - 1.0 #---从[0,1]转到[-1,1]
                curr_grid_in_prev_frame_lst.append(curr_grid_in_prev_frame)

            curr_grid_in_prev_frame = torch.stack(curr_grid_in_prev_frame_lst, dim=0) #--torch.Size([2, 80, 120, 8, 3])
            
            #------本质是三元线性插值 torch.Size([2, 256, 8, 80, 120])--------------
            prev_img_bev_warp_to_curr_frame = torch.nn.functional.grid_sample(
                prev_bev[i].permute(0, 1, 4, 2, 3),  # torch.Size([2, 256, 8, 80, 120])
                curr_grid_in_prev_frame.permute(0, 3, 1, 2, 4),  # [bs, z, h, w, 3] #--torch.Size([2, 8, 80, 120, 3])
                align_corners=False)
            prev_bev_img = prev_img_bev_warp_to_curr_frame.permute(0, 1, 3, 4, 2).unsqueeze(0) # add bs dim, torch.Size([1, 2, 256, 80, 120, 8])

        return prev_bev_img
        
    def align_radar_prev_bev(self, prev_bev, bev_h, bev_w, **kwargs):
        #---torch.Size([1, 2, 384, 160, 240])
        pc_range = self.cam_encoder.pc_range
        ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=prev_bev.dtype, device=prev_bev.device),
                torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=prev_bev.dtype, device=prev_bev.device),
            )
        ref_y = ref_y / bev_h
        ref_x = ref_x / bev_w

        GROUND_HEIGHT = 0
        grid = torch.stack(
                (ref_x,
                ref_y,
                ref_x.new_full(ref_x.shape, GROUND_HEIGHT),
                ref_x.new_ones(ref_x.shape)), dim=-1) #--torch.Size([160, 240, 4])

        min_x, min_y, min_z, max_x, max_y, max_z = pc_range
        grid[..., 0] = grid[..., 0] * (max_x - min_x) + min_x
        grid[..., 1] = grid[..., 1] * (max_y - min_y) + min_y
        
        grid = grid.reshape(-1, 4)  #----torch.Size([38400, 4])和ref_3d好像是一样的

        bs = prev_bev.shape[0]
        len_queue = prev_bev.shape[1]  #--这里在测试的时候确保之前多少个bev就有多少个+1个ego2global_transform
        assert bs == 1
        for i in range(bs):
            lidar_to_ego = kwargs['img_metas'][i]['lidar2ego_transformation']
            curr_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][-1] #--当前帧的pose

            
            curr_2d_grid_in_prev_frame_lst = [] #-----ref2D
            for j in range(len_queue): #---下面的lidar都是ego
                prev_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][j] #--第一帧开始的pose
                prev_ego_to_curr_ego =  np.linalg.inv(curr_ego_to_global) @ prev_ego_to_global #--前一帧ego到当前ego
                curr_ego_to_prev_ego = np.linalg.inv(prev_ego_to_curr_ego) #--当前ego到前一帧ego
                curr_ego_to_prev_ego = grid.new_tensor(curr_ego_to_prev_ego)
                #------当前的参考点转到前一帧 torch.Size([160,240, 3])
                curr_grid_in_prev_frame = torch.matmul(curr_ego_to_prev_ego, grid.T).T.reshape(bev_h, bev_w, -1)[..., :3]
                curr_grid_in_prev_frame[..., 0] = (curr_grid_in_prev_frame[..., 0] - min_x) / (max_x - min_x)
                curr_grid_in_prev_frame[..., 1] = (curr_grid_in_prev_frame[..., 1] - min_y) / (max_y - min_y)
                
                curr_grid_in_prev_frame = curr_grid_in_prev_frame * 2.0 - 1.0 #---从[0,1]转到[-1,1]
                curr_2d_grid_in_prev_frame_lst.append(curr_grid_in_prev_frame[..., :2])


            curr_2d_grid_in_prev_frame = torch.stack(curr_2d_grid_in_prev_frame_lst, dim=0) #--torch.Size([2, 160, 240, 2])

            #------radar_bev二元线性插值---torch.Size([2, 384, 80, 120])这里变成了80,120了-----------
            pre_radar_bev_warp_to_curr_frame = torch.nn.functional.grid_sample(
                prev_bev[i],#---len_que,c,h,w
                curr_2d_grid_in_prev_frame,# [bs,  h, w, 2]
                align_corners=False)
            prev_bev_radar = pre_radar_bev_warp_to_curr_frame.unsqueeze(0) # add bs dim, torch.Size([1, 2, 384, 160, 240])
        
        return prev_bev_radar
        
    def bev_temporal_fuse(
        self,
        bev_img_feats,
        pts_feats,
        prev_bev,
        bev_h,
        bev_w,
        bev_z,
        
        **kwargs
    ) -> torch.Tensor:
        #---prev_bev={'prev_bev_img':torch.Size([1, 2, 256, 80, 120, 8]),
#                   'prev_bev_radar':torch.Size([1, 2, 384, 160, 240])}
        #--------------------------时序对齐------
        if len(prev_bev['prev_bev_img']) != 0:
            prev_bev_img = self.align_img_prev_bev(prev_bev['prev_bev_img'], bev_h, bev_w, bev_z, **kwargs)
            prev_bev_radar = self.align_radar_prev_bev(prev_bev['prev_bev_radar'], prev_bev['prev_bev_radar'].shape[-2], prev_bev['prev_bev_radar'].shape[-1],**kwargs)

            prev_bev = {'prev_bev_img': prev_bev_img, 'prev_bev_radar': prev_bev_radar}

        bev_embeds = self.temporal_encoder(bev_img_feats, pts_feats, prev_bev)

        return bev_embeds


    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                bev_z,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                pts_feats=None,
                bev_seg_gt=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        #-----------这里生成img_voxel特征---------
        bev_img_feat = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            bev_z,
            grid_length=grid_length,
            bev_pos=bev_pos,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims  torch.Size([1, 76800, 256])
        
        #--------这里加入当前帧的voxel和bev特征以及历史帧的voxel和bev特征--------------
        #--------得到融合时序的当前帧voxel和bev特征---------
        #--------{'temporal_img': temporal_fused_bev_feat_img, 'temporal_radar': temporal_bev_radar}---------
        img_radar_temporal_feat = self.bev_temporal_fuse(bev_img_feat, pts_feats,prev_bev, bev_h, bev_w, bev_z, **kwargs) #-torch.Size([1, 76800, 256])
        
        bev_img_embed = img_radar_temporal_feat['temporal_img'] #--torch.Size([1, 76800, 256])
        bev_radar_feat = img_radar_temporal_feat['temporal_radar'] #--torch.Size([1, 256, 160, 240])

        bev_embed_vox = bev_img_embed.view(1,bev_h*bev_w,bev_z,-1)  #--torch.Size([1, 9600, 8, 256])
        #--------下面是解码到原始voxel分辨率0---------------
        #---voxel_feat:torch.Size([1, 32, 16, 160, 240])voxel_det:torch.Size([1, 1, 16, 160, 240])
        #------10.28去掉了occmask-----------
        

        voxel_feat = self.voxel_decoder(bev_embed_vox)
        voxel_det = None
        #----------这里将voxel_feat和bev_radar_feat进去一起解码
        #----------#---torch.Size([1, 12, 16, 160, 240])

        if self.with_occ:
            
            occupancy,voxel_det = self.seg_decoder(voxel_feat,bev_radar_feat)
            # [bs, w, h, z, class_num] torch.Size([1, 240, 160, 16, 12])
            occupancy = occupancy.permute(0,4,3,2,1)
            if voxel_det is not None:
                voxel_det = voxel_det.permute(0,4,3,2,1) #--torch.Size([1, 240, 160, 16, 1])
        # Add Det Branch 下面应该和detr3d一致----
        #---------obj query初始化的方式，包括ref_points------------
        #TODO
        if self.with_det:
        #-----------输入voxel特征与voxel_det---------
        #---voxel_feat:torch.Size([1, 32, 16, 160, 240])voxel_det:torch.Size([1, 240, 160, 16, 1])
        #---bev_radar_feat:#--torch.Size([1, 256, 160, 240])
            bs = mlvl_feats[0].size(0)
        #-----输出#--torch.Size([1, 256, 160, 240])------
            bev_embed_det = self.occbevfusion2d(voxel_feat,bev_radar_feat)

            #------这里加入bev_mask，乘还是不乘------------------
            if self.bev_seg_aux:
                bev_seg_feat = self.bev_seg_decoder(bev_embed_det)
                # bev_embed_det = bev_embed_det * bev_seg_feat.sigmoid()
            else:
                bev_seg_feat = None


            fusebev_h, fusebev_w = bev_embed_det.shape[-2],bev_embed_det.shape[-1]
            #——------------可视化---------------
            if self.training:
                self.vis_time_bev += 1
                if self.vis_time_bev % self.draw_interval == 0:
                    if self.vis_tag =='vod':
                        from projects.mmdet3d_plugin.models.utils.visual import draw_vod
                        bev_feat = bev_embed_det.clone()
                        bev_seg_clone = bev_seg_feat.clone()
                        bev_seg_mask = bev_seg_clone.sigmoid() > self.bev_seg_decoder.mask_thre
                        draw_vod(bev_feat,bev_seg_mask,bev_seg_gt, kwargs['img_metas'].copy(),self.vis_time_bev)
                    #---------------------------------------
                    elif self.vis_tag == 'tj4d':
                        from projects.mmdet3d_plugin.models.utils.visual import draw_tj4d
                        bev_feat = bev_embed_det.clone()
                        bev_seg_clone = bev_seg_feat.clone()
                        bev_seg_mask = bev_seg_clone.sigmoid() > self.bev_seg_decoder.mask_thre
                        draw_tj4d(bev_feat,bev_seg_mask,bev_seg_gt, kwargs['img_metas'].copy(),self.vis_time_bev)
            
            bev_embed_det = bev_embed_det.view(bs,self.embed_dims,-1).permute(2,0,1) #--torch.Size([38400, 1,  256])
        #----------------------------------------------
            
            query_pos, query = torch.split(
                object_query_embed, self.embed_dims, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1) #---torch.Size([1, 900, 256])
            reference_points = self.reference_points(query_pos)
            reference_points = reference_points.sigmoid() #--torch.Size([1, 900, 3])
            init_reference_out = reference_points

            query = query.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2)
            #----------这里用原始尺寸的voxel取z平均得到bev torch.Size([9600, 1, 256])-------------
            # bev_embed_det = (bev_embed_vox).mean(2).permute(1, 0, 2)
            
            #--------返回中间特征和参考点，如果不使用with_box_refine，则初始参考点和中间的参考点是一样的
            inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed_det,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[fusebev_h, fusebev_w]], device=query.device), #---这里根据融合的bev
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

            inter_references_out = inter_references
            #-----这返回的就是当前帧提取的bev_feat，进入时序融合之前---
            #-----------根据tag返回所需特征----------
            if self.with_occ:
                return bev_img_feat, occupancy, voxel_det, inter_states, init_reference_out, inter_references_out,bev_seg_feat
            return bev_img_feat,voxel_det, inter_states, init_reference_out, inter_references_out,bev_seg_feat
        return bev_img_feat, occupancy,voxel_det

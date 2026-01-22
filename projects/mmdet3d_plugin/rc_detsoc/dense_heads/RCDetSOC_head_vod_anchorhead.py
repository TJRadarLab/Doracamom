import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.cnn import ConvModule
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet3d.core import bbox3d2result, LiDARInstance3DBoxes
from mmdet3d.models import builder
 # ---------------------------------------------
# Modified by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
@HEADS.register_module()
class RCDetSOCHead_vod_anchorhead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 query_init_style='radar+cam',
                 with_box_refine=False,
                 as_two_stage=False,
                 sync_cls_avg_factor=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 code_weights=None,
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 bev_h=30,
                 bev_w=30,
                 bev_z=5,
                 num_classes=19,
                 od_num_classes=9,
                 loss_occ=None,
                 use_mask=False,
                 with_det = False,
                 with_occ = False,
                 num_query = 900,
                 loss_cls = None,
                 loss_bbox = None,
                 loss_iou = None,
                 assigner = None,
                 loss_occupancy_aux = None,
                 loss_det_occ = None,
                 positional_encoding=None,
                 with_MTL_adaptive=True,
                 decoder_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01, requires_grad=True),


                 **kwargs):


        self.with_MTL_adaptive = with_MTL_adaptive
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False
        self.num_classes=num_classes #----OCC类别
        self.od_num_classes=od_num_classes#---------OD类别
        self.use_mask=use_mask
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_occupancy_aux = loss_occupancy_aux
        self.loss_det_occ = loss_det_occ
        
        self.with_det = with_det
        self.with_occ = with_occ
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage


        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(RCDetSOCHead_vod_anchorhead, self).__init__()

        self.query_init_style = query_init_style
        if self.query_init_style in ['radar+cam']:
            self.reduc_conv = ConvModule(
                384, #--384
                256, #--256   改成256
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                inplace=False)

        if self.with_occ:
            self.loss_occ = build_loss(loss_occ)
            if loss_occupancy_aux is not None:
                self.aux_loss = build_loss(loss_occupancy_aux)
    
        if loss_det_occ is not None: 
            self.det_occ_loss = build_loss(loss_det_occ)


    
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
                #--------------加入两类MTL自适应权重---------------
        if self.with_MTL_adaptive:
            if self.transformer.bev_seg_aux:
                self.task_weights = nn.Parameter(torch.ones(2,requires_grad=True))
            else:
                self.task_weights = nn.Parameter(torch.ones(2,requires_grad=True))
        #------------------------------------------


        if not self.as_two_stage:
            if self.query_init_style == 'learned':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w * self.bev_z, self.embed_dims)
        #------------------检测分支----------
        if self.with_det:
            self.decoder = builder.build_head(decoder_cfg)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        self.decoder.init_weights()

    #------------------加入radar_features-------------
    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, pts_feats=None, prev_bev=None, only_bev=False, test=False,bev_seg_gt=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape  #--[[torch.Size([1, 6, 256, 68, 120])]]
        dtype = mlvl_feats[0].dtype
        if self.with_det:
            object_query_embeds = None
        else:
            object_query_embeds = None

    
        if self.query_init_style == 'radar+cam':
            
            radar_bev_feat = F.interpolate(pts_feats[0], (self.bev_h, self.bev_w), mode='bilinear', align_corners=False)
            bev_queries_radar = self.reduc_conv(radar_bev_feat) #--torch.Size([1, 256, 80, 120])
            bev_queries_radar = bev_queries_radar.unsqueeze(-1).repeat(1,1,1,1,self.bev_z) #--torch.Size([1, 256, 80, 120, 8])
            bev_queries_radar = bev_queries_radar.permute(0,2,3,4,1).reshape(-1,self.embed_dims) #--torch.Size([76800, 256])
        
            ref_3d = self.transformer.cam_encoder.get_reference_points(self.bev_h,self.bev_w,self.bev_z, self.bev_z, dim='3d', bs=bs, device=mlvl_feats[0].device, dtype=dtype)
            #----#--（6,1,9600,8,2）(6,1,9600,8)
            ref_points_cam, valid_mask = self.transformer.cam_encoder.point_sampling(ref_3d, self.pc_range, img_metas)
            bev_queries_img = self.backproject_inplace(mlvl_feats[0].clone(),ref_points_cam,valid_mask)
            bev_queries = bev_queries_radar + bev_queries_img

        bev_pos = None
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=None,  
                cls_branches=None, #--这里是None
                img_metas=img_metas,
                prev_bev=prev_bev, #----包含两个模态历史特征
                pts_feats=pts_feats,   #-------加入点云特征------------
                bev_seg_gt=bev_seg_gt
            )
        if self.with_det:
            if self.with_occ:
                bev_embed, occ_outs, voxel_det, hs, init_reference, inter_references,bev_seg_feat = outputs
            else:
                bev_embed, voxel_det, bev_feature, bev_seg_feat = outputs
           
            outputs_od = self.decoder([bev_feature])
            
            if self.with_occ:

                outs = {
                    'bev_embed': bev_embed,
                    'all_cls_scores': outputs_od,
                    'all_bbox_preds': outputs_od,
                    'occ': occ_outs,
                    'det_occ':voxel_det,
                    'bev_seg_feat':bev_seg_feat,
                }
            else:
                outs = {
                    'bev_embed': bev_embed,
                    'all_cls_scores': outputs_od,
                    'all_bbox_preds': outputs_od,
                    'det_occ':voxel_det,
                    'bev_seg_feat':bev_seg_feat,
                }
        else:
            bev_embed, occ_outs,voxel_det = outputs

            outs = {
                'bev_embed': bev_embed,
                'occ':occ_outs,
                'det_occ':voxel_det,
            }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_occ,
             mask_camera,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None,
             bev_seg_gt=None):#----加入bevseg----

        loss_dict=dict()

        if self.with_det:

            outs = preds_dicts['all_cls_scores']
            od_loss_inputs = outs + (gt_bboxes_list, gt_labels_list, img_metas) 
            det_loss_dict = self.decoder.loss(
                *od_loss_inputs,
                gt_bboxes_ignore=None
            )
            loss_dict.update(det_loss_dict)

        #-------改一下------------
        if self.with_occ:
            assert gt_occ.min()>=0 and gt_occ.max()<=11
            occ=preds_dicts['occ']
            loss_ssc,loss_occ = self.loss_single(gt_occ,mask_camera,occ)
            loss_dict['loss_ssc'] = loss_ssc
            loss_dict['loss_occ'] = loss_occ

    # add det occ-----------------------
        if self.transformer.occ_det_aux:
            
            det_occ = preds_dicts['det_occ']
        
            
            voxel_det_gt = (gt_occ>0) #----这里我们0是empty
            voxel_det_gt = voxel_det_gt.long().unsqueeze(-1).to(det_occ.device)
            loss_occ_det = self.transformer.seg_decoder.occ_det_decoder.get_occ_mask_loss(voxel_det_gt, det_occ)
            
            loss_dict.update(loss_occ_det)
        #add bev_seg--------------------------------------
        if self.transformer.bev_seg_aux:
            bev_seg_feat = preds_dicts['bev_seg_feat']
            bev_seg_gt = bev_seg_gt.to(bev_seg_feat.device)
            loss_bevseg_dict = self.transformer.bev_seg_decoder.get_bev_mask_loss(bev_seg_gt, bev_seg_feat)
            loss_dict.update(loss_bevseg_dict)
        
        #-----------自适应权重----------------
        if self.with_MTL_adaptive:
            #-------------anchorhead--------------------------------
            loss_adaptive_dict = dict()
            loss_OD = loss_dict['loss_cls'][0] + loss_dict['loss_bbox'][0] + loss_dict['loss_dir'][0]
            loss_OD = 0.5/(self.task_weights[0] ** 2)*loss_OD +torch.log(1+self.task_weights[0] ** 2) 

            # loss_OCC = loss_dict['loss_occ']+loss_dict['loss_ssc']
            # loss_OCC = 0.5/(self.task_weights[1] ** 2)*loss_OCC +torch.log(1+self.task_weights[1] ** 2) 
            if self.transformer.occ_det_aux:
                loss_occ_mask = loss_dict['occ_mask_ce_loss']+loss_dict['occ_mask_dc_loss']
                loss_occ_mask = 0.5/(self.task_weights[2] ** 2)*loss_occ_mask +torch.log(1+self.task_weights[2] ** 2) 
                loss_adaptive_dict['loss_occ_mask'] = loss_occ_mask
            if self.transformer.bev_seg_aux:
                loss_bevseg = loss_dict['bev_mask_ce_loss']+loss_dict['bev_mask_dc_loss']
                loss_bevseg = 0.5/(self.task_weights[1] ** 2)*loss_bevseg +torch.log(1+self.task_weights[1] ** 2) 
                loss_adaptive_dict['loss_bevseg'] = loss_bevseg
            
            loss_adaptive_dict['loss_OD'] = loss_OD
            # loss_adaptive_dict['loss_OCC'] = loss_OCC
            return loss_adaptive_dict
        
        return loss_dict
    #-----OCC的loss------

    def loss_single(self, gt_occ, mask_camera, occ_pred):

        voxel_semantics = gt_occ.long()
        loss_ssc = self.sem_scal_loss(occ_pred, voxel_semantics.long()) \
                    + self.geo_scal_loss(occ_pred, voxel_semantics.long())
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = occ_pred.reshape(-1, self.num_classes)
        
        loss_occ = self.loss_occ(preds, voxel_semantics)




        return loss_ssc,loss_occ
    
    def geo_scal_loss(self, preds, ssc_target, semantic=True):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        if semantic:
            pred = F.softmax(pred, dim=1)

            # Compute empty and nonempty probabilities
            empty_probs = pred[:, 0, :, :, :]
        else:
            empty_probs = 1 - torch.sigmoid(pred)
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != 0  # 迁移过来occ3d中17代表空 原为0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )
    
    def sem_scal_loss(self, preds, ssc_target):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)   # torch.Size([1, 17, 25, 25, 2])
        loss = 0
        count = 0
        mask = ssc_target != 255    # 剔除255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] ## 原surroundocc适配格式
            # p = pred[:, :, :, :, i] ## 适配bevformer_occ格式

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)


        return occ_score
        


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        ret_list = self.decoder.get_bboxes(*preds_dicts['all_cls_scores'], img_metas, rescale=rescale)
        return ret_list

    def get_voxel_grid(self, bev_h=200,bev_w=200,bev_z=16):
        pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        ref_x, ref_y, ref_z = torch.meshgrid(
                        torch.linspace(0.5, bev_h - 0.5, bev_h),
                        torch.linspace(0.5, bev_w - 0.5, bev_w),
                        torch.linspace(0.5, bev_z - 0.5, bev_z),
                    )
        ref_y = ref_y / bev_h
        ref_x = ref_x / bev_w
        ref_z = ref_z / bev_z
        grid = torch.stack(
                        (ref_x,
                        ref_y,
                        ref_z,
                        ref_x.new_ones(ref_x.shape)), dim=-1)
        min_x, min_y, min_z, max_x, max_y, max_z = pc_range
        grid[..., 0] = grid[..., 0] * (max_x - min_x) + min_x
        grid[..., 1] = grid[..., 1] * (max_y - min_y) + min_y
        grid[..., 2] = grid[..., 2] * (max_z - min_z) + min_z
        return grid


#------------------加入3d-->2d采样生成course query-------------------

    def backproject_inplace(self, features, points, valid):
        '''
        function: 2d feature + predefined point cloud -> 3d volume
        input:
            features: [6, 64, 225, 400] 
            points: [3, 200, 200, 12]
            projection: [6, 3, 4]
        output:
            volume: [64, 200, 200, 12]
        '''
        #--（6,1,9600,8,2）(6,1,9600,8)
        bs, n_images, n_channels, height, width = features.shape #--torch.Size([1, 6, 256, 68, 120])
        ref_points = points.permute(1,0,2,3,4).reshape(bs,n_images,-1,2) #--1,6,76800,2
        valid = valid.permute(1,0,2,3).reshape(bs,n_images,-1).squeeze(0) #--6,76800
        assert bs==1
        x = ref_points[...,0].squeeze(0).squeeze(-1)*width
        x = x.round().long() #--[6,76800]
        y = ref_points[...,1].squeeze(0).squeeze(-1)*height
        y = y.round().long() #-#--[6,76800]


        volume = torch.zeros(
            (n_channels, valid.shape[-1]), device=features.device
        ).type_as(features)
        for i in range(n_images):
            volume[:, valid[i]] = features[0, i, :, y[i, valid[i]].clamp(0, height - 1), x[i, valid[i]].clamp(0, width - 1)]

        return volume.permute(1,0).contiguous()
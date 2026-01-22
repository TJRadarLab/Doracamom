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
 # ---------------------------------------------
# Modified by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
@HEADS.register_module()
class RCDetSOCHead(BaseModule):
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
                 code_weights=[1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
                 with_MTL_adaptive=False,
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
        super(RCDetSOCHead, self).__init__()


                #-----------query初始化方式-------------
        self.query_init_style = query_init_style
        if self.query_init_style in ['radar+cam']:
            self.reduc_conv = ConvModule(
                384, #--384
                256, #--256   改成256
                3,
                padding=1,
                conv_cfg=None,
                # norm_cfg=dict(type='SyncBN',requires_grad=True),#----改成SyncBN--------
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                inplace=False)
        #------------------------------------------这里加入occ损失----------
        if self.with_occ:
            self.loss_occ = build_loss(loss_occ)
            if loss_occupancy_aux is not None:
                self.aux_loss = build_loss(loss_occupancy_aux)
    
        if loss_det_occ is not None: #----这里不管那个任务暂时都学一下------
            self.det_occ_loss = build_loss(loss_det_occ)


        

        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
                #--------------加入两类MTL自适应权重---------------
        if self.with_MTL_adaptive:
            if self.transformer.bev_seg_aux:
                self.task_weights = nn.Parameter(torch.ones(4,requires_grad=True))
            else:
                self.task_weights = nn.Parameter(torch.ones(2,requires_grad=True))
        #------------------------------------------


        if not self.as_two_stage:
            if self.query_init_style == 'learned':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w * self.bev_z, self.embed_dims)
        #------------------检测分支----------
        if self.with_det:
            self.code_weights = code_weights
            self.code_weights = nn.Parameter(torch.tensor(
                    self.code_weights, requires_grad=False), requires_grad=False)
            self.bbox_coder = build_bbox_coder(bbox_coder)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.embed_dims, self.od_num_classes))   #---目标检测类别
            fc_cls = nn.Sequential(*cls_branch)

            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, 10)) #---属性
            reg_branch = nn.Sequential(*reg_branch)
            num_pred = (self.transformer.decoder.num_layers + 1) if \
                    self.as_two_stage else self.transformer.decoder.num_layers
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

            self.loss_cls = build_loss(loss_cls)
            self.loss_bbox = build_loss(loss_bbox)
            self.loss_iou = build_loss(loss_iou)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.with_det:
            if self.loss_cls.use_sigmoid:
                bias_init = bias_init_with_prob(0.01)
                for m in self.cls_branches:
                    nn.init.constant_(m[-1].bias, bias_init)
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
            object_query_embeds = self.query_embedding.weight.to(dtype)
        else:
            object_query_embeds = None

    
        if self.query_init_style == 'radar+cam':
            radar_bev_feat = F.interpolate(pts_feats[0], (self.bev_h, self.bev_w), mode='bilinear', align_corners=False)
            bev_queries_radar = self.reduc_conv(radar_bev_feat) #--torch.Size([1, 256, 80, 120])
            bev_queries_radar = bev_queries_radar.unsqueeze(-1).repeat(1,1,1,1,self.bev_z) #--torch.Size([1, 256, 80, 120, 8])
            bev_queries_radar = bev_queries_radar.permute(0,2,3,4,1).reshape(-1,self.embed_dims) #--torch.Size([76800, 256])
            ref_3d = self.transformer.cam_encoder.get_reference_points(self.bev_h,self.bev_w,self.bev_z, self.bev_z, dim='3d', bs=bs, device=mlvl_feats[0].device, dtype=dtype)
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
                reg_branches=self.reg_branches if self.with_box_refine else None,  
                cls_branches=self.cls_branches if self.as_two_stage else None, #--这里是None
                img_metas=img_metas,
                prev_bev=prev_bev, #----包含两个模态历史特征
                pts_feats=pts_feats,   #-------加入点云特征------------
                bev_seg_gt=bev_seg_gt
            )
        if self.with_det:
            if self.with_occ:
                bev_embed, occ_outs, voxel_det, hs, init_reference, inter_references,bev_seg_feat = outputs
            else:
                bev_embed, voxel_det, hs, init_reference, inter_references,bev_seg_feat = outputs
            hs = hs.permute(0, 2, 1, 3) #--torch.Size([3, 1, 900, 256])
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])
                assert reference.shape[-1] == 3  #---坐标是实际坐标
                #--------xyz网络预测的偏差，再加参考点为每个位置实际比例，再sigmoid比例按范围相乘---
            #----------(cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy)
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                self.pc_range[2]) + self.pc_range[2])
                
                # TODO: check if using sigmoid
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_classes = torch.stack(outputs_classes) #--torch.Size([3, 1, 900, 9])
            outputs_coords = torch.stack(outputs_coords) #--torch.Size([1, 900, 10])这里在解码的时候xyz已经解码
            if self.with_occ:

                outs = {
                    'bev_embed': bev_embed,
                    'all_cls_scores': outputs_classes,
                    'all_bbox_preds': outputs_coords,
                    'occ': occ_outs,
                    'det_occ':voxel_det,
                    'bev_seg_feat':bev_seg_feat,
                }
            else:
                outs = {
                    'bev_embed': bev_embed,
                    'all_cls_scores': outputs_classes,
                    'all_bbox_preds': outputs_coords,
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
            all_cls_scores = preds_dicts['all_cls_scores']
            all_bbox_preds = preds_dicts['all_bbox_preds']
            num_dec_layers = len(all_cls_scores) #---3
            device = gt_labels_list[0].device

            gt_bboxes_list = [torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                dim=1).to(device) for gt_bboxes in gt_bboxes_list]

            all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
            all_gt_bboxes_ignore_list = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]
            losses_cls, losses_bbox = multi_apply(
                self.loss_single_det, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)
            #----------这里只保留了最后一层的loss，原始的包括其他三层的损失-----
            loss_dict['loss_cls'] = losses_cls[-1]
            loss_dict['loss_bbox'] = losses_bbox[-1]
#---------------------------其他层的损失，bevformer加，panoocc未加----------------------------------
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                            losses_bbox[:-1]):
                loss_dict['loss_cls'] += loss_cls_i
                loss_dict['loss_bbox'] += loss_bbox_i
                num_dec_layer += 1

        #---------12类---
        assert gt_occ.min()>=0 and gt_occ.max()<=11

        #-------改一下------------
        if self.with_occ:
            occ=preds_dicts['occ']
            loss_ssc,loss_occ = self.loss_single(gt_occ,mask_camera,occ)
            loss_dict['loss_ssc'] = loss_ssc
            loss_dict['loss_occ'] = loss_occ

    # add det occ-----------------------
        if self.transformer.occ_det_aux:
            
            det_occ = preds_dicts['det_occ']
        
            
            voxel_det_gt = (gt_occ>0) 
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
            loss_adaptive_dict = dict()
            loss_OD = loss_dict['loss_cls'] + loss_dict['loss_bbox']
            loss_OD = 0.5/(self.task_weights[0] ** 2)*loss_OD +torch.log(1+self.task_weights[0] ** 2) 

            loss_OCC = loss_dict['loss_occ']+loss_dict['loss_ssc']
            loss_OCC = 0.5/(self.task_weights[1] ** 2)*loss_OCC +torch.log(1+self.task_weights[1] ** 2) 
            if self.transformer.occ_det_aux:
                loss_occ_mask = loss_dict['occ_mask_ce_loss']+loss_dict['occ_mask_dc_loss']
                loss_occ_mask = 0.5/(self.task_weights[2] ** 2)*loss_occ_mask +torch.log(1+self.task_weights[2] ** 2) 
                loss_adaptive_dict['loss_occ_mask'] = loss_occ_mask
            if self.transformer.bev_seg_aux:
                loss_bevseg = loss_dict['bev_mask_ce_loss']+loss_dict['bev_mask_dc_loss']
                loss_bevseg = 0.5/(self.task_weights[3] ** 2)*loss_bevseg +torch.log(1+self.task_weights[3] ** 2) 
                loss_adaptive_dict['loss_bevseg'] = loss_bevseg
            
            loss_adaptive_dict['loss_OD'] = loss_OD
            loss_adaptive_dict['loss_OCC'] = loss_OCC
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
    
    
#----------------------更换loss------------------
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
        

    def loss_single_det(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss  #---这里又写死了----
        cls_scores = cls_scores.reshape(-1, self.od_num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * 0.0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets这里全分成了类别+1
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.od_num_classes,   #--------这里是类别，给写死了草-----
                                    dtype=torch.long)
        
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

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

        # method2：特征填充，只填充有效特征，重复特征直接覆盖
        volume = torch.zeros(
            (n_channels, valid.shape[-1]), device=features.device
        ).type_as(features)
        for i in range(n_images):
            volume[:, valid[i]] = features[0, i, :, y[i, valid[i]].clamp(0, height - 1), x[i, valid[i]].clamp(0, width - 1)]

        return volume.permute(1,0).contiguous()
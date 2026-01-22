import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.datasets.evaluation_metrics import evaluation_reconstruction, evaluation_semantic, new_evaluation_semantic, \
aug_evaluation_semantic
 # ---------------------------------------------
# Modified by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
@DETECTORS.register_module()
class RCDetSOC_vod(MVXTwoStageDetector):
    """4DRC-DetSOC.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 time_interval=1,
                 bev_h=160,
                 bev_w=160,
                 freeze_img=False,
                 ):

        super(RCDetSOC_vod,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.time_interval = time_interval
        self.bev_h = bev_h
        self.bev_w = bev_w
        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': {'prev_bev_img':[],'prev_bev_radar':[]},
            "ego2global_transform_lst": [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.freeze_img = freeze_img
        self.freeze()

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
    #--------提取点云特征-----------
    def extract_pts_feat(self, pts,len_queue=None):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        if len_queue is not None:
            assert len(pts) == len_queue
            pts_feats = []
            for i in range(len_queue):
                voxels, num_points, coors = self.voxelize(pts[i])
                voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                        )
                batch_size = coors[-1, 0] + 1
                x = self.pts_middle_encoder(voxel_features, coors, batch_size)
                x = self.pts_backbone(x)
                if self.with_pts_neck:
                    x = self.pts_neck(x)
                pts_feats.append(x)
            return pts_feats
        else:
            voxels, num_points, coors = self.voxelize(pts)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                    )
            batch_size = coors[-1, 0] + 1
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if self.with_pts_neck:
                x = self.pts_neck(x)
            return x
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    #----------提取图像2D特征和点云bev特征-------
    @auto_fp16(apply_to=('img','points'),out_fp32=True)
    def extract_feat(self, img, points, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        #--torch.Size([1, 2, 6, 256, 68, 120])  torch.Size([1, 2, 6, 256, 34, 60])torch.Size([1, 2, 6, 256, 17, 30])torch.Size([1, 2, 6, 256, 9, 15])
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue) #--[四个tensor特征图]
        pts_feats = self.extract_pts_feat(points,len_queue=len_queue) #---[[torch.Size([1, 384, 160, 240])],[torch.Size([1, 384, 160, 240])]]
        return dict(
            img_feats=img_feats,
            pts_feats=pts_feats
        )

    def forward_pts_train(self,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_occ,
                          mask_camera,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          pts_feats=None,
                          bev_seg_gt=None):#-----pts_feats,bev_seg
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            img_feats, img_metas, pts_feats=pts_feats,prev_bev=prev_bev,bev_seg_gt=bev_seg_gt)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_occ, mask_camera,outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas,bev_seg_gt=bev_seg_gt)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    #----------加入点云特征------
    def obtain_history_bev(self, imgs_queue, points_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        is_training = self.training
        self.eval()
        #--------------目前只支持batch=1----------------
        prev_bev_lst_cam = []
        prev_bev_lst_radar = []
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W) #--torch.Size([2, 6, 3, 544, 960])
            #---------图像和点云queue_feature-------------
            img_pts_feat_dict = self.extract_feat(img=imgs_queue, points=points_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_pts_feat_dict['img_feats']] #--4个featmap
                pts_feats = img_pts_feat_dict['pts_feats'][i] 
                prev_bev_lst_radar.append(pts_feats[0])
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, pts_feats=pts_feats,only_bev=True) #--torch.Size([1, 76800, 256])
                prev_bev = prev_bev.permute(0, 2, 1)
                prev_bev = prev_bev.reshape(prev_bev.shape[0], -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z) #--torch.Size([1, 256, 80, 120, 8])
                prev_bev_lst_cam.append(prev_bev)
        if is_training:
            self.train()
        # (bs, num_queue, embed_dims, H, W,Z) torch.Size([1, 3, 256, 80, 120, 8])
        return dict(prev_bev_img=torch.stack(prev_bev_lst_cam, dim=1),
                prev_bev_radar=torch.stack(prev_bev_lst_radar, dim=1))
    
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_occ=None,
                      mask_lidar=None,
                      mask_camera=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img = img.unsqueeze(1)
        len_queue = 1

        if True:
            prev_bev = {'prev_bev_img':[],'prev_bev_radar':[]}
        else:
#----------prev_bev={'pre_bev_img':torch.Size([1, 2, 256, 80, 120, 8]),
#                   'prev_bev_radar':torch.Size([1, 2, 384, 160, 240])}
                                
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_points, prev_img_metas)  #---包含prev_voxel和prev_radar_bev
            
        #----------这里改成如果除了第一帧外其他都有前bev就可以提时序特征--------------
        if not all([img_metas[0][i]['prev_bev_exists'] for i in range(1, len_queue)]):
            prev_bev = {'prev_bev_img':[],'prev_bev_radar':[]}
        #-----------如果不用时序训练也让prev_bev=None----------------
        if not self.video_test_mode:
            prev_bev = {'prev_bev_img':[],'prev_bev_radar':[]}
        img_metas = img_metas
        # if not img_metas[0]['prev_bev_exists']: #---这里感觉有点问题，之前有好几个前bevlist，不能保证每个都在一个场景？
        #     prev_bev = None
        #-----------加入img,radar当前帧特征-----------------
        img_pts_feats = self.extract_feat(img=img, points=points, img_metas=img_metas)
        #------------加入bev_seg真值----------
        device = img_pts_feats['img_feats'][0].device
        batch_size = img_pts_feats['img_feats'][0].size(0)
        gt_bboxes_3d_filtered = [gt_bboxes_3d[i][gt_labels_3d[i] != -1] for i in range(batch_size)] # filter out the ignored labels
        gt_bev_mask = self.generate_bev_mask(gt_bboxes_3d_filtered, batch_size, self.bev_h,self.bev_w) # B H W
        gt_bev_mask = gt_bev_mask.to(device)



        #-----------------------------------
        losses = dict()
        losses_pts = self.forward_pts_train(img_pts_feats['img_feats'], gt_bboxes_3d,
                                            gt_labels_3d, gt_occ, mask_camera, img_metas,
                                            gt_bboxes_ignore, prev_bev,img_pts_feats['pts_feats'],gt_bev_mask)
        #----------------------
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas,
                     img=None,
                     points=None,
                     gt_occ=None,  #--这里进来带括号[],TTApipeline的原因
                     mask_lidar=None,
                     mask_camera=None,
                     **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else [img[0].unsqueeze(1)] #--torch.Size([1, 6, 3, 544, 960])

        # if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
        #     # the first sample of each scene is truncated
        #     self.prev_frame_info['prev_bev'] = {'prev_bev_img':[],'prev_bev_radar':[]}
        #     self.prev_frame_info["ego2global_transformation_lst"] = []
        # # update idx
        # self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = {'prev_bev_img':[],'prev_bev_radar':[]}
            self.prev_frame_info["ego2global_transformation_lst"] = []

        # # Get the delta of ego position and angle between two timestamps.
        # tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        # tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # # if self.prev_frame_info['prev_bev'] is not None: #--[]也不是None
        # if len(self.prev_frame_info['prev_bev']['prev_bev_img']) != 0: #--[]也不是None
        #     img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
        #     img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        # else:
        #     img_metas[0][0]['can_bus'][-1] = 0
        #     img_metas[0][0]['can_bus'][:3] = 0

        # #-------这里要改一下，默认间隔是1也可----------
        # #-------每次输入之后，第一帧是当前帧-------
        # #----这里先加入当前帧的ego2global变换矩阵，保证当前帧肯定有ego pose，前面的egopose根据pre_bev数量在temporal_encoder里面循环
        # self.prev_frame_info["ego2global_transformation_lst"].append(img_metas[0][0]["ego2global_transformation"])

        # # img_metas[0][0]["ego2global_transform_lst"] = self.prev_frame_info["ego2global_transformation_lst"][-1::-1][::-1]
        # # prev_bev = self.prev_frame_info['prev_bev'][-1:: -1][:: -1]
        # img_metas[0][0]["ego2global_transform_lst"] = self.prev_frame_info["ego2global_transformation_lst"]
        
        prev_bev = copy.deepcopy(self.prev_frame_info['prev_bev'])

        # prev_bev['prev_bev_img'] = torch.stack(prev_bev['prev_bev_img'], dim=1) if len(prev_bev['prev_bev_img']) > 0 else []
        # prev_bev['prev_bev_radar'] = torch.stack(prev_bev['prev_bev_radar'], dim=1) if len(prev_bev['prev_bev_radar']) > 0 else []
        new_prev_img_bev, new_prev_radar_bev, prediction_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=prev_bev, gt_occ=gt_occ, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.

        # self.prev_frame_info['prev_pos'] = tmp_pos
        # self.prev_frame_info['prev_angle'] = tmp_angle
        # new_prev_img_bev = new_prev_img_bev.permute(0, 2, 1).reshape(1, -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z)
        
        # self.prev_frame_info['prev_bev']['prev_bev_img'].append(new_prev_img_bev)
        # self.prev_frame_info['prev_bev']['prev_bev_radar'].append(new_prev_radar_bev)

        # # while len(self.prev_frame_info["prev_bev"]) >= self.pts_bbox_head.transformer.temporal_encoder.num_bev_queue * 1:
        # while len(self.prev_frame_info["prev_bev"]['prev_bev_img']) >= self.pts_bbox_head.transformer.temporal_encoder.num_bev_queue:
            
        #     self.prev_frame_info["prev_bev"]['prev_bev_img'].pop(0)
        #     self.prev_frame_info["prev_bev"]['prev_bev_radar'].pop(0)

        #     self.prev_frame_info["ego2global_transformation_lst"].pop(0)

        return prediction_results

    def simple_test_pts(self, img_feats, pts_feats,img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(img_feats, img_metas, pts_feats=pts_feats,prev_bev=prev_bev, test=True)
        if 'occ' in outs.keys():
            occ = self.pts_bbox_head.get_occ(
                outs, img_metas, rescale=rescale) #--torch.Size([1, 240, 160, 16])
            
        else:
            occ = None
        if 'all_cls_scores' in outs.keys():
            bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale) #--rescale没有用到
            bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
                        ] #---------tensor转到cpu--------
        else:
            bbox_results = None
        return outs['bev_embed'], occ, bbox_results

    def simple_test(self, img_metas, img=None, points=None,prev_bev=None,gt_occ=None, rescale=False):
        """Test function without augmentaiton."""
        img_pts_feat_dict = self.extract_feat(img=img, points=points, img_metas=img_metas)
        prediction_results = {}
        new_prev_radar_bev = img_pts_feat_dict['pts_feats'][0].clone() #--torch.Size([1, 384, 160, 240])
        new_prev_img_bev, occ, bbox_results = self.simple_test_pts(
            img_pts_feat_dict['img_feats'], img_pts_feat_dict['pts_feats'],img_metas, prev_bev, rescale=rescale)
        #-----如果有OD的结果---------
        if bbox_results is not None:
            bbox_list = [dict() for i in range(len(img_metas))]
            for result_dict, pts_bbox in zip(bbox_list, bbox_results):
                result_dict['pts_bbox'] = pts_bbox
            prediction_results['bbox_results'] = bbox_list
        #----处理OCC---------------
        if occ is not None:
            occ_eval_results = aug_evaluation_semantic(occ, gt_occ[0], img_metas[0], self.pts_bbox_head.num_classes)
            prediction_results['occ_results'] = occ_eval_results
        return  new_prev_img_bev, new_prev_radar_bev, prediction_results
    

#----------------生成BEV_mask真值---------------------------
    def generate_bev_mask(self, gt_bboxes_3d, batch_size,bev_h,bev_w):
        # As long as it is occupied, it is 1
        gt_bev_mask = []
        if len(gt_bboxes_3d) != 0:
            pc_range = torch.tensor(self.pts_bbox_head.pc_range)

            bev_grid_shape = (bev_h,bev_w)
            bev_cell_size = torch.tensor([(pc_range[4]-pc_range[1])/bev_h, (pc_range[3]-pc_range[0])/bev_w])
            for bsid in range(len(gt_bboxes_3d)):
                bev_mask = torch.zeros(bev_grid_shape)
                bbox_corners = gt_bboxes_3d[bsid].corners[:, [0,2,4,6],:2] # bev corners
                num_rectangles = bbox_corners.shape[0]
                bbox_corners[:,:,0] = (bbox_corners[:,:,0] - pc_range[0])/bev_cell_size[1] # id_num, 4, 2
                bbox_corners[:,:,1] = (bbox_corners[:,:,1] - pc_range[1])/bev_cell_size[0] # id_num, 4, 2
                
                # precise bur slow method
                # grid_min = torch.clip(torch.floor(torch.min(bbox_corners, axis=1).values).to(torch.int64), 0, bev_grid_shape[0] - 1)
                # grid_max = torch.clip(torch.ceil (torch.max(bbox_corners, axis=1).values).to(torch.int64), 0, bev_grid_shape[1] - 1)
                grid_min = torch.floor(torch.min(bbox_corners, axis=1).values).to(torch.int64)
                grid_min[:,0] = torch.clamp(grid_min[:,0],min=0,max=bev_grid_shape[1]-1)
                grid_min[:,1] = torch.clamp(grid_min[:,1],min=0,max=bev_grid_shape[0]-1)
                grid_max = torch.ceil(torch.max(bbox_corners, axis=1).values).to(torch.int64)
                grid_max[:,0] = torch.clamp(grid_max[:,0],min=0,max=bev_grid_shape[1]-1)
                grid_max[:,1] = torch.clamp(grid_max[:,1],min=0,max=bev_grid_shape[0]-1)

                
                for i in range(num_rectangles):
                    bev_mask[grid_min[i, 1]:grid_max[i, 1], grid_min[i, 0]:grid_max[i, 0]] = True
                
                gt_bev_mask.append(bev_mask)
            gt_bev_mask = torch.stack(gt_bev_mask, dim=0).unsqueeze(1) # B 1 H W
        else:
            gt_bev_mask = torch.zeros((batch_size, 1, self.bev_grid_shape[0], self.bev_grid_shape[1]))
        gt_bev_mask = gt_bev_mask.to(torch.bool)
        return gt_bev_mask
    

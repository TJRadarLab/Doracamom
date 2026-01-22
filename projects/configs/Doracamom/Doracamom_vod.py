
 # ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
use_pos = True
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2.76] #---自车/lidar坐标系后面改成60,40
voxel_size = [0.16, 0.16, 5.76]
occ_size = [160, 160, 18]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #---这里fcos3d的pretrain是torchvision的R50
#------- 标签映射-----
class_names = ['Pedestrian', 'Cyclist', 'Car']
#--------VOD没用到-------------
use_semantic = True
occ_class_names = [
     'car', 'pedestrian', 'rider', 'large_vehicle',
     'cycle',
    'road_obstacle',
    'traffic_fence',
    'driveable_surface',
    'sidewalk',
    'vegetation',
    'manmade'
]
occ_num_class = 12
#-------------------------
od_num_classes=3
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True) #---这里不一定使用，model读取之后判断是否有ptsbackbone

#---encode/decoder layer---
_encoder_num_layer = 4 
_decoder_num_layer = 6
#--------------------
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
# Voxel Queries Size
bev_h_ = 80
bev_w_ = 80
bev_z_ = 9   
# Temporal 
queue_length = 1 # 

up_rate = [1,1,1]

#-----OD与OCC控制--------
with_det = True # whether use detection branch
with_occ = False # whether use occupancy branch
num_query = 300
bev_seg_aux = True #----------辅助任务
occ_det_aux = False
with_MTL_adaptive = False
model = dict(
    type='RCDetSOC_vod',
    use_grid_mask=False,
    video_test_mode=False,
    # freeze_img=True,#---------
    # time_interval = time_interval,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),

    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_voxel_layer=dict(
        max_num_points=10, #----这里设置成10/16
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),

    pts_voxel_encoder=dict( # 这里改了
        type='RadarPillarFeatureNet_vod',
        in_channels=5,  ## 这里输入由4变5
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range= point_cloud_range,
        with_velocity_snr_center=True,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        ),

    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 320]), #---这里修改成440,560。y,x/voxelsize
    
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),

    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),


    pts_bbox_head=dict(
        type='RCDetSOCHead_vod_anchorhead',
        with_MTL_adaptive=with_MTL_adaptive,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        num_classes=occ_num_class,
        od_num_classes=od_num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=False, #---这里原始是True,使用每次迭代的中间参考点
        as_two_stage=False,
        use_mask=False,#---无camera_mask
        with_det = with_det,
        with_occ = with_occ,
        num_query = num_query,
        query_init_style = 'radar+cam',
        
        transformer=dict(
            type='RCDetSOCTransformer_vod_anchorhead',
            num_cams=1,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=False, #----默认False
            embed_dims=_dim_,
            with_det = with_det,
            with_occ = with_occ,
            bev_seg_aux = bev_seg_aux,#----------辅助任务
            occ_det_aux = occ_det_aux,#----------辅助任务
            num_feature_levels = _num_levels_,
            cam_encoder=dict(
                type='OccupancyEncoderV1',
                num_layers=_encoder_num_layer, #---
                pc_range=point_cloud_range,
                num_points_in_pillar=bev_z_,
                return_intermediate=False,
                ego = False, #---ref参考点，这里也没什么变化
                transformerlayers=dict(
                    type='OccupancyLayerV1',
                    attn_cfgs=[
                        dict(
                            type='OccSpatialAttentionV1',
                            pc_range=point_cloud_range,
                            num_cams=1,
                            deformable_attention=dict(
                                type='MSDeformableAttention3DV1',
                                embed_dims=_dim_,
                                num_points= bev_z_,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm','conv'))),#----修改成conv3d
            temporal_encoder=dict(
                    type="OccTemporalEncoderV1",
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    bev_z=bev_z_,
                    num_bev_queue=queue_length, 
                    embed_dims=_dim_,
                    num_block=1,
                    block_type="self_attention",
                    conv_cfg=dict(type='Conv3d'),
                    
                    conv_cfg_radar=dict(type='Conv2d'),
                    use_pos = use_pos,
                    
                    ),
            voxel_decoder = dict(
                type='VoxelDecoderV1',
                occ_size=occ_size,
                bev_h=bev_h_,
                bev_w=bev_w_,
                bev_z=bev_z_,
                embed_dim = _dim_,
                out_dim = _dim_//8,
                ), #--这里传入occ_size判断上采样的比例,用来得到OCC分辨率
            seg_decoder = dict(
                type='MLP_DecoderV1',
                num_classes = occ_num_class, #----occ_num_class
                out_dim = _dim_//8,
                inter_up_rate = up_rate,
                occ_det_aux = occ_det_aux
            ),
            occbevfusion2d = dict(
                type='OccbevFusion2D',
                img_channels=256, 
                radar_channels=256,
                img_bev_conv_channel=576,
            )),
        decoder_cfg=dict(
                type='Anchor3DHead',
                num_classes=3,
                in_channels=256,  ###gai
                feat_channels=256, ##gai
                use_direction_classifier=True,
                anchor_generator=dict(
                    type='Anchor3DRangeGenerator',
                    ranges=[
                        [0, -25.6, -0.6, 51.2, 25.6, -0.6],
                        [0, -25.6, -0.6, 51.2, 25.6, -0.6],
                        [0, -25.6, -1.78, 51.2, 25.6, -1.78],
                    ],
                    sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
                    rotations=[0, 1.57],
                    reshape_out=False),
                assigner_per_size=False, ###
                diff_rad_by_sin=True,
                assign_per_class=False,
                bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                loss_dir=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
                train_cfg=dict(
                        assigner=[
                            dict(  # for Pedestrian
                                type='MaxIoUAssigner',
                                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                                pos_iou_thr=0.5,
                                neg_iou_thr=0.35,
                                min_pos_iou=0.35,
                                ignore_iof_thr=-1),
                            dict(  # for Cyclist
                                type='MaxIoUAssigner',
                                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                                pos_iou_thr=0.5,
                                neg_iou_thr=0.35,
                                min_pos_iou=0.35,
                                ignore_iof_thr=-1),
                            dict(  # for Car
                                type='MaxIoUAssigner',
                                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                                pos_iou_thr=0.6,
                                neg_iou_thr=0.45,
                                min_pos_iou=0.45,
                                ignore_iof_thr=-1),
                        ],
                        allowed_border=0,
                        pos_weight=-1,
                        debug=False),
                test_cfg=dict(
                        use_rotate_nms=True,
                        nms_across_levels=False,
                        nms_thr=0.01,
                        score_thr=0.1,
                        min_bbox_size=0,
                        nms_pre=100,
                        max_num=50)
                                ),    
                    
        positional_encoding=dict(
            type='Learned3DPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            z_num_embed = bev_z_,
            ),

    # model training and testing settings
    train_cfg=None,
    test_cfg=None,))

dataset_type = 'KittiDataset'
data_root = 'data/vod/'
file_client_args = dict(backend='disk')
img_scale = (1936, 1216)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5]),
    
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadImageFromFile'),
    dict(
    type='Resize',
    img_scale=img_scale,  #  按范围resize
    multiscale_mode='value',  ## 距离内随机采样
    keep_ratio=True),
    
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),

    dict(type='Normalize', **img_norm_cfg), #----normalization-------
    # dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),#----尺度缩放到0.5
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d','img','points']) #---加入radar points---
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5]),
    dict(type='LoadImageFromFile'),
    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale, #----分辨率改-
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=img_scale, multiscale_mode='value', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
             dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img','points']) #---这里还是读入occ_gt评估，而不是直接生成结果文件评估
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',        
        ),
    val=dict(type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

evaluation = dict(interval=1, pipeline=test_pipeline) #----eval评估间隔，改24
# learning policy


# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     min_lr_ratio=1e-3)
# learning policy
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=1.0 / 30,
    step=[12,14]
    )
optimizer = dict(
    type='AdamW',
    lr=8e-5, #--2e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            # 'pts_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) #_--35
momentum_config = None
log_config = dict(
    interval=50, #---50
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

total_epochs = 16
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/Doracamom_vod/doracamom_vod_1222'
load_img_from_and_not_change_state_dict = 'ckpts/r50_fcos3d_pretrain.pth'
load_pts_from = '/mnt/zhenglianqing/vod-mmdet3d/work_dirs/vod-Radarpillarnet/epoch_79.pth'
resume_from = None
load_from=None
workflow = [('train', 1)]




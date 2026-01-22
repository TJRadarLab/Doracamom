
 # ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-60.0, -40.0, -3.0, 60.0, 40.0, 5.0] #---自车/lidar坐标系后面改成60,40
voxel_size = [0.25, 0.25, 8]
occ_size = [240, 160, 16]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #---这里fcos3d的pretrain是torchvision的R50
#------- 标签映射-----
class_names = [
    'car', 'pedestrian', 'rider', 'large_vehicle']
#-----OCC标签类别,这里真值生成还没映射-------------
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
od_num_classes=4
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True) #---这里不一定使用，model读取之后判断是否有ptsbackbone

#---encode/decoder layer---
_encoder_num_layer = 3
_decoder_num_layer = 3
#--------------------
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
# Voxel Queries Size
bev_h_ = 80
bev_w_ = 120
bev_z_ = 8   #---先1米，1米，0.5米显存不够，只能都是1米
# Temporal 
queue_length = 4 # 

up_rate = [1,1,1]
with_det = True # whether use detection branch
with_occ = True # whether use occupancy branch
num_query = 900
bev_seg_aux = True #----------辅助任务
occ_det_aux = True
with_MTL_adaptive=False
model = dict(
    type='RCDetSOC',
    use_grid_mask=False,
    video_test_mode=True,
    queue_length=queue_length,
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
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNet',
        in_channels=7, #---输入点云七维信息xyz,vx,vy,power,snr
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        with_velocity_snr_center=True,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),

    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 480]), #---这里修改成440,560。y,x/voxelsize
    
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
        type='RCDetSOCHead',
        with_MTL_adaptive=with_MTL_adaptive,#-----------MTL权重---
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

        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],#------权重
        num_query = num_query,
        query_init_style = 'radar+cam',
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_occ= dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0), #---权重默认是10
        loss_det_occ= dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0), #--前景
        # loss_occupancy_aux = dict(
        #     type = 'Lovasz3DLoss',
        #     ignore = 0,#忽略的标签这里能否忽略？？
        #     loss_weight=10.0), 
        transformer=dict(
            type='RCDetSOCTransformer',
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
                ego = True, #---ref参考点，这里也没什么变化
                transformerlayers=dict(
                    type='OccupancyLayerV1',
                    attn_cfgs=[
                        dict(
                            type='OccSpatialAttentionV1',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3DV1',
                                embed_dims=_dim_,
                                num_points= 8,
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
            ),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=_decoder_num_layer,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-70, -50, -10.0, 70, 50, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=4), #----这里类别要改4类
        positional_encoding=dict(
            type='Learned3DPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            z_num_embed = bev_z_,
            ),
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),

    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range)))))

dataset_type = 'CustomNewScenesDataset_MTL'
data_root = 'data/NewScenes_Final/' #---数据路径
file_client_args = dict(backend='disk')
#--生成的格式x y z vx_r_comp vy_r_comp, power,snr,time_diff,Vr[每个radar坐标下],radar_ID
#-使用x y z vx_r_comp vy_r_comp, power,snr,time_diff补偿的速度，旋转到ego/lidar坐标系
radar_use_dims = [0, 1, 2, 3, 4, 5, 6]

train_pipeline = [
    #--读取多帧radar点云，进行速度补偿，旋转到ego坐标系，过滤范围外的点云
    dict(
    type='LoadRadarPointsMultiSweeps',
    load_dim=8,
    sweeps_num=3,
    use_dim=radar_use_dims,
    file_client_args=file_client_args,
    max_num=40000, #--没用到_pad_or_drop
    pc_range=point_cloud_range),
    dict(type='LoadOccupancy_Newscenes', use_semantic=use_semantic, class_names=occ_class_names,occ_size=occ_size),#--读取OCC真值
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadMultiViewImageFromFiles_newsc', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),

    dict(type='NormalizeMultiviewImage', **img_norm_cfg), #----normalization-------
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),#----尺度缩放到0.5
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d','img','points','gt_occ']) #---加入radar points---
]

test_pipeline = [
    dict(type='LoadOccupancy_Newscenes', use_semantic=use_semantic, class_names=occ_class_names,occ_size=occ_size),#--读取OCC真值
    dict(
    type='LoadRadarPointsMultiSweeps',
    load_dim=8,
    sweeps_num=3,
    use_dim=radar_use_dims,
    file_client_args=file_client_args,
    max_num=40000,
    pc_range=point_cloud_range),
    dict(type='LoadMultiViewImageFromFiles_newsc', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080), #----分辨率改-
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img','points','gt_occ']) #---这里还是读入occ_gt评估，而不是直接生成结果文件评估
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'newscenes-final_infos_temporal_occ_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR',
        #--------------加入OCC需要的参数----
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        occ_class_names=occ_class_names,        
        ),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'newscenes-final_infos_temporal_occ_val.pkl',
             pipeline=test_pipeline,  
             bev_size=(bev_h_, bev_w_),
             classes=class_names, 
             modality=input_modality,
        #--------------加入OCC需要的参数----
             occ_size=occ_size,
             pc_range=point_cloud_range,
             use_semantic=use_semantic,
             occ_class_names=occ_class_names, 
             samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'newscenes-final_infos_temporal_occ_val.pkl',
              pipeline=test_pipeline, 
              bev_size=(bev_h_, bev_w_),
              classes=class_names,
              modality=input_modality,
        #--------------加入OCC需要的参数----
              occ_size=occ_size,
              pc_range=point_cloud_range,
              use_semantic=use_semantic,
              occ_class_names=occ_class_names,
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

evaluation = dict(interval=2, pipeline=test_pipeline) #----eval评估间隔，改24
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 30,
#     step=[20,23]
#     )
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
optimizer = dict(
    type='AdamW',
    lr=2e-4, #--2e-4
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
checkpoint_config = dict(interval=1, max_keep_ckpts=6)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/doracamom_20241120/final/Doracamom_1120_final'
load_img_from_and_not_change_state_dict = 'ckpts/r50_fcos3d_pretrain.pth'
load_from = 'ckpts/radarpillarnet.pth'
resume_from = None

workflow = [('train', 1)]




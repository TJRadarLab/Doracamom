# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
#-----后面检测范围要改一下和OCC一致------------
point_cloud_range = [-72, -56, -3.0, 72, 56, 5.0] #---自车/lidar坐标系后面改成五十？
voxel_size = [0.25, 0.25, 8]

# For newScenes we usually do 8-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'engineering_vehicle',
    'rider', 'pedestrian','tricyclist', 'light_truck'
]
# Input modality for newScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=True)

#-------这里加入整个项目------------
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

#----model------------

model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=16, #----这里设置成16
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=8, #---输入点云八维信息
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[448, 576]), #---这里修改成440,560。y,x/voxelsize
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
        type='Anchor3DHead',
        num_classes=9, #---暂时是9类
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator', #---这里后期需修改，根据目标中心点,并统计训练集目标的平均尺寸
            ranges=[
                [-72, -56, -1.80032795, 72, 56, -1.80032795],
                [-72, -56, -1.74440365, 72, 56, -1.74440365],
                [-72, -56, -1.68526504, 72, 56, -1.68526504],
                [-72, -56, -1.67339111, 72, 56, -1.67339111],
                [-72, -56, -1.61785072, 72, 56, -1.61785072],
                [-72, -56, -1.80984986, 72, 56, -1.80984986],
                [-72, -56, -1.763965, 72, 56, -1.763965],
            ],
            sizes=[
                [1.95017717, 4.60718145, 1.72270761],  # car
                [2.4560939, 10.73778078, 2.73004906],  # truck/bus
                [2.87427237, 20.01320693, 3.81509561],  # trailer/engineering_vehicle
                [0.60058911, 1.68452161, 1.27192197],  # rider
                [0.66344886, 0.7256437, 1.75748069],  # pedestrian
                [0.60058911, 2.68452161, 1.27192197],  # tricyclist
                [2.4560939, 6.73778078, 2.73004906],  # light_truck
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))

#-------dataset----------
dataset_type = 'NewScenesDataset' #---dataset
data_root = 'data/newscenes-mini/' #---数据路径
file_client_args = dict(backend='disk')
#--生成的格式x y z vx_r_comp vy_r_comp, power,snr,time_diff,Vr[每个radar坐标下],radar_ID
#-使用x y z vx_r_comp vy_r_comp, power,snr,time_diff补偿的速度，旋转到ego/lidar坐标系
radar_use_dims = [0, 1, 2, 3, 4, 5, 6, 7]
train_pipeline = [
#--读取多帧radar点云，进行速度补偿，旋转到lidar坐标系，过滤范围外的点云
    dict(
    type='LoadRadarPointsMultiSweeps',
    load_dim=8,
    sweeps_num=3,
    use_dim=radar_use_dims,
    file_client_args=file_client_args,
    max_num=40000, #--没用到_pad_or_drop
    pc_range=point_cloud_range),

    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d','points']) #---加入radar points---
]

test_pipeline = [
    
    dict(
    type='LoadRadarPointsMultiSweeps',
    load_dim=8,
    sweeps_num=3,
    use_dim=radar_use_dims,
    file_client_args=file_client_args,
    max_num=30000,
    pc_range=point_cloud_range),

    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080), #----分辨率改-
        pts_scale_ratio=1,
        flip=False,
        transforms=[
    
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['points']) #---加入radar points
        ])
]

#------------val和test-----------
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'newscenes-mini_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality, #----get_data_info中增加相应字段模态信息----
        test_mode=False,
        use_valid_flag=True,  #----在get_anno_info增加mask过滤掉无用目标/这里训练和测试时根据图像是否可视
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'newscenes-mini_infos_temporal_val.pkl',
             pipeline=test_pipeline,  
             classes=class_names, 
             modality=input_modality, 
             samples_per_gpu=1),
    test=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'newscenes-mini_infos_temporal_val.pkl',
             pipeline=test_pipeline,  
             classes=class_names, 
             modality=input_modality, 
             samples_per_gpu=1),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

evaluation = dict(interval=24, pipeline=test_pipeline) #----eval评估间隔，改24
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[20, 23])
momentum_config = None
log_config = dict(
    interval=5, #---50
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# For NewScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
total_epochs = 24
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None

load_from = None
resume_from = None
workflow = [('train', 1)]



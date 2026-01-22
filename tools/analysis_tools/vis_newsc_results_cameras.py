
# ------------------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ------------------------------------------------------
import os
import mmcv
import numpy as np
import pickle
from matplotlib import rcParams
from matplotlib.axes import Axes
from newscenes_devkit.eval.detection.render import visualize_sample
from newscenes_devkit.data_classes import Box
from pyquaternion import Quaternion
from newscenes_devkit.eval.detection.data_classes import DetectionBox
from newscenes_devkit.newscenes import NewScenes
from newscenes_devkit.eval.detection.utils import category_to_detection_name

from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadRadarPointsMultiSweeps
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
# from projects.mmdet3d_plugin.core.points.radar_points import RadarPoints
# from mmdet3d.core.points import LiDARPoints

def lidiar_render(sample_token, pred_results,point_cloud_range,radar_points, lidar_points, out_path=None):
    bbox_gt_list = []
    bbox_pred_list = []
    gt_boxes = newsc.get_annotation_box(sample_token)  #---gt_bboxes
    for i, box in enumerate(gt_boxes):
        #--过滤与映射gt类别
        detection_name = category_to_detection_name(box.name)
        if detection_name is None:
            continue
        #-----过滤gt范围----------
        if abs(box.center[0])>abs(point_cloud_range[0]) or abs(box.center[1])>abs(point_cloud_range[1]):
            continue
        #--过滤visibility---
        if box.visibility == 0:
            continue

        bbox_gt_list.append(box)

    #---------pred box在dataset中范围已经过滤过，class映射过了，这里直接添加即可-----
    bbox_pred = pred_results['results'][sample_token]
    for content in bbox_pred:
        box_pred = Box(
            center=list(content['translation']),
            size=list(content['size']),
            orientation=Quaternion(content['rotation']),
            velocity=tuple(content['velocity']+[0]),
            name=content['detection_name'],
            score=float(content['detection_score']),)
        #-----过滤pred范围----------
        if abs(box_pred.center[0])>abs(point_cloud_range[0]) or abs(box_pred.center[1])>abs(point_cloud_range[1]):
            continue
        bbox_pred_list.append(box_pred)

    print('green is ground truth')
    print('blue is the predited result')
    visualize_sample(sample_token, bbox_gt_list, bbox_pred_list, radar_points, lidar_points,point_cloud_range,conf_th=0.3,savepath=out_path+'_bev')
    

if __name__ == '__main__':
    newsc = NewScenes(version='v1.0-trainval', dataroot='data/NewScenes_Final', verbose=True)
    point_cloud_range = [-60,-40,-3,60,40,5] #----目标范围xyz
    pp_4dradar_results = mmcv.load('work_dirs/NewScenes_Final/pointpillars_newscenes_4DRadar_newscenesfinal/val_result/Thu_Aug_29_14_25_29_2024/pts_bbox/results_newsc.json')
    pp_lidar_results = mmcv.load('work_dirs/pointpillars_newscenes_LiDAR/val_result/Thu_Apr_25_10_53_59_2024/pts_bbox/results_newsc.json')

    sample_token_list = list(pp_4dradar_results['results'].keys())

    #----读取val的pkl，用来读lidar和radar的路径
    data_infos = mmcv.load('data/NewScenes_Final/newscenes-final_infos_temporal_val.pkl')
    
    for id in [100,500,1000,2000]:
        out_path = "/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/newscenesfinal/" + sample_token_list[id]
        for data in data_infos['infos']:
            if data['token'] == sample_token_list[id]:
                info = data
                break
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            radars=info['radars'],
        )
        #--------读lidar点云---------
        lidar_loader = LoadPointsFromFile(coord_type='LIDAR',load_dim=6,use_dim=4)
        input_dict = lidar_loader(input_dict)
        lidar_points_mask = input_dict['points'].in_range_3d(point_cloud_range)
        lidar_points = input_dict['points'][lidar_points_mask].tensor.numpy()
        #--------读radar点云---------  
        radar_loader = LoadRadarPointsMultiSweeps(load_dim=8,
                 use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=point_cloud_range, 
                 test_mode=False)     
        input_dict = radar_loader(input_dict)
        radar_points = input_dict['points'].tensor.numpy()
        # lidiar_render(sample_token_list[id],pp_4dradar_results, point_cloud_range, radar_points, lidar_points, out_path=out_path+'_radar')
        # lidiar_render(sample_token_list[id],pp_lidar_results, point_cloud_range, radar_points, lidar_points, out_path=out_path+'_lidar')
        
        lidiar_render(sample_token_list[id],pp_4dradar_results, point_cloud_range, radar_points, lidar_points=lidar_points, out_path=out_path+'_radar_lidar')

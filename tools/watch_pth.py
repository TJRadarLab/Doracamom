import torch
import pickle
import json

# model = torch.load('/mnt/zhenglianqing/bevformer_noted/work_dirs/MTL_experiment/panoOCC/epoch_48.pth')
# print('debug')
# with open('/mnt/zhenglianqing/bevformer_noted/data/newscenes-mini/newscenes-mini_infos_temporal_val.pkl', 'rb') as f:
#     data = pickle.load(f)
# with open('/mnt/zhenglianqing/bevformer_noted/data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
#     data2 = pickle.load(f)
with open('/mnt/zhenglianqing/bevformer_noted/work_dirs/NewScenes_Final/pointpillars_newscenes_4DRadar_newscenesfinal/val_result/Sat_Aug_31_21_26_01_2024/pts_bbox/results_newsc.json', 'r') as f:
    data3 = json.load(f)
print('debug')
# import numpy as np
# from pyquaternion import Quaternion
# from scipy.spatial.transform import Rotation
# #------------判断是不是正交阵------------
# def check_is_3d_rotation_matrix(matrix: np.ndarray, precision=1.e-5) -> bool:
#     if matrix.shape != (3, 3):
#         return False
#     return np.allclose(matrix.dot(matrix.T), np.identity(3), atol=precision) and \
#         np.allclose(np.linalg.det(matrix), 1, atol=precision)



# # 定义一个4x4矩阵
# matrix = np.array([
#       [-0.032621, 0.053255, 0.998031, 2.107089],
#       [-0.999439, -0.005676, -0.032364, 0.225863],
#       [0.003941, -0.998538, 0.053412, 1.500539],
#       [0.0, 0.0, 0.0, 1.0]
#     ])

# # 提取旋转矩阵和平移向量
# rotation_matrix = matrix[:3, :3]
# x = check_is_3d_rotation_matrix(rotation_matrix)
# print(x)
# # # 计算四元数
# quaternion = Quaternion(matrix=rotation_matrix,atol=1e-4)
# quaternion_elements = quaternion.elements.astype(float)
# # # 创建平移矩阵
# # translation_matrix = np.eye(4)
# # translation_matrix[:3, 3] = translation_vector
# # rotation_matrix2 = Quaternion(quaternion_elements).rotation_matrix

# # # 打印结果
# # print("四元数：", quaternion)
# # print(quaternion_elements)
# # # Calculate the difference between the two rotation matrices
# # difference_matrix = np.subtract(rotation_matrix, rotation_matrix2)

# # # Round the difference matrix to the tenth decimal place
# # rounded_difference_matrix = np.round(difference_matrix, decimals=10)

# # print(rounded_difference_matrix)


# # quaternion = Quaternion(axis=[0, 0, 1], radians=-3.132266003289848)
# # quaternion_elements = quaternion.elements.astype(float)
# print(quaternion_elements)
# import json 

# def read_json(path):
#     with open(path, 'r') as f:
#         return json.load(f)


# result = read_json('/mnt/zhenglianqing/bevformer_noted/test/bevformer_small/Wed_Jan_10_22_36_05_2024/pts_bbox/results_nusc.json')
# print('debug')


# if __name__ == '__main__':
#     root_path = 'data/newscenes-mini'
#     out_path = 'data/newscenes-mini'
#     info_prefix = 'newscenes-mini'
#     version = 'v1.0-mini'
#     max_sweeps = 2  #---lidar历史两帧
#     from newscenes_devkit.newscenes_converter import create_newscenes_infos


#     create_newscenes_infos(root_path,
#                           out_path,
#                           info_prefix,
#                           version,
#                           max_sweeps)
    
    
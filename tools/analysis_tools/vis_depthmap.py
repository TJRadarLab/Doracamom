import numpy as np
import matplotlib.pyplot as plt

def load_sparse_depth_bin(file_path):
    # 读取.bin文件
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    
    # 假设每个深度点包含三个值：u, v, d
    # 每个点的格式为 (u, v, d)
    points = data.reshape(-1, 3)
    
    return points

def visualize_sparse_depth(points,lidar_path):
    # 提取 u, v, d
    u = points[:, 0]
    v = points[:, 1]
    d = points[:, 2]

    # 创建一个稀疏的深度图
    plt.figure(figsize=(12, 6))
    plt.scatter(u, v, c=d, cmap='viridis', s=1)
    plt.colorbar(label='Depth (m)')
    plt.title('Sparse Depth Map')
    plt.xlabel('u (Pixel)')
    plt.ylabel('v (Pixel)')
    plt.gca().invert_yaxis()  # 反转y轴
    name = lidar_path.split('/')[-2]+'_'+lidar_path.split('/')[-1].split('.')[0]
    plt.savefig(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/depth/{name}_depth.png')

# 示例使用
lidar_path = '/mnt/zhenglianqing/bevformer_noted/data/NewScenes_Final/1693730429633415/depth_gt/camera_front/1693730429666666.jpg.bin'  # 替换为你的.bin文件路径
sparse_depth_points = load_sparse_depth_bin(lidar_path)
visualize_sparse_depth(sparse_depth_points,lidar_path)
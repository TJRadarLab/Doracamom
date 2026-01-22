# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Written by [TONGJI] [Long Yang] & [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 对数据集预处理生成json文件
# ---------------------------------------------
import os
import json
import shutil
def generate_sample_json(root_dir):
    sample = []
    sample_data = []
    dirnames = sorted(next(os.walk(root_dir))[1])
    scene_idx = 0
    for dirname in dirnames:
        scene_dir = os.path.join(root_dir, dirname)
        annotation_dir = os.path.join(scene_dir, 'annotations')
        sync_files_dir = os.path.join(scene_dir, 'sync_files')

        if os.path.exists(annotation_dir):
            annotation_files = sorted([f.rstrip('.json') for f in os.listdir(annotation_dir) if f.endswith('.json')])
            for i, annotation_file in enumerate(annotation_files):
                timestamp_token = annotation_file
                pre_token = annotation_files[i-1] if i > 0 else ''
                next_token = annotation_files[i+1] if i < len(annotation_files) - 1 else ''
                scene_token = dirname
                scene_name = f'scene-{scene_idx:04d}'
                sample.append({
                    'token': timestamp_token,
                    'frame_idx': i,
                    'prev': pre_token,
                    'next': next_token,
                    'timestamp': timestamp_token,
                    'scene_token': scene_token,
                    'scene_name': scene_name
                })
            scene_idx += 1

        if os.path.exists(sync_files_dir):
            sync_files = sorted([f.rstrip('.json') for f in os.listdir(sync_files_dir) if f.endswith('.json')])
            for i, sync_file in enumerate(sync_files):
                sync_file = sync_files[i] + '.json'
                sync_token = sync_file.rstrip('.json')
                pre_token = sync_files[i-1] if i > 0 else ''
                next_token = sync_files[i+1] if i < len(sync_files) - 1 else ''
                is_key_frame = sync_token in annotation_files

                with open(os.path.join(sync_files_dir, sync_file), 'r') as f:
                    sync_file_content = json.load(f)

                for key in sync_file_content:
                    if key == 'ego_pose':
                        continue
                    for subkey in sync_file_content[key]:
                        extension = '.bin' if key in ['lidar', 'radars'] else '.jpg'
                        sync_file_content[key][subkey] = os.path.join(dirname, key, subkey, sync_file_content[key][subkey] + extension)

                sync_data_item = {
                    'token': sync_token,
                    'prev': pre_token,
                    'next': next_token,
                    'timestamp': sync_token,
                    'is_key_frame': is_key_frame,
                }
                sync_data_item.update(sync_file_content)
                sample_data.append(sync_data_item)

    with open(os.path.join(root_dir,'sample.json'), 'w') as f:
        json.dump(sample, f, indent=4)

    with open(os.path.join(root_dir,'sample_data.json'), 'w') as f:
        json.dump(sample_data, f, indent=4)

def generate_anno_json(root_dir):

    annos = []
    dirnames = sorted(next(os.walk(root_dir))[1])
    for dirname in dirnames:
        scene_dir = os.path.join(root_dir, dirname)
        annotation_dir = os.path.join(scene_dir, 'annotations')

        if os.path.exists(annotation_dir):
            annotation_files = sorted([f.rstrip('.json') for f in os.listdir(annotation_dir) if f.endswith('.json')])
            for i, annotation_file in enumerate(annotation_files):
                timestamp_token = annotation_file
                scene_token = dirname
                with open(os.path.join(annotation_dir, annotation_file + '.json'), 'r') as f:
                    anno = json.load(f)
                annos.append({
                    'token': timestamp_token,
                    'scene_token': scene_token,
                    'annotations': anno,
                })

    with open(os.path.join(root_dir,'annotations.json'), 'w') as f:
        json.dump(annos, f, indent=4)



def generate_egopose_json(root_dir):

    ego_pose = []
    dirnames = sorted(next(os.walk(root_dir))[1])
    for dirname in dirnames:
        scene_dir = os.path.join(root_dir, dirname)
        ego_pose_file = os.path.join(scene_dir, 'icp_pose.txt')

        if os.path.exists(ego_pose_file):
            with open(ego_pose_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(',')
                    timestamp = line[0]
                    pose = [float(x) for x in line[1:]]
                    ego_pose.append({
                        'token': timestamp,
                        'scene_token': dirname,
                        'pose': pose,
                    })

    with open(os.path.join(root_dir,'ego_pose.json'), 'w') as f:
        json.dump(ego_pose, f, indent=4)



def generate_imu_json(root_dir):
    imu_data = []
    dirnames = sorted(next(os.walk(root_dir))[1])
    for dirname in dirnames:
        print('Now generating imu_json for', dirname)
        scene_dir = os.path.join(root_dir, dirname)
        imu_file = os.path.join(scene_dir, 'imu.txt')
        ego_velocity_file = os.path.join(scene_dir, 'ego_velocity.txt')
        if os.path.exists(imu_file) and os.path.exists(ego_velocity_file):
            with open(imu_file, 'r') as f:
                imu_original_data = f.readlines()
            
            with open(ego_velocity_file, 'r') as f:
                ego_velocity_data = f.readlines()
            
            #---判断两个文件行数是否相等以及首位时间戳是否一致--
            if len(imu_original_data) != len(ego_velocity_data):
                print(f'imu and ego_velocity file line number not equal in {dirname}')
                return
            if imu_original_data[0][:16].split(',')[0] != ego_velocity_data[0].split(',')[0]:
                print(f'imu and ego_velocity file first timestamp not equal in {dirname}')
                return
            if imu_original_data[-1].split(',')[0][:16] != ego_velocity_data[-1].split(',')[0]:
                print(f'imu and ego_velocity file last timestamp not equal in {dirname}')
                return
            #-----------------------------------------------------------------------
            #--time,$GPCHC,GPSWeek,GPSTime,Heading,Pitch,Roll,gyro x,gyro y,gyro z,acc x,acc y,acc z,
            #--Lattitude,Longitude,Altitude,Ve,Vn,Vu,V,NSV1,NSV2,Status,Age,WarmingCs<CR><LF>
            for i in range(len(imu_original_data)):
                imu = imu_original_data[i]
                ego_velocity = ego_velocity_data[i]

                imu = imu.strip().split(',')
                ego_velocity = ego_velocity.strip().split(',')
                timestamp = ego_velocity[0]
                gyro_xyz = [float(imu[7]), float(imu[8]), float(imu[9])]
                acc_xyz = [float(imu[10]), float(imu[11]), float(imu[12])]
                velocity_enu = [float(imu[16]),float(imu[17]),float(imu[18])]
                velocity_ego = [float(x) for x in ego_velocity[1:]]
                imu_data.append({
                    'token': timestamp,
                    'scene_token': dirname,
                    'gyro_xyz': gyro_xyz,
                    'acc_xyz':acc_xyz,
                    'velocity_enu':velocity_enu,
                    'velocity_ego':velocity_ego,
                })

    with open(os.path.join(root_dir,'imu_data.json'), 'w') as f:
        json.dump(imu_data, f, indent=4)


def generate_sensor_calibration_json(root_dir):

    ego_pose = []
    dirnames = sorted(next(os.walk(root_dir))[1])
    for dirname in dirnames:
        scene_dir = os.path.join(root_dir, dirname)
        ego_pose_file = os.path.join(scene_dir, 'icp_pose.txt')

        if os.path.exists(ego_pose_file):
            with open(ego_pose_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(',')
                    timestamp = line[0]
                    pose = [float(x) for x in line[1:]]
                    ego_pose.append({
                        'token': timestamp,
                        'scene_token': dirname,
                        'pose': pose,
                    })

    with open(os.path.join(root_dir,'ego_pose.json'), 'w') as f:
        json.dump(ego_pose, f, indent=4)

if __name__ == '__main__':
    root_dir = '/mnt/zhenglianqing/bevformer_noted/data/newscenes-mini'  # 场景包根目录
    
    #-----生成sample.json和sample_data.json
    generate_sample_json(root_dir)
    
    
    #-----生成annotations.json
    # generate_anno_json(root_dir)

    #-----生成ego_pose.json----
    # generate_egopose_json(root_dir)

    #---生成imu_data.json----
    # generate_imu_json(root_dir)

    #-----移动到v1.0-mini目录下---
    json_file_folder = os.path.join(root_dir, 'v1.0-mini')
    if not os.path.exists(json_file_folder):
        os.makedirs(json_file_folder)
        print(f"文件夹 '{json_file_folder}' 创建成功")
    else:
        print(f"文件夹 '{json_file_folder}' 已存在")
    json_files = ['sample.json', 'sample_data.json', 'annotations.json', 
                  'ego_pose.json', 'imu_data.json','scene_split.json','sensor_calibration.json']
    for json_file in json_files:
        file_path = os.path.join(root_dir, json_file)
        shutil.move(file_path, os.path.join(json_file_folder, json_file))
        print(f"文件 '{json_file}' 已成功移动到 '{json_file_folder}' 文件夹中")
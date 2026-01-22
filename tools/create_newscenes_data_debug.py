# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------

from newscenes_devkit import newscenes_converter as newscenes_converter
import argparse
from os import path as osp
import sys
sys.path.append('.')


def newscenes_data_prep(root_path,
                       out_dir,
                       info_prefix,
                       version,
                       max_sweeps=2):
    """Prepare data related to newScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    #-------------主要在这个函数里----------------
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
    else:
        #---------------这里取上边生成好的pkl文件-------------
        info_train_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version)
        # create_groundtruth_database(dataset_name, root_path, info_prefix,
        #                             f'{out_dir}/{info_prefix}_infos_train.pkl')

if __name__ == '__main__':
    root_path = 'data/newscenes-mini'
    out_path = 'data/newscenes-mini'
    info_prefix = 'newscenes-mini'

    version = 'v1.0-mini'
    max_sweeps = 2  #---lidar历史两帧,radar历史两帧

    newscenes_data_prep(root_path,
                          out_path,
                          info_prefix,
                          version,
                          max_sweeps)
    #-------------只运行这个生成pkl------------------------------
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        #-------先生成train和val--------------
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        #----------再生成test-----------------
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)

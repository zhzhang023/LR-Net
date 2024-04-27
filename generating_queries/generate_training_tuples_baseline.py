# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import tqdm

from data.oxford import TrainingTuple
# Import test set boundaries
from generating_queries.generate_test_sets import P1, P2, P3, P4, check_in_test_set

# Test set boundaries
P = [P1, P2, P3, P4]

RUNS_FOLDER = "oxford/"
FILENAME = "pointcloud_locations_20m_10overlap.csv"
POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    # 找出所有location对应的 positive 和 negative 数据索引
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        # 当前位置
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        # 该位置的点云文件路径
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        # 分割出该文件的 timestamp
        scan_filename = os.path.split(query)[1]
        # 检测读到的是不是.bin文件
        assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])
        # 取该位置下的 positive 和 negative 索引
        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]
        # 在 positive 中去掉 当前位置的索引， 当前位置不能算自己的positive
        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # query是个 dictionary，里面存的键是当前位置的索引，值是 TrainingTuple的数据形式
        # TrainingTumple 主要存储以下信息： 当前的索引ID，时间戳，文件全路径，positive索引列表，negative索引列表，当前的位置（north， east）
        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


if __name__ == '__main__':

    data_root='/media/joey/DATASET/PNVLAD_Dataset/benchmark_datasets/'
    assert os.path.exists(data_root), f"Cannot access dataset root folder: {data_root}"
    base_path = data_root

    #RUN FOLDER是具体场景的文件夹，这里列出该场景下所有文件
    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
    folders = []

    # All runs are used for training (both full and partial)
    # 计算总共的文件夹数，每个文件夹就是一次RUN的数据，讲文件夹名放入列表folders内
    index_list = range(len(all_folders) - 1)
    print("Number of runs: " + str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])
    print(folders)

    # 定义一个表格用于数据管理
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in tqdm.tqdm(folders):
        # 读取具体文件夹中的 excel 表格，读取文件名和 north east 数据
        df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, FILENAME), sep=',')
        # 将 timestamp修改为正确的文件路径，并将 timestamp 列 改为 file
        df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        # 遍历所有的行，测试是不是属于test范围，将整体 df_location 分为 train 和 test 两个不同的frame
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)
    #输出一下 train 和 test 的数量多少
    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    # 进入为 train 和 test 数据生成 query dict
    construct_query_dict(df_train, base_path, "spv_training_queries_baseline.pickle", ind_nn_r=10)
    construct_query_dict(df_test, base_path, "spv_test_queries_baseline.pickle", ind_nn_r=10)

# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse

# For training and test data splits
X_WIDTH = 150
Y_WIDTH = 150
# 用于区分各自场景test部分的坐标
# For Oxford
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]

# For University Sector
P5 = [363621.292362, 142864.19756]
P6 = [364788.795462, 143125.746609]
P7 = [363597.507711, 144011.414174]

# For Residential Area
P8 = [360895.486453, 144999.915143]
P9 = [362357.024536, 144894.825301]
P10 = [361368.907155, 145209.663042]

P_DICT = {"oxford": [P1, P2, P3, P4], "university": [P5, P6, P7], "residential": [P8, P9, P10], "business": []}

# 看来用于测试的database和query的
def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
    '''
    :param base_path: 数据集根目录
    :param runs_folder: 场景文件夹（因为有 oxford 和 3个inhouse场景的区分）
    :param folders: 场景下对应的不同时间的观测数据文件夹
    :param pointcloud_fols: 这里 oxford 用的是20m文件夹，好像是用来做 database的
    :param filename: 点云 timestamp north east 的 csv 文件
    :param p: 用于甄别 test 部分的点
    :param output_name: 输出场景名
    :return:
    '''
    database_trees = []
    test_trees = []
    # 这次遍历 主要是给每个文件夹建立树
    for folder in folders:
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            # 每个 Row 里包含 file地址 north east 坐标
            # entire business district is in the test set
            if output_name == "business": # 这是按场景添加的 business test数据，整个business都在test set里
                df_test = df_test.append(row, ignore_index=True)
            elif check_in_test_set(row['northing'], row['easting'], p):# 否则其他的就要test一下
                df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # 将读取的csv文件中的时间改成 文件路径，并将 timestamp标签改成 file
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + \
                                    df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})# timestamp 重命名为 file
        # 这里为 database 和 test 添加数据，test需要验证一下，database就直接添加。
        # 这个set里有 query（文件路径） north east
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            time_stamp = row['file'].split('/')[-1].split('.')[0]
            if output_name == "business":
                test[len(test.keys())] = {'query': row['file'], 'timestamp':time_stamp,'northing': row['northing'], 'easting': row['easting']}
            elif check_in_test_set(row['northing'], row['easting'], p):
                test[len(test.keys())] = {'query': row['file'], 'timestamp':time_stamp,'northing': row['northing'], 'easting': row['easting']}
            database[len(database.keys())] = {'query': row['file'], 'timestamp':time_stamp,'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    # 这样遍历过之后，tree和set里的数据都对应了
    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j: # 如果是同一个run下面的数据的话就不算，主要看不同时期数据之间的检索
                continue
            for key in range(len(test_sets[j].keys())):#j代表的是场景，key遍历的就是场景下的帧数据
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=25) # 这个index计算的是当前帧在其他run之中的对应帧数
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, base_path, output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, base_path, output_name + '_evaluation_query.pickle')


if __name__ == '__main__':
    import model.Config as cfg
    print('Dataset root: {}'.format(cfg.dataset_folder))

    assert os.path.exists(cfg.dataset_folder), f"Cannot access dataset root folder: {cfg.dataset_folder}"
    base_path = cfg.dataset_folder

    # For Oxford
    # Oxford只用部分场景的做了测试
    # Oxford测试用了20m文件夹的部分
    folders = []
    runs_folder = "oxford/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    # 并不是所有的文件夹下都有20m那个文件夹的，这里的索引应该是找出这些文件架
    index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_20m/",
                                      "pointcloud_locations_20m.csv", P_DICT["oxford"], "oxford")

    # For University Sector
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    uni_index = range(10, 15)
    for index in uni_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["university"], "university")

    # For Residential Area
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    res_index = range(5, 10)
    for index in res_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["residential"], "residential")

    # For Business District
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    bus_index = range(5)
    for index in bus_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["business"], "business")

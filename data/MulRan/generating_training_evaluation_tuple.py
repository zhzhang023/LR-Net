import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

BASE_DIR = '/media/joey/DATASET/MulRan'
Scene = ['DCC', 'KAIST', 'RiverSide', 'Sejong']
Train_split = ['DCC01', 'DCC02', 'Riverside_01', 'Riverside_02']
Eval_split = ['DCC03', 'Riverside_03','KAIST01','KAIST02','KAIST03']
Train=True
Eval=False

Mode='original_pc'

Train_min_dis=10
Train_max_dis=50
Eval_dis=25

def construct_query_dict(df_centroids,return_base_index=0,scene=None,queries=None):
    tree = KDTree(df_centroids[['north','east']])
    # 对于室内场景，2m内可以算positive，20m外应该就看不见了，可以是negative
    ind_nn = tree.query_radius(df_centroids[['north','east']],r=Train_min_dis)
    ind_r = tree.query_radius(df_centroids[['north','east']], r=Train_max_dis)
    if Mode=='original_pc':
        ask_key='lidar_old'
    else:
        raise NotImplementedError
    if queries is None:
        queries = {}
    for i in tqdm(range(len(ind_nn))):
        anchor_pos = np.array(df_centroids.iloc[i][['north', 'east']])
        query = df_centroids.iloc[i][ask_key]
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.npy', f"Expected .npy file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])
        # data within 10m except query
        positives = ind_nn[i]
        non_negatives = ind_r[i]
        # 在 positive 中去掉 当前位置的索引， 当前位置不能算自己的positive
        positives = positives[positives != i]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)
        positives=positives+return_base_index
        non_negatives=non_negatives+return_base_index
        queries[i+return_base_index] = TrainingTuple(id=i+return_base_index, timestamp=timestamp, rel_scan_filepath=query,
                                   positives=positives, non_negatives=non_negatives, position=anchor_pos,
                                   scene=scene)
    base_index_new=return_base_index+len(ind_nn)-1
    return queries,base_index_new

def construct_query_and_database_sets_for_onescene(df_locations,scene=None,use_origin_pc=True,time_base=None):
    # 主要功能： 对当前输入场景的 df——location 生成这个场景对应的 database 和 query 的 Pickle。
    df_database = pd.DataFrame(columns=['time_stap', 'north', 'east', 'lidar_new', 'lidar_old','scene'])
    df_test = pd.DataFrame(columns=['time_stap', 'north', 'east', 'lidar_new', 'lidar_old','scene'])
    if Mode == 'original_pc':
        ask_key = 'lidar_old'
    else:
        raise NotImplementedError
    for index, row in df_locations.iterrows():
        ts_temp = row['time_stap']
        north_temp = row['north']
        east_temp = row['east']
        lidar_path_temp=row[ask_key]
        scene_temp=row['scene']
        if scene_temp in Eval_split:  #
            df_test = df_test.append(row, ignore_index=True)
        df_database = df_database.append(row, ignore_index=True)
    database_tree = KDTree(df_database[['north', 'east']])
    database = {}
    test = {}
    for index, row in df_locations.iterrows():
        scene_temp = row['scene']
        # entire business district is in the test set
        if scene_temp in Eval_split:  #
            test[len(test.keys())] = {'query': row[ask_key], 'northing': row['north'], 'easting': row['east'],'timestamp':row['time_stap']}
        database[len(database.keys())] = {'query': row[ask_key], 'northing': row['north'],
                                          'easting': row['east'],'timestamp':row['time_stap']}
    for key in range(len(test.keys())):
        coor = np.array([[test[key]["northing"], test[key]["easting"]]])
        index = database_tree.query_radius(coor, r=Eval_dis)  # 这个index计算的是当前帧在其他run之中的对应帧数
        # indices of the positive matches in database i of each query (key) in test set j
        test[key][0] = index[0].tolist()

    return database, test

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray,scene:str):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
        self.scene=scene

if __name__ == '__main__':
    file_name = 'Timestap_and_location.xlsx'

    all_folders=sorted(os.listdir(BASE_DIR)) # 每个场景只有3次run
    folders=[]
    index_list = range(len(all_folders))
    print("Number of Scenes: "+str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])
    folders=[t for t in folders if '.pickle' not in t]
    print(folders)
    # Generate Training pickles
    if Train:
        print('Generate Training Pickles...')
        Traning_queries = {}
        train_base_index = 0
        for i in range(len(folders)):
            folder = folders[i]
            folder_path=os.path.join(BASE_DIR,folder)
            print('proceesing training scenes:' + folder)
            df_train = pd.DataFrame(columns=['time_stap', 'north', 'east', 'lidar_new', 'lidar_old','scene'])
            df_locations=pd.read_excel(os.path.join(BASE_DIR,folder,file_name),index_col=0)
            for index, row in tqdm(df_locations.iterrows()):
                ts_temp=row['time_stap']
                north_temp=row['north']
                east_temp=row['east']
                lidar_new_path_temp=row['lidar_new']
                lidar_old_path_temp = row['lidar_old']
                scene_temp=row['scene']
                if scene_temp not in Train_split:
                    continue
                df_train=df_train.append(row,ignore_index=True)
            if len(df_train['time_stap'])==0:
                print('Not in training list,skipping '+folder)
                continue
            print("Number of training submaps in" + folder + ":" + str(len(df_train['time_stap'])))
            training_file_path = folder + "_training_queries_baseline.pickle"
            Traning_queries, train_base_index = construct_query_dict(df_train, return_base_index=train_base_index,
                                                                     scene=folder, queries=Traning_queries,
                                                                     )
        if Mode=='original_pc':
            training_file_path = os.path.join(BASE_DIR, "training_queries_baseline.pickle")
        else:
            raise NotImplementedError
        with open(training_file_path, 'wb') as handle:
            pickle.dump(Traning_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('FINISH!')
    else:
        print('Skipping train process..')

    # Evaluation
    # Eval的pickle是每个场景都有一个单独的
    if Eval:
        print('Start Eval...')
        for i in range(len(folders)):
            folder=folders[i]
            if os.path.exists(os.path.join(BASE_DIR, folder + "_evaluation_database_"+str(Eval_dis)+ '_' + Mode +".picklee")):
                print('Already exist eval file for scene: '+folder)
                continue
            print('processing evaluation scenes:'+folder)
            df_locations = pd.read_excel(os.path.join(
                BASE_DIR, folder, file_name), index_col=0)
            database_file, query_file = construct_query_and_database_sets_for_onescene(df_locations, scene=folder)

            database_file_path = os.path.join(BASE_DIR, folder + "_evaluation_database_"+str(Eval_dis)+ '_' + Mode+".pickle")
            query_file_path = os.path.join(BASE_DIR, folder + "_evaluation_query_"+str(Eval_dis)+ '_' + Mode+".pickle")
            with open(database_file_path, 'wb') as handle:
                pickle.dump(database_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(query_file_path, 'wb') as handle:
                pickle.dump(query_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('FINISH SCENE:'+folder)
    else:
        print('Skipping Eval')



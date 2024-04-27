import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from tqdm import tqdm

BASE_DIR='/media/joey/DATASET/KITTI/dataset/sequences'
Scenes = ['00', '02', '05', '06', '07', '08']
Using_idx=[0,1,2,3,4,5]

def random_sampling(pc, num_sample, replace=False):
    choices = np.random.choice(len(pc), num_sample, replace=replace).tolist()
    database=[]
    test=[]
    for i in range(len(pc)):
        if i in choices:
            test.append(pc[i])
        else:
            database.append(pc[i])
    # test=[pc[choices[t]] for t in range(choices.shape[0])]
    return database,test

def construct_query_and_database_sets_for_onescene(df_locations,scene=None,test_num=500):
    df_database = pd.DataFrame(columns=['time_stap', 'north', 'east', 'lidar'])
    df_test = pd.DataFrame(columns=['time_stap', 'north', 'east', 'lidar'])
    database=[]
    scene_database = {}
    scene_test = {}
    for index, row in df_locations.iterrows():
        database.append(row)
    print('All %d Scenes in sequence' %len(database))
    database,test=random_sampling(database,test_num,replace=True)
    q_North=[]
    q_East=[]
    d_North=[]
    d_East=[]
    for t in tqdm(range(len(database))):
        df_database = df_database.append(database[t], ignore_index=True)
        time_stamp=database[t]['time_stap']
        if time_stamp=='0:00:00':
            time=0
        else:
            tt=time_stamp.split(':')
            day=float(tt[0])
            hour=float(tt[1])
            minsec=tt[2].split('.')
            if len(minsec)==1:
                min=float(minsec[0])
                sec=0
            else:
                min = float(minsec[0])
                sec = float(minsec[1]) / 100000
            time=day*86400+hour*3600+min*60+sec
        scene_database[len(scene_database.keys())] = {'query': database[t]['lidar'], 'northing': database[t]['north'],
                                                      'easting': database[t]['east'],'time_stamp':time}
        d_North.append(float(database[t]['north']))
        d_East.append(float(database[t]['east']))

    for t in tqdm(range(len(test))):
        df_test = df_test.append(test[t], ignore_index=True)
        time_stamp = test[t]['time_stap']
        if time_stamp == '0:00:00':
            time = 0
        else:
            tt = time_stamp.split(':')
            day = float(tt[0])
            hour = float(tt[1])
            minsec = tt[2].split('.')
            if len(minsec) == 1:
                min = float(minsec[0])
                sec = 0
            else:
                min = float(minsec[0])
                sec = float(minsec[1]) / 100000
            time = day * 86400 + hour * 3600 + min * 60 + sec
        scene_test[len(scene_test.keys())] = {'query': test[t]['lidar'], 'northing': test[t]['north'],
                                                      'easting': test[t]['east'],'time_stamp':time}
        q_North.append(float(test[t]['north']))
        q_East.append(float(test[t]['east']))

    fig = plt.figure()
    ax = fig.gca()
    figure = ax.plot(d_North, d_East, c='r')
    plt.scatter(q_North, q_East, marker='o', facecolors='none',
                edgecolors='m', s=300, linewidths=3, alpha=1)
    plt.show()
    database_tree = KDTree(df_database[['north', 'east']])
    test_tree = KDTree(df_test[['north', 'east']])

    for key in range(len(scene_test.keys())):
        coor = np.array([[scene_test[key]["northing"], scene_test[key]["easting"]]])
        index = database_tree.query_radius(coor, r=25)  # 这个index计算的是当前帧在其他run之中的对应帧数
        # indices of the positive matches in database i of each query (key) in test set j
        scene_test[key][0] = (index[0]).tolist()
    return scene_database, scene_test

if __name__ == '__main__':
    file_name = 'Timestap_and_location.xlsx'

    print("Number of Scenes: " + str(len(Using_idx)))

    for i in range(len(Using_idx)):
        scene_temp=Scenes[Using_idx[i]]
        print('processing evaluation scene: '+scene_temp)
        df_locations = pd.read_excel(os.path.join(
            BASE_DIR, scene_temp, file_name), index_col=0)
        database,test = construct_query_and_database_sets_for_onescene(df_locations,scene_temp)
        database_path=os.path.join(BASE_DIR,scene_temp+'_evaluation_database.pickle')
        query_path = os.path.join(BASE_DIR, scene_temp + '_evaluation_query.pickle')
        with open(database_path, 'wb') as handle:
            pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(query_path, 'wb') as handle:
            pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Finish processing: '+scene_temp)
    print('ALL FINISH!')


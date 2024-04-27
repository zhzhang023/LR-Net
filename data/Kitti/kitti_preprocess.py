import os
import sys
import numpy as np
import pykitti
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pc_util as util

Data_root='/media/joey/DATASET/KITTI/dataset'

def Process_One_Scene(scene,show=False,save_dir=None,format='npy'):
    Time_stamp = []
    North = []
    East = []
    LiDAR = []
    dataset = pykitti.odometry(Data_root, sequence=scene)
    for i in tqdm(range(dataset.__len__())):
        # add location
        pose_temp = dataset.poses[i]
        trans_temp = pose_temp[0:3, 3]
        North.append(trans_temp[0])
        East.append(trans_temp[1])
        # add timestamp
        time_temp=str(dataset.timestamps[i])
        Time_stamp.append(time_temp)
        # add lidar_path
        lidar_temp = dataset.velo_files[i]
        if format=='bin':
            LiDAR.append(lidar_temp)
        elif format=='npy':
            time_stap_temp = lidar_temp.split('/')[-1].split('.')[0]
            save_path_temp = save_dir + '/' + time_stap_temp + '.npy'
            data_temp = dataset.get_velo(i)
            if not os.path.exists(save_path_temp):
                data_temp = dataset.get_velo(i)
                np.save(save_path_temp, data_temp)
            LiDAR.append(save_path_temp)
        elif format=='ply':
            time_stap_temp = lidar_temp.split('/')[-1].split('.')[0]
            save_path_temp = save_dir + '/' + time_stap_temp + '.ply'
            if not os.path.exists(save_path_temp):
                data_temp = dataset.get_velo(i)
                util.write_ply(data_temp,save_path_temp)
        else:
            raise NotImplementedError
    if show:
        fig = plt.figure()
        ax = fig.gca()
        figure = ax.plot(North, East, c='r')
        plt.show()
    result = pd.DataFrame(
        {'time_stap': Time_stamp, 'north': North, 'east': East, 'lidar': LiDAR})
    # Saving
    save_path=os.path.join(Data_root,'sequences',scene)
    Save_path = os.path.join(save_path, 'Timestap_and_location.xlsx')
    result.to_excel(Save_path)

if __name__ == '__main__':
    print('Pre-processing kitti dataset...')
    Use_Scenes=['00','02','05','06','07','08']
    for i in range(len(Use_Scenes)):
        print('Prosessing sequence: '+Use_Scenes[i])
        save_path_temp=os.path.join(Data_root,'sequences',Use_Scenes[i],'npyfile')
        if not os.path.exists(save_path_temp):
            os.mkdir(save_path_temp)
        Process_One_Scene(Use_Scenes[i],show=True,save_dir=save_path_temp,format='npy')
        print('Down Saving Scene: '+Use_Scenes[i])



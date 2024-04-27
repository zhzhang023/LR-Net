import os
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def find_nearest(list,value):
    array=np.asarray(list,dtype=float)
    idx=(np.abs(array-value)).argmin()
    return idx,array[idx]

def Precess_One_Scene(data_root,folders,show_traj=False,pose_file_name='pd_northing_easting.csv'):
    Time_stamp=[]
    North=[]
    East=[]
    LiDAR_old=[]
    LiDAR_new=[]
    Scene=[]
    for folder in folders:
        print('processing scene:'+folder)
        run_path = os.path.join(data_root,folder)
        global_pose_file=os.path.join(run_path, pose_file_name)
        count = 0
        with open(global_pose_file, 'r') as f1:
            f1 = list(csv.reader(f1))
            for global_pose_line in f1:
                if count == 0:
                    count += 1
                    continue
                Time_stamp.append(str(global_pose_line[2]))
                North.append(float(global_pose_line[3]))
                East.append(float(global_pose_line[4]))
                Scene.append(str(folder))
                lidar_old_temp=os.path.join(run_path,'Ouster',str(global_pose_line[2])+'.bin')
                lidar_old_temp_npy=os.path.join(run_path,'Ouster',str(global_pose_line[2])+'.npy')
                if not os.path.exists(lidar_old_temp_npy):
                    pc_temp = np.fromfile(lidar_old_temp, dtype=np.float32).reshape([-1, 4])
                    pc_idx = np.where(pc_temp[:, 0] != 0)[0]
                    pc_temp=pc_temp[pc_idx,:]
                    np.save(lidar_old_temp_npy,pc_temp)
                lidar_new_temp = os.path.join(run_path, 'Ouster_4096', str(global_pose_line[2]) + '.npy')
                LiDAR_old.append(lidar_old_temp_npy)
                LiDAR_new.append(lidar_new_temp)
    if show_traj:
        fig = plt.figure()
        ax = fig.gca()
        figure = ax.plot(North, East, c='r')
        plt.show()
    result = pd.DataFrame({'time_stap': Time_stamp, 'north': North, 'east': East,
                           'lidar_new': LiDAR_new, 'lidar_old': LiDAR_old, 'scene':Scene})
    # Saving
    Save_path = os.path.join(data_root, 'Timestap_and_location.xlsx')
    result.to_excel(Save_path)

if __name__ == '__main__':
    data_root='/media/joey/DATASET/MulRan'
    Scene=['DCC','RiverSide','KAIST']
    Scene_Idx=2
    # 给每个场景生成 Timestamp North 和 east 的pickle
    data_root=os.path.join(data_root,Scene[Scene_Idx])
    assert os.path.exists(data_root), f"Cannot access dataset root folder: {data_root}"
    all_folders = sorted(os.listdir(data_root))
    folders = []
    # 计算 root 路径下文件夹数量，该数量代表run的次数
    index_list = range(len(all_folders))
    print("Number of runs: " + str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])
    print(folders)
    Precess_One_Scene(data_root,folders,show_traj=True)
    print('Down!')
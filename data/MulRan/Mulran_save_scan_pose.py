import glob
import os
import numpy as np
import csv
from matplotlib import pyplot as plt
import sys

basedir = '/media/joey/DATASET/MulRan_temp/'
def findNnPoseUsingTime(target_time, all_times, data_poses):
    time_diff = np.abs(all_times - target_time)
    nn_idx = np.argmin(time_diff)
    return data_poses[nn_idx]

sequences = ['KAIST/KAIST01', 'KAIST/KAIST02', 'KAIST/KAIST03']
for sequence in sequences:
    sequence_path = basedir + sequence + '/Ouster/'
    scan_names = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))

    North_all=[]
    East_all=[]
    with open(basedir + sequence + '/global_pose.csv', newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    for i in range(len(data_poses)):
        North_all.append(float(data_poses[i][4]))
        East_all.append(float(data_poses[i][8]))
    data_poses_ts = np.asarray([int(t) for t in np.asarray(data_poses)[:, 0]])
    
    North=[]
    East=[]
    for scan_name in scan_names:
        scan_time = int(scan_name.split('/')[-1].split('.')[0])
        scan_pose = findNnPoseUsingTime(scan_time, data_poses_ts, data_poses)
        writing_pose=[scan_pose[0],scan_time,scan_pose[4],scan_pose[8]]
        North.append(float(scan_pose[4]))
        East.append(float(scan_pose[8]))
        with open(basedir + sequence + '/pd_northing_easting.csv', 'a', newline='') as csvfile:
            posewriter = csv.writer(
                csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            posewriter.writerow(writing_pose)
    print('Finish scene:'+sequence)
# [LR-Net] Lightweight and Rotation-Invariant Place Recognition Network for Large-scale Raw Point Clouds


## [Paper]: 
* Paper: [Lightweight and rotation-invariant place recognition network for large-scale raw point clouds](https://doi.org/10.1016/j.isprsjprs.2024.04.030) 

### Introduction
 The paper presents a lightweight and rotation-invariant place recognition network, named **LR-Net**, designed to handle large-scale raw point clouds. 
 
 Based on our experiments, we observed that most existing methods face challenges in achieving rotation invariance and are hindered by a large number of parameters and complex data preprocessing steps, such as ground point removal and extensive down-sampling. These limitations hinder their practical implementation in complex real-world scenarios.

 LR-Net tackles rotation invariance by efficiently capturing sparse local neighborhood information and generating low-dimensional rotation-invariant features through the analysis of spatial distribution and positional information with local region.
 Then, we enhance the network's feature perception by incorporating residual MLP and attention mechanism, the GM-pooling function is utilized to aggregate the discriminative 3D point cloud descriptor.

 LR-Net has a lightweight model size of 0.4M parameters, and eliminates the need for data preprocessing, as it can directly process raw point cloud for stable place recignition and localization. 
 Evaluation on standard benchmarks proves that LR-Net outperforms current state-of-the-art.  


### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 1.9.0  on Ubuntu 18.04 with CUDA 11.7.

### Datasets

**LR-Net** is trained and evaluated on four large-scale datasets: Oxford_RobotCar, NUS In-house, MulRan and Kitti odometry.
We divided the datasets into: **Standardized datasets** and **Raw point cloud datasets**

**Standardized datasets:** 

Oxford RobotCar and In-house (U.S., R.A., B.D.) datasets introduced in
*PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition* paper ([link](https://arxiv.org/pdf/1804.03492)).

For dataset description see PointNetVLAD paper or github repository ([link](https://github.com/mikacuy/pointnetvlad)).

You can download training and evaluation datasets from 
[here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) 
([alternative link](https://drive.google.com/file/d/1-1HA9Etw2PpZ8zHd3cjrfiZa8xzbp41J/view?usp=sharing)). 

**Raw point cloud datasets:**

*The MulRan dataset([link](https://sites.google.com/view/mulran-pr/home))* consists of scans collected from the Ouster-64 sensor in various environments across South Korea, including the Dajeon Convention Center (DCC), Korea Advanced Institute of Science and Technology (KAIST), Riverside, and Sejong City. 
The dataset contains 12 sequences, 9 of which we used for evaluation. The training was carried out on the DCC01, DCC02, Riverside01, and Riverside02 sequences, while the remaining sequences of DCC, Riverside, and KAIST were used for evaluation.

*The Kitti odometry dataset([link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php))* contains 11 sequences of Velodyne HDL-64E LiDAR scans collected in Karlsruhe, Germany. 
We chose six sequences with revisits (00, 02, 05, 06, 07, and 08) as unseen test sets to assess the cross-dataset generalization ability of our method. 

#### It is important to note that the raw point cloud datasets were not subjected to any preprocessing operations, such as ground point removal or downsampling on the submaps.

## Results
The Training an Testing procedures, as well as experimental results, will be updated after the formal publication of the paper.

## Acknowledgement
Some of our code refers to [PointNetVLAD](https://github.com/mikacuy/pointnetvlad)


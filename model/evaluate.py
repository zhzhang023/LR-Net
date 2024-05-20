import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
import torch
import tqdm
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
import time
import matplotlib.pyplot as plt
from loading_pointclouds import *
import pc_util as util
import os
import sys
from scipy.spatial.transform import Rotation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

PRESENTATION=False

def evaluate(model,device,silent=True):
    assert len(cfg.eval_database_files) == len(cfg.eval_query_files)
    stats = {}
    for database_file, query_file in zip(cfg.eval_database_files, cfg.eval_query_files):
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)
        p = os.path.join(cfg.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(cfg.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, database_sets, query_sets, silent=silent)
        stats[location_name] = temp

    return stats

def evaluate_dataset(model, device, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    database_information = []
    query_information = []

    model.eval()
    time_means = []

    Nice_Precision=[]
    Nice_Recall=[]
    F1_max=[]

    for set in tqdm.tqdm(database_sets, disable=silent):
        # set=database_sets[set]
        latent_vector_temp, _,positions,timestamp= get_latent_vectors(model, set, device,b=True)
        database_embeddings.append(latent_vector_temp)
        database_inform_temp={}
        database_inform_temp['embedding'] = latent_vector_temp
        database_inform_temp['position'] = positions
        database_inform_temp['timestamp'] = timestamp
        database_information.append(database_inform_temp)

    for set in tqdm.tqdm(query_sets, disable=silent):
        latent_vector_temp, time_mean,positions,timestamp = get_latent_vectors(model, set, device,b=False)
        time_means.append(time_mean)
        query_embeddings.append(latent_vector_temp)
        query_inform_temp = {}
        query_inform_temp['embedding'] = latent_vector_temp
        query_inform_temp['position'] = positions
        query_inform_temp['timestamp'] = timestamp
        query_information.append(query_inform_temp)

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
            np_temp,nr_temp,f1_temp=get_pr_curve(i,j,database_information,query_information,query_sets)
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets)
            Nice_Precision.append(np_temp)
            Nice_Recall.append(nr_temp)
            F1_max.append(f1_temp)

            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    time_mean = np.mean(time_means)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity, 'time': time_mean}
    NP_final=np.average(Nice_Precision)
    NR_final=np.average(Nice_Recall)
    F1_final=np.average(F1_max)
    print('Precision:%f'%NP_final)
    print('Recall:%f' % NR_final)
    print('F1-max:%f' % F1_final)
    return stats


'''
#### Jitter and DownSample ####
'''
def jitter_pointcloud(cloud_data, sigma=0.05, clip=0.1):
    # 随机给点云增加噪声
    N, C = cloud_data.shape
    assert (clip > 0)
    # jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data = np.clip(np.random.normal(0.0, scale=sigma, size=(N, C)),
                            a_min=-1 * clip, a_max=clip)
    jittered_data += cloud_data
    return jittered_data


def random_sampling(pc, num_sample, color=None, replace=True, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if color is not None:
        assert color.shape[0] == pc.shape[0]
    if replace is None: replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        if color is not None:
            return pc[choices], color[choices], choices
        else:
            return pc[choices], choices
    else:
        if color is not None:
            return pc[choices], color[choices]
        else:
            return pc[choices]

def random_outlier(pc,num_noise):
    num_noise=int(num_noise)
    pc_max=np.max(pc,axis=0)
    pc_min=np.min(pc,axis=0)
    noise_data=np.ones((num_noise,3))
    for i in range(num_noise):
        noise_data[i,0]=np.random.uniform(pc_min[0],pc_max[0])
        noise_data[i, 1] = np.random.uniform(pc_min[1], pc_max[1])
        noise_data[i, 2] = np.random.uniform(pc_min[2], pc_max[2])
    new_data=np.concatenate((noise_data,pc),axis=0).tolist()
    np.random.shuffle(new_data)
    new_data=np.array(new_data)
    return new_data

'''
#### Jitter and DownSample ####
'''
'''
# Rotate around Z-axis
'''
def random_rotation(pc,factor):
    anglex = 0.0001 * np.pi
    angley = 0.0001 * np.pi
    # anglez = np.random.uniform() * np.pi / factor
    anglez = np.pi / factor

    rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
    target_cloud = rotation_ab.apply(pc)
    return target_cloud


def load_pc(file_name,tensor=True,b=False):
    # returns Nx3 matrix
    file_path = os.path.join(cfg.dataset_folder, file_name)
    pc = np.fromfile(file_path, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    assert pc.shape[0] == cfg.num_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    if cfg.Noise:
        pc = jitter_pointcloud(pc, sigma=cfg.Jitter_factor)
    if cfg.Density:
        pc = random_sampling(pc, cfg.DownSample_Num, replace=False)  # down_sample
        pc = random_sampling(pc, 4096, replace=True)  # up_sample
    if cfg.Outlier:
        pc = random_sampling(pc, 4096 - cfg.Outlier_num)
        pc = random_outlier(pc, cfg.Outlier_num)
        # pc = random_sampling(pc, 4096, replace=True)
    if not b:
        if cfg.Rotate:
            pc=random_rotation(pc,cfg.Factor)
    if tensor:
        pc = torch.tensor(pc, dtype=torch.float)
    return pc


def get_latent_vectors(model, set, device,b=False):
    # Adapted from original PointNetVLAD code

    model.eval()
    embeddings_l = []
    position_l=[]
    timestamp_l=[]
    time_use = []
    for elem_ndx in set:
        x = load_pc(set[elem_ndx]['query'],b=b)
        p = np.asarray([set[elem_ndx]["northing"], set[elem_ndx]["easting"]])
        t = set[elem_ndx]["timestamp"]
        with torch.no_grad():
            # coords are (n_clouds, num_points, channels) tensor
            # query_pc = x.numpy()
            # coords = get_Sparse_Tensor(query_pc)
            bcoords = sparse_collate_fn([{'coords':x}])
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            batch = bcoords['coords'].to(device)
            start = time.time()
            embedding = model(batch)
            end = time.time()
            # embedding is (1, 1024) tensor
            if cfg.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings
        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)
        time_temp = (end - start)
        time_use.append(time_temp)
        position_l.append(p)
        timestamp_l.append(t)
    embeddings = np.vstack(embeddings_l)
    positions = np.vstack(position_l)
    timestamp = np.vstack(timestamp_l)
    time_eval_mean = np.mean(time_use)
    return embeddings, time_eval_mean, positions, timestamp


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]  # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                    if PRESENTATION:
                        # Draw thr BaseMap
                        north = []
                        east = []
                        for w in range(len(database_sets[m])):
                            temp = database_sets[m][w]
                            temp_north = temp['northing']
                            temp_east = temp['easting']
                            north.append(temp_north)
                            east.append(temp_east)
                        fig = plt.figure(dpi=500,frameon=False)
                        ax = fig.gca()
                        # draw the figure, the color is r = read
                        figure = ax.plot(north, east, c='k')
                        # Get Query Position and Top 1 position
                        Query_temp = query_sets[n][i]
                        TopOne_Temp = database_sets[m][indices[0][j]]
                        # TopTwo_Temp = database_sets[m][indices[0][1]]
                        # TopThree_Temp = database_sets[m][indices[0][2]]

                        plt.scatter(Query_temp['northing'], Query_temp['easting'], color='r', marker='*', s=300)
                        plt.scatter(TopOne_Temp['northing'], TopOne_Temp['easting'],marker='o',facecolors='none',
                                    edgecolors='m', s=300, linewidths=3,alpha=1)
                        # plt.scatter(TopTwo_Temp['northing'], TopTwo_Temp['easting'], marker='o',facecolors='none',
                        #             edgecolors='b', s=300, linewidths=3,alpha=1)
                        # plt.scatter(TopThree_Temp['northing'], TopThree_Temp['easting'], marker='o',facecolors='none',
                        #             edgecolors='k', s=200, linewidths=2,alpha=1)

                        # plt.show()
                        ### Make File Path ###
                        Save_path_temp = os.path.join(BASE_DIR, 'PRESENTATION_SAVE')
                        if not os.path.exists(Save_path_temp):
                            os.mkdir(Save_path_temp)
                        Save_path_temp = os.path.join(Save_path_temp, cfg.Dataset)
                        if not os.path.exists(Save_path_temp):
                            os.mkdir(Save_path_temp)
                        file_name_temp = Query_temp['query'].split('/')[-1].split('.')[0]
                        Save_path_temp = os.path.join(Save_path_temp, file_name_temp)
                        if not os.path.exists(Save_path_temp):
                            os.mkdir(Save_path_temp)
                        pic_file_name_temp = file_name_temp + '.jpg'
                        Save_fig_dir = os.path.join(Save_path_temp, pic_file_name_temp)
                        plt.savefig(Save_fig_dir)
                        ### Save pc file ###
                        PC_query_temp = load_pc(Query_temp['query'],tensor=False)
                        PC_query_filedir = os.path.join(Save_path_temp, 'Query_PC.jpg')
                        PC_TopOne_temp = load_pc_file(TopOne_Temp['query'])
                        PC_TopOne_filedir = os.path.join(Save_path_temp, 'TopOne_PC.jpg')
                        # PC_TopTwo_temp = load_pc_file(TopTwo_Temp['query'])
                        # PC_TopTwo_filedir = os.path.join(Save_path_temp, 'TopTwo_PC.jpg')
                        # PC_TopThree_temp = load_pc_file(TopThree_Temp['query'])
                        # PC_TopThree_filedir = os.path.join(Save_path_temp, 'TopThree_PC.jpg')

                        util.pyplot_draw_point_cloud(PC_query_temp, PC_query_filedir)
                        util.pyplot_draw_point_cloud(PC_TopOne_temp, PC_TopOne_filedir)
                        # util.pyplot_draw_point_cloud(PC_TopTwo_temp, PC_TopTwo_filedir)
                        # util.pyplot_draw_point_cloud(PC_TopThree_temp, PC_TopThree_filedir)
                        # plt.scatter(TopTwo_Temp['northing'], TopTwo_Temp['easting'], marker='o',facecolors='none',
                        #             edgecolors='b', s=300, linewidths=3,alpha=1)
                        # plt.scatter(TopThree_Temp['northing'], TopThree_Temp['easting'], marker='o',facecolors='none',
                        #             edgecolors='k', s=200, linewidths=2,alpha=1)
                recall[j] += 1
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'TIME:: {:.3f}   Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['time'], stats[database_name]['ave_one_percent_recall'],
                       stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])

def get_pr_curve(m,n,database_information, query_information,query_sets,show_pr_curve=False):
    database_inf=database_information[m]
    query_inf=query_information[n]
    # 读取 database，包括编码后的 vectors，position 和 timestamp
    database_output = database_inf['embedding']
    database_position = database_inf['position']
    database_timestamp = database_inf['timestamp']
    # 读取 query，包括编码后的 vectors，position 和 timestamp
    queries_output = query_inf['embedding']
    queries_position = query_inf['position']
    queries_timestamp = query_inf['timestamp']
    # 设置预制 和 TP TN FP FN 数组
    thresholds = np.linspace(0.001, 1.0, int(1000))
    num_thresholds = len(thresholds)
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    # 建立树，用来找最近
    database_nbrs = KDTree(database_output)
    # 遍历query
    for i in range(len(queries_output)):
        # 当前query的一些信息
        query_details = query_sets[n][i]  # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        query_position_temp = queries_position[i]
        query_timestamp_temp = queries_timestamp[i]
        # 当前 query 在数据集中找最近
        '''
        问题可能在这里，计算distance有问题，应该是 cosine distance
        '''
        feat_dists = cdist(np.array([queries_output[i]]), database_output,
                           metric='cosine').reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
        # distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=25)
        # 记录一些参数,包括最近的特征距离，和最近的indice
        # min_dist=distances[0][0]
        # nearest_idx=indices[0][0]
        database_position_temp = database_position[nearest_idx]
        database_timestamp_temp = database_timestamp[nearest_idx]
        p_dist=np.linalg.norm(query_position_temp - database_position_temp)
        # 根据 thresholds 生成 pr—curve
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]
            if (min_dist < threshold):  # Positive Prediction
                # 如果此时还在3m内，就是True
                if p_dist <= 25:
                    num_true_positive[thres_idx] += 1
                # 20m外就是 False，那中间的？就不计算了？
                elif p_dist > 25:
                    num_false_positive[thres_idx] += 1
            else:  # Negative Prediction
                # 对于negative,判定不是3m 和 20m， 而是看在不在 is revisited
                # 这个 is revisited 是 存储的 json 文件中的
                if len(true_neighbors) == 0:
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1
    # 计算后续具体指标
    F1max = 0.0
    Precisions, Recalls = [], []
    Nice_Precision=0.0
    Nice_Racall=0.0
    for ithThres in range(num_thresholds):
        nTrueNegative = num_true_negative[ithThres]
        nFalsePositive = num_false_positive[ithThres]
        nTruePositive = num_true_positive[ithThres]
        nFalseNegative = num_false_negative[ithThres]

        Precision = 0.0
        Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)

            F1 = 2 * Precision * Recall * (1 / (Precision + Recall))

        if F1 > F1max:
            F1max = F1
            F1_TN = nTrueNegative
            F1_FP = nFalsePositive
            F1_TP = nTruePositive
            F1_FN = nFalseNegative
            F1_thresh_id = ithThres
            Nice_Precision = F1_TP / (F1_TP + F1_FP)
            Nice_Racall = F1_TP / (F1_TP + F1_FN)
        Precisions.append(Precision)
        Recalls.append(Recall)
    if show_pr_curve:
        plt.title('Seq: ' + 'test' +
                  '    F1Max: ' + "%.4f" % (F1max))
        plt.plot(Recalls, Precisions, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis([0, 1, 0, 1.1])
        plt.xticks(np.arange(0, 1.01, step=0.1))
        plt.grid(True)
        plt.show()
    return Nice_Precision,Nice_Racall,F1max

if __name__ == '__main__':
    print('start evaluating...')
    from model.LR_Core import LRCore
    import model.Config as cfg

    model = LRCore()
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    resume_filename=os.path.join(cfg.weights_path,'MulRan_NG_'+cfg.model+'_current_best.ckpt')

    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    starting_epoch = checkpoint['epoch']
    print('starting_epoch:%d'%starting_epoch)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    stats = evaluate(model,device,silent=False)
    print_eval_stats(stats)

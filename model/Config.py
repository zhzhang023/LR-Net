''' ## DataLoader ##'''
Datasets = ['Oxford','MulRan']
Index=0
Training_dataset=Datasets[Index]
if Training_dataset == 'Oxford':
    dataset_folder = '/media/joey/DATASET/PNVLAD_Dataset/benchmark_datasets/'
    train_file = 'training_queries_baseline.pickle'
elif Training_dataset == 'MulRan':
    dataset_folder = '/media/joey/DATASET/MulRan'
    train_file = 'training_queries_baseline.pickle'
else:
    raise NotImplementedError

val_file=None
num_points=4096

batch_size = 16
batch_size_limit = 80
batch_expansion_rate = 1.4
batch_expansion_th = 0.7

quantization_size = 0.01
num_workers=4
aug_mode = 1
''' ## DataLoader ##'''

''' ## Ablation ##'''
Use_CA=True
Use_LA=True
Use_Pre=True
Use_Pro=True
''' ## Ablation ##'''

''' ## Model ##'''
model='LR-Net'
feature_size=256
in_channel=3
drop=0.1
''' ## Model ##'''

''' ## Train ##'''
Resume=False
weights_path='/home/joey/Linux-Project-File/LR-Net/model_v2/weights'


lr = 1e-3
epochs = 60
scheduler='MultiStepLR'
scheduler_milestones = [30,50,70]
weight_decay = 1e-3

loss = 'BatchHardTripletMarginLoss'
normalize_embeddings = False
margin = 0.2
maigin_expansion_rate=1
margin_threshold=0.21

Save_model_every_epoch=True
''' ## Train ##'''

''' ## Eval ##'''
Dataset='DCC_25'
Noise=False
Jitter_factor=0.003
Density=False
DownSample_Num=50000
Outlier=False
Outlier_num=1000
Rotate=False
Factor=6

EVAL_DATABASE_DATASETS={'Oxford':'oxford_evaluation_database.pickle',
               'BS':'business_evaluation_database.pickle',
               'RA':'residential_evaluation_database.pickle',
               'US':'university_evaluation_database.pickle',
                # MulRan
                'DCC_25':'DCC_evaluation_database_25.pickle',
                'RiverSide_25':'RiverSide_evaluation_database_25.pickle',
                'KAIST_25':'KAIST_evaluation_database_25.pickle',
                # KITTI
                'Kitti_00':'00_evaluation_database.pickle',
                'Kitti_02':'02_evaluation_database.pickle',
                'Kitti_05':'05_evaluation_database.pickle',
                'Kitti_06':'06_evaluation_database.pickle',
                'Kitti_07':'07_evaluation_database.pickle',
                'Kitti_08':'08_evaluation_database.pickle',
                        }
EVAL_QUERY_DATASETS={'Oxford':'oxford_evaluation_query.pickle',
               'BS':'business_evaluation_query.pickle',
               'RA':'residential_evaluation_query.pickle',
               'US':'university_evaluation_query.pickle',
             # MulRan
             'DCC_25':'DCC_evaluation_query_25.pickle',
             'RiverSide_25':'RiverSide_evaluation_query_25.pickle',
             'KAIST_25':'KAIST_evaluation_query_25.pickle',
             # Kitti
             'Kitti_00':'00_evaluation_query.pickle',
             'Kitti_02':'02_evaluation_query.pickle',
             'Kitti_05': '05_evaluation_query.pickle',
             'Kitti_06': '06_evaluation_query.pickle',
             'Kitti_07':'07_evaluation_query.pickle',
             'Kitti_08': '08_evaluation_query.pickle',
             }

eval_database_files = [EVAL_DATABASE_DATASETS[Dataset]]
eval_query_files = [EVAL_QUERY_DATASETS[Dataset]]
''' ## Eval ##'''

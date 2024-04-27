import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from data.oxford import TrainTransform,TrainSetTransform
from torchsparse.utils.collate import sparse_collate_fn
from torch.utils.data import DataLoader
import model.Config as cfg
from data.samplers import BatchSampler

# assert cfg.Datasets[cfg.Index]=='MulRan'

def random_sampling(pc, num_sample, replace=None, return_choices=False,weight=None):
    """
    Input: N x C,
    output: num_sample x C
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    if weight is not None:
        weight=weight.detach().cpu().numpy()
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace,p=weight)
    if return_choices:
        return pc[choices], torch.from_numpy(choices)
    else:
        return pc[choices]

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

class MulRanDataset(Dataset):
    def __init__(self,dataset_path,query_filename,transform=None,set_transform=None):
        super(MulRanDataset, self).__init__()
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path,query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries = pickle.load(open(self.query_filepath, 'rb'))
        self.n_points = 30000
        print('{} queries in the dataset'.format(len(self)))
    def __len__(self):
        return len(self.queries)
    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = self.queries[ndx].rel_scan_filepath
        query_pc = self.load_pc(file_pathname)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        out_dict = {
            'pc': query_pc,
            'ndx': ndx,
        }
        return out_dict
    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.dataset_path, filename)
        pc = np.load(file_path)[...,:3]

        if pc.shape[0]!=self.n_points:
            pc=random_sampling(pc,self.n_points)
        pc = torch.tensor(pc, dtype=torch.float)
        return pc

def make_datasets(dataset_folder,train_file):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(1)
    train_set_transform = TrainSetTransform(1)

    datasets['train'] = MulRanDataset(dataset_folder, train_file, train_transform,
                                      set_transform=train_set_transform)
    val_transform = None
    if cfg.val_file is not None:
        datasets['val'] = MulRanDataset(cfg.dataset_folder, cfg.val_file, val_transform)
    return datasets

def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

def make_collate_fn(dataset, quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        batch = sparse_collate_fn(data_list)
        batch_sparsepc=batch['pc']
        labels=batch['ndx']

        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        return batch_sparsepc, positives_mask, negatives_mask

    return collate_fn

def make_mulran_dataloaders():
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(cfg.dataset_folder,cfg.train_file)

    dataloders = {}
    if cfg.Resume:
        weights_path=cfg.weights_path
        resume_filename = os.path.join(weights_path, cfg.Training_dataset+'_model_current_best.ckpt')
        checkpoint = torch.load(resume_filename)
        train_sampler=checkpoint['sampler']
    else:
        train_sampler = BatchSampler(datasets['train'], batch_size=cfg.batch_size,
                                     batch_size_limit=cfg.batch_size_limit,
                                     batch_expansion_rate=cfg.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  cfg.quantization_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=cfg.num_workers, pin_memory=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=cfg.batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'], cfg.quantization_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=cfg.num_workers, pin_memory=True)

    return dataloders

if __name__ == '__main__':
    print('running some test...')
    tt=make_mulran_dataloaders()
    for batch, positives_mask, negatives_mask in tt['train']:
        # batch is (batch_size, n_points, 3) tensor
        # labels is list with indexes of elements forming a batch
        batch_stats = {}
        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        tt=0
    tt=0

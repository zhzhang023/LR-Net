import os
import numpy as np
from torch.utils.data import DataLoader
from data.oxford import OxfordDataset, TrainTransform, TrainSetTransform
from data.samplers import BatchSampler
import model.Config as cfg
import torch
from torchsparse.utils.collate import sparse_collate_fn

def make_datasets():
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(cfg.aug_mode)
    train_set_transform = TrainSetTransform(cfg.aug_mode)

    datasets['train'] = OxfordDataset(cfg.dataset_folder, cfg.train_file, train_transform,
                                      set_transform=train_set_transform,v_size=cfg.quantization_size)
    val_transform = None
    if cfg.val_file is not None:
        datasets['val'] = OxfordDataset(cfg.dataset_folder, cfg.val_file, val_transform)
    return datasets

def make_collate_fn(dataset: OxfordDataset):
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

def make_dataloaders():
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets()

    dataloders = {}
    if cfg.Resume:
        weights_path=cfg.weights_path
        resume_filename = os.path.join(weights_path, cfg.Training_dataset+'_RIP-Net_current_best.ckpt')
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


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e


if __name__ == '__main__':
    print('running some test...')

    test_loader=make_dataloaders()
    for batch, positives_mask, negatives_mask in test_loader['train']:
        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        tt=batch['features']
        tt=0
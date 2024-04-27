import os
from datetime import datetime
import numpy as np
import torch
import pickle
import tqdm
import pathlib
import time

from torch.utils.tensorboard import SummaryWriter
from model.evaluate import evaluate, print_eval_stats
import model.Config as cfg
from model.loss import make_loss
from model.LR_Core import LRCore
# from model.LWR_AddSample import LWR_AddSample

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                       stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']))
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))

def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats

def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path

def do_train(dataloaders, debug=False, visualize=False):
    # Create model class
    s = get_datetime()
    model=LRCore()
    # 模型按照日期重命名
    model_name = 'model_' + cfg.model + '_' + s
    print('Model name: {}'.format(model_name))
    # 确认weight文件夹路径
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    # 验证 model 程序中是否有 print_info 属性，有就按照这个info输出，没有就输出模型的modelsize
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # 确定部署设备
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    print('Model device: {}'.format(device))

    loss_fn = make_loss()

    # Training elements
    if cfg.weight_decay is None or cfg.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler is None:
        scheduler = None
    else:
        if cfg.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs + 1,
                                                                   eta_min=cfg.min_lr)
        elif cfg.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(cfg.scheduler))

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

    now = datetime.now()
    logdir = os.path.join("../tf_logs", now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)

    ###########################################################################
    #
    ###########################################################################
    # 看下 dataloader中有没有加载 val
    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}

    best_rnz=10000
    best_loss=10000
    margin_expand=False
    # 是否继续训练
    if cfg.Resume:
        resume_filename = os.path.join(weights_path, cfg.Training_dataset+'_'+cfg.model+'_current_best.ckpt')
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch']
        model.load_state_dict(saved_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.scheduler_milestones, gamma=0.1)
        scheduler.load_state_dict(checkpoint['scheduler'])
        cfg.margin=checkpoint['margin']
        # dataloaders['train'].batch_sampler=checkpoint['sampler']
        print('starting_epoch:%d,current_margin:%f,current batch size:%d'%(starting_epoch,cfg.margin,dataloaders['train'].batch_sampler.batch_size))
        print('current lr:%f'%optimizer.state_dict()['param_groups'][0]['lr'])
    else:
        starting_epoch = 0
    # 开始遍历 epoch
    for epoch in tqdm.tqdm(range(starting_epoch, cfg.epochs + 1)):
        print('Epoch: %d'%epoch)
        print('current lr:%f'%optimizer.state_dict()['param_groups'][0]['lr'])
        # 这里注意一下，每个 epoch 里都要遍历 phases 中的内容。如果有 val 的话也要跑
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_stats = []  # running stats for the current epoch
            count_batches = 0
            # 遍历 dataloader ，开始训练
            for batch, positives_mask, negatives_mask in dataloaders[phase]:
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}
                # 这里有点意思，这个debug参数是原作设计，专门用于检验是否程序出错的，会控制只运行2个batch
                if debug and count_batches > 2:
                    break

                # batch = {e: batch[e].to(device) for e in batch}
                batch = batch.to(device)
                # 这里注意下，这个训练主要是在一个batch 中进行的，构造batch的时候，彼此互为 positive 和 negative
                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()
                # 如果没有 positive 或 negatie，这个batch就要被skip
                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()
                if visualize:
                    # visualize_batch(batch)
                    pass

                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    start = time.time()
                    embeddings = model(batch)
                    end = time.time()
                    time_temp = (end - start)/dataloaders['train'].batch_sampler.batch_size

                    loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = loss.item()
                    # print('Current batch:%d, Current loss: %f, time_each_data:%f, margin=%f'%(count_batches,loss.item(),time_temp,cfg.margin))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
                # ******* PHASE END *******
                # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

            # ******* EPOCH END *******

        if scheduler is not None:
            scheduler.step()

        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)

        if 'num_triplets' in stats['train'][-1]:
            nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
            if 'val' in phases:
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
            writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        elif 'num_pairs' in stats['train'][-1]:
            nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                          'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
            if 'val' in phases:
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
            writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

        if cfg.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < cfg.batch_expansion_th:
                    if dataloaders['train'].batch_sampler.batch_size >= cfg.batch_size_limit:
                        if cfg.margin <= cfg.margin_threshold:
                            margin_expand=True
                            cfg.margin=cfg.margin*cfg.maigin_expansion_rate
                            loss_fn=make_loss()
                        else:
                            margin_expand=False
                    dataloaders['train'].batch_sampler.expand_batch()
        # 每个epcoh都要保存模型，可以根据Triplet per batch 中 non-zero 的情况进行保存
        # 注意，还需要保存 当前的 batchsize 和 margin
        if cfg.Save_model_every_epoch:
            epoch_train_stats = stats['train'][-1]
            rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
            if rnz<best_rnz or epoch_train_stats['loss']<best_loss or margin_expand:
                best_rnz=rnz
                best_loss=epoch_train_stats['loss']
                current_best_model_path = os.path.join(weights_path, cfg.Training_dataset+'_'+cfg.model+'_current_best.ckpt')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                    'margin':cfg.margin,
                    'sampler':dataloaders['train'].batch_sampler
                },
                    current_best_model_path)
                print('Current best! model is saved in:'+current_best_model_path )

    print('')

    # Save final model weights
    final_model_path = model_pathname +'_'+cfg.Training_dataset+ '_final.ckpt'
    torch.save(model.state_dict(), final_model_path)

    stats = {'train_stats': stats}

    # Evaluate the final modelLWR
    model.eval()
    final_eval_stats = evaluate(model, device)
    print('Final model:')
    print_eval_stats(final_eval_stats)
    stats['eval'] = {'final': final_eval_stats}
    print('')

    # Pickle training stats and parameters
    # 这里注意下。training 过程中的网络参数，存储成的是 pickle 格式。
    pickle_path = model_pathname + '_'+cfg.Training_dataset+'_stats.pickle'
    pickle.dump(stats, open(pickle_path, "wb"))

    # Append key experimental metrics to experiment summary file
    _, model_name = os.path.split(model_pathname)
    prefix = "{}".format(model_name)
    export_eval_stats(cfg.Training_dataset+"_experiment_results.txt", prefix, final_eval_stats)

def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in ['oxford', 'university', 'residential', 'business']:
            ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)

if __name__ == '__main__':
    print('start trainning...')
    # Oxford
    from data.dataset_utils import make_dataloaders
    # NaverLabs
    from data.MulRan.MulRan import make_mulran_dataloaders

    if cfg.Training_dataset == 'Oxford':
        dataloaders = make_dataloaders()
    elif cfg.Training_dataset == 'MulRan':
        dataloaders = make_mulran_dataloaders()
    else:
        raise NotImplementedError
    print('Training model on Scene: '+cfg.Training_dataset)
    do_train(dataloaders,  debug=False, visualize=False)


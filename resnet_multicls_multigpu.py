# %%
import os, sys
from pathlib import Path
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from func.trainer import load_train_objs, prepare_dataloader, Trainer
from func.utility import loggergen

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank: int, 
         world_size: int, 
         save_every: int = None, 
         max_epochs: int = None, 
         batch_size: int = None):

    # create log
    prj_dir = Path('/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data')
    log_dir = prj_dir.joinpath('log')
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = loggergen(log_dir)

    # set up the data directory
    datadir= Path('/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification')
    train_dataset_dir = datadir.joinpath('train')
    # train_csv = datadir.joinpath('train.csv')
    train_csv = datadir.joinpath('train_select_1K.csv')
    
    logger.info(f'Dataset Directory: {datadir}')
    logger.info(f'Dataset CSV Name: {train_csv}')
    logger.info(f'Project Directory: {prj_dir}')
    
    # set up the parameters
    lr = 1e-3
    input_ch_ct = 4 # 3 or 4
    weight_decay = 1e-5
    max_epochs = 100
    val_interval = 3
    save_every = 1
    batch_size = 24
    num_workers = 7
    debug_size = 100
    debug_size = None

    mean = [0.0540, 0.0530, 0.0804, 0.0806]
    std = [0.1420, 0.0831, 0.1272, 0.1229]
    if isinstance(input_ch_ct, int): 
        mean = mean[:input_ch_ct]
    elif isinstance(input_ch_ct, list): 
        std = [std[ch] for ch in input_ch_ct]

    # setup the process groups
    logger.info('setup multiple GPUs')
    setup(rank, world_size)
    
    # prepare the dataloader 
    logger.info('generate training, validation datasets, and the model')
    train_ds, val_ds, model = load_train_objs(
                            input_ch_ct = input_ch_ct,
                            input_csv = train_csv, 
                            data_root = train_dataset_dir,
                            debug_size = debug_size, 
                            deterministic = True, 
                            mean = mean, std = std)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    train_loader = prepare_dataloader(train_ds, batch_size, num_workers)
    val_loader = prepare_dataloader(val_ds, batch_size, num_workers)
    trainer = Trainer(
                model = model, 
                train_loader = train_loader, 
                val_loader = val_loader,
                optimizer = optimizer, 
                gpu_id = rank, 
                max_epochs = max_epochs,
                val_interval = val_interval,
                data_dir = prj_dir)
    
    trainer.train(max_epochs)
    
    
    print('Finished Training')
    cleanup()

if __name__ == '__main__':
    # suppose we have 3 gpus
    world_size = 2
    mp.spawn(
        main,
        args=([world_size]),
        nprocs=world_size
    )
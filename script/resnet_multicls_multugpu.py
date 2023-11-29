# %%

import os, sys
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from datetime import datetime
now = datetime.now()

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from HPASCDataset import HPASCDataset

transform = transforms.Compose(
    [
    transforms.Resize((2048, 2048)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])  

datadir= Path('/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification')
train_dataset_dir = datadir.joinpath('train')
# train_csv = datadir.joinpath('train.csv')
train_csv = datadir.joinpath('train_select_1K.csv')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Net(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 509 * 509, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 19)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        data_dir: Path,
        val_interval: int, 
        save_every: int = None,
        max_epochs: int = 500,
        
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.val_interval = val_interval
        self.model = DDP(model, device_ids=[gpu_id])
        self.max_epochs = max_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = max_epochs)
        self.criterion = nn.BCEWithLogitsLoss()
        
        timestamp = now.strftime("%d_%m_%Y_%H_%M")
        self.data_dir = data_dir
        log_dir = self.data_dir.joinpath('log', timestamp)
        self.writer = SummaryWriter(log_dir)

        checkpoint_dir = self.data_dir.joinpath('checkpoints', timestamp)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.best_metric_epoch = -1
        self.epoch_loss_values = []
        self.min_val_loss = -1.0

    def _run_batch(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        step = 0
        for i, batch_data  in tqdm(enumerate(self.train_loader, 0)):
            step_start = time.time()
            inputs, labels = batch_data 
            inputs = inputs.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            loss = self._run_batch(inputs, labels)
            epoch_loss += loss.item()
            print(
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
            step += 1

        self.lr_scheduler.step()
        epoch_loss /= step
        self.epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if self.gpu_id == 0:
            self.writer.add_scalar('Loss/train', epoch_loss, epoch+1)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = self.checkpoint_dir.joinpath('best_checkpoint.pth')
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _run_val(self, epoch): 
        epoch_val_loss = 0.0
        best_epoch = 0
        self.model.eval()
        # val_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            val_step = 0
            for val_data in self.val_loader:
                val_inputs, val_labels = val_data
                val_inputs = val_inputs.to(self.gpu_id)
                val_labels = val_labels.to(self.gpu_id)
                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(val_outputs, val_labels)
                epoch_val_loss += val_loss.item()
                val_step += 1
        
        epoch_val_loss /= val_step
        self.writer.add_scalar('Loss/valid', epoch_val_loss, epoch+1)
        if  self.min_val_loss > epoch_val_loss and epoch > 0: 
            print(f'Validation Loss Decreased({self.min_val_loss:.6f}--->{epoch_val_loss:.6f}) \t Saving The Model')
            self.min_val_loss = epoch_val_loss
            self.best_metric_epoch = epoch + 1
            print(f'Best Metric Epoch: {self.best_metric_epoch}')
            # Saving State Dict
            ckp = self.model.module.state_dict()
            PATH = self.checkpoint_dir.joinpath('best_checkpoint.pth')
            torch.save(ckp, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        elif self.min_val_loss == -1.0:
            self.min_val_loss = epoch_val_loss
            print(self.min_val_loss)
            ckp = self.model.module.state_dict()
            PATH = self.checkpoint_dir.joinpath('best_checkpoint.pth')
            torch.save(ckp, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)
            if self.gpu_id == 0 and (epoch + 1) % self.val_interval == 0:
                self._run_val(epoch)
        print(self.best_metric_epoch, self.min_val_loss)


def cleanup():
    dist.destroy_process_group()

def load_train_objs(input_ch, debug_size):
    HPA_dataset = HPASCDataset(
                    input_csv = train_csv, 
                    root = train_dataset_dir, 
                    split = 'train', 
                    transform = transform,
                    input_ch = input_ch, 
                    n_class = 19, 
                    debug_size = debug_size,
                    )
    torch.manual_seed(1947)
    train_ds, val_ds = random_split(HPA_dataset, [0.9, 0.1])
    print("train_ds size:", len(train_ds))
    print("val_ds size:", len(val_ds))

    model = Net(input_ch)
    return train_ds, val_ds, model

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, 
         world_size: int, 
         save_every: int = None, 
         max_epochs: int = None, 
         batch_size: int = None):

    lr = 1e-3
    input_ch = 4 # 3 or 4
    weight_decay = 1e-5
    max_epochs = 100
    val_interval = 3
    save_every = 1
    batch_size = 24
    num_workers = 7
    data_dir = Path('/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data')
    debug_size = None
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    train_ds, val_ds, model = load_train_objs(input_ch, debug_size)
    
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
                data_dir = data_dir)
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
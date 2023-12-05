from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
now = datetime.now()

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, ToTensor, Resize, v2

from torch.utils.tensorboard import SummaryWriter

from func.HPASCDataset import HPASCDataset, TrDataset
from func.nets import simple_net, custom_resnet

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
        
        timestamp = now.strftime("%Y_%m_%d_%H_%M")
        self.data_dir = data_dir
        log_dir = self.data_dir.joinpath('training_log', timestamp)
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
        b_sz = len(next(iter(self.train_loader))['image'][0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        step = 0
        for i, batch_data  in tqdm(enumerate(self.train_loader, 0)):
            step_start = time.time()
            # inputs, labels = batch_data
            inputs = batch_data['image']
            labels = batch_data['label']
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
                # val_inputs, val_labels = val_data
                val_inputs = val_data['image']
                val_labels = val_data['label']
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

def load_train_objs(input_ch_ct, 
                    input_csv, 
                    data_root,
                    # transform_compose = 'default', 
                    split_ratio = [0.8, 0.1, 0.1], 
                    n_class = 19, 
                    sample_size = None, 
                    deterministic = False,
                    mean = None, std = None, 
                    model_input_size = (1024, 1024)
                    ):

    if mean is None: mean = 0.5
    if std is None: mean = 0.5

    tf_dataset = Compose(
        [
            ToTensor(), 
            v2.Resize(size = model_input_size, antialias = True),
        ]
    )

    HPA_dataset = HPASCDataset(
                        input_csv = input_csv, 
                        root = data_root, 
                        input_ch_ct = input_ch_ct,
                        transform = tf_dataset, 
                        n_class = n_class, 
                        sample_size = sample_size,
                        )
    
    if deterministic: torch.manual_seed(1947)
    train_ds, val_ds, test_ds = random_split(HPA_dataset, split_ratio)
    
    tv_v1_train = Compose(
            [   
                v2.RandomResizedCrop(size = model_input_size, antialias=True),
                v2.RandomHorizontalFlip(p = 0.5),
                v2.RandomVerticalFlip(p = 0.5),
                v2.RandomRotation(degrees=(0, 180)),
                v2.Normalize(mean = mean, std = std),
            ]
        )
        
    tv_v1_val = Compose(
            [   
                v2.Resize(size = model_input_size, antialias = True),
                v2.Normalize(mean = mean, std = std),
            ]
        )

    train_ds = TrDataset(train_ds, tv_v1_train)
    val_ds = TrDataset(val_ds, tv_v1_val)
    test_ds = TrDataset(test_ds, tv_v1_val)

    print("train_ds size:", len(train_ds))
    print("val_ds size:", len(val_ds))
    print("val_ds size:", len(test_ds))
    
    # load model
    # model = simple_net(input_ch_ct, model_input_size)
    model = custom_resnet(input_ch_ct)
    return train_ds, val_ds, test_ds, model

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int, sampler = 'DDP'):
    if sampler == 'DDP':
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            pin_memory = True,
            num_workers = num_workers,
            shuffle = False,
            sampler = DistributedSampler(dataset)
        )
    elif sampler is None:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            pin_memory = True,
            num_workers = num_workers,
            shuffle = False,
        )
    return dataloader
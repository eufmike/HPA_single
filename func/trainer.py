from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
from typing import Union

now = datetime.now()

from pprint import pformat

import torch
import torchvision

torchvision.disable_beta_transforms_warning()
import torch.nn as nn
import torch.optim as optim

torch.set_float32_matmul_precision("high")

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, ToTensor, Resize, v2

from torch.utils.tensorboard import SummaryWriter

from func.utility import loggergen
from func.HPASCDataset import HPASCDataset, TrDataset
from func.nets import simple_net, custom_resnet


def get_value_channel(value, input_ch_ct, default):
    # print(input_ch_ct)
    if value is not None:
        if isinstance(input_ch_ct, int):
            value = value[:input_ch_ct]
        elif isinstance(input_ch_ct, list):
            value = [value[ch] for ch in input_ch_ct]
        else:
            raise ValueError("input_ch_ct must be an integer or a list of integers")
    else:
        return default
    return value


class Train_Project:
    def __init__(
        self,
        config: dict,
    ) -> None:

        self.config = config

        try:
            self.prjdir = Path(self.config["dir"]["project"])
        except KeyError:
            raise ValueError("project directory must be provided in the config file")

        # create log
        self.log_dir = self.prjdir.joinpath("log")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.logger = loggergen(self.log_dir)

        self.logger.info(pformat(self.config))

        try:
            self.dataload_config = self.config["train"]["dataload"]
        except KeyError:
            raise ValueError("dataload config must be provided in the config file")

        self.input_ch_ct = self.dataload_config.get("input_ch_ct")
        self.ch_mean = get_value_channel(
            self.dataload_config.get("mean"), self.input_ch_ct, 0.5
        )
        self.ch_std = get_value_channel(
            self.dataload_config.get("std"), self.input_ch_ct, 0.25
        )

        try:
            self.dataset_csv = Path(self.dataload_config.get("dataset_csv"))
        except KeyError:
            raise ValueError("dataset_csv must be provided in the config file")

        try:
            self.datadir = Path(self.config["dir"]["data"])
        except KeyError:
            raise ValueError("data directory must be provided in the config file")

        self.train_datasetdir = self.datadir.joinpath("train")

        if not self.dataset_csv.is_absolute():
            self.dataset_csv = self.datadir.joinpath(self.dataset_csv)

        # load_train_obj
        self.split_ratio = self.dataload_config.get("split_ratio", [0.8, 0.1, 0.1])
        self.n_class = self.dataload_config.get("n_class", 19)
        self.sample_size = self.dataload_config.get("sample_size")
        self.deterministic_status = self.dataload_config.get("deterministic")
        self.model_input_size = tuple(self.dataload_config.get("model_input_size"))
        # dataloader
        self.num_workers = self.dataload_config.get("num_workers", 8)
        self.batch_size = self.dataload_config.get("batch_size", 10)
        self.sampler = self.dataload_config.get("sampler")
        return

    def load_train_objs(
        self,
        # deterministic = False,
        # model_input_size = (1024, 1024),
    ):

        tf_dataset = Compose(
            [
                ToTensor(),
                v2.Resize(size=self.model_input_size, antialias=True),
            ]
        )

        HPA_dataset = HPASCDataset(
            input_csv=self.dataset_csv,
            root=self.train_datasetdir,
            input_ch_ct=self.input_ch_ct,
            transform=tf_dataset,
            n_class=self.n_class,
            sample_size=self.sample_size,
        )

        if self.deterministic_status:
            torch.manual_seed(1947)
        train_ds, val_ds, test_ds = random_split(HPA_dataset, self.split_ratio)

        tv_v1_train = Compose(
            [
                v2.RandomResizedCrop(size=self.model_input_size, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(0, 180)),
                v2.Normalize(mean=self.ch_mean, std=self.ch_std),
            ]
        )

        tv_v1_val = Compose(
            [
                v2.Resize(size=self.model_input_size, antialias=True),
                v2.Normalize(mean=self.ch_mean, std=self.ch_std),
            ]
        )

        self.train_ds = TrDataset(train_ds, tv_v1_train)
        self.val_ds = TrDataset(val_ds, tv_v1_val)
        self.test_ds = TrDataset(test_ds, tv_v1_val)

        self.logger.info(f"train_ds size: {len(train_ds)}")
        self.logger.info(f"val_ds size: {len(val_ds)}")
        self.logger.info(f"test_ds size: {len(test_ds)}")

        # load model
        # model = simple_net(input_ch_ct, model_input_size)
        self.model = custom_resnet(self.input_ch_ct)
        self.model = torch.compile(self.model)
        return

    def _prepare_dataloader(self, dataset, batch_size, num_workers):
        self.logger.info(f"batch_size:{batch_size}")
        self.logger.info(f"num_workers:{num_workers}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=True,
        )
        return dataloader

    def prepare_dataloader(self):
        self.train_loader = self._prepare_dataloader(
            self.train_ds,
            self.batch_size,
            self.num_workers,
        )
        self.val_loader = self._prepare_dataloader(
            self.val_ds, self.batch_size, self.num_workers
        )
        self.logger.info(len(self.train_loader))
        self.logger.info(len(self.val_loader))
        return


class Trainer:
    def __init__(self, train_prj: Train_Project, gpu_device: list):

        # load from train_prj
        self.train_loader = train_prj.train_loader
        self.val_loader = train_prj.val_loader
        self.datadir = train_prj.datadir
        self.logger = train_prj.logger

        timestamp = now.strftime("%Y_%m_%d_%H_%M")
        self.tblog_dir = train_prj.prjdir.joinpath("training_tblog", timestamp)
        self.writer = SummaryWriter(self.tblog_dir)
        print(self.tblog_dir)

        self.checkpoint_dir = train_prj.prjdir.joinpath("checkpoints", timestamp)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(self.checkpoint_dir)

        # load train config
        try:
            self.datatrain_config = train_prj.config["train"]["datatrain"]
        except KeyError:
            raise ValueError("datatrain config must be provided in the config file")

        # get device
        self.gpu_count = train_prj.config.get("GPU_num", 1)

        if gpu_device is None:
            if self.gpu_count > len(gpu_device):
                self.gpu_count = len(gpu_device)
            self.device = gpu_device[: self.gpu_count]
        else:
            self.device = [0]
        # load model and send to device
        self.device = self.device[0]  # only for this code to call single GPU
        self.model = train_prj.model.to(self.device)

        self.lr = float(self.datatrain_config.get("lr", 1e-3))
        self.weight_decay = float(self.datatrain_config.get("weight_decay", 1e-5))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.save_every = self.datatrain_config.get("save_every")
        self.val_interval = self.datatrain_config.get("val_interval")
        self.max_epochs = self.datatrain_config.get("max_epochs")
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_metric_epoch = -1
        self.epoch_loss_values = []
        self.min_val_loss = -1.0
        return

    def _run_batch(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        self.model.train()
        b_sz = len(next(iter(self.train_loader))["image"])
        self.logger.info(
            f"[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}"
        )
        # self.train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        step = 0
        for i, batch_data in tqdm(enumerate(self.train_loader)):
            step_start = time.time()
            # inputs, labels = batch_data
            inputs = batch_data["image"]
            labels = batch_data["label"]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            loss = self._run_batch(inputs, labels)
            epoch_loss += loss.item()
            self.logger.info(
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
            step += 1

        self.lr_scheduler.step()
        epoch_loss /= step
        self.epoch_loss_values.append(epoch_loss)
        self.logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        self.writer.add_scalar("Loss/train", epoch_loss, epoch + 1)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        ckppath = self.checkpoint_dir.joinpath("best_checkpoint.pth")
        torch.save(ckp, ckppath)
        self.logger.info(f"Epoch {epoch} | Training checkpoint saved at {ckppath}")

    def _run_val(self, epoch):
        self.logger.info("start validation")
        epoch_val_loss = 0.0
        self.best_epoch = 0
        self.model.eval()
        # val_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            val_step = 0
            for val_data in self.val_loader:
                # val_inputs, val_labels = val_data
                val_inputs = val_data["image"]
                val_labels = val_data["label"]
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)
                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(val_outputs, val_labels)
                epoch_val_loss += val_loss.item()
                val_step += 1

        epoch_val_loss /= val_step
        self.writer.add_scalar("Loss/valid", epoch_val_loss, epoch + 1)
        if self.min_val_loss > epoch_val_loss and epoch > 0:
            self.logger.info(
                f"Validation Loss Decreased({self.min_val_loss:.6f}--->{epoch_val_loss:.6f}) \t Saving The Model"
            )
            self.min_val_loss = epoch_val_loss
            self.best_metric_epoch = epoch + 1
            self.logger.info(f"Best Metric Epoch: {self.best_metric_epoch}")
        elif self.min_val_loss == -1.0:
            self.min_val_loss = epoch_val_loss
            self.logger.info(self.min_val_loss)
        else:
            return
        ckp = self.model.state_dict()
        ckppath = self.checkpoint_dir.joinpath("best_checkpoint.pth")
        torch.save(ckp, ckppath)
        self.logger.info(f"Epoch {epoch} | Training checkpoint saved at {ckppath}")
        self.best_epoch = epoch
        return

    def train(self):
        for epoch in range(self.max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)
            if (epoch + 1) % self.val_interval == 0:
                self._run_val(epoch)
        self.logger.info(self.best_metric_epoch, self.min_val_loss)

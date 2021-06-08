import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from PIL import Image
from PyTorch_VAE.models import BaseVAE
from PyTorch_VAE.models.types_ import *
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from PyTorch_VAE.utils import data_loader
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from Models.datasets import ConcatDataset, HolyGrailDataset


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, dataset_path, auto_crop=False, channel="*",split = 0.8) -> None:
        super(VAEXperiment, self).__init__()
        self.save_hyperparameters()
        self.auto_crop = auto_crop
        self.channel = channel
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.split = split
        self.labels_df = params['df_path']
        self.img_size  = params['img_size']
        self.datasets_parent_dir = dataset_path
        # print(f"img size as params in VAEexp {params['img_size']}")
        self.img_transforms = [transforms.functional.to_tensor, transforms.CenterCrop((256, 128)),
                               transforms.Resize((self.img_size[0], 128)),transforms.Normalize((0.41,0.42,0.42),(0.07,0.07,0.07))]
        if params['auto_crop'] == "True" or params['auto_crop']:
            self.img_transforms.append(transforms.RandomCrop((self.img_size[0], self.img_size[1])))
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def prepare_data(self):
        if self.params['dataset'] == 'nisuyGavia':
            index_train, index_val = train_test_split(self.labels_df[self.labels_df['train']].index, train_size=0.9,
                                                      random_state=4)
            self.train_dataset = HolyGrailDataset(self.labels_df.loc[index_train], self.datasets_parent_dir,
                                                  auto_crop=self.auto_crop, channel=self.channel, load2ram=False)
            self.val_origin_dataset = HolyGrailDataset(self.labels_df.loc[index_val], self.datasets_parent_dir,
                                                       auto_crop=self.auto_crop, channel=self.channel, load2ram=False)
            self.test_sens_dataset = HolyGrailDataset(self.labels_df[self.labels_df.train_sens == False],
                                                      self.datasets_parent_dir, auto_crop=self.auto_crop,
                                                      channel=self.channel, load2ram=False)
            self.test_time_dataset = HolyGrailDataset(self.labels_df[self.labels_df.train_time == False],
                                                      self.datasets_parent_dir, auto_crop=self.auto_crop,
                                                      channel=self.channel, load2ram=False)
            self.val_dataset = ConcatDataset(self.val_origin_dataset, self.test_sens_dataset, self.test_time_dataset)
        elif self.params['dataset'] == 'allData':
            self.train_dataset = torchvision.datasets.ImageFolder(self.params['data_path'],transforms.Compose(self.img_transforms))
            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            thr = int(np.floor(self.split * num_train))
            train_idx, valid_idx = indices[:thr], indices[thr:]
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.logger.experiment.log({key + "_valid": val.item() for key, val in val_loss.items()})
        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        # transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'nisuyGavia' or self.params['dataset'] == 'allData':
            dataset = self.train_dataset
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'], num_workers=64,
                          shuffle=False,sampler = self.train_sampler,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        # transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader = DataLoader(CelebA(root=self.params['data_path'],
                                                       split="test",
                                                       transform=transform,
                                                       download=False),
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)

        elif self.params['dataset'] == 'nisuyGavia':
            self.sample_dataloader = DataLoader(self.val_origin_dataset, batch_size=self.params['batch_size'],
                                                num_workers=64,
                                                shuffle=True,
                                                drop_last=True)
        elif self.params['dataset'] == 'allData':
            self.sample_dataloader = DataLoader(self.train_dataset,
                          batch_size=self.params['batch_size'], num_workers=64,
                          shuffle=False,sampler = self.valid_sampler,
                          drop_last=True)
        else:
            raise ValueError('Undefined dataset type')
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform

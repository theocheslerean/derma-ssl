'''
    Author: Theodor Cheslerean-Boghiu
    Date: May 26th 2023
    Version 1.0
'''
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

import torchmetrics as tm
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed

from torchvision import models

class SSL_Model(pl.LightningModule):
    def __init__(
            self,
            batch_size: int = 64,
            logits_size: int = 128,
            vit_type: str = 'vit_small_patch16_224',
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            loss_temperature: float = 0.1,
            world_size: int = 1,
            class_weights: list = None,
        ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.world_size = world_size
        self.batch_size = batch_size
        
        self.encoder = timm.create_model(vit_type, pretrained=True, global_pool='token', drop_rate=0.25)
        self.n_latent_features = self.encoder.head.in_features
        self.encoder.head = torch.nn.Identity()

        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear1',torch.nn.Linear(in_features=self.n_latent_features, out_features=4*self.n_latent_features)),
            ('silu1',torch.nn.SiLU()),
            ('linear2',torch.nn.Linear(in_features=4*self.n_latent_features, out_features=logits_size))]))
               
        self.loss = losses.NTXentLoss(temperature=loss_temperature)
        
        if world_size > 1:
            self.loss = distributed.DistributedLossWrapper(self.loss, efficient=True)
        
        self.indices = torch.arange(0, batch_size)
        self.labels = torch.cat((self.indices, self.indices))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        # extract the class token as the image representation that will be used for the projection and similarity computation step
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        a, b = batch
        
        _, a_image, _ = a
        _, b_image, _ = b
               
        a_image_feats = self(a_image)
        b_image_feats = self(b_image)
        
        embeddings = torch.cat((a_image_feats, b_image_feats))
        loss = self.loss(embeddings, self.labels)
        
        self.log("train/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        a, b = batch
        
        _, a_image, _ = a
        _, b_image, _ = b
        
        a_image_feats = self(a_image)
        b_image_feats = self(b_image)
        
        # pytorch metric learning loss
        embeddings = torch.cat((a_image_feats, b_image_feats))
        loss = self.loss(embeddings, self.labels)
        
        self.log("val/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        #TODO: add an additional parameter in the constructor to provide the optimizer from CLI
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=25, T_mult=1, eta_min=self.learning_rate/100, verbose=True)
        return [optimizer], [scheduler]
    
class SL_Model(pl.LightningModule):
    def __init__(
            self,
            batch_size: int = 256,
            logits_size: int = 128,
            vit_type: str = 'vit_small_patch16_224',
            learning_rate: float = 1e-4,
            labels: list = None,
            class_weights: list = None,
            ssl_pretrained: bool=False,
            ckpt_path: str = None
        ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.labels = labels
        
        self.encoder = timm.create_model(vit_type, pretrained=True, global_pool='token', drop_rate=0.25)
        self.n_latent_features = self.encoder.head.in_features
        self.encoder.head = torch.nn.Identity()

        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear1',torch.nn.Linear(in_features=self.n_latent_features, out_features=logits_size))]))
        
        if ssl_pretrained:
            print(f"Loading pretrained model")
            assert ckpt_path is not None, "Please provide a checkpoint file path to load from."
            pretrained_model = torch.load(ckpt_path)
            split_state_dict = {'encoder' : {}, 'classifier' : {}}
            for key, value in pretrained_model['state_dict'].items():
                if key.split('.')[0] == 'encoder':
                    split_state_dict['encoder']['.'.join(key.split('.')[1:])] = value
                elif key.split('.')[0] == 'classifier':
                    split_state_dict['classifier']['.'.join(key.split('.')[1:])] = value
            
            self.encoder.load_state_dict(split_state_dict['encoder'])

        self.loss = nn.CrossEntropyLoss(torch.tensor(class_weights))
        
        macro_metrics = tm.MetricCollection({
                "acc": MulticlassAccuracy(num_classes=logits_size, average='micro'),
                "auc": MulticlassAUROC(num_classes=logits_size, average='macro'),
            })
        per_class_metrics = tm.MetricCollection({
                "per_class_acc": MulticlassAccuracy(num_classes=logits_size, average='none'),
                "per_class_auc": MulticlassAUROC(num_classes=logits_size, average='none'),
            })
        self.macro_train_metrics = macro_metrics.clone(prefix="train/")
        self.macro_val_metrics = macro_metrics.clone(prefix="val/")
        
        self.per_class_train_metrics = per_class_metrics.clone(prefix="train/")
        self.per_class_val_metrics = per_class_metrics.clone(prefix="val/")

    def forward(self, x):
        embedding = self.encoder(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx):
        _, image, label = batch
               
        output = self(image)
        
        loss = self.loss(output, label)
        
        self.macro_train_metrics(output, label)
        self.per_class_train_metrics(output, label)
        
        self.log("train/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
        return loss

    def on_train_epoch_end(self):
        per_class_metrics = self.per_class_train_metrics.compute()

        for key, value in per_class_metrics.items():
            metrics_dict = {}
            for i, metric in enumerate(value):
                metrics_dict[str(self.labels[i])] = metric 
            self.logger.experiment.add_scalars(str(key),
                        metrics_dict,
                        global_step=self.current_epoch)
        
        self.log_dict(self.macro_train_metrics.compute(),
            prog_bar=False,
            logger=True)
        self.per_class_train_metrics.reset()
        self.macro_train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        _, image, label = batch
        
        output = self(image)
        
        loss = self.loss(output, label)
        
        self.macro_val_metrics(output, label)
        self.per_class_val_metrics(output, label)
        
        self.log("val/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
        return loss

    def on_validation_epoch_end(self):
        per_class_metrics = self.per_class_val_metrics.compute()

        for key, value in per_class_metrics.items():
            metrics_dict = {}
            for i, metric in enumerate(value):
                metrics_dict[str(self.labels[i])] = metric 
            self.logger.experiment.add_scalars(str(key),
                        metrics_dict,
                        global_step=self.current_epoch)
            
        self.log_dict(self.macro_val_metrics.compute(),
            prog_bar=True,
            logger=True)
        self.per_class_val_metrics.reset()
        self.macro_val_metrics.reset()

    def configure_optimizers(self):
        #TODO: add an additional parameter in the constructor to provide the optimizer from CLI
        lr_dict = [
            {
                'params' : self.encoder.parameters(),
                'lr' : 1e-5
            },
            {
                'params' : self.classifier.parameters()
            }
        ]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer#, [scheduler]

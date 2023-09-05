'''
    Author: Theodor Cheslerean-Boghiu
    Date: May 26th 2023
    Version 1.0
'''
from collections import OrderedDict

import copy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

import torchmetrics as tm
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC

class ClassificationModel(pl.LightningModule):
    def __init__(
            self,
            batch_size: int = 256,
            logits_size: int = 128,
            learning_rate: float = 1e-4,
            labels: list = None,
            class_weights: list = None,
            ssl_pretrained: bool=False,
            ckpt_path: str = None
        ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.labels = labels
        
        print(f'model: output size {logits_size}')
               
        self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)

        split_state_dict = {}
        if ssl_pretrained:
            # assert ckpt_path is not None, "Please provide a checkpoint file path to load from."
            print(f"Loading pretrained model from {'/u/home/boghiu/github/dino/lightning_logs/version_19097/checkpoints/last.ckpt'}")
            pretrained_model = torch.load('/u/home/boghiu/github/dino/lightning_logs/version_19097/checkpoints/last.ckpt')
            split_state_dict = {'encoder' : {}, 'encoder_classifier' : {}}
            for key, value in pretrained_model['state_dict'].items():
                if key.split('.')[0] == 'teacher_backbone':
                    split_state_dict['encoder']['.'.join(key.split('.')[1:])] = value
                
            self.encoder.load_state_dict(split_state_dict['encoder'])

        for param in self.encoder.parameters():
            param.requires_grad = True

        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(in_features=self.encoder.embed_dim, out_features=logits_size))]))
            
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.classifier.apply(init_weights)

        self.loss = nn.CrossEntropyLoss(torch.tensor(class_weights))
        
        macro_metrics = tm.MetricCollection({
                "acc": MulticlassAccuracy(num_classes=logits_size, average='macro'),
                "auc": MulticlassAUROC(num_classes=logits_size, average='macro'),
                "rec": MulticlassRecall(num_classes=logits_size, average='macro'),
                "pre": MulticlassPrecision(num_classes=logits_size, average='macro'),
            })
        per_class_metrics = tm.MetricCollection({
                "per_class_acc": MulticlassAccuracy(num_classes=logits_size, average='none'),
                "per_class_auc": MulticlassAUROC(num_classes=logits_size, average='none'),
                "per_class_rec": MulticlassRecall(num_classes=logits_size, average='none'),
                "per_class_pre": MulticlassPrecision(num_classes=logits_size, average='none'),
            })
        self.macro_train_metrics = macro_metrics.clone(prefix="train/")
        self.macro_val_metrics = macro_metrics.clone(prefix="val/")
        
        self.per_class_train_metrics = per_class_metrics.clone(prefix="train/")
        self.per_class_val_metrics = per_class_metrics.clone(prefix="val/")

    def forward(self, x):
        embedding = self.encoder(x).flatten(start_dim=1)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

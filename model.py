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

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed

from torchvision import models

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from tformer_models.transformer import drop_path


class DINO_Model(pl.LightningModule):
    def __init__(
            self,
            batch_size: int = 64,
            logits_size: int = 128,
            vit_type: str = 'dino_vits16',
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


        self.student = torch.hub.load('facebookresearch/dino:main', vit_type, pretrained=True)
        self.n_latent_features = self.student.embed_dim
        self.student.fc = torch.nn.Identity()
        self.student.head = torch.nn.Identity()

        state_dict = {}
        model_timm = timm.create_model('vit_base_patch16_224', pretrained=True, global_pool='token', drop_rate=0.3)
        for key, _ in self.student.state_dict().items():
            state_dict[key] = model_timm.state_dict()[key]
        self.student.load_state_dict(state_dict)

        self.teacher = copy.deepcopy(self.student)

        self.student_classifier = DINOProjectionHead(
            self.n_latent_features, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_classifier = DINOProjectionHead(
            self.n_latent_features, 512, 64, 2048)

        self.loss = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
            
        deactivate_requires_grad(self.teacher)
        deactivate_requires_grad(self.teacher_classifier)

    def forward(self, x):
        y = self.student(x).flatten(start_dim=1)
        return self.student_classifier(y)

    def forward_teacher(self, x):
        y = self.teacher(x).flatten(start_dim=1)
        return self.teacher_classifier(y)


    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student, self.teacher, m=momentum)
        update_momentum(self.student_classifier, self.teacher_classifier, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.loss(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def on_after_backward(self):
        self.student_classifier.cancel_last_layer_gradients(current_epoch=self.current_epoch)


    def validation_step(self, batch, batch_idx):
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.loss(teacher_out, student_out, epoch=self.current_epoch)
        self.log("val/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return [optimizer]#, [scheduler]
    

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

class DINO_SL_Model(pl.LightningModule):
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
        
        self.encoder = torch.hub.load('facebookresearch/dino:main', vit_type, pretrained=True)
        self.n_latent_features = self.encoder.embed_dim
        self.encoder.fc = torch.nn.Identity()
        self.encoder.head = torch.nn.Identity()

        # Loading weights from the timm pretrained model as those are trained in a supervised manner on imagenet 21k
        state_dict = {}
        model_timm = timm.create_model('vit_base_patch16_224', pretrained=True, global_pool='token', drop_rate=0.2)
        for key, _ in self.encoder.state_dict().items():
            state_dict[key] = model_timm.state_dict()[key]
        self.encoder.load_state_dict(state_dict)

        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(in_features=768, out_features=logits_size))]))
            
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.classifier.apply(init_weights)
        
        if ssl_pretrained:
            assert ckpt_path is not None, "Please provide a checkpoint file path to load from."
            print(f"Loading pretrained model from {ckpt_path}")
            pretrained_model = torch.load(ckpt_path)
            split_state_dict = {'encoder' : {}, 'encoder_classifier' : {}}
            for key, value in pretrained_model['state_dict'].items():
                if key.split('.')[0] == 'teacher':
                    split_state_dict['encoder']['.'.join(key.split('.')[1:])] = value
                elif key.split('.')[0] == 'student_classifier':
                    split_state_dict['encoder_classifier']['.'.join(key.split('.')[1:])] = value
            
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        return optimizer

class DINO_SL_Model_v2(pl.LightningModule):
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
        # self.encoder = timm.create_model('vit_small_patch16_224', pretrained=True, global_pool='token')
        # self.encoder.embed_dim = self.encoder.head.in_features
        # self.encoder.head = torch.nn.Identity()
        # if ssl_pretrained:
        #     pretrained_model = torch.load('/u/home/boghiu/github/dino/lightning_logs/version_19097/checkpoints/last.ckpt')
        #     self.encoder.load_state_dict(pretrained_model['state_dict'])

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

import math
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import torchmetrics

import factories
import vision_modules as vnn
import evaluation
import retrieval_metrics as rm
import triplet_margin_loss

class BasicCnnEncoder(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super(BasicCnnEncoder, self).__init__()

        input_channels = 3

        block = vnn.ConvNormAct

        stem_out_channels = 16
        stage_count = 3
        stage_channels = [hidden_size//4, hidden_size//2, hidden_size]
        stage_kernel_sizes = [3, 3, 3]
        stage_strides = [2, 2, 2]
        stage_paddings = [0, 0, 0]
        final_channels = hidden_size * 2

        stem = block(input_channels, stem_out_channels, 3, 
                stride=1, padding=1)

        in_channels = stem_out_channels

        body = nn.ModuleList()
        for i in range(stage_count):
            out_channels = stage_channels[i]
            kernel_size = stage_kernel_sizes[i]
            stride = stage_strides[i]
            padding = stage_paddings[i]

            body.append(block(
                    in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding))
            in_channels = out_channels
        body.append(vnn.ConvNormAct(in_channels, final_channels, 1))

        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()

        self.net = nn.Sequential(
                stem,
                *body,
                pool,
                flatten)

        self.embedding_size = final_channels

    def forward(self, x):
        x = self.net(x)
        return x

class ResNetEncoder(nn.Module):
    def __init__(self, hidden_size, architecture_name,
            pretrained=False, progress=False, **kwargs):
        super(ResNetEncoder, self).__init__()

        architecture = getattr(torchvision.models, architecture_name)
        self.net = architecture(pretrained, progress, **kwargs)

        self.conv1 = nn.Conv2d(self.net.fc.in_features, hidden_size, 1)

        self.net.fc = nn.Sequential()

        self.embedding_size = hidden_size

    def forward(self, x):
        return self._forward_impl(x)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.conv1(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class GruEncoder(nn.Module):
    def __init__(self, hidden_size, vocabulary_size, token_embedding_size,
            num_layers=1, bidirectional=True, aggregation='sum'):
        super(GruEncoder, self).__init__()

        self.token_embedding = nn.Embedding(
                vocabulary_size, token_embedding_size)
        token_embedding_std = math.sqrt(1./token_embedding_size)
        nn.init.normal_(self.token_embedding.weight,
                mean=1., std=token_embedding_std)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.aggregation = aggregation

        self.gru = nn.GRU(token_embedding_size, self.hidden_size, 
                self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        
        self.embedding_size = hidden_size * (1 + 
                self.bidirectional * (self.aggregation=='cat'))

    def forward(self, x, lengths):
        batch_size = lengths.size(0)
        x = self.token_embedding(x)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), 
                batch_first=True, enforce_sorted=True)

        x, h = self.gru(x)

        # Get the last output of the RNN for each direction.
        h = h.view(
                self.num_layers, 1+self.bidirectional, 
                batch_size, self.hidden_size
            )[-1]\
            .transpose(0, 1)\
            .reshape(batch_size, (1+self.bidirectional)*self.hidden_size)

        if self.bidirectional and self.aggregation == 'sum':
            h = h.view(batch_size, -1, self.hidden_size)
            h = h[:,0,:] + h[:,1,:]

        return h

class L2Normalization(nn.Module):
    def __init__(self, dim=1):
        super(L2Normalization, self).__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.matmul(x1, x2.t())

class VSE(pl.LightningModule):
    def __init__(self, embedding_size, 
            image_encoder_cfg, text_encoder_cfg,
            loss_cfg,
            optimizer_cfg, scheduler_cfg,
            modality_normalization: bool = True, # normalize encoder features
            joint_normalization: bool = True, # normalize projection features
            single_positive: bool = False,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = factories.VisionModelFactory.create(
                image_encoder_cfg['name'],
                embedding_size,
                **image_encoder_cfg['kwargs'])
        self.image_projection = nn.Linear(
                self.image_encoder.embedding_size, embedding_size)

        self.text_encoder = factories.TextModelFactory.create(
                text_encoder_cfg['name'],
                embedding_size,
                **text_encoder_cfg['kwargs'])

        self.modality_normalization = modality_normalization
        self.joint_normalization = joint_normalization

        self.normalization = L2Normalization(dim=1)

        self.similarity = DotProduct()
        self.loss = triplet_margin_loss.HardestTripletMarginLoss(
                margin=loss_cfg['margin'],
                hardest_fraction=loss_cfg['hardest_fraction']) 

        ks = [1, 5, 10]
        query_target_strs = ['image2text', 'text2image']

        batch_metrics = torchmetrics.MetricCollection({
            "MeanCrossRecall@1": rm.CrossmodalRecallAtK(1, 1)
        })
        self.training_batch_metrics = batch_metrics.clone(prefix="train/")
        self.validation_batch_metrics = batch_metrics.clone(prefix="val/")

        self.validation_binary_metrics = nn.ModuleDict({
            qt: torchmetrics.MetricCollection({
                f"{qt}_Retrieval@{k}": rm.RetrievalAtK(k)
                for k in ks
            }, prefix='val/')
            for qt in query_target_strs})
        self.validation_ratio_metrics = nn.ModuleDict({
            qt: torchmetrics.MetricCollection({
                **{
                    f"{qt}_Recall@{k}": rm.RecallAtK(k)
                    for k in ks
                },
                f"{qt}_MeanRank": rm.MeanRank(),
            }, prefix='val/')
            for qt in query_target_strs
        })

        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg

        self.profiler = pl.profiler.PassThroughProfiler()

    def forward(self, images, texts, lengths, **kwargs):
        image_embeddings = None
        if images is not None:
            with self.profiler.profile('image_forward'):
                image_embeddings = self.image_encoder(images)
                if self.modality_normalization:
                    image_embeddings = self.normalization(image_embeddings)
                #image_embeddings = self.image_projection(image_embeddings)
                if self.joint_normalization:
                    image_embeddings = self.normalization(image_embeddings)

        text_embeddings = None
        if texts is not None:
            with self.profiler.profile('text_forward'):
                text_embeddings = self.text_encoder(texts, lengths)
                if self.joint_normalization:
                    text_embeddings = self.normalization(text_embeddings)

        return image_embeddings, text_embeddings

    def _shared_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        lengths = batch['lengths']
        positive_pairs = batch['positive_pairs']

        images, texts = self(images, texts, lengths)

        return {
            'images': images,
            'texts': texts,
            'positive_pairs': positive_pairs,
        }

    def _shared_step_end(self, step_outputs, metrics=None):
        images = step_outputs['images']
        texts = step_outputs['texts']
        positive_pairs = step_outputs['positive_pairs']

        scores = self.similarity(images, texts)

        targets, indices_images, indices_texts = \
                rm.positive_sparse2dense(
                    positive_pairs, [scores.size(0), scores.size(1)])
            
        loss = self.loss(scores, targets)

        metric_values = {}
        if metrics is not None:
            metric_values = metrics(scores, targets)

        return loss, metric_values

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def training_step_end(self, training_step_outputs):
        loss, metric_values = self._shared_step_end(
            training_step_outputs, self.training_batch_metrics)

        self.log('train/loss', loss, on_step=True, on_epoch=False)
        self.log_dict(metric_values, on_step=True, on_epoch=False)

        return {
            'loss': loss, 
            **metric_values
        }

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)

    def validation_step_end(self, validation_step_outputs):
        loss, batch_metric_values = self._shared_step_end(
            validation_step_outputs, self.validation_batch_metrics)

        self.log('val/loss', loss, on_epoch=True)
        self.log_dict(batch_metric_values, on_epoch=True)

        return validation_step_outputs

    def validation_epoch_end(self, validation_step_outputs):
        i2t_metric_values = evaluation.batch_retrieval(
            validation_step_outputs,
            source_key='images', target_key='texts', 
            swap_positive_pairs=False,
            score_fn=self.similarity, 
            metrics={
                'binary': self.validation_binary_metrics['image2text'],
                'ratio': self.validation_ratio_metrics['image2text'],
            }
        )
        t2i_metric_values = evaluation.batch_retrieval(
            validation_step_outputs,
            source_key='texts', target_key='images', 
            swap_positive_pairs=True,
            score_fn=self.similarity, 
            metrics={
                'binary': self.validation_binary_metrics['text2image'],
                'ratio': self.validation_ratio_metrics['text2image'],
            }
        )

        self.log_dict(i2t_metric_values['binary'], on_epoch=True)
        self.log_dict(i2t_metric_values['ratio'], on_epoch=True)
        self.log_dict(t2i_metric_values['binary'], on_epoch=True)
        self.log_dict(t2i_metric_values['ratio'], on_epoch=True)

    def configure_optimizers(self):
        optimizer = factories.OptimizerFactory.create(
                self._optimizer_cfg['name'], 
                self.parameters(),
                **self._optimizer_cfg['kwargs'])

        scheduler_cfgs = self._scheduler_cfg
        if not isinstance(self._scheduler_cfg, list):
            scheduler_cfgs = [scheduler_cfgs]

        schedulers = []
        for scheduler_cfg in scheduler_cfgs:
            scheduler_dict = {
                'scheduler': factories.SchedulerFactory.create(
                    scheduler_cfg['name'],
                    optimizer,
                    **scheduler_cfg['kwargs']),
                **scheduler_cfg['pl_kwargs']
            }
            schedulers.append(scheduler_dict)

        return [optimizer], schedulers

class Unfreezing(pl.callbacks.Callback):
    """
        Freezes the given modules on training start, and unfreezes them 
        once their unfreezing epoch is reached.
    """
    def __init__(self, milestones: Dict[str, int], verbose: bool = False):
        super().__init__()
        self.milestones = milestones
        self.verbose = verbose

    def _get_object(self, pl_module, attr_str):
        attrs = attr_str.split('.')
        current_object = pl_module
        for attr in attrs:
            current_object = getattr(current_object, attr)
        return current_object

    def on_train_start(self, trainer, pl_module):
        for attr in self.milestones:
            self._get_object(pl_module, attr).requires_grad_(False)
            if self.verbose:
                print(f"Freezing {attr}")

    def on_train_epoch_start(self, trainer, pl_module):
        for attr, epoch in self.milestones.items():
            if epoch == trainer.current_epoch:
                self._get_object(pl_module, attr).requires_grad_(True)
                if self.verbose:
                    print(f"Unfreezing {attr}")


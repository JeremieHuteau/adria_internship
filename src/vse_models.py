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
import retrieval_metrics

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

        self.embedding_size = self.net.fc.in_features
        self.net.fc = nn.Sequential()

    def forward(self, x):
        x = self.net(x)
        return x

class GruEncoder(nn.Module):
    def __init__(self, hidden_size, vocabulary_size, token_embedding_size,
            num_layers=1, bidirectional=True):
        super(GruEncoder, self).__init__()

        self.token_embedding = nn.Embedding(
                vocabulary_size, token_embedding_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(token_embedding_size, self.hidden_size, 
                self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        
        self.embedding_size = hidden_size * (1+self.bidirectional)

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

class HardestFractionDecay(pl.callbacks.Callback):
    _schedules = ['cosine', 'linear']
    def __init__(self, 
            total_steps: int, 
            min_fraction: float = 0.0,
            schedule: str = 'cosine',
        ):
        super().__init__()
        self.step_count = 0
        self.total_steps = total_steps
        self.min_fraction = min_fraction

        if schedule not in self._schedules:
            raise ValueError(f"schedule must be one of {self._schedules}")
        self.schedule = schedule

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, 
            batch_idx, dataloader_idx):
        if self.step_count > self.total_steps:
            return
        if self.step_count == 0:
            self.initial_hardest_fraction = pl_module.loss.hardest_fraction
        self.step_count += 1

        steps_fraction = self.step_count / self.total_steps

        if self.schedule == 'linear':
            decay = 1 - steps_fraction
        if self.schedule == 'cosine':
            decay = math.cos((math.pi / 2) * steps_fraction)

        new_fraction = max(self.min_fraction, 
                self.initial_hardest_fraction * decay)
        pl_module.loss.hardest_fraction = new_fraction

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
        self.text_projection = nn.Linear(
                self.text_encoder.embedding_size, embedding_size)

        self.modality_normalization = modality_normalization
        self.joint_normalization = joint_normalization

        self.normalization = L2Normalization(dim=1)

        self.similarity = DotProduct()

        self.loss = retrieval_metrics.HardestTripletMarginLoss(
                margin=loss_cfg['margin'],
                hardest_fraction=loss_cfg['hardest_fraction'], 
                single_positive=single_positive)
        
        k = 1
        self.r_at_k_str = f'R@{k}'
        batch_metrics = torchmetrics.MetricCollection({
                self.r_at_k_str: retrieval_metrics.CrossmodalRecallAtK(
                    k, k, matrix_preds=single_positive)
                })
        self.train_metrics = batch_metrics.clone(prefix="train/")
        self.validation_batch_metrics = batch_metrics.clone(prefix="val/")

        image_metrics = torchmetrics.MetricCollection({
                f'I2T_{self.r_at_k_str}': retrieval_metrics.RecallAtK(k),
                })
        text_metrics = torchmetrics.MetricCollection({
                f'T2I_{self.r_at_k_str}': retrieval_metrics.RecallAtK(k),
                })
        self.validation_image_metrics = image_metrics.clone(prefix="val/")
        self.validation_text_metrics = text_metrics.clone(prefix="val/")

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
                image_embeddings = self.image_projection(image_embeddings)
                if self.joint_normalization:
                    image_embeddings = self.normalization(image_embeddings)

        text_embeddings = None
        if texts is not None:
            with self.profiler.profile('text_forward'):
                text_embeddings = self.text_encoder(texts, lengths)
                if self.modality_normalization:
                    text_embeddings = self.normalization(text_embeddings)
                text_embeddings = self.text_projection(text_embeddings)
                if self.joint_normalization:
                    text_embeddings = self.normalization(text_embeddings)

        return image_embeddings, text_embeddings

    def _shared_step(self, batch, batch_idx, metrics=None):
        with self.profiler.profile('shared_step'):
            images = batch['images']
            texts = batch['texts']
            lengths = batch['lengths']

            with self.profiler.profile('forward'):
                images, texts = self(images, texts, lengths)
            with self.profiler.profile('similarity'):
                scores = self.similarity(images, texts)

            positive_pairs = batch['positive_pairs']
            targets, indices_images, indices_texts = \
                    retrieval_metrics.positive_sparse2dense(
                        positive_pairs, [scores.size(0), scores.size(1)])

            with self.profiler.profile('loss'):
                loss = self.loss(scores, targets, indices_images, indices_texts)
            with self.profiler.profile('recall'):
                metric_values = {}
                if metrics is not None:
                    metric_values = metrics(scores, targets, indices_images, indices_texts)


        return loss, metric_values

    def training_step(self, batch, batch_idx):
        loss, metric_values = self._shared_step(
                batch, batch_idx, self.train_metrics)
        
        self.log('train/loss', loss)
        self.log_dict(metric_values)

        self.log('loss.hardest_fraction', self.loss.hardest_fraction)

        return {
            'loss': loss, 
            **metric_values
        }

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        lengths = batch['lengths']

        images, texts = self(images, texts, lengths)
        scores = self.similarity(images, texts)

        positive_pairs = batch['positive_pairs']
        targets, indices_images, indices_texts = \
                retrieval_metrics.positive_sparse2dense(
                    positive_pairs, [scores.size(0), scores.size(1)])

        loss = self.loss(scores, targets, indices_images, indices_texts)
        batch_metric_values = self.validation_batch_metrics(
                scores, targets, indices_images, indices_texts)
        self.log('val/loss', loss, on_epoch=True)
        self.log_dict(batch_metric_values, on_epoch=True)

        embedded_batch = {
            'images': images,
            'texts': texts,
            'positive_pairs': positive_pairs,
        }

        return embedded_batch

    def validation_epoch_end(self, validation_step_outputs):
        embedded_batches = validation_step_outputs

        i2t_metric_values = evaluation.batch_retrieval(
            embedded_batches,
            source_key='images', target_key='texts', 
            swap_positive_pairs=False,
            score_fn=self.similarity, 
            metrics=self.validation_image_metrics
        )
        t2i_metric_values = evaluation.batch_retrieval(
            embedded_batches,
            source_key='texts', target_key='images', 
            swap_positive_pairs=True,
            score_fn=self.similarity, 
            metrics=self.validation_text_metrics
        )

        self.log_dict({
                **i2t_metric_values,
                **t2i_metric_values}, 
            on_epoch=True)


    def configure_optimizers(self):
        optimizer = factories.OptimizerFactory.create(
                self._optimizer_cfg['name'], 
                self.parameters(),
                **self._optimizer_cfg['kwargs'])
        scheduler = {
                'scheduler': factories.SchedulerFactory.create(
                    self._scheduler_cfg['name'],
                    optimizer,
                    **self._scheduler_cfg['kwargs']),
                **self._scheduler_cfg['pl_kwargs']}
        return [optimizer], [scheduler]

